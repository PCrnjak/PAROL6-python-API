"""What a jog or servo stream may do, driven through the controller's UDP
socket and its loop against the fake serial: an unhomed arm takes only
``jog_j``; a ``jog_j`` into a joint limit ramps that joint to rest short of
it while the others carry on; a servo stream runs at the speed it asked for
and brakes to a hold when its client goes silent."""

import math
import socket
import time

import numpy as np
import pytest

from parol6.config import HOME_ANGLES_DEG, INTERVAL_S, LIMITS, steps_to_rad
from parol6.protocol.wire import (
    JogJCmd,
    JogLCmd,
    OkMsg,
    ServoJCmd,
    ServoLCmd,
    SetShapesCmd,
    ShapeWire,
)
from parol6.server.state import get_fkine_se3
from parol6.utils.error_codes import ErrorCode
from pinokin import se3_rpy
from tests.integration.controller_loop import push, ready, send, tick, tick_until
from waldoctl import Box

pytestmark = pytest.mark.integration


def _q_rad(state) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_rad(state.Position_in, out)
    return out


def test_an_unhomed_arm_takes_a_joint_jog_and_nothing_else(controller):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=False)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for cmd in (
            JogLCmd(velocities=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.0),
            ServoJCmd(angles=[10.0, -90.0, 180.0, 0.0, 0.0, 180.0]),
            ServoLCmd(pose=[200.0, 0.0, 200.0, 180.0, 0.0, 180.0]),
        ):
            before = state.Position_in.copy()
            push(controller, sock, cmd)
            tick_until(
                controller,
                state,
                lambda: state.error is not None,
                f"{type(cmd).__name__} was not refused unhomed",
            )
            assert state.error is not None
            assert state.error.code == int(ErrorCode.MOTN_NOT_HOMED), state.error
            for _ in range(20):
                tick(controller, state)
            assert np.array_equal(state.Position_in, before), (
                "the refused stream moved the arm"
            )
            state.error = None


def test_a_joint_jog_stops_short_of_the_limit_one_joint_at_a_time(controller):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True)
    lo, hi = LIMITS.joint.position.rad[:, 0], LIMITS.joint.position.rad[:, 1]
    start = _q_rad(state)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # J1 and J6 together, J1 towards its limit; the jog outlives the
        # travel to it. (The timer is wall-clock; the loop runs faster.)
        push(
            controller,
            sock,
            JogJCmd(speeds=[1.0, 0.0, 0.0, 0.0, 0.0, 0.3], duration=30.0),
        )
        deadline = time.monotonic() + 20.0
        still = 0
        last = start.copy()
        while time.monotonic() < deadline:
            tick(controller, state)
            q = _q_rad(state)
            if np.allclose(q, last, atol=1e-6):
                still += 1
                if still >= 20 and q[0] > start[0] + 0.1:
                    break
            else:
                still = 0
            last = q
        else:
            pytest.fail("J1 never came to rest against its limit")
        q = _q_rad(state)
        assert q[0] < hi[0], f"J1 ran into its limit: {q[0]:.4f} >= {hi[0]:.4f}"
        assert q[0] > hi[0] - 0.1, (
            f"J1 stopped {math.degrees(hi[0] - q[0]):.1f}° short of its limit"
        )
        # J6 was still being driven while J1 ramped down, and came to rest
        # only against ITS limit: a jog that stopped whole with J1 would
        # leave it far short.
        assert hi[5] - 0.1 < q[5] < hi[5], (
            f"J6 stopped {math.degrees(hi[5] - q[5]):.1f}° short of its own "
            "limit: it stopped with J1 instead of carrying on"
        )
        assert q[5] > lo[5]


def test_a_joint_joining_a_streamed_jog_does_not_carry_another_past_its_limit(
    controller,
):
    """A jog streamed as the UI streams it — one datagram every other tick —
    with J6 joining just as J1 nears its limit: J1's brake is its own, not
    stretched to finish with J6's ramp, and each datagram continues the
    motion the lookahead has measured instead of restarting it from rest."""
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True)
    hi = LIMITS.joint.position.rad[:, 1]
    peak = _q_rad(state)[0]
    joined = False
    still = 0
    last = _q_rad(state)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for i in range(3000):
            if i % 2 == 0:
                speeds = [0.3, 0.0, 0.0, 0.0, 0.0, 1.0 if joined else 0.0]
                push(controller, sock, JogJCmd(speeds=speeds, duration=0.5))
            tick(controller, state)
            q = _q_rad(state)
            peak = max(peak, q[0])
            if not joined and q[0] >= hi[0] - 0.09:
                joined = True
            if joined and abs(q[0] - last[0]) < 1e-7:
                still += 1
                if still >= 50:
                    break
            else:
                still = 0
            last = q
        else:
            pytest.fail("J1 never came to rest against its limit")
    assert joined, "J1 never approached its limit"
    assert peak < hi[0], (
        f"J1 ran into its limit when J6 joined: peak {peak:.4f} >= {hi[0]:.4f}"
    )


def test_a_servo_stream_runs_at_its_speed_and_holds_when_its_client_goes_silent(
    controller,
):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=list(HOME_ANGLES_DEG))
    start = _q_rad(state)
    target = np.degrees(start).tolist()
    target[0] += 20.0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # One datagram, then silence: the stream runs at 30% and brakes
        # after the grace instead of running on to a 20° target.
        push(controller, sock, ServoJCmd(angles=target, speed=0.3))
        peak = 0.0
        prev = start.copy()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            tick(controller, state)
            q = _q_rad(state)
            peak = max(peak, abs(q[0] - prev[0]) / INTERVAL_S)
            prev = q
            time.sleep(INTERVAL_S)
        assert peak <= LIMITS.joint.hard.velocity[0] * 0.3 * 1.05, (
            f"J1 ran at {peak:.3f} rad/s against a 30% ceiling of "
            f"{LIMITS.joint.hard.velocity[0] * 0.3:.3f} rad/s"
        )
        assert peak > 0.01, "the servo stream never moved the arm"
        end = _q_rad(state)
        assert start[0] + 0.02 < end[0] < math.radians(target[0]) - 0.02, (
            f"the silent stream did not brake short of its target: {math.degrees(end[0]):.2f}° "
            f"of {target[0]:.2f}°"
        )
        assert controller._executor.active_command is None, (
            "the stream did not end at the hold"
        )


def test_a_jog_l_braked_short_of_a_keep_out_ends_in_error(controller):
    """A cartesian jog heading into a keep-out brakes to rest short of it and
    ends FAILED, with the collision latched as the error ``error()`` reads,
    like a refused planned move: it neither runs on nor ends silently."""
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=[0.0, -90.0, 180.0, 0.0, 0.0, 180.0])
    # A slab 10 cm above the wrist, straight up the jog's path.
    slab = Box(name="slab", x=0.10, y=0.10, z=0.04, pose=(0.237, 0.0, 0.43, 0, 0, 0))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        reply = send(
            controller,
            state,
            sock,
            SetShapesCmd(shapes=[ShapeWire(*slab.to_wire())]),
            1,
        )
        assert isinstance(reply, OkMsg), reply
        up = JogLCmd(velocities=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0], duration=0.5)
        for i in range(1500):
            if i % 2 == 0:
                push(controller, sock, up)
            tick(controller, state)
            if state.error is not None:
                break
        else:
            pytest.fail("the jog never stopped at the keep-out")
        assert state.error.code == int(ErrorCode.SYS_SELF_COLLISION), state.error
        assert "slab" in state.error.cause, state.error.cause
        assert controller._executor.active_command is None
        held = state.Position_in.copy()
        for _ in range(50):
            tick(controller, state)
            assert state.error is not None, "the collision error did not latch"
        assert np.abs(state.Position_in - held).max() <= 1, (
            "the arm moved on after the jog ended"
        )


def test_a_servo_l_stream_through_an_unreachable_pose_resumes(controller):
    """A servo_l stream that asks for a pose the solver cannot reach brakes
    and holds; once the stream moves on to a pose it can reach, it tracks
    that one — the stream is not over, and nothing is reported failed."""
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=[0.0, -90.0, 180.0, 0.0, 0.0, 180.0])
    tcp = get_fkine_se3(state)
    rpy = np.zeros(3)
    se3_rpy(tcp, rpy)
    start = [*(tcp[:3, 3] * 1000.0).tolist(), *np.degrees(rpy).tolist()]
    out_of_reach = list(start)
    out_of_reach[0] += 600.0
    target = list(start)
    target[0] += 40.0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for pose, ticks in ((out_of_reach, 200), (target, 400)):
            for i in range(ticks):
                if i % 2 == 0:
                    push(controller, sock, ServoLCmd(pose=pose))
                tick(controller, state)
                assert state.error is None, f"the stream failed: {state.error}"
    reached = get_fkine_se3(state)[:3, 3] * 1000.0
    assert np.linalg.norm(reached - np.asarray(target[:3])) < 1.0, (
        f"the stream did not resume to {target[:3]}: it holds at {reached}"
    )
