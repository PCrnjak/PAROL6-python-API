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

from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.protocol.wire import JogJCmd, JogLCmd, ServoJCmd, ServoLCmd
from parol6.utils.error_codes import ErrorCode
from tests.integration.controller_loop import push, ready, tick, tick_until

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
        # J6 was still being driven while J1 ramped down, and stops only at
        # ITS limit or the timer — here it is still short of both.
        assert q[5] > start[5] + 0.1, "J6 stopped with J1 instead of carrying on"
        assert lo[5] < q[5] < hi[5]


def test_a_servo_stream_runs_at_its_speed_and_holds_when_its_client_goes_silent(
    controller,
):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True)
    start = _q_rad(state)
    target = np.degrees(start).tolist()
    target[0] += 40.0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # One datagram, then silence: the stream runs at 30% and brakes
        # after the grace instead of running on to a 40° target.
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
