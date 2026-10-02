"""Streams driven through the controller's UDP socket and its loop against
the fake serial, past the points where one stream hands over to the next:
a stream something else ended leaves nothing behind for the next one; a
joint jog stays inside its limits and leaves them again; a servo stream
holds each joint to its hardware speed, keeps out of keep-outs, turns the
way its targets turn and resumes when its client does; a cartesian jog
moves in the frame it is given as the tool now stands; and a timed jog
lasts its duration however many periods the loop drops."""

import math
import socket

import numpy as np
import pytest

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.servo_commands import SERVO_GRACE_S
from parol6.config import HOME_ANGLES_DEG, INTERVAL_S, LIMITS, steps_to_rad
from parol6.protocol.wire import (
    CommandCode,
    JogJCmd,
    JogLCmd,
    OkMsg,
    ServoJCmd,
    ServoLCmd,
    SetShapesCmd,
    ShapeWire,
    StopCmd,
    TeleportCmd,
)
from parol6.server.state import get_fkine_se3
from parol6.utils.error_codes import ErrorCode
from pinokin import se3_rpy
from tests.integration.controller_loop import (
    VirtualClock,
    drain,
    push,
    ready,
    send,
    tick,
)
from waldoctl import Box

pytestmark = pytest.mark.integration

STANDBY = [float(v) for v in HOME_ANGLES_DEG]
CLEAR_OF_THE_WRIST = [90.0, -80.0, 190.0, 0.0, 30.0, 180.0]


def _measured(state) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_rad(state.Position_in, out)
    return out


def _commanded(state) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_rad(state.Position_out, out)
    return out


def _wire_pose(pose: np.ndarray) -> list[float]:
    rpy = np.zeros(3)
    se3_rpy(pose, rpy)
    return [*(pose[:3, 3] * 1000.0).tolist(), *np.degrees(rpy).tolist()]


def _turned_deg(before: np.ndarray, after: np.ndarray) -> float:
    cos = (np.trace(before.T @ after) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, cos))))


def _stream(controller, state, clock, sock, cmd, ticks: int, until=None) -> bool:
    """Send *cmd* every other tick, as a UI streams it, for *ticks* ticks or
    until *until* holds; whether it did."""
    for i in range(ticks):
        if i % 2 == 0:
            push(controller, sock, cmd)
        clock.tick(controller, state)
        if until is not None and until():
            return True
    return False


def _settle(controller, state, clock) -> None:
    """Tick until the stream in flight has run out and ended."""
    for _ in range(round(5.0 / INTERVAL_S)):
        clock.tick(controller, state)
        if controller._executor.active_command is None:
            return
    pytest.fail("the stream never ended")


def test_a_joint_stream_cut_off_by_a_stop_or_a_teleport_restarts_from_the_arm(
    controller, monkeypatch
):
    """Whatever ends a joint stream, the next one starts from the arm at
    rest: a jog turned round after a stop does not first run on the way the
    stopped one was going, and a jog or servo stream after a teleport does
    not drag the arm back towards where the teleport took it from."""
    state = controller.state_manager.get_state()
    clock = VirtualClock(monkeypatch)
    away = JogJCmd(speeds=[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=0.5)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)

        def jog_away() -> None:
            ready(controller, state, homed=True, at_deg=STANDBY)
            assert _stream(
                controller,
                state,
                clock,
                sock,
                away,
                300,
                until=lambda: math.degrees(_measured(state)[0]) <= STANDBY[0] - 20.0,
            ), "the jog never moved J1"

        jog_away()
        reply = send(controller, state, sock, StopCmd(), 1)
        assert isinstance(reply, OkMsg), reply
        stopped = math.degrees(_measured(state)[0])
        lowest = stopped
        back = JogJCmd(speeds=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=0.5)
        for i in range(40):
            if i % 2 == 0:
                push(controller, sock, back)
            clock.tick(controller, state)
            if state.Command_out == CommandCode.MOVE:
                lowest = min(lowest, math.degrees(_commanded(state)[0]))
        assert lowest > stopped - 0.2, (
            f"the jog turned round after the stop first ran J1 "
            f"{stopped - lowest:.1f}° further the way the stopped jog was going"
        )
        _settle(controller, state, clock)

        for req_id, follow in enumerate(
            (
                JogJCmd(speeds=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration=0.5),
                ServoJCmd(angles=[STANDBY[0] + 5.0, *STANDBY[1:]]),
            ),
            start=2,
        ):
            jog_away()
            reply = send(controller, state, sock, TeleportCmd(angles=STANDBY), req_id)
            assert isinstance(reply, OkMsg), reply
            lowest = STANDBY[0]
            for i in range(30):
                if i % 2 == 0:
                    push(controller, sock, follow)
                clock.tick(controller, state)
                if state.Command_out == CommandCode.MOVE:
                    lowest = min(lowest, math.degrees(_commanded(state)[0]))
            assert lowest > STANDBY[0] - 0.2, (
                f"the {type(follow).__name__} after the teleport commanded J1 back "
                f"to {lowest:.1f}°, towards where the teleport took the arm from"
            )
            _settle(controller, state, clock)


def test_a_joint_backed_off_its_limit_jogs_towards_it_again(controller, monkeypatch):
    """A joint a jog stopped at its limit is held there only while the jog
    pushes into it: backed off and jogged towards the limit again within
    the same stream, which J6 keeps going, it follows the jog again rather
    than staying frozen until the stream ends."""
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=STANDBY)
    clock = VirtualClock(monkeypatch)
    hi = LIMITS.joint.position.rad[0, 1]

    def jog(j1: float) -> JogJCmd:
        return JogJCmd(speeds=[j1, 0.0, 0.0, 0.0, 0.0, 0.3], duration=0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        last = _measured(state)[0]
        still = 0
        for i in range(600):
            if i % 2 == 0:
                push(controller, sock, jog(1.0))
            clock.tick(controller, state)
            q = _measured(state)[0]
            still = still + 1 if abs(q - last) < 1e-7 else 0
            last = q
            if still >= 10 and q > hi - 0.1:
                break
        else:
            pytest.fail("J1 never came to rest at its limit")
        assert _stream(
            controller,
            state,
            clock,
            sock,
            jog(-1.0),
            300,
            until=lambda: _measured(state)[0] < hi - 0.5,
        ), "J1 never backed off its limit"
        low = _measured(state)[0]
        for i in range(100):
            if i % 2 == 0:
                push(controller, sock, jog(1.0))
            clock.tick(controller, state)
            low = min(low, _measured(state)[0])
        rise = _measured(state)[0] - low
    assert rise > 0.1, (
        f"jogged towards its limit again, J1 rose {math.degrees(rise):.2f}° and "
        f"stayed {math.degrees(hi - low - rise):.1f}° short of it"
    )


def test_a_joint_jog_braking_for_its_limit_stays_inside_it_when_its_accel_drops(
    controller, monkeypatch
):
    """The next datagram of a jog braking for a limit may carry a lower
    accel (a jog-accel slider moved, a script's next jog_j at its default);
    the brake still ends inside the limit. The commanded position is what
    is checked: the fake serial clamps the measured one at the limit."""
    state = controller.state_manager.get_state()
    clock = VirtualClock(monkeypatch)
    hi = LIMITS.joint.position.rad[0, 1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for accel in (0.5, 0.3):
            ready(controller, state, homed=True, at_deg=[-60.0, *STANDBY[1:]])
            peak = -math.inf
            top = 0.0
            braking = False
            prev = None
            for i in range(round(3.5 / INTERVAL_S)):
                if i % 2 == 0:
                    push(
                        controller,
                        sock,
                        JogJCmd(
                            speeds=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            duration=0.5,
                            accel=accel if braking else 1.0,
                        ),
                    )
                clock.tick(controller, state)
                if state.Command_out != CommandCode.MOVE:
                    prev = None
                    continue
                q = _commanded(state)[0]
                peak = max(peak, q)
                if prev is not None:
                    speed = (q - prev) / INTERVAL_S
                    top = max(top, speed)
                    braking = braking or (top > 1.0 and speed < top - 0.05)
                prev = q
            assert braking, "J1 never braked for its limit"
            assert peak <= hi + 1e-3, (
                f"with its accel dropped to {accel} as it braked, J1 was commanded "
                f"{math.degrees(peak - hi):.2f}° past its limit"
            )
            _settle(controller, state, clock)


def test_a_servo_l_braking_as_its_client_goes_silent_keeps_each_joint_to_its_speed(
    controller, monkeypatch
):
    """Close to the wrist singularity a tilt out of the arm's plane asks J4
    and J6 for a large, fast swing; the stream moves each joint no further
    per tick than its hardware allows, so the commanded joints lag the
    solution. When the client goes silent the stream brakes, and the brake
    is held to the same limit: it does not close that lag in one tick."""
    state = controller.state_manager.get_state()
    tilted = np.zeros((4, 4), dtype=np.float64, order="F")
    PAROL6_ROBOT.robot.fkine_into(
        np.radians([90.0, -90.0, 180.0, 60.0, 20.0, 120.0]), tilted
    )
    ready(controller, state, homed=True, at_deg=[90.0, -90.0, 180.0, 0.0, 2.0, 180.0])
    clock = VirtualClock(monkeypatch)
    limit = LIMITS.joint.hard.velocity_steps * INTERVAL_S
    worst = np.zeros(6, dtype=np.int64)
    prev = state.Position_in.astype(np.int64)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for i in range(round(1.5 / INTERVAL_S)):
            if i < 6 and i % 2 == 0:
                push(controller, sock, ServoLCmd(pose=_wire_pose(tilted)))
            clock.tick(controller, state)
            if state.Command_out == CommandCode.MOVE:
                out = state.Position_out.astype(np.int64)
                np.maximum(worst, np.abs(out - prev), out=worst)
                prev = out
    assert controller._executor.active_command is None, (
        "the silent stream never braked to its end"
    )
    assert worst[3] > 0.5 * limit[3], (
        "the tilt never swung J4 at its speed limit: the stream never lagged"
    )
    for j in range(6):
        assert worst[j] <= limit[j] + 1, (
            f"J{j + 1} was commanded {worst[j]} steps in one tick; its hardware "
            f"allows {limit[j]:.0f}"
        )


def test_a_servo_l_stream_resending_its_target_after_going_silent_carries_on_to_it(
    controller, monkeypatch
):
    """A client that goes silent past the grace and comes back resending the
    target it was on resumes its stream: the brake gives way and the tool
    goes on to the target, rather than stopping short and ending there."""
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=STANDBY)
    clock = VirtualClock(monkeypatch)
    goal = get_fkine_se3(state).copy()
    goal[2, 3] -= 0.1
    target = ServoLCmd(pose=_wire_pose(goal))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        _stream(controller, state, clock, sock, target, 20)
        for _ in range(round(SERVO_GRACE_S / INTERVAL_S) + 3):
            clock.tick(controller, state)
        for i in range(600):
            if i % 5 == 0:
                push(controller, sock, target)
            clock.tick(controller, state)
            short = np.linalg.norm(get_fkine_se3(state)[:3, 3] - goal[:3, 3]) * 1000.0
            if short < 1.0:
                break
            assert controller._executor.active_command is not None, (
                f"the stream ended {short:.1f} mm short of the target its client "
                "kept resending"
            )
        else:
            pytest.fail("the resumed stream never reached its target")


def test_a_servo_stream_stops_short_of_a_keep_out(controller, monkeypatch):
    """A servo stream, joint or cartesian, heading into a keep-out stops the
    arm short of it with the collision standing as the error, as a jog
    does: it does not drive the arm in."""
    state = controller.state_manager.get_state()
    checker = PAROL6_ROBOT.collision
    assert checker is not None
    # A slab under the wrist at standby, where both streams head.
    slab = Box(name="slab", x=0.10, y=0.10, z=0.04, pose=(0.0, 0.237, 0.23, 0, 0, 0))

    def in_slab(q: np.ndarray) -> bool:
        return any(
            "slab" in name for pair in checker.colliding_pairs(q) for name in pair
        )

    clock = VirtualClock(monkeypatch)
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
        ready(controller, state, homed=True, at_deg=STANDBY)
        assert not in_slab(_measured(state)), "the arm starts inside the slab"
        below = get_fkine_se3(state).copy()
        below[2, 3] -= 0.104
        for cmd in (
            ServoJCmd(angles=[90.0, -90.0, 150.0, 0.0, 0.0, 180.0]),
            ServoLCmd(pose=_wire_pose(below)),
        ):
            name = type(cmd).__name__
            ready(controller, state, homed=True, at_deg=STANDBY)
            refused = None
            for i in range(300):
                if i % 2 == 0:
                    push(controller, sock, cmd)
                clock.tick(controller, state)
                assert not in_slab(_measured(state)), (
                    f"the {name} stream drove the arm into the slab"
                )
                if state.error is not None and state.error.code == int(
                    ErrorCode.SYS_SELF_COLLISION
                ):
                    refused = state.error
            assert refused is not None, f"the {name} stream was never stopped"
            assert "slab" in refused.cause, refused.cause
            _settle(controller, state, clock)


def test_a_jog_l_moves_in_the_frame_it_is_given_as_the_tool_now_stands(
    controller, monkeypatch
):
    """The axis switches within one stream, as a UI streams them: after the
    tool has turned about world X, a world-Y jog turns it about world Y;
    after it has turned about its own Z, a tool-X jog moves it along its X
    as it now stands, not as it stood when the stream began."""
    # The tool's Z lies along world X, so turning about either is J6 alone.
    begin = [0.0, -60.0, 190.0, 0.0, 20.0, 180.0]
    state = controller.state_manager.get_state()
    clock = VirtualClock(monkeypatch)
    rest = [0.0] * 6
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for frame, turn, then in (
            ("WRF", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
            ("TRF", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ):
            ready(controller, state, homed=True, at_deg=begin)
            start = get_fkine_se3(state)[:3, :3].copy()
            assert _stream(
                controller,
                state,
                clock,
                sock,
                JogLCmd(velocities=turn, duration=0.5, frame=frame),
                400,
                until=lambda: _turned_deg(start, get_fkine_se3(state)[:3, :3]) >= 75.0,
            ), f"the {frame} jog never turned the tool"
            # Brought to rest and set off again, all one stream.
            for velocities, ticks in ((rest, 50), (then, 30)):
                _stream(
                    controller,
                    state,
                    clock,
                    sock,
                    JogLCmd(velocities=velocities, duration=0.5, frame=frame),
                    ticks,
                )
            before = get_fkine_se3(state).copy()
            _stream(
                controller,
                state,
                clock,
                sock,
                JogLCmd(velocities=then, duration=0.5, frame=frame),
                30,
            )
            after = get_fkine_se3(state).copy()
            if frame == "WRF":
                turned = _turned_deg(before[:3, :3], after[:3, :3])
                assert turned > 5.0, f"the world-Y jog turned the tool {turned:.1f}°"
                d = after[:3, :3] @ before[:3, :3].T
                axis = np.array(
                    [d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]
                )
                axis /= np.linalg.norm(axis)
                off = math.degrees(math.acos(min(1.0, abs(axis[1]))))
                assert off < 5.0, (
                    f"the world-Y jog turned the tool about an axis {off:.1f}° off "
                    f"world Y: {np.round(axis, 3)}"
                )
            else:
                moved = after[:3, 3] - before[:3, 3]
                length = float(np.linalg.norm(moved))
                assert length > 0.005, (
                    f"the tool-X jog moved the tool {length * 1000.0:.1f} mm"
                )
                cos = float(np.dot(moved, after[:3, 0])) / length
                off = math.degrees(math.acos(min(1.0, max(-1.0, cos))))
                assert off < 5.0, (
                    f"the tool-X jog moved the tool {off:.1f}° off its X axis"
                )
            _settle(controller, state, clock)


def test_a_servo_l_stream_turning_the_tool_past_half_a_turn_keeps_turning(
    controller, monkeypatch
):
    """A servo stream whose targets turn the tool steadily about its own Z
    keeps turning it the same way once it is more than half a turn from
    where the stream began, as J6's travel allows, rather than swinging it
    back the other way through the start."""
    begin = [90.0, -60.0, 190.0, 0.0, 20.0, 10.0]
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=begin)
    clock = VirtualClock(monkeypatch)
    start = get_fkine_se3(state).copy()
    turn = np.eye(4)
    j6 = peak = begin[5]
    back = 0.0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # 1° a datagram, one every other tick; the last one held until the
        # tool is there.
        for k in range(280):
            a = math.radians(min(k, 220))
            turn[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
            push(controller, sock, ServoLCmd(pose=_wire_pose(start @ turn)))
            for _ in range(2):
                clock.tick(controller, state)
                j6 = math.degrees(_measured(state)[5])
                peak = max(peak, j6)
                back = max(back, peak - j6)
    assert back < 0.5, (
        f"the tool turned back {back:.1f}° after J6 reached {peak:.1f}°, "
        f"{peak - begin[5]:.1f}° from where the stream began"
    )
    assert abs(j6 - (begin[5] + 220.0)) < 1.0, (
        f"J6 ended at {j6:.1f}°, not the {begin[5] + 220.0:.1f}° the stream "
        "turned it to"
    )


def test_a_jog_l_after_a_cancelled_cartesian_stream_moves_the_arm(
    controller, monkeypatch
):
    """A cartesian jog following a cartesian stream something else ended,
    a stop, a teleport, or the jog itself cutting a servo stream off, starts
    from the arm and moves it; it does not end at once where it began."""
    state = controller.state_manager.get_state()
    clock = VirtualClock(monkeypatch)
    down = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        for req_id, cut_off_by in enumerate(("a stop", "a teleport", "the jog"), 1):
            ready(controller, state, homed=True, at_deg=CLEAR_OF_THE_WRIST)
            if cut_off_by == "the jog":
                goal = get_fkine_se3(state).copy()
                goal[2, 3] -= 0.06
                servo = ServoLCmd(pose=_wire_pose(goal))
                _stream(controller, state, clock, sock, servo, 20)
            else:
                jog = JogLCmd(velocities=down, duration=0.5)
                _stream(controller, state, clock, sock, jog, 30)
                cancel = (
                    StopCmd()
                    if cut_off_by == "a stop"
                    else TeleportCmd(angles=CLEAR_OF_THE_WRIST)
                )
                reply = send(controller, state, sock, cancel, req_id)
                assert isinstance(reply, OkMsg), reply
                tick(controller, state)
            before = get_fkine_se3(state)[2, 3]
            push(controller, sock, JogLCmd(velocities=down, duration=0.5))
            # Loopback can deliver the jog a tick or two late (macOS): read it
            # before waiting for the stream to end.
            drain(controller, state, sock, 100 + req_id)
            _settle(controller, state, clock)
            dropped = (before - get_fkine_se3(state)[2, 3]) * 1000.0
            assert dropped > 10.0, (
                f"after {cut_off_by} cut the stream off, a 0.5 s jog down moved the "
                f"tool {dropped:.1f} mm"
            )


def test_a_timed_jog_l_lasts_its_duration_on_a_loop_that_drops_periods(
    controller, monkeypatch
):
    """A loop that overruns skips the periods it missed, so wall time runs on
    while a jog advances one interval per tick. A timed jog still ends
    where its preview does: its duration is counted in the ticks that move
    the arm, not in the wall time the loop fell behind by."""
    from parol6.client.dry_run_client import DryRunRobotClient

    velocities = [1.0, -1.0, -1.0, 0.0, 0.0, 0.0]
    duration = 0.6
    preview = DryRunRobotClient(initial_joints_deg=CLEAR_OF_THE_WRIST)
    assert preview.jog_l(
        "WRF",
        axes=["X", "Y", "Z"],
        speeds_list=velocities[:3],
        duration=duration,
        accel=1.0,
    )
    assert preview.plan().blocks[0].error is None
    expected = np.asarray(preview.pose()[:3])

    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=CLEAR_OF_THE_WRIST)
    clock = VirtualClock(monkeypatch)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        push(
            controller,
            sock,
            JogLCmd(velocities=velocities, duration=duration, accel=1.0),
        )
        for _ in range(round((duration + 1.0) / INTERVAL_S)):
            clock.tick(controller, state)
            # The period after each tick is the one the loop overran.
            clock.now += INTERVAL_S
    miss = float(np.linalg.norm(get_fkine_se3(state)[:3, 3] * 1000.0 - expected))
    assert miss < 3.0, f"previewed {expected}, the jog ended {miss:.1f} mm away"
