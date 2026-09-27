"""What the motion pipeline owes when it drops, fails, refuses or holds
work: commands a failed move or a jog drops take their effects with them —
the TCP the planner already applied, the tool actions sent for a dropped
tool selection — and fail as cancelled; a command that failed to plan
reports that failure to any later wait; a stream refused on an unhomed arm
fails nothing but itself; a blend chain sent move by move while earlier
motion plays is held until it can be rounded; and a blend hold that is not
a positive duration is refused."""

import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from parol6 import MotionError, RobotClient
from parol6.protocol.wire import (
    HomeCmd,
    JogLCmd,
    OkMsg,
    SelectToolCmd,
    ServoLCmd,
    ToolActionCmd,
)
from parol6.utils.error_codes import ErrorCode
from tests.conftest import wait_until
from tests.integration.controller_loop import push, ready, send, tick_for, tick_until

pytestmark = pytest.mark.integration

#: Fraction of the planned-move linear ceiling (0.2 m/s) the blend chain runs at.
SPEED = 0.25


def _offset(pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    return [pose[0] + dx, pose[1] + dy, pose[2] + dz, *pose[3:]]


def _out_of_reach(pose: list[float]) -> list[float]:
    return _offset(pose, 1000.0, 0.0, 0.0)


def _assert_a_move_to_where_it_is_stays(client: RobotClient) -> None:
    """A move_l to the pose the arm reads goes nowhere only when the planner
    solves it against the TCP the controller reports the pose for."""
    start = client.angles()
    pose = client.pose()
    assert start is not None and pose is not None
    index = client.move_l(pose, duration=1.0, wait=False)
    assert index >= 0 and client.wait_command(index, timeout=10.0)
    after = client.angles()
    assert after is not None
    assert np.allclose(after, start, atol=0.5), (
        f"the planner kept a TCP the controller dropped: {start} -> {after}"
    )


def test_a_tcp_change_a_jog_or_a_failed_move_drops_is_dropped_by_the_planner_too(
    client: RobotClient, server_proc
):
    """The planner applies a TCP change when it plans it, while the command
    is still queued. A jog that takes the arm from the queue, or a move that
    fails ahead of the change, drops it on the controller; the planner must
    let it go as well, or every later plan is solved against a TCP the
    controller never applied."""
    assert client.delay(3.0) >= 0
    assert client.set_tcp_transform(0, 0, 40, 0, 90, 0) >= 0
    assert client.jog_j(0, 0.2, duration=0.2) == 1
    assert client.wait_motion(timeout=5.0)
    assert client.tcp_transform() == pytest.approx([0] * 6)
    _assert_a_move_to_where_it_is_stays(client)

    pose = client.pose()
    assert pose is not None
    # The delay holds the failure back until the change is queued behind it.
    assert client.delay(1.0) >= 0
    assert client.move_l(_out_of_reach(pose), wait=False) >= 0
    dropped = client.set_tcp_offset(0, 0, 40)
    assert dropped >= 0
    with pytest.raises(MotionError) as cancelled:
        client.wait_command(dropped, timeout=5.0)
    assert cancelled.value.robot_error.code == ErrorCode.MOTN_CANCELLED
    assert client.tcp_offset() == pytest.approx([0, 0, 0])
    _assert_a_move_to_where_it_is_stays(client)


def test_a_tool_action_sent_for_a_selection_a_failed_move_drops_is_cancelled_with_it(
    client: RobotClient, server_proc
):
    """A tool action sent right behind the select_tool that fits its tool
    belongs to that selection: when a move failing ahead of both drops the
    selection, the action fails as cancelled with it, and the tool still
    fitted takes its own actions again."""
    fitted = client.select_tool("PNEUMATIC")
    assert fitted >= 0 and client.wait_command(fitted, timeout=10.0)
    pose = client.pose()
    assert pose is not None
    # The delay holds the failure back until both are queued behind it.
    assert client.delay(1.5) >= 0
    assert client.move_l(_out_of_reach(pose), wait=False) >= 0
    selecting = client.select_tool("SSG-48")
    calibrating = client.tool_action("SSG-48", "calibrate", wait=False)
    assert selecting >= 0 and calibrating >= 0

    for index in (selecting, calibrating):
        with pytest.raises(MotionError) as cancelled:
            client.wait_command(index, timeout=5.0)
        assert cancelled.value.robot_error.code == ErrorCode.MOTN_CANCELLED, (
            cancelled.value
        )
        assert cancelled.value.command_index == index
    tools = client.tools()
    assert tools is not None and tools.tool == "PNEUMATIC"
    opening = client.tool_action("PNEUMATIC", "open", wait=False)
    assert opening >= 0 and client.wait_command(opening, timeout=5.0)


def test_a_wait_on_a_move_that_failed_to_plan_raises_its_failure_after_the_error_clears(
    client: RobotClient, server_proc
):
    """The next accepted command clears the standing error, not the failed
    command's outcome: a wait on it asked afterwards still raises the error
    it failed with, attributed to it."""
    pose = client.pose()
    start = client.angles()
    assert pose is not None and start is not None
    failing = client.move_l(_out_of_reach(pose), wait=False)
    assert failing >= 0
    wait_until(
        lambda: client.error() is not None, 10.0, "the unreachable move never failed"
    )
    failure = client.error()
    assert failure is not None

    target = [start[0] + 5.0, *start[1:]]
    assert client.move_j(target, duration=0.5, timeout=10.0) >= 0
    assert client.error() is None
    with pytest.raises(MotionError) as failed:
        client.wait_command(failing, timeout=2.0)
    assert failed.value.robot_error.code == failure.code
    assert failed.value.command_index == failing


def test_a_stream_refused_unhomed_is_not_the_failure_of_the_tool_action_beside_it(
    controller,
):
    """A cartesian jog refused on an unhomed arm leaves its refusal standing
    as its own, not against the calibration queued before it."""
    state = controller.state_manager.get_state()
    controller._planner.start()
    ready(controller, state, homed=False)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        selected = send(controller, state, sock, SelectToolCmd(tool_name="SSG-48"), 1)
        assert isinstance(selected, OkMsg), selected
        tick_for(
            controller,
            state,
            lambda: state.current_tool == "SSG-48",
            "the select_tool never ran",
            seconds=30.0,
        )
        calibrate = send(
            controller,
            state,
            sock,
            ToolActionCmd(tool_key="SSG-48", action="calibrate", params=[]),
            2,
        )
        assert isinstance(calibrate, OkMsg) and calibrate.index is not None, calibrate
        calibrating = calibrate.index

        push(
            controller,
            sock,
            JogLCmd(velocities=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.0),
        )
        tick_until(
            controller,
            state,
            lambda: state.error is not None,
            "the unhomed jog_l was not refused",
        )
        assert state.error is not None
        assert state.error.code == int(ErrorCode.MOTN_NOT_HOMED), state.error
        assert not state.command_completed(calibrating), (
            "the calibration finished before the refusal"
        )
        # wait_command reads a standing error at or below its own index as
        # that command's failure.
        assert state.error.command_index > calibrating, (
            f"the jog's refusal stands against index {state.error.command_index}, "
            f"failing a wait on the calibration ({calibrating})"
        )
        # Wall time, not a tick budget: the calibration reaches the loop
        # through the planner process, like all queued work.
        tick_for(
            controller,
            state,
            lambda: state.command_completed(calibrating),
            "the calibration never completed",
            seconds=30.0,
        )
        assert state.command_failure(calibrating) is None


def test_a_cartesian_stream_refused_unhomed_leaves_the_home_running(controller):
    """A jog_l or servo_l that reaches an arm while it homes is refused for
    want of the reference the home is establishing; it does not take the
    arm from the home and cancel it first."""
    state = controller.state_manager.get_state()
    controller._planner.start()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        for req_id, stream in enumerate(
            (
                JogLCmd(velocities=[0.3, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.0),
                ServoLCmd(pose=[200.0, 0.0, 200.0, 180.0, 0.0, 180.0]),
            ),
            start=1,
        ):
            name = type(stream).__name__
            ready(controller, state, homed=False)
            reply = send(controller, state, sock, HomeCmd(), req_id)
            assert isinstance(reply, OkMsg) and reply.index is not None, reply
            homing = reply.index
            tick_for(
                controller,
                state,
                lambda: state.executing_command_index == homing,
                "the home never started",
                seconds=30.0,
            )

            push(controller, sock, stream)
            tick_until(
                controller,
                state,
                lambda: state.error is not None,
                f"{name} was not refused unhomed",
            )
            assert state.error is not None
            assert state.error.code == int(ErrorCode.MOTN_NOT_HOMED), state.error
            assert state.command_failure(homing) is None, (
                f"{name} cancelled the home: {state.command_failure(homing)}"
            )
            tick_until(
                controller,
                state,
                lambda: state.command_completed(homing) and all(state.Homed_in[:6]),
                f"the home never completed after {name} was refused",
                ticks=500,
            )


def test_a_blend_hold_that_is_not_a_positive_duration_is_refused_at_import():
    """PAROL6_BLEND_HOLD_S is how long the planner waits for the rest of a
    blend chain: NaN or infinity kills the planner, zero or less spins it
    and plans every move alone. Such a value is refused where it is read."""
    root = Path(__file__).resolve().parents[2]
    for value in ("nan", "inf", "-0.5", "0"):
        result = subprocess.run(
            [sys.executable, "-c", "import parol6.config"],
            cwd=root,
            env={**os.environ, "PAROL6_BLEND_HOLD_S": value},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode != 0, f"PAROL6_BLEND_HOLD_S={value} was accepted"
        assert "ValueError" in result.stderr, result.stderr[-2000:]
        assert "PAROL6_BLEND_HOLD_S" in result.stderr, result.stderr[-2000:]


def test_a_blend_chain_sent_move_by_move_while_a_move_plays_is_one_motion(
    client: RobotClient, server_proc
):
    """A script sends a blended chain one move at a time, each further apart
    than the blend hold, while the move before the chain still plays: the
    chain is held until its last move and rounds its corners, instead of
    each move being planned alone once the queue falls quiet and stopping
    at every corner."""
    assert client.select_profile("TOPPRA") > 0
    pose = client.pose()
    assert pose is not None
    top = pose[2]
    lowered = _offset(pose, 0.0, 0.0, -40.0)
    corners = [_offset(pose, 40.0, 0.0, -40.0), _offset(pose, 40.0, 40.0, -40.0)]
    end = _offset(pose, 0.0, 40.0, -40.0)
    radius = 12.0
    path: list[np.ndarray] = []

    def tracing(done):
        def record(s) -> bool:
            path.append(s.pose[[3, 7, 11]])
            return done(s)

        return record

    assert client.move_l(lowered, duration=4.0, wait=False) >= 0
    # Each move goes out once the long one is 12 mm further down, about a
    # second after the one before: twice the hold the test server runs.
    last = -1
    for depth, target, r in (
        (4.0, corners[0], radius),
        (16.0, corners[1], radius),
        (28.0, end, 0.0),
    ):
        assert client.wait_status(
            tracing(lambda s, d=depth: s.pose[11] < top - d), timeout=10.0
        ), f"the long move never came {depth} mm down"
        last = client.move_l(target, speed=SPEED, r=r, wait=False)
        assert last >= 0
    assert client.wait_status(
        tracing(lambda s: s.completed_index >= last), timeout=20.0
    ), "the chain never completed"

    kept = [path[0]]
    for p in path[1:]:
        if np.linalg.norm(p - kept[-1]) > 1e-6:
            kept.append(p)
    pts = np.asarray(kept)
    for corner in corners:
        miss = float(np.min(np.linalg.norm(pts - np.array(corner[:3]), axis=1)))
        assert 1.0 < miss <= radius + 0.5, (
            f"the chain passed {miss:.2f} mm from its corner at {corner[:3]}: "
            "a move planned alone stops at its end"
        )

    bottom = np.array(lowered[:3])
    finish = np.array(end[:3])
    arrived = np.flatnonzero(np.linalg.norm(pts - bottom, axis=1) < 0.5)
    assert arrived.size, "the long move never arrived"
    chain = pts[arrived[0] :]
    body = chain[
        (np.linalg.norm(chain - bottom, axis=1) > 8.0)
        & (np.linalg.norm(chain - finish, axis=1) > 8.0)
    ]
    steps = np.linalg.norm(np.diff(body, axis=0), axis=1)
    assert steps.min() > 0.3, "the chain comes to rest between its moves"
