"""A tool action is queued work: it takes its turn in the command queue
behind the motion sent ahead of it, the arm holds still while it runs, and
the motion sent after it waits for it. What discards the queue — a stop, an
e-stop, a reset, a teleport, a stream taking the arm — discards the tool
actions in it and halts the one running where the jaws are; a pause holds
them; a tool action refused in its turn drops what is queued behind it.
Only ``tool.stop()`` acts at once, ahead of the queue."""

import socket
import time

import numpy as np
import pytest

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import MotionError, RobotClient
from parol6.config import INTERVAL_S, steps_to_deg
from parol6.protocol.wire import (
    DelayCmd,
    MoveJCmd,
    OkMsg,
    SelectToolCmd,
    ToolActionCmd,
)
from parol6.utils.error_codes import ErrorCode
from tests.conftest import wait_until
from tests.integration.controller_loop import ready, send, tick, tick_for

pytestmark = pytest.mark.integration


def _offset(pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    return [pose[0] + dx, pose[1] + dy, pose[2] + dz, *pose[3:]]


def _open_at(client: RobotClient, angles: list[float]) -> None:
    """Put the arm at *angles* with the jaws fully open. The gripper opens
    them first, so the snap lands where it is already driving them."""
    opening = client.tool.open()
    assert opening >= 0 and client.wait_command(opening, timeout=15.0)
    assert client.teleport(angles, tool_positions=[0.0]) == 1
    assert client.wait_status(
        lambda s: (
            s.tool_status.positions[0] == 0.0
            and np.allclose(s.angles, angles, atol=0.05)
        ),
        timeout=2.0,
    ), "the arm never landed with the jaws open"


def _fit_ssg48_open(client: RobotClient):
    """Select and calibrate the SSG-48 and open it, sent back to back: each
    waits for the one ahead of it."""
    selecting = client.select_tool("SSG-48")
    calibrating = client.tool.calibrate()
    assert min(selecting, calibrating) >= 0
    angles = client.angles()
    assert angles is not None
    _open_at(client, angles)
    assert client.wait_command(calibrating, timeout=1.0)
    return client.tool


def _trace(client: RobotClient, until, timeout: float):
    """TCP positions (mm) and jaw positions, frame by frame, until *until*
    holds or *timeout* runs out, and whether it held."""
    tcp: list[np.ndarray] = []
    jaws: list[float] = []

    def record(s) -> bool:
        jaw = float(s.tool_status.positions[0])
        tcp.append(s.pose[[3, 7, 11]])
        jaws.append(jaw)
        return until(s)

    held = client.wait_status(record, timeout=timeout)
    return np.asarray(tcp), np.asarray(jaws), held


def _assert_cancelled(client: RobotClient, index: int, by: str) -> None:
    with pytest.raises(MotionError) as cancelled:
        client.wait_command(index, timeout=2.0)
    assert cancelled.value.robot_error.code == ErrorCode.MOTN_CANCELLED, (
        f"{by}: {cancelled.value}"
    )
    assert cancelled.value.command_index == index, by


def _angles_deg(state) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_deg(state.Position_in, out)
    return out


def test_a_tool_action_takes_its_turn_between_the_moves_around_it(
    client: RobotClient, server_proc
):
    """move_l(A, r) → close → move_l(B), sent back to back: the jaws stay
    open while the arm travels to A, the arm comes to rest at A — a tool
    action ends a blend — and holds there while the jaws close, and leaves
    for B only once they have."""
    tool = _fit_ssg48_open(client)
    pose = client.pose()
    assert pose is not None
    a = _offset(pose, 0.0, 0.0, -40.0)
    b = _offset(pose, 40.0, 0.0, -40.0)

    reaching = client.move_l(a, duration=3.0, r=12.0, wait=False)
    closing = tool.close(speed=0.1)
    leaving = client.move_l(b, duration=1.5, wait=False)
    assert min(reaching, closing, leaving) >= 0
    tcp, jaws, done = _trace(
        client, lambda s: s.completed_index >= leaving, timeout=20.0
    )
    assert done, "the move sent after the close never finished"

    from_a = np.linalg.norm(tcp - np.array(a[:3]), axis=1)
    at_a = from_a < 0.5
    arrived = int(np.argmax(at_a)) if at_a.any() else len(at_a)
    early = jaws[:arrived].max(initial=0.0)
    assert early < 0.01, (
        f"the jaws closed to {early:.2f} while the arm was still on its way to A"
    )
    assert at_a.any(), (
        f"the arm blended past A, {from_a.min():.2f} mm from it at the closest"
    )
    under_way = (jaws > 0.02) & (jaws < 0.97)
    assert under_way.any(), "the close was never seen under way"
    assert at_a[under_way].all(), (
        f"the arm moved {from_a[under_way].max():.2f} mm off A while the jaws "
        "were closing"
    )
    left = ~at_a & (np.arange(len(at_a)) > arrived)
    assert left.any(), "the arm never left A for B"
    assert jaws[left].min() > 0.97, (
        f"the arm left A for B with the jaws at {jaws[left].min():.2f}"
    )
    assert np.linalg.norm(tcp[-1] - np.array(b[:3])) < 0.5
    for index in (reaching, closing, leaving):
        assert client.wait_command(index, timeout=1.0)


def test_a_wait_on_a_tool_action_covers_the_motion_queued_ahead_of_it(
    client: RobotClient, server_proc
):
    """The close runs only once the move sent ahead of it has, so a wait on
    the close returns with the arm already there."""
    tool = _fit_ssg48_open(client)
    pose = client.pose()
    assert pose is not None
    a = _offset(pose, 0.0, 0.0, -40.0)

    reaching = client.move_l(a, duration=3.0, wait=False)
    closing = tool.close(speed=0.1)
    assert min(reaching, closing) >= 0
    assert client.wait_command(closing, timeout=15.0)
    here = client.pose()
    assert here is not None
    off = float(np.linalg.norm(np.array(here[:3]) - np.array(a[:3])))
    assert off < 0.5, (
        f"the wait on the close returned with the arm {off:.1f} mm short of "
        "the move queued ahead of it"
    )
    assert client.wait_command(reaching, timeout=1.0)
    assert tool.status().positions[0] > 0.97


def test_a_paused_queue_holds_and_lists_the_tool_actions_in_it(
    client: RobotClient, server_proc
):
    """Under a pause the queue lists the tool actions among the moves, and
    they wait with them — the jaws do not move — until it resumes, when
    all of it runs in order."""
    tool = _fit_ssg48_open(client)
    pose = client.pose()
    assert pose is not None
    a = _offset(pose, 0.0, 0.0, -20.0)
    try:
        assert client.pause() == 1
        closing = tool.close()
        reaching = client.move_l(a, duration=1.0, wait=False)
        opening = tool.open()
        assert min(closing, reaching, opening) >= 0
        wait_until(
            lambda: len(client.queue() or []) >= 3,
            5.0,
            "the paused queue never listed the tool actions around the move",
        )
        listed = client.queue()
        assert listed == ["tool_action", "move_l", "tool_action"], listed

        # Fixed observation window: "held" has no condition to poll for.
        tcp, jaws, _ = _trace(client, lambda s: False, timeout=0.5)
        assert jaws.size, "no status arrived under the pause"
        assert jaws.max() < 0.01, f"the jaws closed to {jaws.max():.2f} under the pause"
        assert np.linalg.norm(tcp - np.array(pose[:3]), axis=1).max() < 0.5
        assert not client.wait_command(closing, timeout=0.2)

        assert client.resume() == 1
        assert client.wait_command(opening, timeout=15.0)
        for index in (closing, reaching):
            assert client.wait_command(index, timeout=1.0)
        here = client.pose()
        assert here is not None
        assert np.linalg.norm(np.array(here[:3]) - np.array(a[:3])) < 0.5
    finally:
        client.stop()
        client.resume()


def test_a_stop_estop_reset_or_teleport_drops_queued_tool_actions_and_halts_the_running_one(
    client: RobotClient, server_proc
):
    """Each one fails a close still queued behind the move under way as
    MOTN_CANCELLED before the jaws move at all, and halts a close under way
    where the jaws are, failing it MOTN_CANCELLED too."""
    tool = _fit_ssg48_open(client)
    start = client.angles()
    pose = client.pose()
    assert start is not None and pose is not None
    a = _offset(pose, 0.0, 0.0, -40.0)

    def teleport_here() -> int:
        here = client.angles()
        assert here is not None
        return client.teleport(here)

    def reenable() -> None:
        assert client.reset() == 1

    def refit() -> None:
        selected = client.select_tool("SSG-48")
        assert selected >= 0 and client.wait_command(selected, timeout=10.0)

    for name, discard, recover in (
        ("stop", client.stop, lambda: None),
        ("estop", client.estop, reenable),
        ("teleport", teleport_here, lambda: None),
        # Last: it also drops the tool and the test motion profile.
        ("reset_state", client.reset_state, refit),
    ):
        _open_at(client, start)
        reaching = client.move_l(a, duration=3.0, wait=False)
        closing = tool.close(speed=0.1)
        assert min(reaching, closing) >= 0
        assert client.wait_status(
            lambda s: np.linalg.norm(s.pose[[3, 7, 11]] - np.array(pose[:3])) > 2.0,
            timeout=5.0,
        ), f"{name}: the move never got under way"
        assert discard() == 1
        recover()
        jaw = tool.status().positions[0]
        assert jaw < 0.01, (
            f"{name}: the close queued behind the move ran, the jaws at {jaw:.2f}"
        )
        for index in (reaching, closing):
            _assert_cancelled(client, index, name)

        _open_at(client, start)
        closing = tool.close(speed=0.05)
        assert closing >= 0
        assert client.wait_status(
            lambda s: 0.15 < s.tool_status.positions[0] < 0.6, timeout=10.0
        ), f"{name}: the jaws never got under way"
        assert discard() == 1
        recover()
        _assert_cancelled(client, closing, name)
        # Fixed observation window: "held" has no condition to poll for.
        _, jaws, _ = _trace(client, lambda s: False, timeout=0.5)
        assert jaws.size, f"{name}: no status arrived after the halt"
        assert 0.1 < jaws[0] < 0.9, f"{name}: the jaws ran on to {jaws[0]:.2f}"
        assert jaws.max() - jaws.min() < 0.02, (
            f"{name}: the jaws kept moving, {jaws.min():.2f}..{jaws.max():.2f}"
        )


def test_a_jog_takes_the_queue_from_the_tool_actions_in_it(
    client: RobotClient, server_proc
):
    """A jog preempts the queue, tool actions included: the close under way
    and the open queued behind it fail MOTN_CANCELLED, and the jaws stay
    where the jog found them."""
    tool = _fit_ssg48_open(client)
    closing = tool.close(speed=0.05)
    opening = tool.open(speed=0.05)
    assert min(closing, opening) >= 0
    assert client.wait_status(
        lambda s: 0.15 < s.tool_status.positions[0] < 0.6, timeout=10.0
    ), "the jaws never got under way"

    assert client.jog_j(0, 0.2, duration=0.2) == 1
    for index in (closing, opening):
        _assert_cancelled(client, index, "a jog")
    # Fixed observation window: "held" has no condition to poll for.
    _, jaws, _ = _trace(client, lambda s: False, timeout=0.5)
    assert jaws.size, "no status arrived after the jog"
    assert 0.1 < jaws[0] < 0.9, f"the jaws ran on to {jaws[0]:.2f}"
    assert jaws.max() - jaws.min() < 0.02, (
        f"the jaws kept moving under the jog, {jaws.min():.2f}..{jaws.max():.2f}"
    )


def test_a_tool_stop_halts_the_close_at_once_and_keeps_the_move_queued_behind_it(
    client: RobotClient, server_proc
):
    """``tool.stop()`` is not queued: it halts the close under way where the
    jaws are, failing it MOTN_CANCELLED by a tool stop, while the move
    queued behind the close is kept and runs once the close has ended."""
    tool = _fit_ssg48_open(client)
    pose = client.pose()
    assert pose is not None
    a = _offset(pose, 0.0, 0.0, -20.0)

    closing = tool.close(speed=0.05)
    reaching = client.move_l(a, duration=1.0, wait=False)
    assert min(closing, reaching) >= 0
    tcp, _, under_way = _trace(
        client, lambda s: s.tool_status.positions[0] > 0.2, timeout=10.0
    )
    assert under_way, "the jaws never got under way"
    drift = float(np.linalg.norm(tcp - np.array(pose[:3]), axis=1).max())
    assert drift < 0.5, (
        f"the move queued behind the close moved the arm {drift:.1f} mm while "
        "the jaws were closing"
    )

    stopping = tool.stop()
    assert stopping >= 0
    with pytest.raises(MotionError) as stopped:
        client.wait_command(closing, timeout=2.0)
    assert stopped.value.robot_error.code == ErrorCode.MOTN_CANCELLED
    assert stopped.value.command_index == closing
    assert "a tool stop" in stopped.value.robot_error.cause
    held = tool.status().positions[0]

    assert client.wait_command(reaching, timeout=10.0), (
        "the move queued behind the close never ran"
    )
    assert client.wait_command(stopping, timeout=2.0)
    here = client.pose()
    assert here is not None
    assert np.linalg.norm(np.array(here[:3]) - np.array(a[:3])) < 0.5
    later = tool.status().positions[0]
    assert 0.1 < held < 0.9, f"the jaws ran on to {held:.2f}"
    assert abs(later - held) < 0.02, (
        f"the jaws moved from {held:.2f} to {later:.2f} after the tool stop"
    )


def test_a_tool_action_refused_in_its_turn_fails_alone_and_drops_what_is_queued_behind_it(
    controller,
):
    """A jaw move on a gripper never calibrated is refused when its turn
    comes — once the move ahead of it has run — under its own index; the
    commands queued behind it fail MOTN_CANCELLED and the arm stays where
    the move ahead left it."""
    state = controller.state_manager.get_state()
    controller._planner.start()
    standby = [float(v) for v in PAROL6_ROBOT.joint.standby_deg]
    ready(controller, state, homed=True, at_deg=standby)
    a = [standby[0] + 10.0, *standby[1:]]
    b = [standby[0] + 20.0, *standby[1:]]

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
        indices = []
        for req_id, cmd in enumerate(
            (
                MoveJCmd(angles=a, duration=1.0),
                ToolActionCmd(tool_key="SSG-48", action="move", params=[1.0, 0.5, 0.5]),
                MoveJCmd(angles=b, duration=1.0),
                DelayCmd(seconds=0.2),
            ),
            start=2,
        ):
            reply = send(controller, state, sock, cmd, req_id)
            assert isinstance(reply, OkMsg) and reply.index is not None, reply
            indices.append(reply.index)
        reaching, closing, leaving, dwelling = indices

        tick_for(
            controller,
            state,
            lambda: state.command_failure(closing) is not None,
            "the close on a never-calibrated gripper was not refused",
            seconds=30.0,
        )
        assert state.command_completed(reaching), (
            "the close was refused before the move queued ahead of it had run"
        )
        failure = state.command_failure(closing)
        assert failure is not None
        assert failure.code == int(ErrorCode.COMM_VALIDATION_ERROR), failure
        assert "not calibrated" in failure.cause
        assert failure.command_index == closing

        tick_for(
            controller,
            state,
            lambda: all(
                state.command_failure(i) is not None for i in (leaving, dwelling)
            ),
            "the commands queued behind the refused close were not dropped",
            seconds=5.0,
        )
        for index in (leaving, dwelling):
            dropped = state.command_failure(index)
            assert dropped is not None
            assert dropped.code == int(ErrorCode.MOTN_CANCELLED), dropped
            assert dropped.command_index == index

        # Fixed observation window: "stays put" has no condition to poll for.
        for _ in range(round(1.0 / INTERVAL_S)):
            tick(controller, state)
            time.sleep(INTERVAL_S)
        assert np.allclose(_angles_deg(state), a, atol=0.05), (
            f"the arm left A after the refusal: {_angles_deg(state)}"
        )
        assert state.gripper_hw.feedback_position == 0, "the jaws must not have moved"
