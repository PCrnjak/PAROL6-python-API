"""A teleport is a system command on the simulator: the controller answers
it once the pose is applied, the pose is exact so an unreferenced arm reads
referenced afterwards, whatever was driving the arm stops there, and what
cannot be applied is refused before anything moves. The arm lands before
any motion read with the teleport, which starts from there; the failure the
arm was left in stays behind; and the tool positions it takes are the ones
status reports, in the convention status reports them in."""

import socket
import time

import numpy as np
import pytest

from parol6 import MotionError, RobotClient
from parol6.config import INTERVAL_S, steps_to_deg
from parol6.protocol.wire import (
    ErrorMsg,
    JogJCmd,
    MoveJCmd,
    OkMsg,
    TeleportCmd,
    decode_message,
)
from parol6.utils.error_catalog import RobotError
from parol6.utils.error_codes import ErrorCode
from tests.integration.controller_loop import (
    VirtualClock,
    push,
    ready,
    send,
    tick,
    tick_until,
)
from waldoctl import ActionState, Box

pytestmark = pytest.mark.integration


def _angles_deg(state) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_deg(state.Position_in, out)
    return out


def _reply_index(sock: socket.socket, req_id: int) -> int:
    """The index acknowledged to request ``req_id``, among the replies
    already waiting on ``sock``."""
    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            pytest.fail(f"no acknowledgement of request {req_id}")
        reply = decode_message(data)
        if isinstance(reply, OkMsg) and reply.req_id == req_id:
            assert reply.index is not None, reply
            return reply.index


def test_a_teleport_is_acked_once_applied_and_references_the_arm(controller):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=False)
    target = [10.0, -80.0, 170.0, 5.0, -10.0, 175.0]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        reply = send(controller, state, sock, TeleportCmd(angles=target), 1)
        assert isinstance(reply, OkMsg), reply
        tick_until(
            controller,
            state,
            lambda: bool(state.Homed_in[:6].all()),
            "the teleported arm never read referenced",
        )
        assert np.allclose(_angles_deg(state), target, atol=0.05)

        # A jog is driving the arm; a teleport ends it where it lands.
        push(
            controller,
            sock,
            JogJCmd(speeds=[0.0, 0.0, 0.0, 0.0, 0.0, -0.8], duration=5.0),
        )
        tick_until(
            controller,
            state,
            lambda: _angles_deg(state)[5] < 174.0,
            "the jog never moved the wrist",
        )
        reply = send(controller, state, sock, TeleportCmd(angles=target), 2)
        assert isinstance(reply, OkMsg), reply
        for _ in range(30):
            tick(controller, state)
        assert np.allclose(_angles_deg(state), target, atol=0.05), (
            "the jog carried on after the teleport"
        )

        # Tool positions for a tool that is not fitted are refused, and the
        # arm stays where it is.
        reply = send(
            controller,
            state,
            sock,
            TeleportCmd(
                angles=[0.0, -90.0, 180.0, 0.0, 0.0, 180.0], tool_positions=[0.5]
            ),
            3,
        )
        assert isinstance(reply, ErrorMsg), reply
        assert RobotError.from_wire(reply.message).code == int(
            ErrorCode.COMM_VALIDATION_ERROR
        )
        for _ in range(5):
            tick(controller, state)
        assert np.allclose(_angles_deg(state), target, atol=0.05)

    # What the hard limits and [0, 1] exclude never reaches the wire.
    with pytest.raises(ValueError):
        TeleportCmd(angles=[10.0, -80.0, 170.0, 5.0, -10.0, 1000.0])
    with pytest.raises(ValueError):
        TeleportCmd(angles=[float("nan"), -80.0, 170.0, 5.0, -10.0, 175.0])
    with pytest.raises(ValueError):
        TeleportCmd(angles=target, tool_positions=[1.5])


def test_a_teleport_read_with_motion_lands_before_the_motion_starts(
    controller, monkeypatch
):
    """A teleport and the command sent straight after it, read in one batch:
    the arm lands where the teleport puts it, and the jog or the planned
    move starts from there, not from where the arm stood before."""
    state = controller.state_manager.get_state()
    controller._planner.start()
    # The jog's duration is timed in ticks.
    clock = VirtualClock(monkeypatch)
    start = [0.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    landing = [30.0, -80.0, 170.0, 5.0, -10.0, 150.0]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)

        ready(controller, state, homed=True, at_deg=start)
        push(controller, sock, TeleportCmd(angles=landing), 1)
        push(
            controller,
            sock,
            JogJCmd(speeds=[0.0, 0.0, 0.0, 0.0, 0.0, -0.5], duration=0.5),
        )
        for _ in range(round(1.5 / INTERVAL_S)):
            clock.tick(controller, state)
        q = _angles_deg(state)
        assert np.allclose(q[:5], landing[:5], atol=0.05), (
            f"the arm never landed at {landing}: it reads {q}"
        )
        assert landing[5] - 60.0 < q[5] < landing[5] - 1.0, (
            f"the jog did not run from the landing: J6 reads {q[5]:.1f}°"
        )

        ready(controller, state, homed=True, at_deg=start)
        goal = [40.0, *landing[1:]]
        push(controller, sock, TeleportCmd(angles=landing), 2)
        push(controller, sock, MoveJCmd(angles=goal, duration=1.0), 3)
        tick(controller, state)
        moving = _reply_index(sock, 3)
        tick_until(
            controller,
            state,
            lambda: abs(_angles_deg(state)[0] - landing[0]) < 0.05,
            "the arm never landed before the move",
        )
        lowest = landing[0]
        deadline = time.monotonic() + 30.0
        while not state.command_completed(moving):
            assert time.monotonic() < deadline, "the move never completed"
            tick(controller, state)
            lowest = min(lowest, float(_angles_deg(state)[0]))
            time.sleep(INTERVAL_S)
        assert lowest > landing[0] - 0.5, (
            f"the move started from the pose before the teleport: J1 went back "
            f"to {lowest:.1f}° on its way from {landing[0]}° to {goal[0]}°"
        )
        assert np.allclose(_angles_deg(state), goal, atol=0.05)


def test_a_teleport_leaves_the_failure_the_arm_was_in_behind(
    client: RobotClient, server_proc
):
    """A scrub teleports the arm out of the state a failed program left it
    in: the collision a move was refused for — the error, the ERROR state,
    the colliding links — does not follow the arm to where it lands."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    start = client.angles()
    assert start is not None
    target = [0.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    wrist = PAROL6_ROBOT.robot.fkine(np.radians(target))[:3, 3]
    blocker = Box(
        name="blocker",
        x=0.25,
        y=0.25,
        z=0.25,
        pose=(float(wrist[0]), float(wrist[1]), float(wrist[2]), 0, 0, 0),
    )
    assert client.set_shapes([blocker]) == 1
    try:
        with pytest.raises(MotionError) as refused:
            client.move_j(target, duration=1.5, wait=True)
        assert refused.value.robot_error.code == ErrorCode.SYS_SELF_COLLISION
        assert client.wait_status(
            lambda s: (
                s.error is not None
                and s.action_state == ActionState.ERROR
                and s.collision_active
            ),
            timeout=2.0,
        ), "the refusal never reached status"
    finally:
        assert client.set_shapes([]) == 1

    assert client.teleport(start) == 1
    assert client.error() is None, "the refusal still stands after the teleport"
    assert client.wait_status(
        lambda s: (
            s.error is None
            and s.action_state != ActionState.ERROR
            and not s.collision_active
            and not s.collision_pairs
        ),
        timeout=2.0,
    ), "status still reports the failure after the teleport"


def test_a_teleport_takes_the_tool_positions_status_reports_for_every_tool(
    client: RobotClient, server_proc
):
    """A scrub replays a recorded keyframe — the angles, and the tool
    positions status reported then — whichever tool was fitted: the
    teleport takes them back and the arm lands."""
    for i, tool in enumerate(("NONE", "PNEUMATIC", "SSG-48", "MSG", "VACUUM")):
        selected = client.select_tool(tool)
        assert selected >= 0 and client.wait_command(selected, timeout=10.0)
        assert client.wait_status(
            lambda s, t=tool: s.tool_status.key == t, timeout=2.0
        ), f"status never reported {tool} fitted"
        status = client.status()
        assert status is not None and status.tool_status.key == tool
        landing = [10.0 + 5.0 * i, -80.0, 170.0, 5.0, -10.0, 175.0]
        assert (
            client.teleport(landing, tool_positions=list(status.tool_status.positions))
            == 1
        ), tool
        assert client.wait_status(
            lambda s, at=landing: np.allclose(s.angles, at, atol=0.05), timeout=2.0
        ), f"the arm never landed with {tool} fitted"


def test_a_teleported_pneumatic_jaw_reads_back_where_it_was_put(
    client: RobotClient, server_proc
):
    """Tool positions go in as status reads them out — 1.0 closed, 0.0
    open — so a pneumatic jaw teleported to where its valve holds it stays
    there, rather than snapping to the far end and stroking back."""
    selected = client.select_tool("PNEUMATIC")
    assert selected >= 0 and client.wait_command(selected, timeout=10.0)
    angles = client.angles()
    assert angles is not None

    # Open first: only a stroke overwrites the jaw reading an earlier tool
    # left in the simulator.
    for action, position in (("open", 0.0), ("close", 1.0)):
        acting = client.tool_action("PNEUMATIC", action, wait=False)
        assert acting >= 0 and client.wait_command(acting, timeout=5.0)
        assert client.wait_status(
            lambda s, p=position: abs(s.tool_status.positions[0] - p) < 0.01,
            timeout=2.0,
        ), f"the jaw never reached {position} on {action}"
        assert client.teleport(angles, tool_positions=[position]) == 1

        readings: list[float] = []

        def record(s) -> bool:
            readings.append(float(s.tool_status.positions[0]))
            return False

        # Fixed observation window: a wrong snap sets off a 0.15 s stroke
        # back, and "stays there" has no condition to poll for.
        client.wait_status(record, timeout=0.5)
        assert readings, "no status arrived after the teleport"
        assert max(abs(r - position) for r in readings) < 0.05, (
            f"teleported to {position} with the valve on {action}, the jaw read "
            f"{min(readings):.2f}..{max(readings):.2f}"
        )
