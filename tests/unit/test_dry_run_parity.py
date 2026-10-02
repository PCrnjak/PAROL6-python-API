"""A program previews as it runs: the dry run accepts, refuses and cancels
what the live client and controller accept, refuse and cancel."""

import asyncio

import numpy as np
import pytest

from parol6 import AsyncRobotClient
from parol6.client.dry_run_client import DryRunRobotClient
from parol6.tools import get_registry
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import MotionError
from tests.conftest import free_udp_port

HOME = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
W1 = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]


def _jaws_at_end(client: DryRunRobotClient, index: int) -> float:
    record = client.plan()
    block = record.blocks[index]
    assert block.error is None and block.rows > 0
    return float(record.tool_closed[block.start_row + block.rows - 1])


def test_tool_verbs_run_as_the_live_tools_run_them():
    """The selected tool offers the live tool's verbs and properties, in the
    forms the live tool takes them, and each sends the action the live one
    sends."""
    client = DryRunRobotClient(initial_joints_deg=HOME, initial_gripper_calibrated=True)
    assert client.select_tool("SSG-48") >= 0
    tool = client.tool

    index = tool.set_position(position=0.4)
    assert _jaws_at_end(client, index) == pytest.approx(0.4, abs=0.05)
    assert tool.status().position == pytest.approx(0.4)

    assert tool.current_range == get_registry()["SSG-48"].current_range
    assert tool.is_open(0.3) and not tool.is_open(0.7)
    tool.action_l(False)
    assert tool.status().position == 1.0
    tool.action_l(True)
    assert tool.status().position == 0.0

    # A valve is open or shut: a move to 0.7 closes it, as the live valve does.
    assert client.select_tool("PNEUMATIC") >= 0
    index = client.tool_action("PNEUMATIC", "move", [0.7])
    assert _jaws_at_end(client, index) > 0.9
    assert client.tool.status().position == 1.0

    assert client.select_tool("msg") >= 0
    assert client.tool.key == "MSG"


def test_malformed_motion_is_refused_as_live_refuses_it():
    """A jog frame or axis list the live client rejects, a keyword the live
    signature does not have, and a frame the controller's decoder rejects
    are refused in the preview too, and nothing moves."""
    preview = DryRunRobotClient(initial_joints_deg=HOME)
    before = preview.angles()

    async def live(call) -> None:
        client = AsyncRobotClient(host="127.0.0.1", port=free_udp_port())
        try:
            await call(client)
        finally:
            await client.close()

    def jog_wrf(c):
        return c.jog_l("wrf", "X", 0.5, 0.5)

    def jog_mismatch(c):
        return c.jog_l("WRF", axes=["X", "Y"], speeds_list=[0.5], duration=0.5)

    for call, raised in ((jog_wrf, ValueError), (jog_mismatch, ValueError)):
        with pytest.raises(raised) as live_refusal:
            asyncio.run(live(call))
        with pytest.raises(raised) as preview_refusal:
            call(preview)
        assert str(preview_refusal.value) == str(live_refusal.value)

    with pytest.raises(TypeError):
        preview.servo_j(before, sped=0.5)
    with pytest.raises(TypeError):
        preview.servo_l(preview.pose(), acel=0.5)

    pose = preview.pose()
    for refused_move in (
        lambda: preview.move_l([0, 0, -10, 0, 0, 0], frame="trf", rel=True, speed=0.5),
        lambda: preview.move_s(
            [[pose[0] + 5, *pose[1:]], [pose[0] + 10, pose[1] + 5, *pose[2:]]],
            frame="wrf",
            speed=0.5,
        ),
    ):
        with pytest.raises(MotionError) as refusal:
            refused_move()
        assert refusal.value.code == ErrorCode.COMM_VALIDATION_ERROR
        block = preview.plan().blocks[-1]
        assert block.rows == 0 and block.error is not None
        assert block.error.code == ErrorCode.COMM_VALIDATION_ERROR

    assert preview.angles() == pytest.approx(before, abs=1e-6)


def test_a_cancel_fails_the_blend_chain_it_drops():
    """A blended move still held for its blend when the program stops,
    e-stops, resets, switches to the simulator or teleports never runs: it
    fails as cancelled, as the controller fails it, rather than reading as
    done."""
    cancels = {
        "stop": lambda c: c.stop(),
        "estop": lambda c: c.estop(),
        "reset_state": lambda c: c.reset_state(),
        "simulator": lambda c: c.simulator(True),
        "teleport": lambda c: c.teleport(HOME),
    }
    for name, cancel in cancels.items():
        client = DryRunRobotClient(initial_joints_deg=HOME)
        pose = client.pose()
        chain = client.move_l([pose[0], pose[1], pose[2] - 20.0, *pose[3:]], r=20)
        cancel(client)
        assert not client.wait_command(chain), name
        record = client.plan()
        block = record.blocks[chain]
        assert block.rows == 0, name
        assert block.error is not None, name
        assert block.error.code == ErrorCode.MOTN_CANCELLED, name
        assert record.stop == "failed", name


def test_teleport_is_refused_and_applied_as_the_controller_does():
    """A teleport is refused while the controller is disabled and when its
    tool positions do not match the fitted tool, leaving the arm where it
    was; an accepted one puts the jaws where it says."""
    client = DryRunRobotClient(initial_joints_deg=HOME)
    client.estop()
    with pytest.raises(MotionError) as refusal:
        client.teleport(W1)
    assert refusal.value.code == ErrorCode.SYS_CONTROLLER_DISABLED
    assert client.plan().blocks[-1].error is not None
    assert client.angles() == pytest.approx(HOME, abs=0.01)
    client.reset()

    with pytest.raises(MotionError) as refusal:
        client.teleport(W1, tool_positions=[1.0])
    assert refusal.value.code == ErrorCode.COMM_VALIDATION_ERROR
    assert client.angles() == pytest.approx(HOME, abs=0.01)

    assert client.select_tool("SSG-48") >= 0
    assert client.teleport(W1, tool_positions=[1.0]) == 1
    record = client.plan()
    assert record.tool_closed[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(np.degrees(record.joints_rad[-1]), W1, atol=0.01)
    assert client.tool.status().positions == (1.0,)


def test_a_transport_switch_forgets_the_gripper_calibration():
    """The gripper behind a new transport has not been calibrated: after a
    simulator toggle or a hardware connect, a jaw move is refused until the
    program calibrates again, as the controller refuses it."""
    for switch in (
        lambda c: c.simulator(True),
        lambda c: c.connect_hardware("/dev/ttyUSB0"),
    ):
        client = DryRunRobotClient(initial_joints_deg=HOME)
        assert client.select_tool("SSG-48") >= 0
        assert client.wait_command(client.tool.calibrate())
        switch(client)
        index = client.tool.close()
        error = client.plan().blocks[index].error
        assert error is not None and error.code == ErrorCode.COMM_VALIDATION_ERROR
        assert "not calibrated" in error.cause
