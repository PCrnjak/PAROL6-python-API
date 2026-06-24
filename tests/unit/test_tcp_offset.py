"""Tests for the TCP offset API."""

import numpy as np

from parol6.protocol.wire import SelectToolCmd, SetTcpOffsetCmd


# ── apply_tool with tcp_offset_m ─────────────────────────────────────────


def test_apply_tool_with_zero_offset():
    """apply_tool with zero offset should produce same FK as without offset."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    q = np.zeros(6, dtype=np.float64)

    PAROL6_ROBOT.apply_tool("SSG-48")
    fk_without = PAROL6_ROBOT.robot.fkine(q).copy()

    PAROL6_ROBOT.apply_tool("SSG-48", tcp_offset_m=(0.0, 0.0, 0.0))
    fk_with_zero = PAROL6_ROBOT.robot.fkine(q).copy()

    np.testing.assert_allclose(fk_without, fk_with_zero, atol=1e-12)
    PAROL6_ROBOT.apply_tool("NONE")


def test_apply_tool_with_nonzero_offset():
    """apply_tool with offset should shift the FK TCP position."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    q = np.zeros(6, dtype=np.float64)

    PAROL6_ROBOT.apply_tool("SSG-48")
    fk_base = PAROL6_ROBOT.robot.fkine(q).copy()

    PAROL6_ROBOT.apply_tool("SSG-48", tcp_offset_m=(0.0, 0.0, -0.190))
    fk_offset = PAROL6_ROBOT.robot.fkine(q).copy()

    # FK results should differ (TCP position shifted)
    assert not np.allclose(fk_base[:3, 3], fk_offset[:3, 3], atol=1e-6)

    PAROL6_ROBOT.apply_tool("NONE")


# ── DryRunRobotClient TCP offset ─────────────────────────────────────────


def test_dry_run_set_tcp_offset():
    """DryRunRobotClient should track TCP offset."""
    from parol6.client.dry_run_client import DryRunRobotClient

    client = DryRunRobotClient()
    assert client.tcp_offset() == [0.0, 0.0, 0.0]

    client.set_tcp_offset(x=0, y=0, z=-190)
    assert client.tcp_offset() == [0.0, 0.0, -190.0]

    # Reset
    client.set_tcp_offset(x=0, y=0, z=0)
    assert client.tcp_offset() == [0.0, 0.0, 0.0]


def test_dry_run_select_tool_resets_tcp_offset():
    """Selecting a new tool should reset TCP offset to zero."""
    from parol6.client.dry_run_client import DryRunRobotClient

    client = DryRunRobotClient()
    client.select_tool("SSG-48")
    client.set_tcp_offset(x=0, y=0, z=-190)
    assert client.tcp_offset() == [0.0, 0.0, -190.0]

    # Selecting a tool resets offset
    client.select_tool("SSG-48")
    assert client.tcp_offset() == [0.0, 0.0, 0.0]


# ── Variant TCP resolution ───────────────────────────────────────────────


def test_get_tool_transform_variant_honors_rpy():
    """Variant tcp_origin/tcp_rpy override the tool transform field-independently
    (matching the client-side ToolSpec semantics)."""
    from waldoctl import ToolVariant

    from parol6.tools import _TOOL_REGISTRY, ToolConfig, get_tool_transform

    cfg = ToolConfig(
        name="Test",
        description="",
        transform=np.eye(4),
        variants=(
            ToolVariant(
                key="angled",
                display_name="Angled",
                tcp_origin=(0.012, 0.0, 0.09),
                tcp_rpy=(0.0, 0.26, 0.0),
            ),
            ToolVariant(key="rpy_only", display_name="RPY", tcp_rpy=(0.0, 0.26, 0.0)),
        ),
    )
    _TOOL_REGISTRY["_TEST_VARIANT_TOOL"] = cfg
    try:
        T = get_tool_transform("_TEST_VARIANT_TOOL", variant_key="angled")
        np.testing.assert_allclose(T[:3, 3], (0.012, 0.0, 0.09), atol=1e-12)
        assert np.isclose(T[0, 0], np.cos(0.26), atol=1e-9)

        T = get_tool_transform("_TEST_VARIANT_TOOL", variant_key="rpy_only")
        np.testing.assert_allclose(T[:3, 3], (0.0, 0.0, 0.0), atol=1e-12)
        assert np.isclose(T[0, 0], np.cos(0.26), atol=1e-9)
    finally:
        del _TOOL_REGISTRY["_TEST_VARIANT_TOOL"]


# ── Planner routing (regression: SetTcpOffsetCmd must reach the planner) ──


def test_set_tcp_offset_command_is_motion_command():
    """SetTcpOffsetCommand must inherit from MotionCommand, not SystemCommand.

    SystemCommands execute in the controller process and bypass the planner
    subprocess. If SetTcpOffsetCommand is a SystemCommand, the planner's
    own robot model never sees the offset, so subsequent trajectory IK
    runs against the stale TCP — TRF rotations end up pivoting around the
    flange instead of the offset point.
    """
    from parol6.commands.base import MotionCommand, SystemCommand
    from parol6.commands.system_commands import SetTcpOffsetCommand

    assert issubclass(SetTcpOffsetCommand, MotionCommand)
    assert not issubclass(SetTcpOffsetCommand, SystemCommand)


def test_planner_updates_state_on_set_tcp_offset():
    """TrajectoryPlanner.process(SetTcpOffsetCmd) must update planner state
    so subsequent move_l calls use the new TCP for IK.
    """
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT
    from parol6.server.motion_planner import TrajectoryPlanner

    planner = TrajectoryPlanner()
    planner.process(SelectToolCmd(tool_name="SSG-48", variant_key=""))
    assert planner.state.tcp_offset_m == (0.0, 0.0, 0.0)

    planner.process(SetTcpOffsetCmd(x=0.0, y=0.0, z=-190.0))
    assert planner.state.tcp_offset_m == (0.0, 0.0, -0.190)

    # Cleanup global robot state mutated by apply_tool
    PAROL6_ROBOT.apply_tool("NONE")
