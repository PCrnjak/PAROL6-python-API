"""Tests for the TCP offset API."""

import numpy as np

from parol6.protocol.wire import SetTcpOffsetCmd, TcpOffsetResultStruct


# ── Wire round-trip ──────────────────────────────────────────────────────


def test_set_tcp_offset_cmd_fields():
    cmd = SetTcpOffsetCmd(x=1.0, y=2.0, z=-190.0)
    assert cmd.x == 1.0
    assert cmd.y == 2.0
    assert cmd.z == -190.0


def test_set_tcp_offset_cmd_defaults():
    cmd = SetTcpOffsetCmd()
    assert cmd.x == 0.0
    assert cmd.y == 0.0
    assert cmd.z == 0.0


def test_tcp_offset_result_struct():
    result = TcpOffsetResultStruct(x=10.0, y=20.0, z=-190.0)
    assert result.x == 10.0
    assert result.y == 20.0
    assert result.z == -190.0


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
