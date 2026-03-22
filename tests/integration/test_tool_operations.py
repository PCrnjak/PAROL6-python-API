"""
Integration tests for tool operations.

Tests tool switching, gripper methods, and tool registry through the client API
with a running controller (FAKE_SERIAL mode).
"""

import pytest
import pytest_asyncio

from waldoctl import (
    ElectricGripperTool,
    GripperType,
    PneumaticGripperTool,
    ToolType,
)


# ---------------------------------------------------------------------------
# Async client fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_client(server_proc, ports):
    """Async client for tool action testing."""
    client = server_proc.create_async_client(
        host=ports.server_ip, port=ports.server_port
    )
    await client.wait_ready(timeout=5.0)
    yield server_proc, client
    await client.close()


# ===========================================================================
# Tool Switching (sync client)
# ===========================================================================


@pytest.mark.integration
class TestToolSwitching:
    """Test tool switching via the client API."""

    def test_set_get_tool_cycle(self, client, server_proc):
        """Cycle through all registered tools and verify get_tool reflects each switch."""
        # Default after reset should be NONE
        result = client.get_tool()
        assert result is not None
        assert result.tool == "NONE"

        # Available should include all registered tools
        for expected in ("PNEUMATIC", "SSG-48", "MSG"):
            assert expected in result.available, (
                f"{expected} not in available tools: {result.available}"
            )

        # Switch to each tool and verify
        for tool_name in ("PNEUMATIC", "SSG-48", "MSG", "NONE"):
            assert client.set_tool(tool_name) >= 0
            assert client.wait_motion_complete(timeout=5.0)
            result = client.get_tool()
            assert result is not None
            assert result.tool == tool_name

    def test_tool_persists_across_motion(self, client, server_proc):
        """Tool setting should survive a joint move."""
        assert client.set_tool("PNEUMATIC") >= 0
        assert client.wait_motion_complete(timeout=5.0)

        idx = client.moveJ([0, -45, 180, 0, 0, 180], speed=0.5)
        assert idx >= 0
        assert client.wait_motion_complete(timeout=10.0)

        result = client.get_tool()
        assert result is not None
        assert result.tool == "PNEUMATIC"


# ===========================================================================
# Pneumatic Gripper Methods (async, via client.tool)
# ===========================================================================


@pytest.mark.integration
class TestPneumaticGripperMethods:
    """Test pneumatic gripper via client.tool()."""

    @pytest.mark.asyncio
    async def test_pneumatic_open_close(self, async_client):
        """Open and close pneumatic gripper via tool methods."""
        robot, client = async_client
        spec = robot.tools["PNEUMATIC"]
        assert isinstance(spec, PneumaticGripperTool)
        assert spec.gripper_type == GripperType.PNEUMATIC
        assert spec.io_port == 1

        await client.set_tool("PNEUMATIC")
        await client.wait_motion_complete(timeout=5.0)

        tool = client.tool

        # Open — command accepted and completes
        idx = await tool.open()
        assert idx >= 0
        assert await client.wait_motion_complete(timeout=5.0)

        # Close — command accepted and completes
        idx = await tool.close()
        assert idx >= 0
        assert await client.wait_motion_complete(timeout=5.0)

    @pytest.mark.asyncio
    async def test_pneumatic_set_position_threshold(self, async_client):
        """set_position uses binary threshold: < 0.5 opens, >= 0.5 closes."""
        robot, client = async_client

        await client.set_tool("PNEUMATIC")
        await client.wait_motion_complete(timeout=5.0)

        tool = client.tool

        # Position 0.8 → dispatches to close()
        idx = await tool.set_position(0.8)
        assert idx >= 0
        assert await client.wait_motion_complete(timeout=5.0)

        # Position 0.2 → dispatches to open()
        idx = await tool.set_position(0.2)
        assert idx >= 0
        assert await client.wait_motion_complete(timeout=5.0)


# ===========================================================================
# SSG-48 Electric Gripper Methods (async, via client.tool)
# ===========================================================================


@pytest.mark.integration
class TestSSG48GripperMethods:
    """Test SSG-48 electric gripper via client.tool()."""

    @pytest.mark.asyncio
    async def test_ssg48_calibrate_and_move(self, async_client):
        """Calibrate and move SSG-48 gripper through tool methods."""
        robot, client = async_client
        spec = robot.tools["SSG-48"]
        assert isinstance(spec, ElectricGripperTool)
        assert spec.gripper_type == GripperType.ELECTRIC

        # Verify parameter ranges
        assert spec.position_range == (0.0, 1.0)
        assert spec.speed_range == (0.0, 1.0)
        assert spec.current_range == (100, 1300)

        await client.set_tool("SSG-48")
        await client.wait_motion_complete(timeout=5.0)

        tool = client.tool

        # Calibrate
        idx = await tool.calibrate()
        assert idx >= 0
        await client.wait_motion_complete(timeout=10.0)

        # Move to half position
        idx = await tool.set_position(0.5, speed=0.7, current=600)
        assert idx >= 0
        await client.wait_motion_complete(timeout=10.0)


# ===========================================================================
# MSG AI Stepper Gripper Methods (async, via client.tool)
# ===========================================================================


@pytest.mark.integration
class TestMSGGripperMethods:
    """Test MSG compliant AI stepper gripper via client.tool()."""

    @pytest.mark.asyncio
    async def test_msg_calibrate_and_move(self, async_client):
        """Calibrate and move MSG gripper through tool methods."""
        robot, client = async_client
        spec = robot.tools["MSG"]
        assert isinstance(spec, ElectricGripperTool)
        assert spec.gripper_type == GripperType.ELECTRIC

        await client.set_tool("MSG")
        await client.wait_motion_complete(timeout=5.0)

        tool = client.tool

        # Calibrate
        idx = await tool.calibrate()
        assert idx >= 0
        await client.wait_motion_complete(timeout=10.0)

        # Move to position
        idx = await tool.set_position(0.3, speed=0.5, current=500)
        assert idx >= 0
        await client.wait_motion_complete(timeout=10.0)


# ===========================================================================
# Tool Registry (no server needed)
# ===========================================================================


@pytest.mark.unit
class TestToolRegistry:
    """Test tool registry completeness — no server required."""

    def test_registry_completeness(self):
        """All expected tools are registered with correct types and properties."""
        from parol6 import Robot

        robot = Robot()
        tools = robot.tools

        # 5 tools registered
        keys = [t.key for t in tools.available]
        assert len(keys) == 5
        for expected in ("NONE", "PNEUMATIC", "SSG-48", "MSG", "VACUUM"):
            assert expected in keys, f"{expected} not in {keys}"

        # Default is NONE
        assert tools.default.key == "NONE"
        assert tools.default.tool_type == ToolType.NONE

        # 4 grippers
        grippers = tools.by_type(ToolType.GRIPPER)
        assert len(grippers) == 4
        gripper_keys = {t.key for t in grippers}
        assert gripper_keys == {"PNEUMATIC", "SSG-48", "MSG", "VACUUM"}

        # Type checks
        assert ToolType.GRIPPER in tools
        assert isinstance(tools["PNEUMATIC"], PneumaticGripperTool)
        assert tools["PNEUMATIC"].gripper_type == GripperType.PNEUMATIC
        assert isinstance(tools["SSG-48"], ElectricGripperTool)
        assert tools["SSG-48"].gripper_type == GripperType.ELECTRIC
        assert isinstance(tools["MSG"], ElectricGripperTool)
        assert tools["MSG"].gripper_type == GripperType.ELECTRIC

        # TCP origins differ
        origins = {t.key: t.tcp_origin for t in tools.available}
        assert origins["NONE"] != origins["PNEUMATIC"]
        assert origins["PNEUMATIC"] != origins["SSG-48"]
        assert origins["SSG-48"] != origins["MSG"]

        # Invalid key raises
        with pytest.raises(KeyError):
            tools["BOGUS"]

        # SSG-48 has meshes
        ssg48 = tools["SSG-48"]
        assert len(ssg48.meshes) == 3  # body + 2 jaws

        # PNEUMATIC has meshes
        pneumatic = tools["PNEUMATIC"]
        assert len(pneumatic.meshes) == 3  # body + 2 jaws
