"""
Unit tests for system command side-effect signaling.

Tests verify that system commands set typed side-effect attributes
(_switch_simulator, _switch_port, _sync_mock) which the controller
reads to trigger infrastructure changes.
"""

from unittest.mock import patch

from parol6.commands.base import ExecutionStatusCode
from parol6.server.state import ControllerState


class TestSystemCommandSideEffects:
    """Test that system commands signal side-effects via typed attributes."""

    def test_simulator_command_sets_switch_simulator(self):
        """Verify SIMULATOR command sets _switch_simulator attribute."""
        from parol6.commands.system_commands import SimulatorCommand
        from parol6.protocol.wire import SimulatorCmd

        cmd = SimulatorCommand(SimulatorCmd(on=True))
        state = ControllerState()

        code = cmd.execute_step(state)

        assert code == ExecutionStatusCode.COMPLETED
        assert cmd._switch_simulator is True

    def test_simulator_command_off(self):
        """Verify SIMULATOR command off sets _switch_simulator=False."""
        from parol6.commands.system_commands import SimulatorCommand
        from parol6.protocol.wire import SimulatorCmd

        cmd = SimulatorCommand(SimulatorCmd(on=False))
        state = ControllerState()

        code = cmd.execute_step(state)

        assert code == ExecutionStatusCode.COMPLETED
        assert cmd._switch_simulator is False

    def test_set_port_command_sets_switch_port(self):
        """Verify SET_PORT command sets _switch_port attribute."""
        from parol6.commands.system_commands import SetSerialPortCommand
        from parol6.protocol.wire import SetPortCmd

        cmd = SetSerialPortCommand(SetPortCmd(port_str="/dev/ttyUSB0"))
        state = ControllerState()

        with patch("parol6.commands.system_commands.save_com_port", return_value=True):
            code = cmd.execute_step(state)

        assert code == ExecutionStatusCode.COMPLETED
        assert cmd._switch_port == "/dev/ttyUSB0"

    def test_set_port_command_fail_leaves_no_side_effect(self):
        """Verify SET_PORT does not set _switch_port on save failure."""
        from parol6.commands.system_commands import SetSerialPortCommand
        from parol6.protocol.wire import SetPortCmd

        cmd = SetSerialPortCommand(SetPortCmd(port_str="/dev/ttyUSB0"))
        state = ControllerState()

        with patch("parol6.commands.system_commands.save_com_port", return_value=False):
            code = cmd.execute_step(state)

        assert code == ExecutionStatusCode.FAILED
        assert cmd._switch_port is None

    def test_reset_command_sets_sync_mock(self):
        """Verify RESET command sets _sync_mock attribute."""
        from parol6.commands.utility_commands import ResetCommand
        from parol6.protocol.wire import ResetCmd

        cmd = ResetCommand(ResetCmd())
        state = ControllerState()

        code = cmd.execute_step(state)

        assert code == ExecutionStatusCode.COMPLETED
        assert cmd._sync_mock is True
