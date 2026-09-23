"""Tests for RESET_STATE command."""

import numpy as np
import pytest

from parol6.commands.utility_commands import ResetStateCommand
from parol6.protocol.wire import ResetStateCmd
from parol6.server.state import ControllerState


class TestResetCommandExecution:
    """ResetStateCommand.tick resets the program-level state and nothing
    physical."""

    def test_reset_restores_program_state_and_keeps_the_physical(self):
        state = ControllerState()
        state.Position_in = np.array(
            [1000, 2000, 3000, 4000, 5000, 6000], dtype=np.int32
        )
        state.Speed_in = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
        state.Homed_in[:] = 1
        state.InOut_out[2] = 1
        state.Gripper_data_out[0] = 128
        state.gripper_calibrated = True
        state.enabled = False
        state.disabled_reason = "E-STOP pressed"
        state.e_stop_active = True
        state.soft_error = True
        state.execution_paused = True
        state.execution_speed = 0.4
        state.motion_profile = "RUCKIG"
        state._current_tool = "GRIPPER"

        cmd = ResetStateCommand(ResetStateCmd())
        cmd.tick(state)
        assert cmd.is_finished is True

        # What the firmware reports and what has been commanded to it stay.
        assert list(state.Position_in) == [1000, 2000, 3000, 4000, 5000, 6000]
        assert list(state.Speed_in) == [10, 20, 30, 40, 50, 60]
        assert all(state.Homed_in[:6])
        assert state.InOut_out[2] == 1
        assert state.Gripper_data_out[0] == 128
        assert state.gripper_calibrated
        # The protective stop stays latched: only reset() clears it.
        assert not state.enabled
        assert state.disabled_reason == "E-STOP pressed"
        assert state.e_stop_active
        # The program-level state is back at its defaults.
        assert state.soft_error is False
        assert state.error is None
        assert not state.execution_paused
        assert state.execution_speed == 1.0
        assert state.motion_profile == "TOPPRA"
        assert state._current_tool == "NONE"

    def test_reset_preserves_connection_state(self):
        """Reset should NOT reset connection-related state."""
        state = ControllerState()
        state.ip = "192.168.1.100"
        state.port = 9999
        state.start_time = 12345.0
        state.ser = "mock_serial"

        cmd = ResetStateCommand(ResetStateCmd())
        cmd.tick(state)

        assert state.ip == "192.168.1.100"
        assert state.port == 9999
        assert state.start_time == 12345.0
        assert state.ser == "mock_serial"


@pytest.mark.integration
class TestResetIntegration:
    """Integration tests for RESET command via client."""

    def test_reset_command_succeeds(self, client, server_proc):
        """Test reset command executes successfully via client."""
        result = client.reset_state()
        assert result > 0

    def test_reset_multiple_times(self, client, server_proc):
        """Test reset can be called multiple times."""
        for _ in range(3):
            result = client.reset_state()
            assert result > 0
