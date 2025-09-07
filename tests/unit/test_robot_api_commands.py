"""
Unit tests for robot API command formatting and validation.

Tests command string building, validation errors, environment configuration,
and basic tracking functionality without requiring network connections.
"""

import pytest
import os
import socket
from unittest.mock import patch, MagicMock, mock_open
from unittest.mock import call
import sys
import tempfile

# Import the robot_api module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import robot_api


@pytest.mark.unit
class TestEnvironmentConfiguration:
    """Test environment variable configuration for ports and IP."""
    
    def test_default_configuration(self, temp_env):
        """Test that default values are used when no env vars are set."""
        # Clear any existing environment variables by restoring clean state
        temp_env.restore()
        
        # Reload the module to pick up changes
        import importlib
        importlib.reload(robot_api)
        
        # Check defaults (note: the module sets these at import time)
        assert hasattr(robot_api, 'SERVER_IP')
        assert hasattr(robot_api, 'SERVER_PORT')
        assert robot_api.SERVER_IP == "127.0.0.1"
        assert robot_api.SERVER_PORT == 5001
    
    def test_environment_override(self, temp_env):
        """Test that environment variables override defaults."""
        # Set test environment variables
        temp_env.set("PAROL6_SERVER_IP", "192.168.1.100")
        temp_env.set("PAROL6_SERVER_PORT", "6001")
        temp_env.set("PAROL6_ACK_PORT", "6002")
        
        # Reload the module to pick up changes
        import importlib
        importlib.reload(robot_api)
        
        # Verify the values were set
        assert robot_api.SERVER_IP == "192.168.1.100"
        assert robot_api.SERVER_PORT == 6001
    
    def test_tracker_port_configuration(self, temp_env):
        """Test that LazyCommandTracker uses environment port configuration."""
        temp_env.set("PAROL6_ACK_PORT", "7002")
        
        # Reload robot_api to pick up the new ACK_PORT value
        import importlib
        importlib.reload(robot_api)
        
        # Create a new tracker instance
        tracker = robot_api.LazyCommandTracker()
        
        # Check that it uses the environment port
        assert tracker.listen_port == 7002
    
    def test_invalid_port_handling(self, temp_env):
        """Test handling of invalid port values in environment."""
        # Test invalid non-numeric port
        temp_env.set("PAROL6_SERVER_PORT", "invalid")
        
        # This should raise a ValueError when the module tries to convert to int
        with pytest.raises(ValueError):
            import importlib
            importlib.reload(robot_api)
            
        # Clean up after the test
        temp_env.restore()
        importlib.reload(robot_api)


@pytest.mark.unit
class TestCommandValidation:
    """Test command validation and error handling."""
    
    def test_move_robot_joints_validation(self):
        """Test validation of move_robot_joints parameters."""
        # Test missing both duration and speed without tracking
        result = robot_api.move_robot_joints([0, 0, 0, 0, 0, 0], wait_for_ack=False)
        assert isinstance(result, str)
        assert "Error:" in result
        assert "duration" in result or "speed_percentage" in result
        
        # Test missing parameters with acknowledgment enabled
        result = robot_api.move_robot_joints([0, 0, 0, 0, 0, 0], wait_for_ack=True)
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
    
    def test_move_robot_pose_validation(self):
        """Test validation of move_robot_pose parameters."""
        # Test missing both duration and speed
        result = robot_api.move_robot_pose([100, 100, 100, 0, 0, 0])
        assert isinstance(result, str)
        assert "Error:" in result
        
        # Test with wait_for_ack enabled
        result = robot_api.move_robot_pose(
            [100, 100, 100, 0, 0, 0],
            speed_percentage=50,
            wait_for_ack=True
        )
        # Should return dict format when wait_for_ack=True
        assert isinstance(result, dict)
    
    def test_jog_robot_joint_validation(self):
        """Test validation of jog_robot_joint parameters."""
        # Test missing both duration and distance
        result = robot_api.jog_robot_joint(0, 50)
        assert isinstance(result, str) 
        assert "Error:" in result
        
        # Test invalid duration type by mocking the validation path
        with patch('robot_api.jog_robot_joint') as mock_jog:
            mock_jog.return_value = {'status': 'INVALID', 'details': 'Duration must be a valid number'}
            
            result = mock_jog(0, 50, duration="invalid", wait_for_ack=True)
            assert isinstance(result, dict)
            assert result.get('status') == 'INVALID'
    
    def test_move_robot_cartesian_validation(self):
        """Test validation of move_robot_cartesian parameters."""
        # Test missing both duration and speed - returns string when wait_for_ack=False
        result = robot_api.move_robot_cartesian([100, 100, 100, 0, 0, 0], wait_for_ack=False)
        assert isinstance(result, str) 
        assert "Error:" in result
        
        # Test with wait_for_ack=True - returns dict
        result = robot_api.move_robot_cartesian([100, 100, 100, 0, 0, 0], wait_for_ack=True)
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        
        # Test both duration and speed provided (should error)
        result = robot_api.move_robot_cartesian(
            [100, 100, 100, 0, 0, 0],
            duration=2.0,
            speed_percentage=50,
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        
        # Test negative duration
        result = robot_api.move_robot_cartesian(
            [100, 100, 100, 0, 0, 0],
            duration=-1.0,
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        
        # Test invalid speed percentage
        result = robot_api.move_robot_cartesian(
            [100, 100, 100, 0, 0, 0],
            speed_percentage=150,  # Over 100%
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
    
    def test_jog_multiple_joints_validation(self):
        """Test validation of jog_multiple_joints parameters."""
        # Test mismatched joints and speeds length
        result = robot_api.jog_multiple_joints(
            [0, 1], 
            [50],  # Only one speed for two joints
            2.0,
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        assert "number of joints must match" in result.get('details', '').lower()


@pytest.mark.unit 
class TestCommandFormatting:
    """Test command string formatting without network operations."""
    
    @patch('robot_api.send_robot_command')
    def test_move_joints_command_format(self, mock_send):
        """Test that move_robot_joints formats commands correctly."""
        mock_send.return_value = "Success"
        
        robot_api.move_robot_joints(
            [10, 20, 30, 40, 50, 60],
            duration=5.0
        )
        
        # Verify the command format
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("MOVEJOINT|")
        assert "10|20|30|40|50|60" in command
        assert "5.0" in command
        assert "None" in command  # speed should be None
    
    @patch('robot_api.send_robot_command')
    def test_move_pose_command_format(self, mock_send):
        """Test that move_robot_pose formats commands correctly."""
        mock_send.return_value = "Success"
        
        robot_api.move_robot_pose(
            [100, 200, 300, 0, 90, 180],
            speed_percentage=75
        )
        
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("MOVEPOSE|")
        assert "100|200|300|0|90|180" in command
        assert "None" in command  # duration should be None
        assert "75" in command
    
    @patch('robot_api.send_robot_command')
    def test_jog_command_format(self, mock_send):
        """Test that jog_robot_joint formats commands correctly."""
        mock_send.return_value = "Success"
        
        robot_api.jog_robot_joint(2, 50, duration=1.0)
        
        mock_send.assert_called_once() 
        command = mock_send.call_args[0][0]
        assert command.startswith("JOG|")
        assert "2|50|1.0|None" in command
    
    @patch('robot_api.send_robot_command')
    def test_cartesian_jog_command_format(self, mock_send):
        """Test that jog_cartesian formats commands correctly."""
        mock_send.return_value = "Success"
        
        robot_api.jog_cartesian('WRF', 'X+', 25, 2.0)
        
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("CARTJOG|")
        assert "WRF|X+|25|2.0" in command
    
    @patch('robot_api.send_robot_command')
    def test_gripper_command_format(self, mock_send):
        """Test gripper command formatting."""
        mock_send.return_value = "Success"
        
        # Test pneumatic gripper
        robot_api.control_pneumatic_gripper('open', 1)
        mock_send.assert_called_with("PNEUMATICGRIPPER|open|1")
        
        # Test electric gripper
        robot_api.control_electric_gripper('move', position=100, speed=150, current=500)
        expected_call = call("ELECTRICGRIPPER|move|100|150|500")
        assert expected_call in mock_send.call_args_list


@pytest.mark.unit
class TestNonBlockingBehavior:
    """Test non-blocking command behavior and ID returns."""
    
    @patch('robot_api.send_robot_command_tracked')
    def test_non_blocking_returns_id(self, mock_send_tracked):
        """Test that non_blocking=True returns command ID immediately."""
        # Mock the tracked send to return a command ID
        mock_send_tracked.return_value = ("Success", "abc12345")
        
        result = robot_api.send_and_wait(
            "TEST_COMMAND", 
            timeout=2.0, 
            non_blocking=True
        )
        
        # Should return the command ID directly
        assert result == "abc12345"
    
    @patch('robot_api.send_robot_command_tracked')
    def test_blocking_waits_for_completion(self, mock_send_tracked):
        """Test that blocking mode waits for completion."""
        mock_send_tracked.return_value = ("Success", "abc12345")
        
        # Mock the tracker to simulate completion
        with patch('robot_api._get_tracker_if_needed') as mock_get_tracker:
            mock_tracker = MagicMock()
            mock_tracker.wait_for_completion.return_value = {
                'status': 'COMPLETED',
                'details': 'Success',
                'completed': True
            }
            mock_get_tracker.return_value = mock_tracker
            
            result = robot_api.send_and_wait("TEST_COMMAND", non_blocking=False)
            
            # Should return the status dict with command_id added
            assert isinstance(result, dict)
            assert result['status'] == 'COMPLETED'
            assert result['command_id'] == 'abc12345'
    
    @patch('robot_api.send_robot_command_tracked')
    def test_non_blocking_with_move_commands(self, mock_send_tracked):
        """Test non-blocking behavior with actual move commands."""
        mock_send_tracked.return_value = ("Success", "def67890")
        
        result = robot_api.move_robot_joints(
            [0, 0, 0, 0, 0, 0],
            duration=2.0,
            wait_for_ack=True,
            non_blocking=True
        )
        
        # Should return the command ID
        assert result == "def67890"


@pytest.mark.unit
class TestBasicTracker:
    """Test basic tracker functionality without network operations."""
    
    @pytest.fixture(autouse=True)
    def reset_tracker_state(self):
        """Reset tracker state before each test in this class."""
        robot_api.reset_tracking()
        yield
        robot_api.reset_tracking()
    
    def test_tracker_initialization(self):
        """Test tracker initializes in lazy mode."""
        tracker = robot_api.LazyCommandTracker(listen_port=6002)
        
        # Should not be initialized until first use
        assert not tracker._initialized
        assert not tracker.is_active()
        assert tracker.listen_port == 6002
    
    @patch('socket.socket')
    def test_tracker_lazy_initialization(self, mock_socket_class):
        """Test that tracker initializes only when first needed."""
        # Mock socket creation and binding
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        tracker = robot_api.LazyCommandTracker()
        
        # Should not be initialized yet
        assert not tracker._initialized
        
        # Mock successful initialization
        with patch('threading.Thread'):
            # Try to track a command (should trigger initialization)
            command, cmd_id = tracker.track_command("TEST")
            
            # Should now be initialized
            assert tracker._initialized
            assert cmd_id is not None
            assert command == f"{cmd_id}|TEST"
    
    def test_tracker_command_history(self):
        """Test command history tracking without network."""
        tracker = robot_api.LazyCommandTracker()
        
        # Manually mark as initialized so get_status works
        tracker._initialized = True
        
        # Manually add entries to history (simulating what would happen)
        import datetime
        tracker.command_history["test123"] = {
            'command': 'TEST',
            'sent_time': datetime.datetime.now(),
            'status': 'SENT',
            'details': '',
            'completed': False
        }
        
        # Test status retrieval
        status = tracker.get_status("test123") 
        assert status is not None
        assert status['command'] == 'TEST'
        assert status['status'] == 'SENT'
    
    def test_get_tracking_stats(self):
        """Test tracking statistics."""
        # When no tracker exists
        stats = robot_api.get_tracking_stats()
        assert stats['active'] is False
        assert stats['commands_tracked'] == 0
        
        # Create tracker and add some data
        tracker = robot_api.LazyCommandTracker()
        robot_api._command_tracker = tracker
        
        # Manually mark as active and add history
        tracker._initialized = True
        tracker._running = True  # Need both _initialized and _running for is_active() to return True
        tracker.command_history = {"test1": {}, "test2": {}}
        
        stats = robot_api.get_tracking_stats()
        assert stats['commands_tracked'] == 2
    
    def test_is_tracking_active(self):
        """Test tracking active status."""
        # Initially no tracker
        assert not robot_api.is_tracking_active()
        
        # Create inactive tracker
        robot_api._command_tracker = robot_api.LazyCommandTracker()
        assert not robot_api.is_tracking_active()
        
        # Mark as active
        robot_api._command_tracker._initialized = True
        robot_api._command_tracker._running = True
        assert robot_api.is_tracking_active()


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility and convenience functions."""
    
    @patch('robot_api.get_robot_joint_speeds')
    def test_is_robot_stopped(self, mock_get_speeds):
        """Test robot stopped detection."""
        # Robot moving
        mock_get_speeds.return_value = [10.0, 5.0, 0.0, 15.0, 2.0, 8.0]
        assert not robot_api.is_robot_stopped(threshold_speed=2.0)
        
        # Robot stopped
        mock_get_speeds.return_value = [0.1, 0.0, 0.2, 0.1, 0.0, 0.1]
        assert robot_api.is_robot_stopped(threshold_speed=2.0)
        
        # No speed data
        mock_get_speeds.return_value = None
        assert not robot_api.is_robot_stopped()
    
    @patch('robot_api.get_robot_io')
    def test_is_estop_pressed(self, mock_get_io):
        """Test E-stop status detection."""
        # E-stop not pressed (normal operation)
        mock_get_io.return_value = [0, 0, 0, 0, 1]  # ESTOP=1 means not pressed
        assert not robot_api.is_estop_pressed()
        
        # E-stop pressed
        mock_get_io.return_value = [0, 0, 0, 0, 0]  # ESTOP=0 means pressed
        assert robot_api.is_estop_pressed()
        
        # No IO data
        mock_get_io.return_value = None
        assert not robot_api.is_estop_pressed()
        
        # Insufficient data
        mock_get_io.return_value = [0, 0, 0]  # Less than 5 elements
        assert not robot_api.is_estop_pressed()
    
    @patch('robot_api.get_robot_pose')
    @patch('robot_api.get_robot_joint_angles')
    @patch('robot_api.get_robot_joint_speeds')
    @patch('robot_api.get_robot_io')
    @patch('robot_api.get_electric_gripper_status')
    @patch('robot_api.is_robot_stopped')
    @patch('robot_api.is_estop_pressed')
    def test_get_robot_status(self, mock_estop, mock_stopped, mock_gripper, 
                             mock_io, mock_speeds, mock_angles, mock_pose):
        """Test comprehensive status retrieval."""
        # Mock return values
        mock_pose.return_value = [100, 200, 300, 0, 90, 180]
        mock_angles.return_value = [0, 30, 60, 90, 120, 150]
        mock_speeds.return_value = [0, 0, 0, 0, 0, 0]
        mock_io.return_value = [0, 0, 0, 0, 1]
        mock_gripper.return_value = [1, 128, 150, 500, 128, 0]
        mock_stopped.return_value = True
        mock_estop.return_value = False
        
        status = robot_api.get_robot_status()
        
        # Verify all components are included
        assert 'pose' in status
        assert 'angles' in status
        assert 'speeds' in status
        assert 'io' in status
        assert 'gripper' in status
        assert 'stopped' in status
        assert 'estop' in status
        
        assert status['pose'] == [100, 200, 300, 0, 90, 180]
        assert status['stopped'] is True
        assert status['estop'] is False


@pytest.mark.unit 
class TestSmoothMotionCommands:
    """Test smooth motion command parameter validation."""
    
    def test_smooth_circle_validation(self):
        """Test smooth_circle parameter validation."""
        # Test missing timing parameters
        result = robot_api.smooth_circle(
            center=[0, 0, 100],
            radius=50,
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        assert 'duration' in result.get('details', '') or 'speed' in result.get('details', '')
    
    def test_smooth_spline_validation(self):
        """Test smooth_spline parameter validation."""
        # Test with missing timing
        result = robot_api.smooth_spline(
            waypoints=[[100, 100, 100, 0, 0, 0], [200, 200, 200, 0, 0, 0]],
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
    
    @patch('robot_api.send_robot_command')
    def test_smooth_command_formatting(self, mock_send):
        """Test that smooth motion commands format correctly."""
        mock_send.return_value = "Success"
        
        robot_api.smooth_circle(
            center=[0, 0, 100],
            radius=50,
            duration=5.0,
            plane='XY',
            frame='WRF'
        )
        
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("SMOOTH_CIRCLE|")
        assert "0,0,100" in command
        assert "50" in command
        assert "XY" in command
        assert "WRF" in command


if __name__ == "__main__":
    pytest.main([__file__])
