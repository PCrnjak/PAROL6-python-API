"""
Unit tests for parol6 client functionality.

Simplified tests focusing on actual parol6 client methods rather than
the old robot_api interface.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the parol6 modules
from parol6 import RobotClient
from parol6.utils.tracking import LazyCommandTracker


@pytest.mark.unit
class TestEnvironmentConfiguration:
    """Test client configuration."""
    
    def test_default_configuration(self):
        """Test that default values are used."""
        client = RobotClient()
        assert client.host == "127.0.0.1"
        assert client.port == 5001
    
    def test_custom_configuration(self):
        """Test that custom values work."""
        client = RobotClient(host="192.168.1.100", port=6001)
        assert client.host == "192.168.1.100"
        assert client.port == 6001
    
    def test_tracker_port_configuration(self):
        """Test that LazyCommandTracker accepts port configuration."""
        tracker = LazyCommandTracker(listen_port=7002)
        assert tracker.listen_port == 7002


@pytest.mark.unit
class TestCommandValidation:
    """Test command validation and error handling."""
    
    def test_move_joints_validation(self):
        """Test validation of move_joints parameters."""
        client = RobotClient()
        
        # Test missing both duration and speed
        with patch.object(client, '_send_tracked', return_value={'status': 'INVALID', 'details': 'Must specify either duration or speed_percentage'}):
            result = client.move_joints([0, 0, 0, 0, 0, 0])
            assert isinstance(result, dict)
            assert result.get('status') == 'INVALID'
    
    def test_move_pose_validation(self):
        """Test validation of move_pose parameters."""
        client = RobotClient()
        
        # Test missing both duration and speed
        with patch.object(client, '_send_tracked', return_value={'status': 'INVALID', 'details': 'Must specify either duration or speed_percentage'}):
            result = client.move_pose([100, 100, 100, 0, 0, 0])
            assert isinstance(result, dict)
            assert result.get('status') == 'INVALID'
    
    def test_jog_joint_validation(self):
        """Test validation of jog_joint parameters."""
        client = RobotClient()
        
        # Test missing both duration and distance
        with patch.object(client, '_send_tracked', return_value={'status': 'INVALID', 'details': 'Must specify either duration or distance_deg'}):
            result = client.jog_joint(0, 50)
            assert isinstance(result, dict)
            assert result.get('status') == 'INVALID'
    
    def test_move_cartesian_validation(self):
        """Test validation of move_cartesian parameters."""
        client = RobotClient()
        
        # Test missing both duration and speed
        with patch.object(client, '_send_tracked', return_value={'status': 'INVALID', 'details': 'Must specify either duration or speed_percentage'}):
            result = client.move_cartesian([100, 100, 100, 0, 0, 0])
            assert isinstance(result, dict)
            assert result.get('status') == 'INVALID'
    
    def test_jog_multiple_validation(self):
        """Test validation of jog_multiple parameters."""
        client = RobotClient()
        
        # Test mismatched joints and speeds length
        with patch.object(client, '_send_tracked', return_value={'status': 'INVALID', 'details': 'Number of joints must match number of speeds'}):
            result = client.jog_multiple([0, 1], [50], 2.0)
            assert isinstance(result, dict)
            assert result.get('status') == 'INVALID'


@pytest.mark.unit 
class TestCommandFormatting:
    """Test command string formatting without network operations."""
    
    @patch.object(RobotClient, '_send_tracked')
    def test_move_joints_command_format(self, mock_send):
        """Test that move_joints formats commands correctly."""
        client = RobotClient()
        mock_send.return_value = "Success"
        
        client.move_joints([10, 20, 30, 40, 50, 60], duration=5.0)
        
        # Verify the command format
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("MOVEJOINT|")
        assert "10|20|30|40|50|60" in command
        assert "5.0" in command
        assert "NONE" in command  # speed should be None
    
    @patch.object(RobotClient, '_send_tracked')
    def test_move_pose_command_format(self, mock_send):
        """Test that move_pose formats commands correctly."""
        client = RobotClient()
        mock_send.return_value = "Success"
        
        client.move_pose([100, 200, 300, 0, 90, 180], speed_percentage=75)
        
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("MOVEPOSE|")
        assert "100|200|300|0|90|180" in command
        assert "NONE" in command  # duration should be None
        assert "75" in command
    
    @patch.object(RobotClient, '_send_tracked')
    def test_jog_command_format(self, mock_send):
        """Test that jog_joint formats commands correctly."""
        client = RobotClient()
        mock_send.return_value = "Success"
        
        client.jog_joint(2, 50, duration=1.0)
        
        mock_send.assert_called_once() 
        command = mock_send.call_args[0][0]
        assert command.startswith("JOG|")
        assert "2|50|1.0|NONE" in command
    
    @patch.object(RobotClient, '_send_tracked')
    def test_cartesian_jog_command_format(self, mock_send):
        """Test that jog_cartesian formats commands correctly."""
        client = RobotClient()
        mock_send.return_value = "Success"
        
        client.jog_cartesian('WRF', 'X+', 25, 2.0)
        
        mock_send.assert_called_once()
        command = mock_send.call_args[0][0]
        assert command.startswith("CARTJOG|")
        assert "WRF|X+|25|2.0" in command
    
    @patch.object(RobotClient, '_send_tracked')
    def test_gripper_command_format(self, mock_send):
        """Test gripper command formatting."""
        client = RobotClient()
        mock_send.return_value = "Success"
        
        # Test pneumatic gripper
        client.control_pneumatic_gripper('open', 1)
        mock_send.assert_called_with("PNEUMATICGRIPPER|open|1", False, 2.0, False)
        
        # Test electric gripper
        client.control_electric_gripper('move', position=100, speed=150, current=500)
        # Check the last call
        last_call = mock_send.call_args_list[-1]
        assert "ELECTRICGRIPPER|move|100|150|500" == last_call[0][0]


@pytest.mark.unit
class TestStatusQueries:
    """Test status query methods."""
    
    @patch.object(RobotClient, '_request')
    def test_get_angles(self, mock_request):
        """Test angle retrieval."""
        client = RobotClient()
        
        # Mock successful response
        mock_request.return_value = "ANGLES|10.0,20.0,30.0,40.0,50.0,60.0"
        result = client.get_angles()
        
        assert result == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        mock_request.assert_called_with("GET_ANGLES", bufsize=1024)
    
    @patch.object(RobotClient, '_request')
    def test_get_io(self, mock_request):
        """Test IO status retrieval."""
        client = RobotClient()
        
        # Mock successful response
        mock_request.return_value = "IO|0,1,0,1,1"
        result = client.get_io()
        
        assert result == [0, 1, 0, 1, 1]
        mock_request.assert_called_with("GET_IO", bufsize=1024)
    
    @patch.object(RobotClient, '_request')
    def test_get_gripper_status(self, mock_request):
        """Test gripper status retrieval."""
        client = RobotClient()
        
        # Mock successful response
        mock_request.return_value = "GRIPPER|1,128,150,500,0,2"
        result = client.get_gripper_status()
        
        assert result == [1, 128, 150, 500, 0, 2]
        mock_request.assert_called_with("GET_GRIPPER", bufsize=1024)


@pytest.mark.unit
class TestBasicTracker:
    """Test basic tracker functionality without network operations."""
    
    def test_tracker_initialization(self):
        """Test tracker initializes in lazy mode."""
        tracker = LazyCommandTracker(listen_port=6002)
        
        # Should not be initialized until first use
        assert not hasattr(tracker, '_initialized') or not tracker._initialized
        assert tracker.listen_port == 6002
    
    @patch('socket.socket')
    def test_tracker_lazy_initialization(self, mock_socket_class):
        """Test that tracker initializes only when first needed."""
        # Mock socket creation and binding
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        tracker = LazyCommandTracker()
        
        # Mock successful initialization
        with patch('threading.Thread'):
            # Try to track a command (should trigger initialization)
            command, cmd_id = tracker.track_command("TEST")
            
            # Should get a command ID
            assert cmd_id is not None
            assert command == f"{cmd_id}|TEST"


if __name__ == "__main__":
    pytest.main([__file__])
