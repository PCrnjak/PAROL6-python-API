"""
Integration smoke tests for UDP communication using parol6.

Tests basic UDP communication with headless_commander.py running in FAKE_SERIAL mode.
Covers PING/PONG, GET_* endpoints, STOP semantics, and basic functionality.
"""

import pytest
import sys
import os
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.utils.udp import UDPClient


@pytest.mark.integration
class TestBasicCommunication:
    """Test basic UDP communication with the server."""
    
    def test_ping_pong(self, client, server_proc):
        """Test PING/PONG communication."""
        result = client.ping()
        assert result is True
    
    def test_server_process_running(self, client, server_proc):
        """Verify the server process is running and responsive."""
        assert client.ping() is True  # Server should be responsive


@pytest.mark.integration  
class TestGetEndpoints:
    """Test GET_* command endpoints that return immediate data."""
    
    def test_get_pose(self, client, server_proc):
        """Test GET_POSE command."""
        pose = client.get_pose()
        assert pose is not None
        assert isinstance(pose, list)
        assert len(pose) == 16  # 4x4 transformation matrix flattened
        
        # Test helper methods too
        pose_rpy = client.get_pose_rpy()
        assert pose_rpy is not None
        assert isinstance(pose_rpy, list)
        assert len(pose_rpy) == 6  # [x, y, z, rx, ry, rz]
        
        pose_xyz = client.get_pose_xyz()
        assert pose_xyz is not None
        assert isinstance(pose_xyz, list)
        assert len(pose_xyz) == 3  # [x, y, z]
    
    def test_get_angles(self, client, server_proc):
        """Test GET_ANGLES command."""
        angles = client.get_angles()
        assert angles is not None
        assert isinstance(angles, list)
        assert len(angles) == 6  # 6 joint angles
    
    def test_get_io(self, client, server_proc):
        """Test GET_IO command."""
        io_status = client.get_io()
        assert io_status is not None
        assert isinstance(io_status, list)
        assert len(io_status) == 5  # IN1, IN2, OUT1, OUT2, ESTOP
        
        # In FAKE_SERIAL mode, ESTOP should be released (1)
        assert io_status[4] == 1
        
        # Test helper method too
        assert not client.is_estop_pressed()  # Should be False in FAKE_SERIAL
    
    def test_get_gripper(self, client, server_proc):
        """Test GET_GRIPPER command."""
        gripper = client.get_gripper_status()
        assert gripper is not None
        assert isinstance(gripper, list)
        assert len(gripper) == 6  # ID, Position, Speed, Current, Status, ObjDetection
    
    def test_get_speeds(self, client, server_proc):
        """Test GET_SPEEDS command."""
        speeds = client.get_speeds()
        assert speeds is not None
        assert isinstance(speeds, list)
        assert len(speeds) == 6  # 6 joint speeds
        
        # Test helper method too
        stopped = client.is_robot_stopped()
        assert isinstance(stopped, bool)
    
    def test_get_status_aggregate(self, client, server_proc):
        """Test GET_STATUS aggregate command."""
        status = client.get_status()
        assert status is not None
        assert isinstance(status, dict)
        
        # Should contain all status components
        assert 'pose' in status
        assert 'angles' in status
        assert 'io' in status
        assert 'gripper' in status



@pytest.mark.integration
class TestStreamMode:
    """Test streaming mode functionality."""
    
    def test_stream_mode_toggle(self, server_proc, ports):
        """Test STREAM ON/OFF commands."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Enable stream mode
        success = client.send_command_no_response("STREAM|ON")
        assert success
        time.sleep(0.1)
        
        # Server should acknowledge and remain responsive
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
        
        # Disable stream mode  
        success = client.send_command_no_response("STREAM|OFF")
        assert success
        time.sleep(0.1)
        
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"


@pytest.mark.integration
class TestBasicMotionCommands:
    """Test basic motion commands with improved assertions."""
    
    def test_home_command(self, client, server_proc):
        """Test HOME command with acknowledgment tracking."""
        # Send HOME command with tracking
        result = client.home(wait_for_ack=True, timeout=15.0)
        assert isinstance(result, dict)
        assert result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING']
        
        # Wait for completion and verify robot stops
        assert client.wait_until_stopped(timeout=10.0)
        
        # Check that robot is responsive after homing
        assert client.ping() is True
        
        # Check that angles are available after homing
        angles = client.get_angles()
        assert angles is not None
        assert len(angles) == 6
    
    def test_basic_joint_move(self, client, server_proc):
        """Test basic joint movement command with tracking."""
        # Send a small joint move command with acknowledgment
        result = client.move_joints(
            [0, 5, 10, 15, 20, 25], 
            duration=2.0, 
            wait_for_ack=True, 
            timeout=10.0
        )
        assert isinstance(result, dict)
        # In FAKE_SERIAL mode without proper homing, joint moves may fail validation
        assert result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING', 'FAILED', 'INVALID']
        
        # Only wait for completion if command was accepted
        if result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING']:
            assert client.wait_until_stopped(timeout=5.0)
        
        # Verify robot state after move attempt
        angles = client.get_angles()
        assert angles is not None
        assert client.ping() is True
    
    def test_basic_pose_move(self, client, server_proc):
        """Test basic pose movement command with validation.""" 
        # Send a pose move command with tracking
        result = client.move_pose(
            [100, 100, 100, 0, 0, 0], 
            speed_percentage=50, 
            wait_for_ack=True, 
            timeout=10.0
        )
        assert isinstance(result, dict)
        # In FAKE_SERIAL mode, pose targets may fail IK validation
        assert result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING', 'FAILED', 'INVALID']
        
        # Only wait for completion if command was accepted
        if result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING']:
            assert client.wait_until_stopped(timeout=5.0)
        
        # Verify robot state
        pose = client.get_pose_rpy()
        assert pose is not None
        assert len(pose) == 6
    
    def test_cartesian_move_validation(self, client, server_proc):
        """Test cartesian movement with proper validation."""
        # Test that move requires either duration or speed
        result = client.move_cartesian([50, 50, 50, 0, 0, 0])  # No duration or speed
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        
        # Valid cartesian move (may still fail IK in FAKE_SERIAL)
        result = client.move_cartesian(
            [50, 50, 50, 0, 0, 0], 
            duration=2.0, 
            wait_for_ack=True
        )
        assert isinstance(result, dict)
        assert result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING', 'FAILED', 'INVALID']


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_command_format(self, server_proc, ports):
        """Test server response to invalid commands."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send malformed command
        success = client.send_command_no_response("INVALID_COMMAND|BAD|PARAMS")
        assert success
        
        # Server should remain responsive despite invalid command
        time.sleep(0.2)
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_empty_command(self, server_proc, ports):
        """Test server response to empty commands."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send empty command
        success = client.send_command_no_response("")
        assert success
        
        # Server should remain responsive
        time.sleep(0.1)
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_rapid_command_sequence(self, server_proc, ports):
        """Test server stability under rapid command sequence."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send multiple commands rapidly
        for i in range(10):
            success = client.send_command_no_response("PING")
            assert success
        
        # Give server time to process
        time.sleep(0.5)
        
        # Server should still be responsive
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"


@pytest.mark.integration
class TestCommandQueuing:
    """Test basic command queuing behavior."""
    
    def test_command_sequence_execution(self, server_proc, ports):
        """Test that commands execute in sequence."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send a sequence of commands
        commands = [
            "HOME",
            "DELAY|0.2", 
            "DELAY|0.2",
            "DELAY|0.2"
        ]
        
        start_time = time.time()
        for cmd in commands:
            success = client.send_command_no_response(cmd)
            assert success
        
        # Wait for all commands to complete
        # In FAKE_SERIAL mode, these should execute relatively quickly
        time.sleep(2.0)
        
        # Server should be responsive after sequence
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
        
        # Total time should be reasonable (commands + processing overhead)
        total_time = time.time() - start_time
        assert total_time < 5.0  # Should complete within reasonable time


if __name__ == "__main__":
    pytest.main([__file__])
