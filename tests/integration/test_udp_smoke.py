"""
Integration smoke tests for UDP communication.

Tests basic UDP communication with headless_commander.py running in FAKE_SERIAL mode.
Covers PING/PONG, GET_* endpoints, STOP semantics, and ENABLE/DISABLE functionality.
"""

import pytest
import sys
import os
import time
from typing import Dict, Any

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.utils import UDPClient, parse_server_response
import robot_api


@pytest.mark.integration
class TestBasicCommunication:
    """Test basic UDP communication with the server."""
    
    def test_ping_pong(self, server_proc, ports):
        """Test PING/PONG communication."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_server_process_running(self, server_proc):
        """Verify the server process is running and responsive."""
        assert server_proc.is_running()
        
        # Process should have initialized without errors
        error_lines = server_proc.get_error_lines()
        # Filter out expected warnings about missing serial port
        serious_errors = [line for line in error_lines 
                         if "ERROR" in line and "Serial" not in line]
        assert len(serious_errors) == 0, f"Unexpected errors in server: {serious_errors}"


@pytest.mark.integration  
class TestGetEndpoints:
    """Test GET_* command endpoints that return immediate data."""
    
    def test_get_pose(self, server_proc, ports):
        """Test GET_POSE command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("GET_POSE", timeout=2.0)
        assert response is not None
        assert response.startswith("POSE|")
        
        parsed = parse_server_response(response)
        assert parsed['type'] == 'POSE'
        assert isinstance(parsed['data'], list)
        assert len(parsed['data']) == 16  # 4x4 transformation matrix flattened
    
    def test_get_angles(self, server_proc, ports):
        """Test GET_ANGLES command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("GET_ANGLES", timeout=2.0)
        assert response is not None
        assert response.startswith("ANGLES|")
        
        parsed = parse_server_response(response)
        assert parsed['type'] == 'ANGLES'
        assert isinstance(parsed['data'], list)
        assert len(parsed['data']) == 6  # 6 joint angles
    
    def test_get_io(self, server_proc, ports):
        """Test GET_IO command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("GET_IO", timeout=2.0)
        assert response is not None
        assert response.startswith("IO|")
        
        parsed = parse_server_response(response)
        assert parsed['type'] == 'IO'
        assert isinstance(parsed['data'], list)
        assert len(parsed['data']) == 5  # IN1, IN2, OUT1, OUT2, ESTOP
        
        # In FAKE_SERIAL mode, ESTOP should be released (1)
        assert parsed['data'][4] == 1
    
    def test_get_gripper(self, server_proc, ports):
        """Test GET_GRIPPER command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("GET_GRIPPER", timeout=2.0)
        assert response is not None
        assert response.startswith("GRIPPER|")
        
        parsed = parse_server_response(response)
        assert parsed['type'] == 'GRIPPER'
        assert isinstance(parsed['data'], list)
        assert len(parsed['data']) == 6  # ID, Position, Speed, Current, Status, ObjDetection
    
    def test_get_speeds(self, server_proc, ports):
        """Test GET_SPEEDS command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("GET_SPEEDS", timeout=2.0)
        assert response is not None
        assert response.startswith("SPEEDS|")
        
        parsed = parse_server_response(response)
        assert parsed['type'] == 'SPEEDS'
        assert isinstance(parsed['data'], list)
        assert len(parsed['data']) == 6  # 6 joint speeds
    
    def test_get_status_aggregate(self, server_proc, ports):
        """Test GET_STATUS aggregate command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        response = client.send_command("GET_STATUS", timeout=2.0)
        assert response is not None
        assert response.startswith("STATUS|")
        
        parsed = parse_server_response(response)
        assert parsed['type'] == 'STATUS'
        assert isinstance(parsed['data'], dict)
        
        # Should contain all status components
        assert 'POSE' in parsed['data']
        assert 'ANGLES' in parsed['data']
        assert 'IO' in parsed['data']
        assert 'GRIPPER' in parsed['data']


@pytest.mark.integration
class TestControlCommands:
    """Test basic control commands."""
    
    def test_stop_command(self, server_proc, ports):
        """Test STOP command functionality."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send STOP command
        success = client.send_command_no_response("STOP")
        assert success
        
        # Give the server a moment to process
        time.sleep(0.1)
        
        # Server should still be responsive after STOP
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_enable_disable_cycle(self, server_proc, ports):
        """Test ENABLE/DISABLE controller functionality."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Test DISABLE command
        success = client.send_command_no_response("DISABLE")
        assert success
        time.sleep(0.1)
        
        # Server should still respond to GET commands when disabled
        response = client.send_command("PING", timeout=2.0) 
        assert response == "PONG"
        
        # Test ENABLE command
        success = client.send_command_no_response("ENABLE")
        assert success
        time.sleep(0.1)
        
        # Should still be responsive
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_clear_error_command(self, server_proc, ports):
        """Test CLEAR_ERROR command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        success = client.send_command_no_response("CLEAR_ERROR")
        assert success
        
        # Server should remain responsive
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"


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
    """Test basic motion commands without acknowledgment tracking."""
    
    def test_home_command(self, server_proc, ports):
        """Test HOME command in FAKE_SERIAL mode."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send HOME command
        success = client.send_command_no_response("HOME")
        assert success
        
        # Give time for homing to process (should be fast in FAKE_SERIAL)
        time.sleep(0.5)
        
        # Check that robot is responsive after homing
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
        
        # Check that joints show as homed (in FAKE_SERIAL mode)
        response = client.send_command("GET_ANGLES", timeout=2.0)
        assert response is not None
        assert response.startswith("ANGLES|")
    
    def test_delay_command(self, server_proc, ports):
        """Test DELAY command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send a short delay command
        success = client.send_command_no_response("DELAY|0.1")
        assert success
        
        # Server should remain responsive
        time.sleep(0.2)
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_basic_joint_move(self, server_proc, ports):
        """Test basic joint movement command."""
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send a small joint move command
        success = client.send_command_no_response("MOVEJOINT|0|5|10|15|20|25|2.0|None")
        assert success
        
        # Give time for move to process
        time.sleep(2.5)
        
        # Server should be responsive after move
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"
    
    def test_basic_pose_move(self, server_proc, ports):
        """Test basic pose movement command.""" 
        client = UDPClient(ports.server_ip, ports.server_port)
        
        # Send a small pose move command
        success = client.send_command_no_response("MOVEPOSE|100|100|100|0|0|0|None|50")
        assert success
        
        # Give time for move to process
        time.sleep(3.0)
        
        # Server should be responsive
        response = client.send_command("PING", timeout=2.0)
        assert response == "PONG"


@pytest.mark.integration
class TestRobotAPIIntegration:
    """Test robot_api.py functions with live server."""
    
    def test_robot_api_get_functions(self, server_proc, robot_api_env):
        """Test robot_api GET functions with live server."""
        # Test get_robot_pose
        pose = robot_api.get_robot_pose()
        assert pose is not None
        assert isinstance(pose, list)
        assert len(pose) == 6  # [x, y, z, rx, ry, rz]
        
        # Test get_robot_joint_angles
        angles = robot_api.get_robot_joint_angles()
        assert angles is not None
        assert isinstance(angles, list)
        assert len(angles) == 6
        
        # Test get_robot_io
        io_status = robot_api.get_robot_io()
        assert io_status is not None
        assert isinstance(io_status, list)
        assert len(io_status) == 5
        assert io_status[4] == 1  # E-stop should be released in FAKE_SERIAL
        
        # Test get_robot_joint_speeds  
        speeds = robot_api.get_robot_joint_speeds()
        assert speeds is not None
        assert isinstance(speeds, list)
        assert len(speeds) == 6
        
        # Test get_electric_gripper_status
        gripper = robot_api.get_electric_gripper_status()
        assert gripper is not None
        assert isinstance(gripper, list)
        assert len(gripper) == 6
    
    def test_robot_api_utility_functions(self, server_proc, robot_api_env):
        """Test robot_api utility functions."""
        # Test is_robot_stopped
        stopped = robot_api.is_robot_stopped()
        assert isinstance(stopped, bool)
        
        # Test is_estop_pressed
        estop = robot_api.is_estop_pressed()
        assert isinstance(estop, bool)
        assert not estop  # Should be False in FAKE_SERIAL mode
        
        # Test get_robot_status
        status = robot_api.get_robot_status()
        assert isinstance(status, dict)
        assert all(key in status for key in ['pose', 'angles', 'speeds', 'io', 'gripper', 'stopped', 'estop'])
    
    def test_basic_untracked_commands(self, server_proc, robot_api_env):
        """Test basic robot_api commands without acknowledgment tracking."""
        # Test home command
        result = robot_api.home_robot()
        assert isinstance(result, str)
        assert "Successfully sent" in result
        
        # Test delay command
        result = robot_api.delay_robot(0.1)
        assert isinstance(result, str)
        assert "Successfully sent" in result
        
        # Test stop command
        result = robot_api.stop_robot_movement()
        assert isinstance(result, str)  
        assert "Successfully sent" in result
        
        # Give commands time to process
        time.sleep(1.0)


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
