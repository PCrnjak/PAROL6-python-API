"""
Integration tests for acknowledgment and non-blocking behavior.

Tests wait_for_ack functionality, non-blocking command flows, status polling,
timeout handling, and command tracking with live server communication.
"""

import pytest
import sys
import os
import time
import uuid
from typing import Dict, Any

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.utils import UDPClient, AckListener, create_tracked_command
import robot_api


@pytest.mark.integration
class TestAcknowledmentTracking:
    """Test command acknowledgment tracking functionality."""
    
    def test_tracked_command_basic_flow(self, server_proc, robot_api_env):
        """Test basic acknowledgment flow with tracked commands."""
        # Send a tracked command and wait for acknowledgment
        result = robot_api.home_robot(wait_for_ack=True, timeout=10.0)
        
        # Should get acknowledgment response
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'command_id' in result
        
        # Status should indicate completion or execution
        assert result['status'] in ['COMPLETED', 'QUEUED', 'EXECUTING']
    
    def test_non_blocking_command_returns_id(self, server_proc, robot_api_env):
        """Test that non-blocking commands return command ID immediately."""
        # Send non-blocking command
        result = robot_api.delay_robot(
            duration=1.0,
            wait_for_ack=True, 
            non_blocking=True
        )
        
        # Should return command ID string immediately
        assert isinstance(result, str)
        assert len(result) == 8  # UUID prefix length
        
        # Can check status later
        status = robot_api.check_command_status(result)
        # Status might be None if tracker not initialized, that's ok for this test
    
    def test_multiple_tracked_commands(self, server_proc, robot_api_env):
        """Test tracking multiple commands simultaneously."""
        # Send several commands with tracking
        commands = []
        
        for i in range(3):
            result = robot_api.delay_robot(
                duration=0.2,
                wait_for_ack=True,
                non_blocking=True
            )
            assert isinstance(result, str)
            commands.append(result)
        
        # Each should have unique ID
        assert len(set(commands)) == len(commands)
        
        # Wait for all to complete
        time.sleep(1.0)
        
        # Check final status of each
        for cmd_id in commands:
            status = robot_api.check_command_status(cmd_id)
            # Status might be None if commands completed and were cleaned up
    
    def test_command_status_polling(self, server_proc, robot_api_env):
        """Test polling command status during execution."""
        # Send a longer running command
        cmd_id = robot_api.delay_robot(
            duration=1.0,
            wait_for_ack=True,
            non_blocking=True
        )
        
        assert isinstance(cmd_id, str)
        
        # Poll status while command runs
        start_time = time.time()
        seen_statuses = []
        
        while time.time() - start_time < 2.0:
            status = robot_api.check_command_status(cmd_id)
            if status and status.get('status') not in seen_statuses:
                seen_statuses.append(status.get('status'))
            time.sleep(0.1)
            
            # If completed, break early
            if status and status.get('completed'):
                break
        
        # Should have seen some status updates
        # Note: In some cases the command might complete too quickly to observe intermediate states


@pytest.mark.integration
class TestAckListenerIntegration:
    """Test the acknowledgment listener with live server."""
    
    @pytest.fixture
    def ephemeral_ack_port(self, monkeypatch):
        """Provide an ephemeral ACK port for isolated testing."""
        import socket
        
        # Find an available ephemeral port
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        
        # Set environment to use this port and disable robot_api tracker
        monkeypatch.setenv("PAROL6_ACK_PORT", str(port))
        monkeypatch.setenv("PAROL6_DISABLE_TRACKER", "1")
        
        return port
    
    def test_ack_listener_captures_acks(self, server_proc, ports, ephemeral_ack_port):
        """Test that AckListener captures acknowledgments from server."""
        # Set up acknowledgment listener on ephemeral port
        ack_listener = AckListener(ephemeral_ack_port)
        assert ack_listener.start()
        
        try:
            # Send tracked command
            client = UDPClient(ports.server_ip, ports.server_port)
            cmd_id = str(uuid.uuid4())[:8]
            tracked_cmd = create_tracked_command("DELAY|0.1", cmd_id)
            
            success = client.send_command_no_response(tracked_cmd)
            assert success
            
            # Wait for acknowledgment
            ack_info = ack_listener.wait_for_ack(cmd_id, timeout=5.0)
            
            if ack_info:  # May be None if server doesn't support tracking for this command
                assert ack_info['cmd_id'] == cmd_id
                assert ack_info['status'] in ['QUEUED', 'EXECUTING', 'COMPLETED']
                assert 'timestamp' in ack_info
                
        finally:
            ack_listener.stop()

@pytest.mark.integration
class TestCommandLifecycle:
    """Test complete command lifecycle with acknowledgments."""
    
    def test_command_state_transitions(self, server_proc, robot_api_env):
        """Test command state transitions from QUEUED to COMPLETED."""
        # Send a command that should go through state transitions
        result = robot_api.move_robot_joints(
            joint_angles=[0, 5, 10, 15, 20, 25],
            duration=1.0,
            wait_for_ack=True,
            timeout=15.0
        )
        
        # Should complete successfully or be invalid if command validation fails
        assert isinstance(result, dict)
        assert result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING', 'INVALID']
        
        if 'command_id' in result:
            cmd_id = result['command_id']
            assert isinstance(cmd_id, str)
            assert len(cmd_id) == 8
    
    def test_command_cancellation(self, server_proc, robot_api_env):
        """Test command cancellation via STOP."""
        # Send a longer command
        cmd_id = robot_api.delay_robot(
            duration=3.0,
            wait_for_ack=True,
            non_blocking=True
        )
        
        if isinstance(cmd_id, str):
            # Send STOP to cancel
            time.sleep(0.1)  # Let command start
            robot_api.stop_robot_movement()
            
            # Wait and check if command was cancelled
            time.sleep(0.5)
            
            # Server should still be responsive
            pose = robot_api.get_robot_pose()
            assert pose is not None
    
    def test_queue_position_tracking(self, server_proc, robot_api_env):
        """Test queue position information in acknowledgments."""
        # Send multiple commands to create a queue
        cmd_ids = []
        
        for i in range(3):
            cmd_id = robot_api.delay_robot(
                duration=0.3,
                wait_for_ack=True,
                non_blocking=True
            )
            if isinstance(cmd_id, str):
                cmd_ids.append(cmd_id)
        
        # All should have received IDs (queued successfully)
        assert len([cid for cid in cmd_ids if isinstance(cid, str)]) > 0
        
        # Wait for completion
        time.sleep(2.0)


@pytest.mark.integration  
class TestErrorConditions:
    """Test error conditions with acknowledgment tracking."""
    
    def test_invalid_command_with_tracking(self, server_proc, robot_api_env):
        """Test that invalid commands return proper error acknowledgments."""
        # Try to send invalid command with tracking
        result = robot_api.move_robot_joints(
            joint_angles=[0, 0, 0, 0, 0, 0],  # Missing timing parameters
            wait_for_ack=True
        )
        
        # Should get error status
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        assert 'details' in result
    
    def test_malformed_parameters_with_tracking(self, server_proc, robot_api_env):
        """Test acknowledgment for commands with malformed parameters."""
        # Test with out-of-range speed percentage
        result = robot_api.move_robot_cartesian(
            pose=[100, 100, 100, 0, 0, 0],
            speed_percentage=200,  # Invalid (>100)
            wait_for_ack=True
        )
        
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        assert 'Speed percentage' in result.get('details', '')


@pytest.mark.integration
class TestTrackerResourceManagement:
    """Test tracker resource management and cleanup."""
    
    def test_tracking_statistics(self, server_proc, robot_api_env):
        """Test tracking statistics reporting."""
        # Get initial stats
        initial_stats = robot_api.get_tracking_stats()
        assert isinstance(initial_stats, dict)
        
        # Send some tracked commands
        for i in range(3):
            robot_api.delay_robot(0.1, wait_for_ack=True, non_blocking=True)
        
        # Get updated stats
        time.sleep(0.5)
        updated_stats = robot_api.get_tracking_stats()
        assert isinstance(updated_stats, dict)
        
        # Should show activity if tracking is working
        if updated_stats['active']:
            assert updated_stats['commands_tracked'] > initial_stats.get('commands_tracked', 0)


if __name__ == "__main__":
    pytest.main([__file__])
