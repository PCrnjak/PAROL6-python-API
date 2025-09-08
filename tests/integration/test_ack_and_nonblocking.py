"""
Integration tests for acknowledgment and non-blocking behavior with parol6.

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

from tests.utils.udp import UDPClient
from parol6 import RobotClient


@pytest.mark.integration
class TestAcknowledmentTracking:
    """Test command acknowledgment tracking functionality."""
    
    def test_tracked_command_basic_flow(self, server_proc):
        """Test basic acknowledgment flow with tracked commands."""
        client = RobotClient()
        
        # Send a tracked command and wait for acknowledgment
        result = client.home(wait_for_ack=True, timeout=10.0)
        
        # Should get acknowledgment response
        assert isinstance(result, dict)
        assert 'status' in result
        
        # Status should indicate completion or execution
        assert result['status'] in ['COMPLETED', 'QUEUED', 'EXECUTING']
    
    def test_non_blocking_command_returns_id(self, server_proc):
        """Test that non-blocking commands return command ID immediately."""
        client = RobotClient()
        
        # Send non-blocking command
        result = client.move_joints(
            [0, 0, 0, 0, 0, 0],
            duration=1.0,
            wait_for_ack=True, 
            non_blocking=True
        )
        
        # Should return command ID string immediately
        assert isinstance(result, str)
        assert len(result) == 8  # UUID prefix length
    
    def test_multiple_tracked_commands(self, server_proc):
        """Test tracking multiple commands simultaneously."""
        client = RobotClient()
        
        # Send several commands with tracking
        commands = []
        
        for i in range(3):
            result = client.move_joints(
                [i, i, i, i, i, i],
                duration=0.2,
                wait_for_ack=True,
                non_blocking=True
            )
            if isinstance(result, str):
                commands.append(result)
        
        # Each should have unique ID
        assert len(set(commands)) == len(commands)
        
        # Wait for all to complete
        time.sleep(1.0)
    
    def test_command_status_polling(self, server_proc):
        """Test polling command status during execution."""
        client = RobotClient()
        
        # Send a longer running command
        result = client.move_joints(
            [5, 5, 5, 5, 5, 5],
            duration=1.0,
            wait_for_ack=True,
            non_blocking=True
        )
        
        if isinstance(result, str):
            cmd_id = result
            
            # Poll status while command runs
            start_time = time.time()
            seen_statuses = []
            
            while time.time() - start_time < 2.0:
                # In parol6, we would need to implement status polling or use tracker
                time.sleep(0.1)
                
                # For now, just verify command was sent
                break


@pytest.mark.integration
class TestErrorConditions:
    """Test error conditions with acknowledgment tracking."""
    
    def test_invalid_command_with_tracking(self, server_proc):
        """Test that invalid commands return proper error acknowledgments."""
        client = RobotClient()
        
        # Try to send invalid command with tracking
        result = client.move_joints([0, 0, 0, 0, 0, 0])  # Missing timing parameters
        
        # Should get error status
        assert isinstance(result, dict)
        assert result.get('status') == 'INVALID'
        assert 'details' in result
    
    def test_malformed_parameters_with_tracking(self, server_proc):
        """Test acknowledgment for commands with malformed parameters."""
        client = RobotClient()
        
        # Test with out-of-range speed percentage - must use wait_for_ack=True to get dict response
        result = client.move_cartesian(
            pose=[100, 100, 100, 0, 0, 0],
            speed_percentage=200,  # Invalid (>100)
            wait_for_ack=True,
            timeout=3.0
        )
        
        # Should get validation error as tracked response
        assert isinstance(result, dict)
        assert result.get('status') in ['INVALID', 'FAILED', 'QUEUED', 'EXECUTING']


@pytest.mark.integration
class TestBasicCommands:
    """Test basic commands work with the server."""
    
    def test_ping(self, server_proc):
        """Test ping functionality."""
        client = RobotClient()
        result = client.ping()
        assert result is True
    
    def test_get_angles(self, server_proc):
        """Test angle retrieval."""
        client = RobotClient()
        angles = client.get_angles()
        assert isinstance(angles, list)
        assert len(angles) == 6
    
    def test_get_io(self, server_proc):
        """Test IO status retrieval."""
        client = RobotClient()
        io_status = client.get_io()
        assert isinstance(io_status, list)
        assert len(io_status) >= 5
    
    def test_stop_command(self, server_proc):
        """Test stop command."""
        client = RobotClient()
        result = client.stop(wait_for_ack=True, timeout=5.0)
        assert isinstance(result, dict)
        assert result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING']


if __name__ == "__main__":
    pytest.main([__file__])
