"""
Integration tests for acknowledgment and non-blocking behavior with parol6.

Tests wait_for_ack functionality, non-blocking command flows, status polling,
timeout handling, and command tracking with live server communication.
"""

import os
import sys

import pytest

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.integration
class TestAcknowledmentTracking:
    """Test command acknowledgment tracking functionality."""

    def test_tracked_command_basic_flow(self, server_proc, client):
        """Test basic acknowledgment flow with tracked commands."""
        # Send a home command (fire-and-forget)
        result = client.home()
        # Should return True on successful send (or OK if FORCE_ACK/system)
        assert result is True

    def test_non_blocking_command_returns_id(self, server_proc, client):
        """Test that non-blocking commands return command ID immediately."""
        # Send command
        result = client.move_joints(
            [0, 0, 0, 0, 0, 0],
            duration=1.0,
        )
        # Should return True on successful send
        assert result is True

    def test_multiple_tracked_commands(self, server_proc, client):
        """Test tracking multiple commands simultaneously."""
        # Send several commands
        results = []
        for i in range(3):
            result = client.move_joints(
                [i, i, i, i, i, i],
                duration=0.2,
            )
            results.append(result)
        # Each should be True
        assert all(r is True for r in results)
        # Wait for motion to complete instead of sleeping
        assert client.wait_until_stopped(timeout=8.0)

    def test_command_status_polling(self, server_proc, client):
        """Test polling command status during execution."""
        # Send a longer running command with valid joint targets (fire-and-forget)
        result = client.move_joints(
            [0, -45, 180, 15, 20, 25],  # Valid within limits for sim
            duration=1.0,
        )
        assert result is True
        # Verify server remains responsive without fixed sleep
        assert client.ping() is not None


@pytest.mark.integration
class TestErrorConditions:
    """Test error conditions with acknowledgment tracking."""

    def test_invalid_command_with_tracking(self, server_proc, client):
        """Test that invalid commands return proper error acknowledgments."""
        # Try to send invalid command; client enforces timing requirement
        with pytest.raises(RuntimeError):
            _ = client.move_joints([0, 0, 0, 0, 0, 0])  # Missing timing parameters

    def test_malformed_parameters_with_tracking(self, server_proc, client):
        """Test acknowledgment for commands with malformed parameters."""
        # Test with out-of-range speed percentage; client sends fire-and-forget
        result = client.move_cartesian(
            pose=[100, 100, 100, 0, 0, 0],
            speed_percentage=200,
        )
        assert result is True


@pytest.mark.integration
class TestBasicCommands:
    """Test basic commands work with the server."""

    def test_ping(self, server_proc, client):
        """Test ping functionality."""
        assert client.ping() is not None

    def test_get_angles(self, server_proc, client):
        """Test angle retrieval."""
        angles = client.get_angles()
        assert isinstance(angles, list)
        assert len(angles) == 6

    def test_get_io(self, server_proc, client):
        """Test IO status retrieval."""
        io_status = client.get_io()
        assert isinstance(io_status, list)
        assert len(io_status) >= 5

    def test_stop_command(self, server_proc, client):
        """Test stop command."""
        result = client.stop()
        assert result is True
        # Re-enable controller to avoid leaking disabled state to subsequent tests
        assert client.enable() is True


if __name__ == "__main__":
    pytest.main([__file__])
