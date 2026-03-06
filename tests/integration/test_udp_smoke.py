"""
Integration smoke tests for UDP communication using parol6.
Covers PING/PONG, GET_* endpoints, STOP semantics, and basic functionality.
"""

import socket

import pytest

from parol6 import RobotClient


@pytest.mark.integration
class TestBasicCommunication:
    """Test basic UDP communication with the server."""

    def test_ping_pong(self, client, server_proc):
        """Test PING/PONG communication."""
        assert client.ping()


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
        from parol6.protocol.wire import StatusResultStruct

        status = client.get_status()
        assert status is not None
        assert isinstance(status, StatusResultStruct)

        # Should contain all status components (as struct attributes)
        assert hasattr(status, "pose")
        assert hasattr(status, "angles")
        assert hasattr(status, "io")
        assert hasattr(status, "tool_status")


@pytest.mark.integration
class TestServoMode:
    """Test servo (real-time) mode functionality.

    stream_on/stream_off were removed in the API redesign.
    Servo commands (servoJ/servoL) replaced streaming mode.
    """

    def test_servo_joint_basic(self, client, server_proc):
        """Test that servoJ command is accepted."""
        # servoJ sends a single real-time joint target
        result = client.servoJ([0, -45, 180, 0, 0, 180], speed=0.5, accel=0.5)
        assert result is True
        assert client.ping() is not None


@pytest.mark.integration
class TestBasicMotionCommands:
    """Test basic motion commands with improved assertions."""

    def test_home_command(self, client, server_proc):
        """Test HOME command (fire-and-forget)."""
        result = client.home()
        assert result >= 0

        # Wait for completion and verify robot stops
        assert client.wait_motion_complete(timeout=15.0)

        # Check that robot is responsive after homing
        assert client.ping() is not None

        # Check that angles are available after homing
        angles = client.get_angles()
        assert angles is not None
        assert len(angles) == 6

    def test_basic_joint_move(self, client, server_proc):
        """Test basic joint movement command (fire-and-forget)."""
        # Use joint angles that are within the robot's limits
        # Joint 2 range: [-145.0088, -3.375]
        # Joint 3 range: [107.866, 287.8675]
        result = client.moveJ(
            [0, -45, 180, 15, 20, 25],  # Valid angles within joint limits
            duration=2.0,
        )
        assert result >= 0

        # Wait for completion and verify robot stops
        assert client.wait_motion_complete(timeout=15.0)

        # Verify robot state after move attempt
        angles = client.get_angles()
        assert angles is not None
        assert client.ping() is not None

    def test_joint_move_with_speed(self, client, server_proc):
        """Test basic joint movement command with validation."""
        result = client.moveJ(
            [80, -80, 170, 5, 5, 190],
            speed=0.5,
        )
        assert result >= 0

        # Wait for completion and verify robot stops
        assert client.wait_motion_complete(timeout=15.0)

        # Verify robot state
        pose = client.get_pose_rpy()
        assert pose is not None
        assert len(pose) == 6

    def test_cartesian_move_validation(self, client, server_proc):
        """Test cartesian movement with proper validation."""
        from parol6.utils.errors import MotionError

        # Test that move requires either duration or speed (struct validates)
        with pytest.raises(ValueError):
            client.moveL([50, 50, 50, 0, 0, 0])  # No duration or speed

        # Unreachable pose — planner surfaces IK failure via MotionError
        with pytest.raises(MotionError):
            client.moveL(
                [50, 50, 50, 0, 0, 0],
                duration=2.0,
            )


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_command_format(self, server_proc, ports):
        """Test server response to invalid binary msgpack commands."""
        from parol6.protocol.wire import MsgType, encode, decode

        # Send invalid command via raw socket with binary msgpack
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2.0)
            # Send an array with invalid command type (9999 is not a valid CmdType)
            msg = encode([9999, "invalid_param"])
            sock.sendto(msg, (ports.server_ip, ports.server_port))

            # Expect error response in array format: [MsgType.ERROR, message]
            data, _ = sock.recvfrom(1024)
            resp = decode(data)
            assert isinstance(resp, (list, tuple))
            assert resp[0] == MsgType.ERROR
            # resp[1] is a RobotError wire list: [cmd_idx, code, title, cause, effect, remedy]
            error_wire = resp[1]
            assert isinstance(error_wire, list)
            assert any("9999" in str(f) or "Invalid" in str(f) for f in error_wire)

        # Server should remain responsive after handling the error
        client = RobotClient(ports.server_ip, ports.server_port)
        assert client.ping() is not None

    def test_halted_motion_raises_motion_error(self, client, server_proc):
        """Motion commands on a halted controller raise MotionError, not -1."""
        from parol6.utils.errors import MotionError

        client.halt()
        try:
            with pytest.raises(MotionError) as exc_info:
                client.home()
            assert exc_info.value.robot_error.code > 0
            assert exc_info.value.robot_error.title
        finally:
            client.resume()

    def test_rapid_command_sequence(self, server_proc, ports):
        """Test server stability under rapid command sequence."""
        client = RobotClient(ports.server_ip, ports.server_port)

        # Send multiple commands rapidly (ping)
        for _ in range(10):
            assert client.ping() is not None

        # Server should still be responsive
        assert client.ping() is not None


@pytest.mark.integration
class TestCommandQueuing:
    """Test basic command queuing behavior."""

    def test_command_sequence_execution(self, server_proc, ports):
        """Test that commands execute in sequence."""
        client = RobotClient(ports.server_ip, ports.server_port)

        start_time = __import__("time").time()

        # Execute sequence using public API
        assert client.home() >= 0
        assert client.delay(0.2) >= 0
        assert client.delay(0.2) >= 0
        assert client.delay(0.2) >= 0

        # Wait for all commands to complete via speeds
        assert client.wait_motion_complete(timeout=10.0)

        # Server should be responsive after sequence
        assert client.ping() is not None

        # Total time should be reasonable (commands + processing overhead)
        total_time = __import__("time").time() - start_time
        assert total_time < 5.0  # Should complete within reasonable time


if __name__ == "__main__":
    pytest.main([__file__])
