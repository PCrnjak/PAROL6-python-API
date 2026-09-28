"""
Integration smoke tests for UDP communication using parol6.
Covers PING/PONG, GET_* endpoints, STOP semantics, and basic functionality.
"""

import math
import socket
import time

import pytest

from parol6 import RobotClient
from parol6.config import LIMITS
from waldoctl import ActionState


@pytest.mark.integration
class TestBasicCommunication:
    """Test basic UDP communication with the server."""

    def test_ping_pong(self, client, server_proc):
        """Test PING/PONG communication."""
        assert client.ping()


@pytest.mark.integration
class TestGetEndpoints:
    """Test GET_* command endpoints that return immediate data."""

    def test_pose(self, client, server_proc):
        """Test POSE command — returns [x, y, z, rx, ry, rz]."""
        pose = client.pose()
        assert pose is not None
        assert isinstance(pose, list)
        assert len(pose) == 6  # [x, y, z, rx, ry, rz]

        pose_xyz = pose[:3]
        assert len(pose_xyz) == 3

    def test_angles(self, client, server_proc):
        """Test ANGLES command."""
        angles = client.angles()
        assert angles is not None
        assert isinstance(angles, list)
        assert len(angles) == 6  # 6 joint angles

    def test_io(self, client, server_proc):
        """Test IO command."""
        io_status = client.io()
        assert io_status is not None
        assert isinstance(io_status, list)
        assert len(io_status) == 5  # IN1, IN2, OUT1, OUT2, ESTOP

        # In FAKE_SERIAL mode, ESTOP should be released (1)
        assert io_status[4] == 1

        # Test helper method too
        assert not client.is_estop_pressed()  # Should be False in FAKE_SERIAL

    def test_joint_speeds_read_deg_per_second(self, client, server_proc):
        """While J1 jogs at full speed, the status stream and
        ``joint_speeds()`` read its jog velocity limit in deg/s, the other
        joints read still, and once the jog ends the arm reads stopped."""
        rate = math.degrees(LIMITS.joint.jog.velocity[0])
        assert client.is_robot_stopped()
        assert client.jog_j(0, -1.0, duration=1.5, accel=1.0) == 1
        assert client.wait_status(
            lambda s: abs(s.speeds[0]) > 0.9 * rate, timeout=2.0
        ), f"the status stream never read J1 near its {rate:.1f} deg/s jog"
        peak = 0.0
        others = 0.0
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            speeds = client.joint_speeds()
            assert speeds is not None
            peak = max(peak, abs(speeds[0]))
            others = max(others, *(abs(v) for v in speeds[1:]))
            time.sleep(0.02)
        assert peak == pytest.approx(rate, rel=0.05), (
            f"J1 read {peak:.2f} at a jog of {rate:.2f} deg/s"
        )
        assert others < 0.5, f"a still joint read {others:.2f} deg/s"
        assert client.wait_motion(timeout=5.0)
        assert client.is_robot_stopped()

    def test_status_aggregate(self, client, server_proc):
        """STATUS reads what the single queries read of the arm at rest, and
        the tool the reset left fitted."""
        status = client.status()
        angles = client.angles()
        pose = client.pose()
        io = client.io()
        assert status is not None and angles and pose and io
        assert status.angles == pytest.approx(angles, abs=1e-6)
        assert [status.pose[3], status.pose[7], status.pose[11]] == pytest.approx(
            pose[:3], abs=1e-3
        )
        assert status.speeds == pytest.approx([0.0] * 6, abs=0.5)
        assert status.io == io
        assert status.tool_status.key == "NONE"

    def test_activity_names_the_move_under_way(self, client, server_proc):
        """ACTIVITY reports the move while it plays and idle once it ends."""
        start = client.angles()
        assert start is not None
        there = [start[0] - 10.0, *start[1:]]
        moving = client.move_j(there, duration=2.0, wait=False)
        assert moving >= 0
        assert client.wait_status(lambda s: s.executing_index == moving, timeout=5.0)
        playing = client.activity()
        assert playing is not None
        assert (playing.state, playing.command) == (ActionState.EXECUTING, "move_j")
        assert client.wait_command(moving, timeout=10.0)
        done = client.activity()
        assert done is not None
        assert (done.state, done.command) == (ActionState.IDLE, "")


@pytest.mark.integration
class TestServoMode:
    """Test servo (real-time) mode functionality.

    stream_on/stream_off were removed in the API redesign.
    Servo commands (servo_j/servo_l) replaced streaming mode.
    """

    def test_servo_joint_basic(self, client, server_proc):
        """Test that servo_j command is accepted."""
        # servo_j sends a single real-time joint target
        result = client.servo_j([0, -45, 180, 0, 0, 180], speed=0.5, accel=0.5)
        assert result > 0
        assert client.ping() is not None


@pytest.mark.integration
class TestBasicMotionCommands:
    """Test basic motion commands with improved assertions."""

    def test_home_command(self, client, server_proc):
        """Test HOME command (fire-and-forget)."""
        result = client.home()
        assert result >= 0

        # Wait for completion and verify robot stops
        assert client.wait_motion(timeout=15.0)

        # Check that robot is responsive after homing
        assert client.ping() is not None

        # Check that angles are available after homing
        angles = client.angles()
        assert angles is not None
        assert len(angles) == 6

    def test_basic_joint_move(self, client, server_proc):
        """Test basic joint movement command (fire-and-forget)."""
        # Use joint angles that are within the robot's limits
        # Joint 2 range: [-145.0088, -3.375]
        # Joint 3 range: [107.866, 287.8675]
        result = client.move_j(
            [0, -45, 180, 15, 20, 25],  # Valid angles within joint limits
            duration=2.0,
        )
        assert result >= 0

        # Wait for completion and verify robot stops
        assert client.wait_motion(timeout=15.0)

        # Verify robot state after move attempt
        angles = client.angles()
        assert angles is not None
        assert client.ping() is not None

    def test_joint_move_with_speed(self, client, server_proc):
        """Test basic joint movement command with validation."""
        result = client.move_j(
            [80, -80, 170, 5, 5, 190],
            speed=0.5,
        )
        assert result >= 0

        # Wait for completion and verify robot stops
        assert client.wait_motion(timeout=15.0)

        # Verify robot state
        pose = client.pose()
        assert pose is not None
        assert len(pose) == 6

    def test_cartesian_move_validation(self, client, server_proc):
        """Test cartesian movement with proper validation."""
        from parol6.utils.errors import MotionError

        # Unreachable pose — planner surfaces IK failure via MotionError
        with pytest.raises(MotionError):
            client.move_l(
                [50, 50, 50, 0, 0, 0],
                duration=2.0,
            )


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_command_format(self, server_proc, ports):
        """A command body the codec cannot read is refused to the id that sent
        it, and a datagram with no request id at all is dropped without
        unsettling the controller."""
        from parol6.protocol.wire import ErrorMsg, decode_message, encode

        req_id = 4242
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(2.0)
            # 9999 is not a CmdType: the envelope is well formed, the body is not.
            body = encode([9999, "invalid_param"])
            sock.sendto(
                req_id.to_bytes(4, "big") + body, (ports.server_ip, ports.server_port)
            )

            reply = decode_message(sock.recvfrom(1024)[0])
            assert isinstance(reply, ErrorMsg)
            assert reply.req_id == req_id, "the refusal must name the request"
            # message is a RobotError wire list: [cmd_idx, code, title, cause, …]
            assert isinstance(reply.message, list)
            assert any("9999" in str(f) or "Invalid" in str(f) for f in reply.message)

            # A datagram too short to carry an id: nothing to reply to.
            sock.sendto(b"\x00\x01", (ports.server_ip, ports.server_port))
            with pytest.raises(socket.timeout):
                sock.recvfrom(1024)

        # Server should remain responsive after handling the error
        client = RobotClient(ports.server_ip, ports.server_port)
        assert client.ping() is not None

    def test_estopped_motion_raises_motion_error(self, client, server_proc):
        """Motion commands on an estopped controller raise MotionError until
        reset() clears the latch."""
        from parol6.utils.errors import MotionError

        client.estop()
        try:
            with pytest.raises(MotionError) as exc_info:
                client.home()
            assert exc_info.value.robot_error.code > 0
            assert exc_info.value.robot_error.title
        finally:
            client.reset()

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
        assert client.wait_motion(timeout=10.0)

        # Server should be responsive after sequence
        assert client.ping() is not None

        # Total time should be reasonable (commands + processing overhead)
        total_time = __import__("time").time() - start_time
        assert total_time < 5.0  # Should complete within reasonable time


if __name__ == "__main__":
    pytest.main([__file__])
