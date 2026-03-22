"""
Integration tests for jog commands at speed extremes.

These tests verify that jog commands actually move the robot at both
the slowest (1%) and fastest (100%) speed settings. This catches issues
where tolerance settings (e.g., IK tolerance) might prevent movement
at certain speeds.
"""

import pytest

from parol6 import RobotClient


@pytest.mark.integration
class TestJogSpeedExtremes:
    """Test jog commands at minimum and maximum speeds."""

    def test_jog_joint_slowest_speed_moves_robot(
        self, client: RobotClient, server_proc
    ):
        """
        Jog at slowest speed (1%) should still move the robot.

        This test ensures that even at minimum speed percentage, the robot
        actually moves. This catches issues where tolerance settings might
        prevent movement at low speeds.
        """
        # Get initial joint angles
        initial_angles = client.get_angles()
        assert initial_angles is not None, "Failed to get initial angles"
        assert len(initial_angles) == 6, "Expected 6 joint angles"

        # Jog J1 at slowest speed (1%) for a short duration
        result = client.jogJ(
            joint=0,
            speed=0.01,  # Slowest speed
            duration=0.5,
        )
        assert result > 0, "Jog command failed to send"

        # Wait for motion to complete plus some settling time
        client.wait_motion_complete(timeout=10, settle_window=1)

        # Get final joint angles
        final_angles = client.get_angles()
        assert final_angles is not None, "Failed to get final angles"

        # Verify J1 actually moved
        angle_change = abs(final_angles[0] - initial_angles[0])
        assert angle_change > 0.001, (
            f"Expected J1 to move at slowest speed (1%), but angle changed by only "
            f"{angle_change:.4f} degrees (initial={initial_angles[0]:.4f}, "
            f"final={final_angles[0]:.4f})"
        )

    def test_jog_joint_fastest_speed_moves_robot(
        self, client: RobotClient, server_proc
    ):
        """
        Jog at fastest speed (100%) should move the robot.

        This test ensures that at maximum speed percentage, the robot
        moves as expected and produces more movement than the slow test.
        """
        # Get initial joint angles
        initial_angles = client.get_angles()
        assert initial_angles is not None, "Failed to get initial angles"
        assert len(initial_angles) == 6, "Expected 6 joint angles"

        # Jog J1 at fastest speed (100%) for a short duration
        result = client.jogJ(
            joint=0,
            speed=1.0,  # Fastest speed
            duration=0.5,
        )
        assert result > 0, "Jog command failed to send"

        # Wait for motion to complete plus some settling time
        client.wait_motion_complete(timeout=10)

        # Get final joint angles
        final_angles = client.get_angles()
        assert final_angles is not None, "Failed to get final angles"

        # Verify J1 actually moved
        angle_change = abs(final_angles[0] - initial_angles[0])
        assert angle_change > 0.1, (
            f"Expected J1 to move significantly at fastest speed (100%), but angle "
            f"changed by only {angle_change:.4f} degrees (initial={initial_angles[0]:.4f}, "
            f"final={final_angles[0]:.4f})"
        )

    def test_jog_faster_speed_moves_more(self, client: RobotClient, server_proc):
        """
        Verify that faster speed produces more movement in the same duration.

        This test compares movement at slow (10%) vs fast (90%) speeds to
        confirm the speed parameter actually affects motion rate.
        """
        # First, jog at slow speed and measure movement
        initial_angles_slow = client.get_angles()
        assert initial_angles_slow is not None

        result = client.jogJ(joint=1, speed=0.1, duration=1.0)
        assert result > 0
        client.wait_motion_complete(timeout=10)

        final_angles_slow = client.get_angles()
        assert final_angles_slow is not None
        slow_movement = abs(final_angles_slow[1] - initial_angles_slow[1])

        # Now jog at fast speed
        initial_angles_fast = client.get_angles()
        assert initial_angles_fast is not None

        result = client.jogJ(joint=1, speed=0.9, duration=1.0)
        assert result > 0
        client.wait_motion_complete(timeout=10)

        final_angles_fast = client.get_angles()
        assert final_angles_fast is not None
        fast_movement = abs(final_angles_fast[1] - initial_angles_fast[1])

        # Fast movement should be significantly more than slow movement
        # Using a ratio check rather than absolute values for robustness
        assert fast_movement > slow_movement * 2, (
            f"Expected fast speed (90%) to move at least 2x more than slow speed (10%), "
            f"but slow={slow_movement:.4f}° and fast={fast_movement:.4f}°"
        )


@pytest.mark.integration
class TestCartesianJogSpeedExtremes:
    """Test Cartesian jog commands at minimum and maximum speeds."""

    def test_cart_jog_slowest_speed_moves_robot(self, client: RobotClient, server_proc):
        """
        Cartesian jog at slowest speed (1%) should still move the robot.

        This test ensures that even at minimum speed percentage, Cartesian
        jogging actually moves the robot. This catches issues where IK
        tolerance settings might prevent movement at low speeds.
        """
        # Get initial pose
        initial_pose = client.get_pose_rpy()
        assert initial_pose is not None, "Failed to get initial pose"
        assert len(initial_pose) == 6, "Expected 6-element pose [x,y,z,rx,ry,rz]"

        # Cartesian jog in +Y direction at slowest speed
        result = client.jogL(
            frame="WRF",
            axis="Y",
            speed=0.02,
            duration=1,
        )
        assert result > 0, "Cartesian jog command failed to send"

        # Wait for motion to complete plus some settling time
        client.wait_motion_complete(timeout=10)

        # Get final pose
        final_pose = client.get_pose_rpy()
        assert final_pose is not None, "Failed to get final pose"

        # Verify position actually changed (check Y coordinate)
        position_change = abs(final_pose[1] - initial_pose[1])
        assert position_change > 0.001, (
            f"Expected Y position to change at slowest cart jog speed (1%), but "
            f"changed by only {position_change:.4f} mm (initial={initial_pose[1]:.4f}, "
            f"final={final_pose[1]:.4f})"
        )

    def test_cart_jog_fastest_speed_moves_robot(self, client: RobotClient, server_proc):
        """
        Cartesian jog at fastest speed (100%) should move the robot significantly.

        This test ensures that at maximum speed percentage, Cartesian jogging
        produces significant movement.
        """
        # Get initial pose
        initial_pose = client.get_pose_rpy()
        assert initial_pose is not None, "Failed to get initial pose"
        assert len(initial_pose) == 6, "Expected 6-element pose [x,y,z,rx,ry,rz]"

        # Cartesian jog in +X direction at fastest speed
        result = client.jogL(
            frame="WRF",
            axis="X",
            speed=1.0,
            duration=0.5,
        )
        assert result > 0, "Cartesian jog command failed to send"

        # Wait for motion to complete plus some settling time
        client.wait_motion_complete(timeout=10)

        # Get final pose
        final_pose = client.get_pose_rpy()
        assert final_pose is not None, "Failed to get final pose"

        # Verify position actually changed significantly
        position_change = abs(final_pose[0] - initial_pose[0])
        assert position_change > 0.5, (
            f"Expected significant X position change at fastest cart jog speed (100%), "
            f"but changed by only {position_change:.4f} mm (initial={initial_pose[0]:.4f}, "
            f"final={final_pose[0]:.4f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
