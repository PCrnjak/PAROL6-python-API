"""
Integration tests for motion profile commands.

Tests SETPROFILE/GETPROFILE through the client API with a running server.
"""

import threading
import time

import numpy as np
import pytest


@pytest.mark.integration
class TestProfileCommands:
    """Test motion profile get/set commands."""

    def test_get_profile_returns_default(self, client, server_proc):
        """Test GETPROFILE returns default profile (TOPPRA) after reset."""
        client.reset()
        profile = client.get_profile()
        assert profile is not None
        assert profile == "TOPPRA"

    def test_set_and_get_profile_roundtrip(self, client, server_proc):
        """Test setting a profile and getting it back."""
        for profile in ["LINEAR", "QUINTIC", "TRAPEZOID", "RUCKIG", "TOPPRA"]:
            assert client.set_profile(profile) > 0
            assert client.get_profile() == profile

    def test_set_profile_case_insensitive(self, client, server_proc):
        """Test that profile names are case-insensitive."""
        assert client.set_profile("linear") > 0
        assert client.get_profile() == "LINEAR"

        assert client.set_profile("Quintic") > 0
        assert client.get_profile() == "QUINTIC"


@pytest.mark.integration
class TestProfileMotionBehavior:
    """Test that different profiles produce correct motion behavior."""

    def test_joint_move_reaches_target_all_profiles(self, client, server_proc):
        """Test that joint moves reach target position with all profiles."""
        target_angles = [10, -50, 190, 5, 10, 15]

        for profile in ["LINEAR", "QUINTIC", "TRAPEZOID", "RUCKIG", "TOPPRA"]:
            # Reset to home first
            client.home(wait=True)

            # Set profile and execute move
            assert client.set_profile(profile) > 0
            result = client.moveJ(target_angles, duration=2.0)
            assert result >= 0
            assert client.wait_motion_complete(timeout=10.0)

            # Verify we reached target (within tolerance)
            angles = client.get_angles()
            assert angles is not None
            for i, (actual, target) in enumerate(zip(angles, target_angles)):
                assert abs(actual - target) < 1.0, (
                    f"Profile {profile}: Joint {i} off target "
                    f"(expected {target}, got {actual})"
                )

    def test_cartesian_move_reaches_target_all_profiles(self, client, server_proc):
        """Test that Cartesian moves reach target position with all profiles.

        Note: RUCKIG automatically falls back to TOPPRA for Cartesian moves.
        """
        # Start from home
        client.home(wait=True)
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Target pose (small offset from start)
        target_pose = [
            start_pose[0],
            start_pose[1] + 20,  # Y + 20mm
            start_pose[2],
            start_pose[3],
            start_pose[4],
            start_pose[5],
        ]

        # All profiles should work for Cartesian moves
        # RUCKIG falls back to TOPPRA automatically
        for profile in ["LINEAR", "QUINTIC", "TRAPEZOID", "RUCKIG", "TOPPRA"]:
            # Reset to home first
            client.home(wait=True)

            # Set profile and execute move
            assert client.set_profile(profile) > 0
            result = client.moveL(target_pose, duration=2.0)
            assert result >= 0
            assert client.wait_motion_complete(timeout=10.0)

            # Verify position reached (within tolerance)
            pose = client.get_pose_rpy()
            assert pose is not None
            assert abs(pose[0] - target_pose[0]) < 1.0, (
                f"Profile {profile}: X position off target "
                f"(expected {target_pose[0]:.1f}, got {pose[0]:.1f})"
            )


@pytest.mark.integration
class TestServoCartesian:
    """Test servo Cartesian motion (replaces old streaming mode tests)."""

    def test_servoL_sequential(self, client, server_proc):
        """Test sequential servoL moves."""
        client.home(wait=True)
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Send a sequence of servo Cartesian commands (fire-and-forget)
        for i in range(5):
            target = [
                start_pose[0] + (i * 5),
                start_pose[1],
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5],
            ]
            result = client.servoL(target, speed=0.5)
            assert result > 0
            time.sleep(0.1)

        assert client.wait_motion_complete(timeout=10.0)

        # Verify robot completed motion
        assert client.is_robot_stopped()


@pytest.mark.integration
class TestCartesianPrecision:
    """Test Cartesian move precision with different profiles."""

    @pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR", "QUINTIC", "TRAPEZOID"])
    def test_cartesian_simple_sequence(self, client, server_proc, profile):
        """
        Test precision of simple Cartesian moves with all profiles.

        All profiles should handle Cartesian paths correctly.
        LINEAR uses uniform time distribution, which may require longer durations.
        """
        client.home(wait=True)
        assert client.set_profile(profile) > 0

        # Get current pose after homing to build moves relative to it
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Use smaller offset for LINEAR to keep duration reasonable
        if profile == "LINEAR":
            offset = 20.0
        else:
            offset = 50.0

        # Move relative to home pose (just Y offset, keep orientation)
        moves = [
            [
                start_pose[0],
                start_pose[1] + offset,
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5],
            ],
            [
                start_pose[0],
                start_pose[1] - offset,
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5],
            ],
        ]

        for target in moves:
            result = client.moveL(target, duration=2.0)
            assert result >= 0
            assert client.wait_motion_complete(timeout=15.0)

        # Verify final pose
        pose = client.get_pose_rpy()
        assert pose is not None
        final_target = moves[-1]

        # Print diagnostic info
        print(f"\nProfile {profile}:")
        print(
            f"  Target:   X={final_target[0]:.2f}, Y={final_target[1]:.2f}, Z={final_target[2]:.2f}"
        )
        print(
            f"            RX={final_target[3]:.2f}, RY={final_target[4]:.2f}, RZ={final_target[5]:.2f}"
        )
        print(f"  Actual:   X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}")
        print(f"            RX={pose[3]:.2f}, RY={pose[4]:.2f}, RZ={pose[5]:.2f}")

        # Check position (X, Y, Z) within 1mm tolerance
        for i, (actual, expected) in enumerate(zip(pose[:3], final_target[:3])):
            assert abs(actual - expected) < 1.0, (
                f"Profile {profile}: Position[{i}] off target "
                f"(expected {expected:.2f}, got {actual:.2f})"
            )

        # Check orientation (RX, RY, RZ) within 1 degree tolerance
        for i, (actual, expected) in enumerate(zip(pose[3:], final_target[3:])):
            assert abs(actual - expected) < 1.0, (
                f"Profile {profile}: Orientation[{i}] off target "
                f"(expected {expected:.2f}, got {actual:.2f})"
            )


def _point_to_line_distance(
    point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> float:
    """Calculate perpendicular distance from a point to a line segment."""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return float(np.linalg.norm(point - line_start))

    line_unit = line_vec / line_len
    point_vec = point - line_start

    # Project point onto line
    projection_len = np.dot(point_vec, line_unit)

    # Clamp to line segment
    projection_len = max(0, min(line_len, projection_len))

    # Find closest point on line
    closest_point = line_start + projection_len * line_unit

    return float(np.linalg.norm(point - closest_point))


def _extract_position_from_pose_matrix(pose_flat: list[float]) -> np.ndarray:
    """Extract XYZ position from flattened 4x4 transformation matrix."""
    # Row-major 4x4 matrix: translation is at indices 3, 7, 11
    return np.array([pose_flat[3], pose_flat[7], pose_flat[11]])


@pytest.mark.integration
class TestTCPPathAccuracy:
    """Test that Cartesian moves follow straight-line TCP paths."""

    @pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR", "QUINTIC", "TRAPEZOID"])
    def test_cartesian_follows_straight_line(self, client, server_proc, profile):
        """
        Verify TCP follows a straight line during Cartesian moves.

        Samples TCP position during motion and checks that all points
        lie within tolerance of the expected straight-line path.
        """
        client.home(wait=True)
        assert client.set_profile(profile) > 0

        start_pose = client.get_pose_rpy()
        assert start_pose is not None
        start_xyz = np.array(start_pose[:3])

        # Move along Y axis - smaller offset for LINEAR to keep duration reasonable
        if profile == "LINEAR":
            offset = 20.0
        else:
            offset = 50.0
        target_pose = [
            start_pose[0],
            start_pose[1] + offset,
            start_pose[2],
            start_pose[3],
            start_pose[4],
            start_pose[5],
        ]
        target_xyz = np.array(target_pose[:3])

        # Collect TCP positions during motion
        sampled_positions: list[np.ndarray] = []
        sampling_done = threading.Event()

        def sample_positions():
            """Background thread to sample TCP positions."""
            while not sampling_done.is_set():
                status = client.get_status()
                if status:
                    pos = _extract_position_from_pose_matrix(status.pose)
                    sampled_positions.append(pos)
                time.sleep(0.02)  # 50 Hz sampling

        # Start sampling thread
        sampler = threading.Thread(target=sample_positions, daemon=True)
        sampler.start()

        # Execute move
        result = client.moveL(target_pose, duration=2.0)
        assert result >= 0
        assert client.wait_motion_complete(timeout=10.0)

        # Stop sampling
        sampling_done.set()
        sampler.join(timeout=1.0)

        # Need at least a few samples to validate path
        assert len(sampled_positions) >= 5, (
            f"Only got {len(sampled_positions)} samples, need at least 5"
        )

        # Calculate max deviation from straight line
        max_deviation = 0.0
        deviations = []
        for pos in sampled_positions:
            dist = _point_to_line_distance(pos, start_xyz, target_xyz)
            deviations.append(dist)
            max_deviation = max(max_deviation, dist)

        # Print diagnostic info
        print(f"\nProfile {profile}:")
        print(f"  Samples collected: {len(sampled_positions)}")
        print(f"  Max path deviation: {max_deviation:.3f} mm")
        print(f"  Mean path deviation: {np.mean(deviations):.3f} mm")

        tolerance_mm = 0.1
        assert max_deviation < tolerance_mm, (
            f"Profile {profile}: TCP deviated {max_deviation:.3f}mm from straight line "
            f"(tolerance: {tolerance_mm}mm)"
        )
