"""
Integration test for servo Cartesian move accuracy.

Tests the servo_l path used for TCP dragging.
Catches bugs where reference pose gets corrupted (e.g., aliasing with FK cache).
"""

import numpy as np
import pytest


def angle_diff(a: float, b: float) -> float:
    """Compute smallest angle difference considering wrapping."""
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def assert_pose_accuracy(
    final_pose: list[float],
    target: list[float],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 1.0,
    context: str = "",
) -> None:
    """Assert that final pose matches target within tolerances."""
    # Position check
    pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target[:3]))
    assert pos_error < pos_tol_mm, (
        f"{context}Position error {pos_error:.3f}mm exceeds {pos_tol_mm}mm tolerance. "
        f"Target: {target[:3]}, Final: {final_pose[:3]}"
    )

    # Orientation check
    for i, axis in enumerate(["RX", "RY", "RZ"]):
        ori_error = angle_diff(final_pose[3 + i], target[3 + i])
        assert ori_error < ori_tol_deg, (
            f"{context}{axis} error {ori_error:.3f}° exceeds {ori_tol_deg}° tolerance. "
            f"Target: {target[3 + i]:.1f}°, Final: {final_pose[3 + i]:.1f}°"
        )


@pytest.mark.integration
class TestServoCartesianAccuracy:
    """Test that servo cartesian moves reach correct targets."""

    def test_servo_l_reaches_target(self, client, server_proc):
        """servo_l move should arrive at the requested target.

        Tests the servo Cartesian path (replaces old stream_on + move_cartesian).
        """
        assert client.resume() > 0
        assert client.home() >= 0
        assert client.wait_motion(timeout=15.0)

        # Get starting pose
        start_pose = client.pose()
        print(f"\nStart pose: {start_pose}")

        # Target: offset from start (like beginning of a TCP drag)
        target = list(start_pose)
        target[0] += 30.0  # +30mm in X

        print(f"Target pose: {target}")

        # Send servo cartesian move (fire-and-forget, no stream mode toggle needed)
        result = client.servo_l(target, speed=1.0)
        assert result > 0

        # Wait for motion to settle
        assert client.wait_motion(timeout=10.0)

        # Verify final pose
        final_pose = client.pose()
        print(f"Final pose:  {final_pose}")

        assert_pose_accuracy(final_pose, target)

    def test_servo_l_sequential_targets(self, client, server_proc):
        """Sequential servo moves should each reach their target.

        Simulates TCP dragging behavior where multiple servo_l commands
        are sent in sequence.
        """
        assert client.resume() > 0
        assert client.home() >= 0
        assert client.wait_motion(timeout=15.0)

        start_pose = client.pose()
        print(f"\nStart pose: {start_pose}")

        # Simulate a drag path: series of small incremental moves
        # This pattern catches bugs where reference pose gets corrupted
        # between moves (like the FK cache aliasing bug)
        offsets = [
            (30.0, 0.0, 0.0),  # +30mm X
            (30.0, 30.0, 0.0),  # +30mm X, +30mm Y
            (30.0, 30.0, -30.0),  # +30mm X, +30mm Y, -30mm Z
            (0.0, 0.0, 0.0),  # hold position
        ]

        for i, (dx, dy, dz) in enumerate(offsets):
            target = list(start_pose)
            target[0] += dx
            target[1] += dy
            target[2] += dz

            print(f"\n--- Move {i + 1}/{len(offsets)} ---")
            print(f"Target: {target[:3]}")

            result = client.servo_l(target, speed=1.0)
            assert result > 0

            # Wait for this move to complete before next
            assert client.wait_motion(timeout=10.0, settle_window=2.0)

            final_pose = client.pose()
            start_pose = final_pose
            print(f"Final:  {final_pose[:3]}")

            assert_pose_accuracy(final_pose, target, context=f"Move {i + 1}: ")
            print(f"Move {i + 1} accurate")

        print("\nAll sequential servo moves reached targets accurately")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
