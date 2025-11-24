"""
Integration test for MoveCart pose accuracy.
Verifies that movecart commands reach the correct final pose.
"""

import os
import sys

import numpy as np
import pytest

# Skip on macOS CI runners due to flakiness
pytestmark = pytest.mark.skipif(
    sys.platform == "darwin" and os.getenv("CI") == "true",
    reason="Flaky on the slow macOS GitHub Actions runners.; skip on CI",
)

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.integration
class TestMoveCartAccuracy:
    """Test that MoveCart commands reach correct final poses."""

    def test_movecart_from_home(self, client, server_proc):
        """Test MoveCart accuracy starting from home position."""
        # Ensure controller is enabled before motion
        assert client.enable() is True
        # Home the robot first
        assert client.home() is True
        assert client.wait_until_stopped(timeout=15.0)

        # Get home pose for reference
        home_pose = client.get_pose_rpy()
        print(f"\nHome pose (mm, deg): {home_pose}")

        # This is in mm for position, degrees for orientation
        target = [0.000, 263, 242, 90, 0, 90]

        # Execute movecart
        result = client.move_cartesian(target, speed_percentage=50)
        assert result is True

        # Wait for completion
        assert client.wait_until_stopped(timeout=15.0)

        # Get final pose
        final_pose = client.get_pose_rpy()
        print(f"Target pose (mm, deg): {target}")
        print(f"Final pose (mm, deg):  {final_pose}")

        # Verify pose accuracy
        # Position tolerance: 1mm
        pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target[:3]))
        print(f"Position error: {pos_error:.3f} mm")
        assert pos_error < 1.0, (
            f"Position error {pos_error:.3f}mm exceeds 1mm tolerance"
        )

        # Orientation tolerance: 1 degree per axis
        # Note: Need to handle angle wrapping for comparison
        def angle_diff(a, b):
            """Compute smallest angle difference considering wrapping."""
            diff = (a - b + 180) % 360 - 180
            return abs(diff)

        for i, axis in enumerate(["RX", "RY", "RZ"]):
            ori_error = angle_diff(final_pose[3 + i], target[3 + i])
            print(f"{axis} error: {ori_error:.3f} deg")
            assert ori_error < 1.0, (
                f"{axis} error {ori_error:.3f}° exceeds 1 degree tolerance"
            )

        print("✓ MoveCart pose accuracy test passed!")

    def test_movecart_multiple_targets(self, client, server_proc):
        """Test MoveCart accuracy with multiple sequential targets."""
        # Ensure controller is enabled before motion
        assert client.enable() is True
        # Home first
        assert client.home() is True
        assert client.wait_until_stopped(timeout=15.0)

        # Define multiple targets to test
        targets = [
            [0.0, 200.0, 250.0, 90.0, 0, 90.0],
            [50.0, 250.0, 200.0, 90, 0, 90.0],
            [0.0, 263.0, 242.0, 90, 0, 90.0],
        ]

        for idx, target in enumerate(targets):
            print(f"\n--- Target {idx + 1}/{len(targets)} ---")
            print(f"Moving to: {target}")

            # Execute movecart
            result = client.move_cartesian(target, speed_percentage=30)
            assert result is True

            # Wait for completion
            assert client.wait_until_stopped(timeout=15.0)

            # Get final pose
            final_pose = client.get_pose_rpy()
            print(f"Achieved:  {final_pose}")

            # Verify position accuracy (1mm tolerance)
            pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target[:3]))
            print(f"Position error: {pos_error:.3f} mm")
            assert pos_error < 1.0, (
                f"Target {idx + 1}: Position error {pos_error:.3f}mm exceeds 1mm"
            )

            # Verify orientation accuracy (1° tolerance per axis)
            def angle_diff(a, b):
                diff = (a - b + 180) % 360 - 180
                return abs(diff)

            for i, axis in enumerate(["RX", "RY", "RZ"]):
                ori_error = angle_diff(final_pose[3 + i], target[3 + i])
                print(f"{axis} error: {ori_error:.3f} deg")
                assert ori_error < 1.0, (
                    f"Target {idx + 1}: {axis} error {ori_error:.3f}° exceeds 1°"
                )

            print(f"✓ Target {idx + 1} reached successfully")

        print("\n✓ All targets reached with required accuracy!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
