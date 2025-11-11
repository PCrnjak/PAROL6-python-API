"""
Integration test for MoveCart idempotence.
Verifies that moving to the current pose results in no motion (angular distance ≈ 0).
"""

import os
import sys

import numpy as np
import pytest

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.integration
class TestMoveCartIdempotence:
    """Test that MoveCart to current pose results in zero movement."""

    def test_movecart_to_current_pose_no_rotation(self, client, server_proc):
        """Test that moving to the current pose results in no rotation."""
        # Home the robot first
        assert client.home() is True
        assert client.wait_until_stopped(timeout=10.0)

        # Get current pose (should be home position)
        current_pose = client.get_pose_rpy()
        print(f"\nCurrent pose at home (mm, deg): {current_pose}")

        # Move to the exact same pose - should result in zero angular distance
        # and effectively be a no-op
        result = client.move_cartesian(current_pose, speed_percentage=50)
        assert result is True

        # Wait for completion (should be instant if duration is ~0)
        assert client.wait_until_stopped(timeout=5.0)

        # Get final pose
        final_pose = client.get_pose_rpy()
        print(f"Final pose after 'move to same' (mm, deg): {final_pose}")

        # Verify pose hasn't changed significantly
        # Position tolerance: 0.1mm (very strict since we didn't move)
        pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(current_pose[:3]))
        print(f"Position error: {pos_error:.4f} mm")
        assert pos_error < 0.1, f"Position changed by {pos_error:.4f}mm when moving to same pose"

        # Orientation tolerance: 0.5 degrees per axis (accounting for numerical precision)
        def angle_diff(a, b):
            """Compute smallest angle difference considering wrapping."""
            diff = (a - b + 180) % 360 - 180
            return abs(diff)

        for i, axis in enumerate(["RX", "RY", "RZ"]):
            ori_error = angle_diff(final_pose[3 + i], current_pose[3 + i])
            print(f"{axis} error: {ori_error:.4f} deg")
            assert ori_error < 0.5, f"{axis} changed by {ori_error:.4f}° when moving to same pose"

        print("✓ MoveCart idempotence test passed - no unwanted rotation!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
