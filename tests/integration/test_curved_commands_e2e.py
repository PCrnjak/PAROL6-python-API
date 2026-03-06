"""
Integration tests for curved motion commands (moveC, moveS, moveP).

Tests command acceptance, completion, and frame handling in FAKE_SERIAL mode.
"""

import numpy as np
import pytest

from parol6.motion.geometry import compute_circle_from_3_points


@pytest.mark.integration
class TestCurvedMotionCommands:
    """Integration tests for moveC, moveS, moveP."""

    @pytest.fixture
    def homed_robot(self, client, server_proc, robot_api_env):
        """Ensure robot is homed before tests."""
        client.set_profile("TOPPRA")
        assert client.home() >= 0
        assert client.wait_motion_complete(timeout=15.0)

    @pytest.fixture
    def home_pose(self, client, homed_robot) -> list[float]:
        """Return the TCP pose at home [x, y, z, rx, ry, rz] (mm, degrees)."""
        pose = client.get_pose_rpy()
        assert pose is not None and len(pose) == 6
        return pose

    def _offset(self, pose: list[float], dx=0, dy=0, dz=0, drz=0) -> list[float]:
        """Small position offset from a base pose, preserving orientation."""
        return [
            pose[0] + dx,
            pose[1] + dy,
            pose[2] + dz,
            pose[3],
            pose[4],
            pose[5] + drz,
        ]

    def test_moveC_basic(self, client, server_proc, robot_api_env, home_pose):
        """Test circular arc through current → via → end."""
        result = client.moveC(
            via=self._offset(home_pose, dx=10, dy=10, dz=-5),
            end=self._offset(home_pose, dx=20),
            duration=2.0,
            frame="WRF",
        )
        assert result >= 0
        assert client.wait_motion_complete(timeout=9.0)
        assert client.is_robot_stopped(threshold_speed=5.0)

    def test_moveC_with_orientation(
        self, client, server_proc, robot_api_env, home_pose
    ):
        """Test arc with small orientation change."""
        result = client.moveC(
            via=self._offset(home_pose, dx=10, dy=10, dz=-5, drz=10),
            end=self._offset(home_pose, dx=20, drz=20),
            duration=2.0,
            frame="WRF",
        )
        assert result >= 0
        assert client.wait_motion_complete(timeout=15.0)
        assert client.is_robot_stopped(threshold_speed=5.0)

    def test_moveC_trf_accepted(self, client, server_proc, robot_api_env, homed_robot):
        """Test that moveC with frame=TRF is accepted and completes."""
        result = client.moveC(
            via=[10, 5, 0, 0, 0, 0],
            end=[20, 0, 0, 0, 0, 0],
            duration=2.0,
            frame="TRF",
        )
        assert result >= 0
        assert client.wait_motion_complete(timeout=15.0)
        assert client.is_robot_stopped(threshold_speed=5.0)

    def test_moveS_basic(self, client, server_proc, robot_api_env, home_pose):
        """Test spline motion through waypoints."""
        waypoints = [
            self._offset(home_pose),
            self._offset(home_pose, dx=20, dy=8, dz=5),
            self._offset(home_pose),
        ]
        result = client.moveS(waypoints=waypoints, duration=3.0, frame="WRF")
        assert result >= 0
        assert client.wait_motion_complete(timeout=15.0)
        assert client.is_robot_stopped(threshold_speed=5.0)

    def test_moveS_trf_accepted(self, client, server_proc, robot_api_env, homed_robot):
        """Test that moveS with frame=TRF is accepted and completes."""
        waypoints = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        result = client.moveS(waypoints=waypoints, duration=3.0, frame="TRF")
        assert result >= 0
        assert client.wait_motion_complete(timeout=15.0)
        assert client.is_robot_stopped(threshold_speed=5.0)

    def test_moveP_basic(self, client, server_proc, robot_api_env, home_pose):
        """Test process move through waypoints with constant TCP speed."""
        waypoints = [
            self._offset(home_pose),
            self._offset(home_pose, dx=5, dy=3),
            self._offset(home_pose),
        ]
        result = client.moveP(waypoints=waypoints, speed=0.3, frame="WRF")
        assert result >= 0
        assert client.wait_motion_complete(timeout=15.0)
        assert client.is_robot_stopped(threshold_speed=5.0)

    def test_moveP_trf_accepted(self, client, server_proc, robot_api_env, homed_robot):
        """Test that moveP with frame=TRF is accepted and completes."""
        waypoints = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        result = client.moveP(waypoints=waypoints, speed=0.3, frame="TRF")
        assert result >= 0
        assert client.wait_motion_complete(timeout=15.0)
        assert client.is_robot_stopped(threshold_speed=5.0)


class TestComputeCircleFrom3Points:
    """Unit tests for compute_circle_from_3_points geometry."""

    def test_known_circle(self):
        """Three points on a known circle produce correct center and radius."""
        # Points on circle centered at (0,0,0) with radius 10 in XY plane
        p1 = np.array([10.0, 0.0, 0.0])
        p2 = np.array([0.0, 10.0, 0.0])
        p3 = np.array([-10.0, 0.0, 0.0])

        center, radius, normal = compute_circle_from_3_points(p1, p2, p3)

        np.testing.assert_allclose(center, [0, 0, 0], atol=1e-10)
        assert abs(radius - 10.0) < 1e-10
        np.testing.assert_allclose(abs(normal), [0, 0, 1], atol=1e-10)

    def test_collinear_raises(self):
        """Collinear points raise ValueError."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="collinear"):
            compute_circle_from_3_points(p1, p2, p3)

    def test_coincident_raises(self):
        """Coincident points raise ValueError."""
        p = np.array([5.0, 5.0, 5.0])

        with pytest.raises(ValueError):
            compute_circle_from_3_points(p, p, p)

    def test_3d_plane(self):
        """Points in a tilted 3D plane produce correct radius."""
        # Circle of radius 5 in a tilted plane
        p1 = np.array([5.0, 0.0, 0.0])
        p2 = np.array([0.0, 5.0, 0.0])
        p3 = np.array([0.0, 0.0, 5.0])

        center, radius, normal = compute_circle_from_3_points(p1, p2, p3)

        # All points should be equidistant from center
        assert abs(np.linalg.norm(p1 - center) - radius) < 1e-10
        assert abs(np.linalg.norm(p2 - center) - radius) < 1e-10
        assert abs(np.linalg.norm(p3 - center) - radius) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
