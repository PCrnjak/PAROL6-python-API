"""Verify that all motion commands users write in scripts work through the dry run client.

The dry run client uses __getattr__ + build_cmd to dispatch calls by mapping
kwargs to wire struct fields. If the client API param names don't match the
struct field names, the kwargs get silently dropped and the command fails.

This test calls every user-facing motion method with the same signatures shown
in the docs / editor auto-complete, ensuring the dry run path doesn't diverge
from the real client.
"""

import pytest

from parol6.client.dry_run_client import DryRunRobotClient

HOME = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
POSE_A = [0.0, 280.0, 200.0, 90.0, 0.0, 90.0]
POSE_B = [50.0, 280.0, 200.0, 90.0, 0.0, 90.0]
POSE_C = [50.0, 280.0, 250.0, 90.0, 0.0, 90.0]
ANGLES_A = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
ANGLES_B = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]


@pytest.fixture
def client():
    return DryRunRobotClient()


class TestDryRunScriptCompat:
    """Every call signature a user can write in a script must work in dry run."""

    def test_home(self, client):
        result = client.home()
        assert result is not None
        assert result.error is None

    def test_move_j_positional(self, client):
        result = client.move_j(ANGLES_A, speed=0.5)
        assert result is not None
        assert result.error is None

    def test_move_j_angles_kwarg(self, client):
        result = client.move_j(angles=ANGLES_A, speed=0.5)
        assert result is not None
        assert result.error is None

    def test_move_j_with_accel(self, client):
        result = client.move_j(ANGLES_A, speed=0.5, accel=0.8)
        assert result is not None
        assert result.error is None

    def test_move_j_with_duration(self, client):
        result = client.move_j(ANGLES_A, duration=2.0)
        assert result is not None
        assert result.error is None

    def test_move_j_relative(self, client):
        result = client.move_j(ANGLES_A, speed=0.5, rel=True)
        assert result is not None

    def test_move_l_positional(self, client):
        result = client.move_l(POSE_A, speed=0.5)
        assert result is not None
        assert result.error is None

    def test_move_l_with_frame(self, client):
        result = client.move_l(POSE_A, speed=0.5, frame="WRF")
        assert result is not None

    def test_move_c(self, client):
        client.move_l(POSE_A, speed=0.5)
        result = client.move_c(via=POSE_B, end=POSE_A, speed=0.5)
        assert result is not None

    def test_move_s(self, client):
        client.move_l(POSE_A, speed=0.5)
        waypoints = [POSE_A, POSE_B, POSE_C, POSE_A]
        result = client.move_s(waypoints=waypoints, speed=0.5)
        assert result is not None

    def test_move_p(self, client):
        client.move_l(POSE_A, speed=0.5)
        waypoints = [POSE_A, POSE_B, POSE_C, POSE_A]
        result = client.move_p(waypoints=waypoints, speed=0.5)
        assert result is not None

    def test_move_j_blend_radius(self, client):
        """Blend radius queues commands; r=0 flushes."""
        r1 = client.move_j(ANGLES_A, speed=0.5, r=10)
        assert r1 is None  # buffered
        r2 = client.move_j(ANGLES_B, speed=0.5, r=0)
        assert r2 is not None  # flushed

    def test_move_l_blend_radius(self, client):
        r1 = client.move_l(POSE_A, speed=0.5, r=15)
        assert r1 is None
        r2 = client.move_l(POSE_B, speed=0.5, r=0)
        assert r2 is not None

    def test_angles(self, client):
        angles = client.angles()
        assert isinstance(angles, list)
        assert len(angles) == 6

    def test_pose(self, client):
        pose = client.pose()
        assert isinstance(pose, list)
        assert len(pose) == 6

    def test_flush(self, client):
        results = client.flush()
        assert isinstance(results, list)

    def test_delay(self, client):
        # Should be a no-op, not raise
        client.delay(1.0)

    def test_wait_motion(self, client):
        client.move_j(ANGLES_A, speed=0.5)
        client.wait_motion()
