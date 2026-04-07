"""Unit tests for DryRunRobotClient blend buffering."""

import numpy as np
import pytest

from parol6.client.dry_run_client import DryRunRobotClient

# Valid PAROL6 joint angles (deg) within limits:
#   J1: [-123, 123], J2: [-145, -3.375], J3: [107.9, 287.9],
#   J4: [-105, 105], J5: [-90, 90], J6: [0, 360]
# Home/standby is [90, -90, 180, 0, 0, 180].
W0 = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
W1 = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
W2 = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]
W3 = [60.0, -60.0, 210.0, 30.0, 30.0, 210.0]


@pytest.fixture
def client():
    return DryRunRobotClient()


class TestDryRunBlend:
    """Tests for blend buffering in DryRunRobotClient."""

    def test_blend_produces_composite(self, client):
        """3x move_j with r > 0 should buffer, then flush returns a single composite."""
        r1 = client.move_j(angles=W1, speed=0.5, r=10)
        assert r1 is None, "r > 0 should buffer, not return immediately"

        r2 = client.move_j(angles=W2, speed=0.5, r=10)
        assert r2 is None, "r > 0 should buffer"

        # r=0 terminates the chain → flush returns composite result
        r3 = client.move_j(angles=W3, speed=0.5, r=0)
        assert r3 is not None, "r=0 after buffered commands should flush and return"
        assert r3.tcp_poses.shape[0] > 0
        assert r3.tcp_poses.shape[1] == 6
        assert r3.error is None

    def test_no_blend_without_radius(self, client):
        """move_j with r=0 should return immediately (no buffering)."""
        result = client.move_j(angles=W1, speed=0.5, r=0)
        assert result is not None, "r=0 should return immediately"
        assert result.tcp_poses.shape[0] > 0
        assert result.error is None

    def test_flush_returns_buffered(self, client):
        """Explicit flush() after buffered commands should return results list."""
        r1 = client.move_j(angles=W1, speed=0.5, r=10)
        assert r1 is None

        r2 = client.move_j(angles=W2, speed=0.5, r=10)
        assert r2 is None

        results = client.flush()
        assert len(results) > 0, "flush() should return buffered results"
        assert results[0].tcp_poses.shape[0] > 0
        assert results[0].error is None

    def test_flush_empty_returns_empty_list(self, client):
        """flush() with no buffered commands should return empty list."""
        assert client.flush() == []

    def test_blended_trajectory_is_longer(self, client):
        """Composite blended trajectory should have longer duration than a single move."""
        single = DryRunRobotClient()
        single_result = single.move_j(angles=W3, speed=0.3, r=0)
        assert single_result is not None

        # Blended chain of 3 moves
        client.move_j(angles=W1, speed=0.3, r=10)
        client.move_j(angles=W2, speed=0.3, r=10)
        r3 = client.move_j(angles=W3, speed=0.3, r=0)
        assert r3 is not None

        assert r3.duration > single_result.duration, (
            f"Blended ({r3.duration:.3f}s) should be longer than single ({single_result.duration:.3f}s)"
        )

    def test_state_updated_after_blend(self, client):
        """Position should reflect the final waypoint after a blended chain."""
        client.move_j(angles=W1, speed=0.5, r=10)
        client.move_j(angles=W2, speed=0.5, r=0)

        angles_after = client.angles()
        assert len(angles_after) == 6
        np.testing.assert_allclose(angles_after, W2, atol=0.5)
