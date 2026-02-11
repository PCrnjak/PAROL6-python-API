"""Integration tests for N-command blend lookahead.

Tests the full pipeline: client → UDP → controller → executor blend peek →
composite trajectory → mock serial execution → final position verification.
"""

import pytest


@pytest.mark.integration
class TestJointBlendLookahead:
    """Joint-space blending with N-command lookahead."""

    def test_three_moveJ_blended_reaches_final_target(self, client, server_proc):
        """Three moveJ with r>0 should blend and reach the last target."""
        targets = [
            [80, -80, 170, 5, 5, 170],
            [70, -70, 160, 10, 10, 160],
            [60, -60, 150, 15, 15, 150],
        ]

        # Queue all without waiting (wait=False so they stack in the queue)
        for t in targets:
            assert client.moveJ(t, speed=0.5, r=30.0, wait=False) >= 0

        # Wait for everything to finish
        assert client.wait_motion_complete(timeout=15.0)

        # Verify final position matches last target
        angles = client.get_angles()
        assert angles is not None
        for i, (actual, expected) in enumerate(zip(angles, targets[-1])):
            assert abs(actual - expected) < 1.0, (
                f"J{i}: expected {expected}, got {actual}"
            )

    def test_moveJ_r0_stops_blend_chain(self, client, server_proc):
        """A moveJ with r=0 in the middle should stop the blend chain."""
        targets = [
            ([80, -80, 170, 5, 5, 170], 30.0),  # blendable
            ([70, -70, 160, 10, 10, 160], 0.0),  # r=0 → hard stop
            ([60, -60, 150, 15, 15, 150], 30.0),  # separate motion
        ]

        for t, r in targets:
            assert client.moveJ(t, speed=0.5, r=r, wait=False) >= 0

        assert client.wait_motion_complete(timeout=15.0)

        angles = client.get_angles()
        assert angles is not None
        for i, (actual, expected) in enumerate(zip(angles, targets[-1][0])):
            assert abs(actual - expected) < 1.0, (
                f"J{i}: expected {expected}, got {actual}"
            )

    def test_two_moveJ_blended(self, client, server_proc):
        """Two moveJ with r>0 should blend (minimum blend chain)."""
        t1 = [80, -80, 170, 5, 5, 170]
        t2 = [70, -70, 160, 10, 10, 160]

        assert client.moveJ(t1, speed=0.5, r=20.0, wait=False) >= 0
        assert client.moveJ(t2, speed=0.5, r=0.0, wait=False) >= 0

        assert client.wait_motion_complete(timeout=15.0)

        angles = client.get_angles()
        assert angles is not None
        for i, (actual, expected) in enumerate(zip(angles, t2)):
            assert abs(actual - expected) < 1.0, (
                f"J{i}: expected {expected}, got {actual}"
            )


@pytest.mark.integration
class TestCartesianBlendLookahead:
    """Cartesian (moveL) blending with N-command lookahead."""

    def test_three_moveL_blended_reaches_final_target(self, client, server_proc):
        """Three moveL with r>0 should blend and reach the last target."""
        start = client.get_pose_rpy()
        assert start is not None

        # Small offsets from current pose (guaranteed reachable)
        targets = [
            [start[0], start[1] + 15, start[2], start[3], start[4], start[5]],
            [start[0], start[1] + 30, start[2], start[3], start[4], start[5]],
            [start[0], start[1] + 45, start[2], start[3], start[4], start[5]],
        ]

        for t in targets:
            assert client.moveL(t, speed=0.5, r=20.0, wait=False) >= 0

        assert client.wait_motion_complete(timeout=15.0)

        final = client.get_pose_rpy()
        assert final is not None
        for i in range(3):
            assert abs(final[i] - targets[-1][i]) < 2.0, (
                f"Axis {i}: expected {targets[-1][i]:.1f}, got {final[i]:.1f}"
            )

    def test_moveL_r0_stops_blend_chain(self, client, server_proc):
        """A moveL with r=0 in the middle should stop the blend chain."""
        start = client.get_pose_rpy()
        assert start is not None

        targets = [
            ([start[0], start[1] + 15, start[2], start[3], start[4], start[5]], 20.0),
            ([start[0], start[1] + 30, start[2], start[3], start[4], start[5]], 0.0),
            ([start[0], start[1] + 45, start[2], start[3], start[4], start[5]], 20.0),
        ]

        for t, r in targets:
            assert client.moveL(t, speed=0.5, r=r, wait=False) >= 0

        assert client.wait_motion_complete(timeout=15.0)

        final = client.get_pose_rpy()
        assert final is not None
        for i in range(3):
            assert abs(final[i] - targets[-1][0][i]) < 2.0


@pytest.mark.integration
class TestMixedTypeBlendTermination:
    """Blend chain should stop at type boundaries."""

    def test_moveJ_then_moveL_executes_separately(self, client, server_proc):
        """moveJ(r>0) followed by moveL should not blend across types."""
        # Small joint move with blend radius
        assert (
            client.moveJ(
                [85, -85, 175, 2, 2, 175],
                speed=0.5,
                r=20.0,
                wait=False,
            )
            >= 0
        )

        # Wait for joint move, then get the pose for a reachable Cartesian target
        assert client.wait_motion_complete(timeout=10.0)
        mid_pose = client.get_pose_rpy()
        assert mid_pose is not None

        # Small Cartesian offset from current position
        final_target = list(mid_pose)
        final_target[1] += 5  # 5mm Y offset — very small, always reachable
        assert client.moveL(final_target, speed=0.5, r=0.0) >= 0

        final = client.get_pose_rpy()
        assert final is not None
        for i in range(3):
            assert abs(final[i] - final_target[i]) < 2.0, (
                f"Axis {i}: expected {final_target[i]:.1f}, got {final[i]:.1f}"
            )
