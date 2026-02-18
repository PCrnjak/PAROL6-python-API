"""Integration tests for N-command blend lookahead.

Tests the full pipeline: client → UDP → controller → planner subprocess →
composite trajectory → segment player → mock serial → final position.
"""

import pytest


@pytest.mark.integration
class TestJointBlendLookahead:
    """Joint-space blending with N-command lookahead."""

    def test_three_moveJ_blended_reaches_final_target(self, client, server_proc):
        """Three moveJ with blend zones should reach the last target."""
        targets = [
            [80, -80, 170, 5, 5, 170],
            [70, -70, 160, 10, 10, 160],
            [60, -60, 150, 15, 15, 150],
        ]

        # r>0 on intermediate commands creates blend zones; r=0 on the last
        # command terminates the chain and triggers immediate planner flush.
        assert client.moveJ(targets[0], speed=0.5, r=30.0, wait=False) >= 0
        assert client.moveJ(targets[1], speed=0.5, r=30.0, wait=False) >= 0
        assert client.moveJ(targets[2], speed=0.5, r=0.0, wait=True, timeout=15.0) >= 0

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
        """Three moveL with blend zones should reach the last target."""
        start = client.get_pose_rpy()
        assert start is not None

        # Small offsets from current pose (guaranteed reachable)
        targets = [
            [start[0], start[1] + 15, start[2], start[3], start[4], start[5]],
            [start[0], start[1] + 30, start[2], start[3], start[4], start[5]],
            [start[0], start[1] + 45, start[2], start[3], start[4], start[5]],
        ]

        # r>0 on intermediate commands creates blend zones; r=0 on the last
        # command terminates the blend chain and triggers immediate flush.
        assert client.moveL(targets[0], speed=0.5, r=20.0, wait=False) >= 0
        assert client.moveL(targets[1], speed=0.5, r=20.0, wait=False) >= 0
        assert client.moveL(targets[2], speed=0.5, r=0.0, wait=True, timeout=15.0) >= 0

        final = client.get_pose_rpy()
        assert final is not None
        for i in range(3):
            assert abs(final[i] - targets[-1][i]) < 2.0, (
                f"Axis {i}: expected {targets[-1][i]:.1f}, got {final[i]:.1f}"
            )

    def test_square_with_rounded_corners(self, client, server_proc):
        """Trace a 20mm square in YZ plane with r=5 rounded corners.

        Path: home → right → down → left → up → right (closed loop + return).
        Verifies position accuracy and orientation preservation through
        4 blended 90-degree direction changes.
        """
        start = client.get_pose_rpy()
        assert start is not None

        side = 20.0
        r = 5.0

        # Build absolute waypoints for the square (Y=right, Z=up)
        def offset(dy: float, dz: float) -> list[float]:
            return [
                start[0],
                start[1] + dy,
                start[2] + dz,
                start[3],
                start[4],
                start[5],
            ]

        right = offset(side, 0)
        down_right = offset(side, -side)
        down_left = offset(0, -side)
        back_home = offset(0, 0)
        back_right = offset(side, 0)  # same as right — closes the loop

        # 5 moveL commands: 4 corners blended (r=5), last terminates chain (r=0)
        assert client.moveL(right, speed=0.3, r=r, wait=False) >= 0
        assert client.moveL(down_right, speed=0.3, r=r, wait=False) >= 0
        assert client.moveL(down_left, speed=0.3, r=r, wait=False) >= 0
        assert client.moveL(back_home, speed=0.3, r=r, wait=False) >= 0
        assert client.moveL(back_right, speed=0.3, r=0.0, wait=False) >= 0

        assert client.wait_motion_complete(timeout=20.0)

        final = client.get_pose_rpy()
        assert final is not None

        # Position: should match back_right within 2mm
        for i in range(3):
            assert abs(final[i] - back_right[i]) < 2.0, (
                f"Axis {i}: expected {back_right[i]:.1f}, got {final[i]:.1f}"
            )

        # Orientation: should be unchanged (pure translation moves)
        for i in range(3, 6):
            diff = abs((final[i] - start[i] + 180) % 360 - 180)
            assert diff < 1.0, (
                f"Orientation axis {i - 3}: drifted {diff:.2f}° (start={start[i]:.1f}, end={final[i]:.1f})"
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
