"""Unit tests for parol6.motion trajectory pipeline."""

import logging
import math

import numpy as np
import pytest

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
import parol6.motion.geometry as geometry
from parol6.client.dry_run_client import DryRunRobotClient
from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import JointPath, ProfileType, Trajectory, TrajectoryBuilder
from parol6.motion.trajectory import (
    IK_MAX_JOINT_STEP_RAD,
    _ik_branch_hop,
)


def _played_rows(client: DryRunRobotClient) -> np.ndarray:
    record = client.plan()
    assert not [block.error for block in record.blocks if block.error]
    # TickIndex decimates rows; acceleration needs every control tick.
    return np.concatenate([chunk.q_rad for chunk in client._chunks if len(chunk.q_rad)])


def _assert_played_acceleration(rows: np.ndarray, fraction: float) -> None:
    at_rest = np.pad(rows, ((1, 1), (0, 0)), mode="edge")
    acceleration = np.diff(at_rest, n=2, axis=0) / INTERVAL_S**2
    ratio = np.max(np.abs(acceleration) / (fraction * LIMITS.joint.hard.acceleration))
    assert ratio <= 1.0, f"Played acceleration reached {ratio:.3f}x the requested limit"


class TestJointPath:
    """Tests for JointPath dataclass."""

    def test_interpolate_two_points(self):
        """Interpolate between two joint configurations."""
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        end = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n_samples = 11

        path = JointPath.interpolate(start, end, n_samples)

        assert len(path) == n_samples
        assert np.allclose(path.positions[0], start)
        assert np.allclose(path.positions[-1], end)
        # Midpoint should be average
        assert np.allclose(path.positions[5], (start + end) / 2)

    def test_interpolate_minimum_samples(self):
        """Interpolate with n_samples < 2 should produce 2 samples."""
        start = np.zeros(6)
        end = np.ones(6)

        path = JointPath.interpolate(start, end, n_samples=1)
        assert len(path) == 2

        path = JointPath.interpolate(start, end, n_samples=0)
        assert len(path) == 2

    def test_sample_at_boundaries(self):
        """Sample at s=0 and s=1 should return exact endpoints."""
        start = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        end = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        path = JointPath.interpolate(start, end, n_samples=10)

        assert np.allclose(path.sample(0.0), start)
        assert np.allclose(path.sample(1.0), end)

    def test_sample_clamps_out_of_bounds(self):
        """Sample at s<0 or s>1 should clamp to boundaries."""
        start = np.zeros(6)
        end = np.ones(6)
        path = JointPath.interpolate(start, end, n_samples=10)

        assert np.allclose(path.sample(-0.5), start)
        assert np.allclose(path.sample(1.5), end)

    def test_append_concatenates_paths(self):
        """Append should concatenate two paths."""
        path1 = JointPath.interpolate(np.zeros(6), np.ones(6), n_samples=5)
        path2 = JointPath.interpolate(np.ones(6), np.ones(6) * 2, n_samples=5)

        combined = path1.append(path2)

        assert len(combined) == 10
        assert np.allclose(combined.positions[:5], path1.positions)
        assert np.allclose(combined.positions[5:], path2.positions)


class TestTrajectoryBuilder:
    """Tests for TrajectoryBuilder."""

    @pytest.fixture
    def simple_joint_path(self) -> JointPath:
        """Create a simple joint path for testing."""
        start = np.array([0.0, -1.57, 3.14, 0.0, 0.0, 0.0])  # ~0, -90deg, 180deg
        end = np.array([0.5, -1.0, 2.5, 0.5, 0.0, 0.5])
        return JointPath.interpolate(start, end, n_samples=50)

    def test_build_linear_profile(self, simple_joint_path):
        """LINEAR profile should produce uniformly spaced trajectory."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.LINEAR,
            duration=1.0,  # 1 second
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration >= 1.0
        # Steps should be int32
        assert trajectory.steps.dtype == np.int32

    @pytest.mark.parametrize(
        "profile,duration,bent",
        [(ProfileType.QUINTIC, 3.0, False), (ProfileType.RUCKIG, 5.0, True)],
    )
    def test_build_profile_with_requested_duration(
        self, simple_joint_path, profile, duration, bent
    ):
        if bent:
            start, end = simple_joint_path.positions[[0, -1]]
            mid = (start + end) / 2
            mid[0] += 0.1
            simple_joint_path = JointPath(positions=np.array([start, mid, end]))
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=profile,
            duration=duration,
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert len(trajectory) > 0
        assert trajectory.duration >= duration

    def test_build_trapezoid_profile(self, simple_joint_path):
        """TRAPEZOID profile should produce trajectory with plateau."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.TRAPEZOID,
            duration=3.0,  # Long duration to stay within velocity limits
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration >= 3.0

    def test_build_ruckig_profile(self, simple_joint_path):
        """RUCKIG profile should produce jerk-limited trajectory."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.RUCKIG,
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration > 0

    def test_velocity_frac_scaling(self, simple_joint_path):
        """Lower velocity_frac should increase duration."""
        # Use TOPPRA which is time-optimal and respects velocity limits
        builder_100 = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.TOPPRA,
            velocity_frac=1.0,
        )
        builder_50 = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.TOPPRA,
            velocity_frac=0.5,
        )

        traj_100 = builder_100.build()
        traj_50 = builder_50.build()

        # At 50% velocity, duration should be longer
        assert traj_50.duration >= traj_100.duration

    def test_single_point_path(self):
        """Single-point path should produce single-step trajectory."""
        path = JointPath(positions=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

        builder = TrajectoryBuilder(
            joint_path=path,
            profile=ProfileType.LINEAR,
        )

        trajectory = builder.build()

        assert len(trajectory) == 1
        assert trajectory.duration == 0.0

    @pytest.mark.parametrize(
        "profile",
        [
            ProfileType.LINEAR,
            ProfileType.QUINTIC,
            ProfileType.TRAPEZOID,
            ProfileType.TOPPRA,
            ProfileType.RUCKIG,
        ],
    )
    def test_trajectory_respects_joint_velocity_limits(
        self, simple_joint_path, profile
    ):
        """All profiles should produce trajectories within joint velocity limits."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=profile,
            dt=INTERVAL_S,
        )
        traj = builder.build()

        if len(traj) < 2:
            return  # Can't check velocity on single-point trajectory

        # Convert steps back to radians for limit checking
        trajectory_rad = np.empty(traj.steps.shape, dtype=np.float64)
        for i in range(len(traj.steps)):
            steps_to_rad(traj.steps[i], trajectory_rad[i])

        # Compute velocities via finite difference
        dt = traj.duration / (len(traj) - 1)
        velocities_rad = np.diff(trajectory_rad, axis=0) / dt

        # Check against velocity limits (with small tolerance for numerical error)
        v_max_rad = LIMITS.joint.hard.velocity
        max_vel_ratio = float(np.max(np.abs(velocities_rad) / v_max_rad))

        assert max_vel_ratio <= 1.05, (
            f"Profile {profile.name}: velocity exceeded limits by "
            f"{(max_vel_ratio - 1) * 100:.1f}%"
        )

    @pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR", "TRAPEZOID", "QUINTIC"])
    def test_blended_chain_is_no_slower_than_stops_and_respects_acceleration(
        self, profile
    ):
        start = [90.0, -80.0, 190.0, 0.0, 30.0, 180.0]
        corner = [110.0, -80.0, 190.0, 0.0, 30.0, 180.0]
        end = [110.0, -65.0, 190.0, 0.0, 30.0, 180.0]
        stopped_rows = 0
        for radius in (0.0, 5.0, 20.0, 50.0):
            client = DryRunRobotClient(initial_joints_deg=start)
            client.select_profile(profile)
            client.move_j(corner, speed=0.5, accel=0.5, r=radius)
            client.move_j(end, speed=0.5, accel=0.5)
            rows = _played_rows(client)
            if radius == 0:
                stopped_rows = len(rows)
            else:
                assert len(rows) <= stopped_rows, (
                    profile,
                    radius,
                    len(rows),
                    stopped_rows,
                )
            _assert_played_acceleration(rows, 0.5)
            np.testing.assert_allclose(
                np.degrees(rows[[0, -1]]), [start, end], atol=0.01
            )

        client = DryRunRobotClient(
            initial_joints_deg=list(PAROL6_ROBOT.joint.standby_deg)
        )
        client.select_profile(profile)
        pose = client.pose()
        client.move_l(
            [pose[0] + 40, pose[1], pose[2] + 40, *pose[3:]], speed=0.5, accel=0.5
        )
        _assert_played_acceleration(_played_rows(client), 0.5)

    @pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR", "TRAPEZOID", "QUINTIC"])
    def test_process_move_holds_a_feasible_tool_speed(self, profile):
        for wrist_tilt in (0.0, 0.4):
            start = np.array(PAROL6_ROBOT.joint.standby_deg, dtype=float)
            start[4] += np.degrees(wrist_tilt)
            client = DryRunRobotClient(initial_joints_deg=start.tolist())
            client.select_profile(profile)
            pose = client.pose()
            client.move_p(
                [
                    [pose[0] + 50, pose[1], *pose[2:]],
                    [pose[0] + 50, pose[1] + 50, *pose[2:]],
                ]
            )
            rows = _played_rows(client)
            _assert_played_acceleration(rows, 0.5)
            tcp = np.array([PAROL6_ROBOT.robot.fkine(q)[:3, 3] for q in rows])
            distance = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
            progress = np.cumsum(distance)
            # A wrist reconfiguration contributes time but no tool travel.
            cruise = distance[
                (progress > 0.2 * progress[-1]) & (progress < 0.8 * progress[-1])
            ]
            assert cruise.min() / cruise.max() >= 0.9, (profile, wrist_tilt)
            moving = np.flatnonzero(distance > 1e-6)
            assert (moving[-1] - moving[0] + 1) * INTERVAL_S < 3.0
            if wrist_tilt:
                middle = distance[int(0.2 * len(distance)) : int(0.8 * len(distance))]
                assert middle.min() / middle.max() >= 0.9


class TestDryRunTimingRegressions:
    START = [90.0, -80.0, 190.0, 0.0, 30.0, 180.0]

    @pytest.mark.parametrize(
        "profile", ["QUINTIC", "TOPPRA", "LINEAR", "TRAPEZOID", "RUCKIG"]
    )
    def test_duration_stretch_scales_acceleration_smoothly(self, profile):
        planned = []
        for duration in (0.0, 5.0):
            client = DryRunRobotClient(initial_joints_deg=self.START)
            client.select_profile(profile)
            target = [100.0, *self.START[1:]]
            if profile == "RUCKIG":
                client.move_j(target, speed=1.0, accel=1.0, duration=duration / 2, r=10)
                target = [100.0, -65.0, *self.START[2:]]
                client.move_j(target, speed=1.0, accel=1.0, duration=duration / 2)
            else:
                client.move_j(target, speed=1.0, accel=1.0, duration=duration)
            rows = _played_rows(client)
            _assert_played_acceleration(rows, 1.0)
            np.testing.assert_allclose(rows[-1], np.radians(target), atol=1e-12)
            planned.append(rows)

        fast, slow = planned
        assert len(slow) == round(5.0 / INTERVAL_S) + 1
        np.testing.assert_array_equal(slow[[0, -1]], fast[[0, -1]])
        stretch = (len(slow) - 1) / (len(fast) - 1)
        fast_peak = np.max(np.abs(np.diff(fast, n=2, axis=0))) / INTERVAL_S**2
        slow_peak = np.max(np.abs(np.diff(slow, n=2, axis=0))) / INTERVAL_S**2
        assert slow_peak <= 1.3 * fast_peak / stretch**2

    def test_quintic_paths_ease_acceleration_at_both_ends(self):
        for chain in (False, True):
            client = DryRunRobotClient(initial_joints_deg=self.START)
            client.select_profile("QUINTIC")
            if chain:
                client.move_j([110.0, *self.START[1:]], speed=0.5, accel=0.5, r=5)
                client.move_j([110.0, -65.0, *self.START[2:]], speed=0.5, accel=0.5)
            else:
                pose = client.pose()
                client.move_l(
                    [pose[0] + 60, pose[1], pose[2] - 40, *pose[3:]],
                    speed=0.5,
                    accel=0.5,
                )
            rows = _played_rows(client)
            _assert_played_acceleration(rows, 0.5)
            acceleration = (
                np.abs(
                    np.diff(np.pad(rows, ((1, 1), (0, 0)), mode="edge"), n=2, axis=0)
                )
                / INTERVAL_S**2
            )
            assert acceleration[[0, -1]].max() < 0.001 * acceleration.max()
            assert acceleration[[1, -2]].max() < 0.02 * acceleration.max()

    def test_process_timing_is_stable_across_sampling_pitches(self, monkeypatch):
        for square in (False, True):
            durations = []
            for pitch in (4.0, 2.0, 1.0, 0.5):
                monkeypatch.setattr(geometry, "PATH_STEP_MM", pitch)
                client = DryRunRobotClient(initial_joints_deg=self.START)
                client.select_profile("TOPPRA")
                pose = client.pose()
                offsets = (
                    [(60, 0), (60, 60), (0, 60), (0, 0)]
                    if square
                    else [(10 * k, 15 if k % 2 else 0) for k in range(1, 11)]
                )
                client.move_p(
                    [[pose[0] + x, pose[1] + y, *pose[2:]] for x, y in offsets],
                    speed=0.5,
                    accel=0.5,
                )
                rows = _played_rows(client)
                _assert_played_acceleration(rows, 0.5)
                durations.append(len(rows) * INTERVAL_S)
            assert max(durations) / min(durations) < 1.08
            assert max(durations) < (5.5 if square else 8.0)

    def test_spline_orientation_change_does_not_stop_at_waypoint(self):
        client = DryRunRobotClient(initial_joints_deg=self.START)
        client.select_profile("TOPPRA")
        pose = client.pose()
        corner = [pose[0] + 60, pose[1], *pose[2:]]
        end = [pose[0] + 60, pose[1] + 60, pose[2], pose[3], pose[4], pose[5] + 30]
        client.move_s([corner, end], speed=0.5)
        rows = _played_rows(client)
        _assert_played_acceleration(rows, 0.5)
        tcp = np.array([PAROL6_ROBOT.robot.fkine(q)[:3, 3] for q in rows]) * 1000
        speeds = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / INTERVAL_S
        middle = speeds[len(speeds) // 5 : 4 * len(speeds) // 5]
        assert middle.min() > 40.0
        assert np.linalg.norm(tcp - np.array(corner[:3]), axis=1).min() < 1.0
        np.testing.assert_allclose(tcp[-1], end[:3], atol=0.1)

    def test_timed_ruckig_uses_requested_ticks_without_extension_warning(self, caplog):
        target = [120.0, -80.0, 180.0, 0.0, 0.0, 180.0]
        for duration in (2.0, 3.0, 3.333, 4.0, 4.005, 7.77):
            caplog.clear()
            client = DryRunRobotClient()
            client.select_profile("RUCKIG")
            with caplog.at_level(logging.WARNING, logger="parol6.motion.trajectory"):
                client.move_j(target, duration=duration, accel=1.0)
                rows = _played_rows(client)
            assert len(rows) == math.ceil(duration / INTERVAL_S - 1e-6) + 1
            assert not [
                record
                for record in caplog.records
                if "Extending duration" in record.message
            ]
            np.testing.assert_allclose(rows[-1], np.radians(target), atol=1e-12)
            _assert_played_acceleration(rows, 1.0)

    def test_corner_timing_respects_acceleration_before_the_post_pass(
        self, monkeypatch
    ):
        raw_ratios = []
        enforce = TrajectoryBuilder._enforce_segment_limits

        def observe(builder, rows):
            acceleration = (
                np.diff(np.pad(rows, ((1, 1), (0, 0)), mode="edge"), n=2, axis=0)
                / INTERVAL_S**2
            )
            raw_ratios.append(float(np.max(np.abs(acceleration) / builder.a_max)))
            return enforce(builder, rows)

        monkeypatch.setattr(TrajectoryBuilder, "_enforce_segment_limits", observe)
        start = np.array([0.0, -1.57, 3.14, 0.0, 0.5, 0.0])
        corner = start + [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
        end = corner + [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
        client = DryRunRobotClient(initial_joints_deg=np.degrees(start).tolist())
        client.select_profile("TOPPRA")
        client.move_j(np.degrees(corner).tolist(), speed=1.0, accel=0.5, r=1.0)
        client.move_j(np.degrees(end).tolist(), speed=1.0, accel=0.5)
        _assert_played_acceleration(_played_rows(client), 0.5)
        assert max(raw_ratios) <= 1.0


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_len_returns_step_count(self):
        """len() should return number of steps."""
        steps = np.zeros((100, 6), dtype=np.int32)
        traj = Trajectory(
            steps=steps, duration=1.0, positions_rad=np.zeros(steps.shape)
        )

        assert len(traj) == 100

    def test_getitem_returns_step(self):
        """Indexing should return individual step."""
        steps = np.arange(60, dtype=np.int32).reshape(10, 6)
        traj = Trajectory(
            steps=steps, duration=1.0, positions_rad=np.zeros(steps.shape)
        )

        assert np.array_equal(traj[0], steps[0])
        assert np.array_equal(traj[5], steps[5])
        assert np.array_equal(traj[-1], steps[-1])


class TestIkBranchHops:
    """A cartesian IK chain that hops branches is refused, never smoothed:
    the hop is where the arm would whip, and interpolating across it
    commands a path nobody checked."""

    @staticmethod
    def _chain(n: int = 40, step: float = 0.01) -> np.ndarray:
        positions = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            positions[i] = i * step
        return positions

    def test_a_smooth_chain_and_one_at_the_step_limit_pass(self):
        assert _ik_branch_hop(self._chain()) is None
        at_limit = self._chain(n=3, step=IK_MAX_JOINT_STEP_RAD)
        assert _ik_branch_hop(at_limit) is None
        assert _ik_branch_hop(self._chain(n=1)) is None

    def test_a_hop_is_reported_at_the_waypoint_it_lands_on(self):
        positions = self._chain()
        positions[20, 3] += IK_MAX_JOINT_STEP_RAD * 1.01
        assert _ik_branch_hop(positions) == 20
        wide = self._chain()
        wide[25:, 4] -= 2.0
        assert _ik_branch_hop(wide) == 25
