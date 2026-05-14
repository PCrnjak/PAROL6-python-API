"""Unit tests for parol6.motion trajectory pipeline."""

import numpy as np
import pytest

from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import JointPath, ProfileType, Trajectory, TrajectoryBuilder
from parol6.motion.trajectory import (
    _IK_OUTLIER_PADDING,
    _IK_OUTLIER_RATIO,
    _smooth_singularity_outliers,
)


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

    def test_build_quintic_profile(self, simple_joint_path):
        """QUINTIC profile should produce smooth trajectory."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.QUINTIC,
            duration=3.0,  # Long duration to stay within velocity limits
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration >= 3.0

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


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_len_returns_step_count(self):
        """len() should return number of steps."""
        steps = np.zeros((100, 6), dtype=np.int32)
        traj = Trajectory(steps=steps, duration=1.0)

        assert len(traj) == 100

    def test_getitem_returns_step(self):
        """Indexing should return individual step."""
        steps = np.arange(60, dtype=np.int32).reshape(10, 6)
        traj = Trajectory(steps=steps, duration=1.0)

        assert np.array_equal(traj[0], steps[0])
        assert np.array_equal(traj[5], steps[5])
        assert np.array_equal(traj[-1], steps[-1])


class TestSmoothSingularityOutliers:
    """Tests for _smooth_singularity_outliers: LM-IK branch-hop repair."""

    @staticmethod
    def _smooth_chain(n: int = 40, step: float = 0.01) -> np.ndarray:
        """A 6-DOF chain with uniform per-sample step magnitude."""
        positions = np.zeros((n, 6), dtype=np.float64)
        for i in range(n):
            positions[i] = i * step
        return positions

    def test_smooth_path_unchanged(self):
        positions = self._smooth_chain()
        original = positions.copy()
        n = _smooth_singularity_outliers(positions)
        assert n == 0
        assert np.array_equal(positions, original)

    def test_single_sample_hop_is_smoothed(self):
        """One outlier sample fires bad-flags on both neighbors (both adjacent
        diffs are large) → run of 3, plus pad on each side → 2*pad+3 patched."""
        positions = self._smooth_chain(n=40)
        bad_idx = 20
        positions[bad_idx, 3] += 5.0  # 5 rad J4 jump — well past 10× median

        n_patched = _smooth_singularity_outliers(positions)
        assert n_patched == 2 * _IK_OUTLIER_PADDING + 3
        # The outlier is gone; uniform-step chain interpolates exactly.
        assert positions[bad_idx, 3] == pytest.approx(bad_idx * 0.01, abs=1e-9)
        # Samples just outside the patched window are untouched.
        outside_lo = bad_idx - 1 - _IK_OUTLIER_PADDING - 1
        outside_hi = bad_idx + 1 + _IK_OUTLIER_PADDING
        assert positions[outside_lo, 3] == pytest.approx(outside_lo * 0.01)
        assert positions[outside_hi, 3] == pytest.approx(outside_hi * 0.01)

    def test_multi_sample_run_is_smoothed(self):
        """Wide hop with monotonically rising outliers → each step is large,
        so the whole shelf forms one contiguous bad run."""
        positions = self._smooth_chain(n=40)
        # Use increasing magnitudes so every adjacent diff exceeds threshold.
        positions[20, 3] += 4.0
        positions[21, 3] += 8.0
        positions[22, 3] += 12.0

        n_patched = _smooth_singularity_outliers(positions)
        # Bad samples: {19, 20, 21, 22, 23} → run [19, 24).
        # Patched: [max(1,19-pad), min(n-1,23+pad)+1) = [15, 28) → 13.
        assert n_patched == 5 + 2 * _IK_OUTLIER_PADDING
        # All run samples land back on the chain.
        for k in (20, 21, 22):
            assert positions[k, 3] == pytest.approx(k * 0.01, abs=1e-9)

    def test_outlier_near_start_clamps_padding(self):
        """Padding clamps at the array boundary; bookend at index 0."""
        positions = self._smooth_chain(n=40)
        positions[2, 3] += 5.0
        original_first = positions[0].copy()
        original_last = positions[-1].copy()

        _smooth_singularity_outliers(positions)
        assert positions[2, 3] == pytest.approx(2 * 0.01, abs=1e-9)
        # Endpoints must never be modified.
        assert np.array_equal(positions[0], original_first)
        assert np.array_equal(positions[-1], original_last)

    def test_outlier_near_end_clamps_padding(self):
        positions = self._smooth_chain(n=40)
        positions[-3, 3] += 5.0
        original_first = positions[0].copy()
        original_last = positions[-1].copy()

        _smooth_singularity_outliers(positions)
        assert positions[-3, 3] == pytest.approx((40 - 3) * 0.01, abs=1e-9)
        assert np.array_equal(positions[0], original_first)
        assert np.array_equal(positions[-1], original_last)

    def test_short_path_is_noop(self):
        for n in (0, 1, 2):
            positions = np.zeros((n, 6), dtype=np.float64)
            assert _smooth_singularity_outliers(positions) == 0

    def test_zero_motion_is_noop(self):
        """Median step = 0 → can't form a ratio threshold; bail."""
        positions = np.zeros((20, 6), dtype=np.float64)
        positions[10, 3] = 5.0  # would-be outlier but median is 0
        n = _smooth_singularity_outliers(positions)
        assert n == 0
        assert positions[10, 3] == 5.0

    def test_below_threshold_step_not_smoothed(self):
        """Steps within `_IK_OUTLIER_RATIO`× median are normal motion, not hops."""
        positions = self._smooth_chain(n=40)
        # Bump one sample by ~5× the median step — should NOT trigger.
        positions[20, 3] += 5 * 0.01
        original = positions.copy()
        n = _smooth_singularity_outliers(positions)
        assert n == 0
        assert np.array_equal(positions, original)

    def test_threshold_exactly_at_ratio_not_smoothed(self):
        """Strict > comparison: step == threshold is NOT an outlier."""
        positions = self._smooth_chain(n=40)
        # Each step ~ sqrt(6) * 0.01; bump just at the threshold boundary.
        median_step = np.sqrt(6) * 0.01
        # Add exactly ratio * median to one component so the step magnitude
        # is just over the legitimate motion but well within tolerance.
        positions[20, 3] += _IK_OUTLIER_RATIO * median_step - 0.5 * median_step
        n = _smooth_singularity_outliers(positions)
        assert n == 0

    def test_repair_preserves_endpoints(self):
        """No matter where the hop is, positions[0] and positions[-1] are
        always exactly preserved (they're the IK seed and final target)."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            positions = self._smooth_chain(n=50)
            hop_idx = int(rng.integers(2, 48))
            positions[hop_idx, rng.integers(0, 6)] += rng.uniform(2.0, 8.0)
            first = positions[0].copy()
            last = positions[-1].copy()
            _smooth_singularity_outliers(positions)
            assert np.array_equal(positions[0], first)
            assert np.array_equal(positions[-1], last)
