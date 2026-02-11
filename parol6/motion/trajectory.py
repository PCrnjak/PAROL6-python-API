"""
Unified trajectory generation pipeline using TOPP-RA for time-optimal path parameterization.

This module provides the shared trajectory infrastructure for all motion commands.
Path generation (geometry) stays in command files; this handles time parameterization.

Pipeline:
  1. Command generates Cartesian poses (for cart commands) or joint targets
  2. JointPath holds uniformly-sampled joint positions
  3. TrajectoryBuilder applies TOPP-RA + motion profile to produce Trajectory
  4. Trajectory contains motor steps ready for tick-by-tick execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from ruckig import InputParameter, OutputParameter, Result, Ruckig

import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from toppra.interpolator import SplineInterpolator

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import INTERVAL_S, LIMITS, rad_to_steps


from pinokin import Damping, IKSolver, se3_from_rpy


logger = logging.getLogger(__name__)


def _rad_to_steps_alloc(rad: NDArray) -> NDArray[np.int32]:
    """Convert radians to steps, allocating output. For planning phase only."""
    out = np.zeros(rad.shape, dtype=np.int32)
    if rad.ndim == 1:
        rad_to_steps(rad, out)
    else:
        # 2D array (trajectory): convert row by row
        for i in range(rad.shape[0]):
            rad_to_steps(rad[i], out[i])
    return out


class ProfileType(Enum):
    """Available trajectory profile types for motion planning."""

    TOPPRA = "toppra"  # Time-optimal path following (default)
    RUCKIG = "ruckig"  # Point-to-point jerk-limited (can't follow Cartesian paths)
    QUINTIC = "quintic"  # Quintic polynomial (C² smooth, predictable shape)
    TRAPEZOID = "trapezoid"  # Trapezoidal velocity profile
    LINEAR = "linear"  # Direct linear interpolation (no smoothing)

    @classmethod
    def from_string(cls, name: str) -> ProfileType:
        """Convert string to ProfileType, case-insensitive."""
        name_upper = name.upper()
        if name_upper == "NONE":
            return cls.LINEAR
        try:
            return cls[name_upper]
        except KeyError:
            logger.warning("Unknown profile type '%s', using TOPPRA", name)
            return cls.TOPPRA


@dataclass
class JointPath:
    """
    Joint-space path uniformly sampled in path space.

    This is the common abstraction for all motion commands. Cartesian commands
    solve IK to produce this; joint commands interpolate directly.

    Attributes:
        positions: (N, 6) array of joint angles in radians
    """

    positions: NDArray[np.float64]  # (N, 6) joint angles in radians

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> NDArray[np.float64]:
        return self.positions[idx]

    @classmethod
    def from_poses(
        cls,
        poses: NDArray[np.float64] | list[np.ndarray],
        seed_q: NDArray[np.float64],
        quiet_logging: bool = True,
    ) -> JointPath:
        """
        Solve IK for poses with seeded chain.

        Each IK solve uses the previous solution as seed, maintaining continuity.

        Args:
            poses: Either (N, 6) array of [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
                   or list of SE3 poses
            seed_q: Initial joint angles for IK seeding (radians)
            quiet_logging: Suppress IK logging

        Returns:
            JointPath with solved joint positions

        Raises:
            IKError: If any pose is unreachable
        """
        from parol6.utils.errors import IKError

        # Convert to list of SE3 (4x4) matrices for batch_ik
        if isinstance(poses, np.ndarray) and poses.ndim == 3:
            # (N, 4, 4) SE3 matrices — create list of views (no data copy)
            se3_poses = [poses[i] for i in range(len(poses))]
        elif isinstance(poses, np.ndarray):
            # (N, 6) [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] — convert to SE3
            n = len(poses)
            se3_poses = [np.empty((4, 4), dtype=np.float64) for _ in range(n)]
            for i, p in enumerate(poses):
                se3_from_rpy(
                    p[0] / 1000.0,
                    p[1] / 1000.0,
                    p[2] / 1000.0,
                    np.radians(p[3]),
                    np.radians(p[4]),
                    np.radians(p[5]),
                    se3_poses[i],
                )
        else:
            se3_poses = poses

        solver = IKSolver(
            PAROL6_ROBOT.robot,
            damping=Damping.Sugihara,
            tol=1e-12,
            lm_lambda=0.0,
            max_iter=10,
            max_restarts=10,
        )
        result = solver.batch_ik(se3_poses, np.asarray(seed_q, dtype=np.float64))

        if not result.all_valid:
            failed = [i for i, v in enumerate(result.valid) if not v]
            raise IKError(
                f"Cartesian path points {failed}/{len(se3_poses)} are unreachable."
            )

        return cls(positions=result.joint_positions)

    @classmethod
    def interpolate(
        cls,
        start_rad: NDArray[np.float64],
        end_rad: NDArray[np.float64],
        n_samples: int,
    ) -> JointPath:
        """
        Direct joint-space linear interpolation (for MovePose/MoveJoint).

        Args:
            start_rad: Starting joint angles in radians
            end_rad: Ending joint angles in radians
            n_samples: Number of samples (minimum 2)

        Returns:
            JointPath with interpolated positions
        """
        n_samples = max(2, n_samples)
        start = np.asarray(start_rad, dtype=np.float64)
        end = np.asarray(end_rad, dtype=np.float64)

        # Vectorized interpolation using broadcasting
        t = np.linspace(0, 1, n_samples).reshape(-1, 1)
        positions = start + t * (end - start)

        return cls(positions=positions)

    def append(self, other: JointPath) -> JointPath:
        """
        Concatenate paths (for path blending).

        Args:
            other: Path to append

        Returns:
            New JointPath with concatenated positions
        """
        combined = np.concatenate([self.positions, other.positions], axis=0)
        return JointPath(positions=combined)

    def sample(self, s: float) -> NDArray[np.float64]:
        """
        Sample path at normalized position s in [0, 1].

        Uses linear interpolation between path points.

        Args:
            s: Path position from 0 (start) to 1 (end)

        Returns:
            Interpolated joint position
        """
        s = np.clip(s, 0.0, 1.0)
        n = len(self.positions)
        if n < 2:
            return self.positions[0].copy()

        idx_float = s * (n - 1)
        idx_lo = int(idx_float)
        idx_hi = min(idx_lo + 1, n - 1)
        frac = idx_float - idx_lo

        return self.positions[idx_lo] * (1 - frac) + self.positions[idx_hi] * frac

    def sample_many(self, s_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Vectorized sampling at multiple path positions.

        Args:
            s_values: Array of path positions from 0 (start) to 1 (end)

        Returns:
            (N, 6) array of interpolated joint positions
        """
        s_values = np.clip(s_values, 0.0, 1.0)
        n = len(self.positions)
        if n < 2:
            return np.tile(self.positions[0], (len(s_values), 1))

        idx_float = s_values * (n - 1)
        idx_lo = idx_float.astype(np.intp)
        idx_hi = np.minimum(idx_lo + 1, n - 1)
        frac = (idx_float - idx_lo).reshape(-1, 1)

        return self.positions[idx_lo] * (1 - frac) + self.positions[idx_hi] * frac


@dataclass
class Trajectory:
    """
    Ready-to-execute trajectory with motor steps at control rate.

    Precomputed trajectories are sent directly to the controller without smoothing.
    StreamingExecutor is only used for online targets (jogging/streaming).

    Attributes:
        steps: (M, 6) motor steps at each control tick
        duration: Actual duration in seconds
    """

    steps: NDArray[np.int32]  # (M, 6) motor steps
    duration: float  # seconds

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> NDArray[np.int32]:
        return self.steps[idx]


class TrajectoryBuilder:
    """
    Converts JointPath to executable Trajectory.

    Uses TOPP-RA to compute maximum allowable path speed, then applies
    the selected motion profile (clamped to TOPP-RA limits).

    All limits come from PAROL6_ROBOT config - no hardcoded fallbacks.
    """

    def __init__(
        self,
        joint_path: JointPath,
        profile: ProfileType | str,
        velocity_frac: float = 1.0,
        accel_frac: float = 1.0,
        jerk_frac: float = 1.0,
        duration: float | None = None,
        dt: float = INTERVAL_S,
        cart_vel_limit: float | None = None,
        cart_acc_limit: float | None = None,
    ):
        """
        Initialize trajectory builder.

        Args:
            joint_path: Path in joint space
            profile: Motion profile to apply
            velocity_frac: Scale joint velocity limits (0.0-1.0), default 1.0
            accel_frac: Scale joint acceleration limits (0.0-1.0), default 1.0
            jerk_frac: Scale jerk limits (0.0-1.0), default 1.0
            duration: Override duration (stretches profile if longer than TOPP-RA min)
            dt: Control loop time step
            cart_vel_limit: Cartesian linear velocity limit in m/s (for Cartesian commands)
            cart_acc_limit: Cartesian linear acceleration limit in m/s² (for Cartesian commands)
        """
        self.joint_path = joint_path
        self.profile = (
            ProfileType.from_string(profile) if isinstance(profile, str) else profile
        )

        # RUCKIG is point-to-point only - if Cartesian limits are set, we need path following
        if self.profile == ProfileType.RUCKIG and (
            cart_vel_limit is not None or cart_acc_limit is not None
        ):
            logger.warning("RUCKIG cannot follow Cartesian paths, using TOPPRA")
            self.profile = ProfileType.TOPPRA

        self.velocity_frac = velocity_frac
        self.accel_frac = accel_frac
        self.jerk_frac = jerk_frac
        self.duration = duration
        self.dt = dt
        self.cart_vel_limit = cart_vel_limit
        self.cart_acc_limit = cart_acc_limit

        # Joint limits scaled by user fractions.
        # Apply 1% safety margin to account for floating-point precision in
        # trajectory libraries and integer rounding in rad→steps conversion.
        limit_margin = 0.99
        self.v_max = LIMITS.joint.hard.velocity * self.velocity_frac * limit_margin
        self.a_max = LIMITS.joint.hard.acceleration * self.accel_frac * limit_margin
        self.j_max = LIMITS.joint.hard.jerk * self.jerk_frac * limit_margin

        # Pre-compute limit arrays for TOPP-RA (avoids allocation per build() call)
        self._vlim = np.column_stack([-self.v_max, self.v_max])
        self._alim = np.column_stack([-self.a_max, self.a_max])

    def build(self) -> Trajectory:
        """
        Generate time-parameterized trajectory.

        Uses TOPP-RA to compute time-optimal trajectory, then samples it directly
        at the control rate. No interpolation of the original joint path is needed
        since TOPP-RA's trajectory already provides smooth, continuous positions.

        For RUCKIG profile: Uses Ruckig for point-to-point motion (ignores path waypoints)
        For other profiles: Uses TOPP-RA trajectory directly

        Returns:
            Trajectory ready for execution
        """
        if len(self.joint_path) < 2:
            # Trivial path - single point
            steps = _rad_to_steps_alloc(
                self.joint_path.positions[0:1]  # Keep 2D shape (1, 6)
            )
            return Trajectory(steps=steps, duration=0.0)

        # Route to appropriate trajectory builder based on profile
        if self.profile == ProfileType.RUCKIG:
            # Point-to-point jerk-limited motion (doesn't follow path waypoints)
            return self._build_ruckig_trajectory()
        elif self.profile == ProfileType.LINEAR:
            # Simple linear interpolation - no velocity smoothing
            return self._build_simple_trajectory()
        elif self.profile == ProfileType.QUINTIC:
            # Quintic polynomial timing
            return self._build_quintic_trajectory()
        elif self.profile == ProfileType.TRAPEZOID:
            # Trapezoidal velocity profile along path
            return self._build_trapezoid_trajectory()
        else:
            # TOPPRA is default - time-optimal path following
            return self._build_toppra_trajectory()

    def _build_toppra_trajectory(self) -> Trajectory:
        """
        Build trajectory using TOPP-RA's time-optimal path parameterization.

        Uses cubic spline interpolation through waypoints and computes
        time-optimal velocity profile respecting joint limits and optional
        Cartesian velocity limits.
        """
        positions = self.joint_path.positions
        n_points = len(positions)

        # Uniform parameterization for spline knots
        ss_waypoints = np.linspace(0.0, 1.0, n_points)

        path = ta.SplineInterpolator(ss_waypoints, positions)

        # Use pre-computed limit arrays for constraints
        joint_vel_constraint = constraint.JointVelocityConstraint(self._vlim)
        joint_acc_constraint = constraint.JointAccelerationConstraint(self._alim)
        constraints = [joint_vel_constraint, joint_acc_constraint]

        # Add Cartesian velocity constraint if specified
        if self.cart_vel_limit is not None and self.cart_vel_limit > 0:
            cart_constraint = self._build_cart_vel_constraint(path, ss_waypoints)
            if cart_constraint is not None:
                constraints.append(cart_constraint)

        try:
            # Use evenly-spaced gridpoints - TOPPRA docs recommend "at least a few times
            # the number of waypoints". Auto-selection can cluster points near
            # discontinuities, causing TOPPRAsd to produce incorrect durations.
            n_gridpoints = n_points * 3
            gridpoints = np.linspace(0.0, 1.0, n_gridpoints)

            # Use TOPPRAsd if duration is specified, otherwise time-optimal TOPPRA
            if self.duration is not None and self.duration > 0:
                instance = algo.TOPPRAsd(constraints, path, gridpoints=gridpoints)
                instance.set_desired_duration(self.duration)
                jnt_traj = instance.compute_trajectory()
                if jnt_traj is not None:
                    duration = self.duration
                    logger.debug(
                        "TrajectoryBuilder: TOPPRAsd target_duration=%.3f, path_len=%d",
                        duration,
                        n_points,
                    )
                else:
                    # Fall back to time-optimal if TOPPRAsd fails
                    logger.warning("TOPPRAsd failed, trying time-optimal TOPPRA")
                    instance = algo.TOPPRA(constraints, path, gridpoints=gridpoints)
                    jnt_traj = instance.compute_trajectory()
            else:
                instance = algo.TOPPRA(constraints, path, gridpoints=gridpoints)
                jnt_traj = instance.compute_trajectory()

            if not isinstance(jnt_traj, SplineInterpolator):
                raise RuntimeError("TOPP-RA failed to compute trajectory")

            duration = float(jnt_traj.duration)

            logger.debug(
                "TrajectoryBuilder: TOPP-RA duration=%.3f, path_len=%d",
                duration,
                n_points,
            )

            # Sample trajectory at control rate, including endpoint (vectorized)
            n_output = max(2, int(np.floor(duration / self.dt)) + 1)
            times = np.arange(n_output - 1) * self.dt
            trajectory_rad = np.empty((n_output, 6), dtype=np.float64)
            trajectory_rad[:-1] = jnt_traj(times)
            trajectory_rad[-1] = jnt_traj(duration)

            logger.debug(
                "TrajectoryBuilder: output_samples=%d, duration=%.3f",
                len(trajectory_rad),
                duration,
            )

            # Convert to motor steps (vectorized)
            steps = _rad_to_steps_alloc(trajectory_rad)

            return Trajectory(steps=steps, duration=duration)

        except Exception as e:
            logger.warning("TOPPRA failed: %s. Falling back to LINEAR profile.", e)
            return self._build_simple_trajectory()

    def _build_simple_trajectory(self) -> Trajectory:
        """
        Build trajectory with simple linear interpolation.

        Uses uniform s-spacing with local slowdown where velocity limits
        would be exceeded. This handles singularities and wrist flips by
        stretching only the affected segments.
        """
        duration = (
            self.duration
            if self.duration and self.duration > 0
            else self._compute_joint_duration_linear()
        )

        # Sample path uniformly
        n_output = max(2, int(np.ceil(duration / self.dt)))
        s_values = np.linspace(0.0, 1.0, n_output)
        trajectory_rad = self.joint_path.sample_many(s_values)

        # Enforce velocity limits by stretching segments where needed
        trajectory_rad, duration = self._enforce_segment_limits(
            trajectory_rad, duration
        )

        # Convert to motor steps
        steps = _rad_to_steps_alloc(trajectory_rad)

        return Trajectory(steps=steps, duration=duration)

    def _is_cartesian_path(self) -> bool:
        """Check if this is a Cartesian path (has Cartesian velocity limits set)."""
        return self.cart_vel_limit is not None and self.cart_vel_limit > 0

    def _compute_s_profile_limits(self) -> tuple[float, float, float]:
        """
        Compute path parameter (s) limits derived from joint limits.

        For a linear path in joint space:
            joint_velocity = joint_delta * (ds/dt)
            joint_acceleration = joint_delta * (d²s/dt²)
            joint_jerk = joint_delta * (d³s/dt³)

        So the s-profile limits are:
            vmax_s = min(v_max[j] / |delta[j]|) for all joints
            amax_s = min(a_max[j] / |delta[j]|) for all joints
            jmax_s = min(j_max[j] / |delta[j]|) for all joints

        Returns:
            (vmax_s, amax_s, jmax_s): Limits for the path parameter profile
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return (1.0, 1.0, 1.0)

        total_delta = np.abs(positions[-1] - positions[0])

        # Avoid division by zero for joints that don't move
        with np.errstate(divide="ignore", invalid="ignore"):
            vmax_s_per_joint = np.where(
                total_delta > 1e-9, self.v_max / total_delta, np.inf
            )
            amax_s_per_joint = np.where(
                total_delta > 1e-9, self.a_max / total_delta, np.inf
            )
            jmax_s_per_joint = np.where(
                total_delta > 1e-9, self.j_max / total_delta, np.inf
            )

        # The limiting joint determines the s-profile limits
        vmax_s = float(np.min(vmax_s_per_joint))
        amax_s = float(np.min(amax_s_per_joint))
        jmax_s = float(np.min(jmax_s_per_joint))

        return (vmax_s, amax_s, jmax_s)

    def _enforce_segment_limits(
        self,
        trajectory_rad: NDArray[np.float64],
        duration: float,
    ) -> tuple[NDArray[np.float64], float]:
        """
        Enforce velocity limits by locally stretching segments that exceed limits.

        Walks through each segment and checks if joint velocities exceed limits.
        Where they do, stretches that segment's time. This handles singularities
        and wrist flips by slowing only where necessary, not globally.

        Args:
            trajectory_rad: Joint positions in radians, shape (N, 6)
            duration: Initial trajectory duration

        Returns:
            (adjusted_trajectory, adjusted_duration): Resampled trajectory with
            locally stretched segments and new total duration
        """
        n_points = len(trajectory_rad)
        if n_points < 2:
            return trajectory_rad, duration

        # Initial uniform time per segment
        initial_dt = duration / (n_points - 1)

        # Compute per-segment joint deltas
        deltas = np.diff(trajectory_rad, axis=0)  # (N-1, 6)

        # For each segment, compute minimum time needed to respect velocity limits
        # time_needed = max(|delta[j]| / v_max[j]) for all joints
        min_segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)

        # Also check acceleration limits between segments
        # This is approximate - we check if velocity change between segments is feasible
        if n_points > 2:
            velocities = deltas / initial_dt  # Approximate velocities per segment
            accel = np.diff(velocities, axis=0) / initial_dt  # (N-2, 6)
            accel_times = np.zeros(n_points - 1)
            # Segments that cause high acceleration need more time
            for i in range(len(accel)):
                max_accel_ratio = np.max(np.abs(accel[i]) / self.a_max)
                if max_accel_ratio > 1.0:
                    # Spread the extra time across adjacent segments
                    stretch = np.sqrt(max_accel_ratio)
                    accel_times[i] = max(accel_times[i], min_segment_times[i] * stretch)
                    accel_times[i + 1] = max(
                        accel_times[i + 1], min_segment_times[i + 1] * stretch
                    )
            min_segment_times = np.maximum(min_segment_times, accel_times)

        # Ensure minimum dt per segment
        min_segment_times = np.maximum(min_segment_times, self.dt)

        # Compute actual segment times: max of initial_dt and min_segment_times
        segment_times = np.maximum(min_segment_times, initial_dt)

        # Check if any stretching was needed
        new_duration = float(np.sum(segment_times))
        if new_duration <= duration * 1.001:  # No significant change
            return trajectory_rad, duration

        logger.warning(
            "Extending duration from %.3fs to %.3fs (%.1f%% increase) to respect velocity/acceleration limits",
            duration,
            new_duration,
            (new_duration / duration - 1) * 100,
        )

        # Resample trajectory at control rate with new timing
        cumulative_times = np.zeros(n_points)
        cumulative_times[1:] = np.cumsum(segment_times)

        n_output = max(2, int(np.ceil(new_duration / self.dt)))
        output_times = np.linspace(0.0, new_duration, n_output)

        # Interpolate each joint
        new_trajectory = np.empty((n_output, 6), dtype=np.float64)
        for j in range(6):
            new_trajectory[:, j] = np.interp(
                output_times, cumulative_times, trajectory_rad[:, j]
            )

        return new_trajectory, new_duration

    def _compute_joint_duration_trapezoid(self) -> float:
        """
        Compute duration for joint paths using trapezoidal profile.

        For each joint, uses InterpolatePy to compute the minimum duration
        for its displacement given its velocity/acceleration limits.
        Returns the maximum (slowest joint determines overall duration).
        """
        from interpolatepy.trapezoidal import (
            TrajectoryParams as TrapParams,
            TrapezoidalTrajectory,
        )

        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = positions[-1] - positions[0]
        max_duration = 0.0

        for j in range(6):
            delta = abs(total_delta[j])
            if delta < 1e-6:
                continue

            params = TrapParams(
                q0=0.0,
                q1=delta,
                v0=0.0,
                v1=0.0,
                vmax=self.v_max[j],
                amax=self.a_max[j],
            )
            _, duration = TrapezoidalTrajectory.generate_trajectory(params)
            max_duration = max(max_duration, duration)

        return max(max_duration, self.dt * 2)

    def _compute_joint_duration_quintic(self) -> float:
        """
        Compute duration for joint paths using quintic polynomial profile.

        For quintic polynomials with zero-velocity endpoints:
        - Peak velocity at t=T/2: v_peak = 1.875 * delta / T
        - Peak acceleration at t=T*(3-sqrt(3))/6: a_peak = 5.77 * delta / T²

        For velocity limit: T = 1.875 * delta / v_max
        For acceleration limit: T = sqrt(5.77 * delta / a_max)

        Returns the maximum duration across all joints.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = np.abs(positions[-1] - positions[0])

        # Velocity-limited duration: T = 1.875 * delta / v_max
        time_vel = 1.875 * total_delta / self.v_max

        # Acceleration-limited duration: T = sqrt(5.77 * delta / a_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            time_acc = np.where(
                self.a_max > 0,
                np.sqrt(5.77 * total_delta / self.a_max),
                0.0,
            )

        # Take the maximum per joint, then across all joints
        time_per_joint = np.maximum(time_vel, time_acc)
        return max(float(np.max(time_per_joint)), self.dt * 2)

    def _compute_joint_duration_linear(self) -> float:
        """
        Compute duration for joint paths using linear interpolation.

        Accounts for both velocity and acceleration limits:
        - For velocity limit: T_vel = delta / v_max
        - For acceleration limit (triangular profile): T_acc = 2 * sqrt(2 * delta / a_max)

        Returns the maximum duration across all joints.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = np.abs(positions[-1] - positions[0])

        # Velocity-limited duration
        time_vel = total_delta / self.v_max

        # Acceleration-limited duration (for triangular velocity profile)
        # Time to reach target with constant accel then decel: T = 2 * sqrt(2 * d / a)
        with np.errstate(divide="ignore", invalid="ignore"):
            time_acc = np.where(
                self.a_max > 0,
                2.0 * np.sqrt(2.0 * total_delta / self.a_max),
                0.0,
            )

        # Take the maximum per joint, then across all joints
        time_per_joint = np.maximum(time_vel, time_acc)
        return max(float(np.max(time_per_joint)), self.dt * 2)

    def _compute_cartesian_duration_from_path(self) -> float:
        """
        Compute duration for Cartesian paths based on per-segment joint requirements.

        This properly handles singularities and wrist flips by analyzing
        the maximum joint movement required in each path segment, not just
        the total start-to-end displacement.

        Returns the sum of minimum segment times, ensuring the path can be
        traversed without violating joint velocity limits at any point.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        # Compute per-segment time based on max joint movement in each segment
        deltas = np.diff(positions, axis=0)  # (N-1, 6)
        segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)

        # Ensure minimum time per segment
        segment_times = np.maximum(segment_times, self.dt)

        return max(float(np.sum(segment_times)), self.dt * 2)

    def _build_quintic_trajectory(self) -> Trajectory:
        """
        Build trajectory with quintic polynomial velocity profile.

        For joint moves: each joint follows its own quintic profile.
        For Cartesian moves: TCP follows quintic profile along path.
        """
        if self._is_cartesian_path():
            return self._build_quintic_trajectory_cartesian()
        else:
            return self._build_quintic_trajectory_joint()

    def _build_quintic_trajectory_joint(self) -> Trajectory:
        """
        Build per-joint quintic trajectory.

        Each joint independently follows a quintic polynomial profile,
        synchronized to finish at the same time.
        """
        from interpolatepy import BoundaryCondition, PolynomialTrajectory, TimeInterval

        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        if self.duration:
            duration = self.duration
        else:
            duration = self._compute_joint_duration_quintic()

        n_output = max(2, int(np.ceil(duration / self.dt)))
        times = np.linspace(0.0, duration, n_output)
        trajectory_rad = np.empty((n_output, 6), dtype=np.float64)

        # Generate quintic profile for each joint
        for j in range(6):
            delta = end_pos[j] - start_pos[j]
            if abs(delta) < 1e-9:
                # Joint doesn't move
                trajectory_rad[:, j] = start_pos[j]
                continue

            # Create quintic trajectory for this joint
            bc_start = BoundaryCondition(
                position=start_pos[j], velocity=0.0, acceleration=0.0
            )
            bc_end = BoundaryCondition(
                position=end_pos[j], velocity=0.0, acceleration=0.0
            )
            interval = TimeInterval(start=0.0, end=duration)
            traj = PolynomialTrajectory.order_5_trajectory(bc_start, bc_end, interval)

            # Sample the trajectory
            for i, t in enumerate(times):
                trajectory_rad[i, j] = traj(t)[0]

        # Enforce velocity limits by stretching segments where needed
        trajectory_rad, duration = self._enforce_segment_limits(
            trajectory_rad, duration
        )

        # Convert to motor steps
        steps = _rad_to_steps_alloc(trajectory_rad)

        return Trajectory(steps=steps, duration=duration)

    def _build_quintic_trajectory_cartesian(self) -> Trajectory:
        """
        Build Cartesian quintic trajectory.

        TCP follows quintic polynomial profile along the path, with local
        slowdown where velocity limits would be exceeded.
        """
        from interpolatepy import BoundaryCondition, PolynomialTrajectory, TimeInterval

        if self.duration:
            duration = self.duration
        else:
            # Use per-segment analysis to handle singularities and wrist flips
            duration = self._compute_cartesian_duration_from_path()

        # Create quintic trajectory from s=0 to s=1 over duration
        bc_start = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
        bc_end = BoundaryCondition(position=1.0, velocity=0.0, acceleration=0.0)
        interval = TimeInterval(start=0.0, end=duration)
        traj = PolynomialTrajectory.order_5_trajectory(bc_start, bc_end, interval)

        n_output = max(2, int(np.ceil(duration / self.dt)))
        times = np.linspace(0.0, duration, n_output)

        # Evaluate quintic trajectory to get profile-shaped s values
        profile_s = np.empty(n_output, dtype=np.float64)
        for i in range(n_output):
            profile_s[i] = traj(float(times[i]))[0]

        # Sample path at quintic-shaped s values
        trajectory_rad = self.joint_path.sample_many(profile_s)

        # Enforce velocity limits by stretching segments where needed
        trajectory_rad, duration = self._enforce_segment_limits(
            trajectory_rad, duration
        )

        # Convert to motor steps
        steps = _rad_to_steps_alloc(trajectory_rad)

        return Trajectory(steps=steps, duration=duration)

    def _build_trapezoid_trajectory(self) -> Trajectory:
        """
        Build trajectory with trapezoidal velocity profile.

        For joint moves: each joint follows its own trapezoidal profile.
        For Cartesian moves: TCP follows trapezoidal profile along path.
        """
        if self._is_cartesian_path():
            return self._build_trapezoid_trajectory_cartesian()
        else:
            return self._build_trapezoid_trajectory_joint()

    def _build_trapezoid_trajectory_joint(self) -> Trajectory:
        """
        Build per-joint trapezoidal trajectory.

        Each joint independently follows a trapezoidal velocity profile,
        synchronized to finish at the same time.
        """
        from interpolatepy.trapezoidal import (
            TrajectoryParams as TrapParams,
            TrapezoidalTrajectory,
        )

        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        if self.duration:
            duration = self.duration
        else:
            duration = self._compute_joint_duration_trapezoid()

        n_output = max(2, int(np.ceil(duration / self.dt)))
        times = np.linspace(0.0, duration, n_output)
        trajectory_rad = np.empty((n_output, 6), dtype=np.float64)

        # Generate trapezoidal profile for each joint
        for j in range(6):
            delta = end_pos[j] - start_pos[j]
            if abs(delta) < 1e-9:
                # Joint doesn't move
                trajectory_rad[:, j] = start_pos[j]
                continue

            # Create trapezoidal profile for this joint
            params = TrapParams(
                q0=start_pos[j],
                q1=end_pos[j],
                v0=0.0,
                v1=0.0,
                vmax=self.v_max[j],
                amax=self.a_max[j],
            )
            traj_fn, profile_duration = TrapezoidalTrajectory.generate_trajectory(
                params
            )

            # Scale times to match synchronized duration
            time_scale = profile_duration / duration if duration > 0 else 1.0

            # Sample the trajectory
            for i, t in enumerate(times):
                trajectory_rad[i, j] = traj_fn(t * time_scale)[0]

        # Enforce velocity limits by stretching segments where needed
        trajectory_rad, duration = self._enforce_segment_limits(
            trajectory_rad, duration
        )

        # Convert to motor steps
        steps = _rad_to_steps_alloc(trajectory_rad)

        return Trajectory(steps=steps, duration=duration)

    def _build_trapezoid_trajectory_cartesian(self) -> Trajectory:
        """
        Build Cartesian trapezoidal trajectory.

        TCP follows trapezoidal velocity profile along the path, with local
        slowdown where velocity limits would be exceeded.
        """
        from interpolatepy.trapezoidal import (
            TrajectoryParams as TrapParams,
            TrapezoidalTrajectory,
        )

        if self.duration:
            duration = self.duration
        else:
            # Use per-segment analysis to handle singularities and wrist flips
            duration = self._compute_cartesian_duration_from_path()

        # Compute s-profile limits from joint limits
        vmax_s, amax_s, _ = self._compute_s_profile_limits()

        # Create trapezoidal profile for path parameter s (0 to 1)
        params = TrapParams(
            q0=0.0,
            q1=1.0,
            v0=0.0,
            v1=0.0,
            vmax=vmax_s,
            amax=amax_s,
        )
        traj_fn, profile_duration = TrapezoidalTrajectory.generate_trajectory(params)

        # If user specified longer duration, scale to match
        if self.duration and self.duration > profile_duration:
            time_scale = profile_duration / self.duration
            duration = self.duration
        else:
            time_scale = 1.0
            duration = profile_duration

        n_output = max(2, int(np.ceil(duration / self.dt)))
        times = np.linspace(0.0, duration, n_output)

        # Evaluate profile at scaled times to get profile-shaped s values
        profile_s = np.array(
            [traj_fn(t * time_scale)[0] for t in times], dtype=np.float64
        )

        # Sample path at trapezoid-shaped s values
        trajectory_rad = self.joint_path.sample_many(profile_s)

        # Enforce velocity limits by stretching segments where needed
        trajectory_rad, duration = self._enforce_segment_limits(
            trajectory_rad, duration
        )

        # Convert to motor steps
        steps = _rad_to_steps_alloc(trajectory_rad)

        return Trajectory(steps=steps, duration=duration)

    def _build_cart_vel_constraint(
        self, path: ta.SplineInterpolator, ss_waypoints: NDArray
    ) -> constraint.JointVelocityConstraintVarying | None:
        """
        Build Cartesian velocity constraint for TOPP-RA using path-tangent method.

        Uses the path tangent (dq/ds) to compute accurate Cartesian velocity limits.
        At each path point s:
        - cart_vel = J_lin @ q_dot = J_lin @ (dq/ds * s_dot)
        - ||cart_vel|| = ||J_lin @ dq/ds|| * |s_dot|
        - For ||cart_vel|| <= v_max: |s_dot| <= v_max / ||J_lin @ dq/ds||

        This is more accurate than the column-norm method as it considers the
        actual direction of motion along the path.

        Args:
            path: The spline path through joint space
            ss_waypoints: Path parameter values at each waypoint

        Returns:
            JointVelocityConstraintVarying with path-dependent limits, or None if error
        """
        if self.cart_vel_limit is None or self.cart_vel_limit <= 0:
            return None

        try:
            robot = PAROL6_ROBOT.robot

            # cart_vel_limit is already in m/s (SI units)
            v_max_m_s = self.cart_vel_limit
            # Use scaled joint limits (respects user's velocity_frac)
            v_max_joint = self.v_max

            # Pre-allocate buffers for velocity limits (avoids per-call allocation)
            vlim_buffer = np.empty((6, 2), dtype=np.float64)
            _jac_buf = np.zeros((6, 6), dtype=np.float64, order="F")

            def vlim_func(s: float) -> NDArray:
                """Compute velocity limits at path position s using path tangent."""
                q = path(s)
                dq_ds = path(s, 1)  # Path tangent (first derivative)

                # Get the linear part of the Jacobian (first 3 rows)
                robot.jacob0_into(q, _jac_buf)
                J_lin = _jac_buf[:3, :]

                # Cartesian velocity per unit s_dot along path tangent
                cart_vel_per_sdot = np.linalg.norm(J_lin @ dq_ds)

                if cart_vel_per_sdot < 1e-6:
                    # Near-zero path tangent (at waypoint or singular), use joint limits
                    vlim_buffer[:, 0] = -v_max_joint
                    vlim_buffer[:, 1] = v_max_joint
                    return vlim_buffer

                # Maximum s_dot to satisfy Cartesian velocity constraint
                max_sdot = v_max_m_s / cart_vel_per_sdot

                # The Cartesian constraint limits s_dot, not individual joint velocities.
                # We scale ALL joint velocity limits uniformly by the ratio of
                # (Cartesian-limited s_dot) / (fastest achievable s_dot from joint limits).
                #
                # This ensures the path velocity respects the Cartesian limit while
                # keeping joints at their relative proportions.
                abs_dq_ds = np.abs(dq_ds)

                # Compute s_dot limit from each joint's velocity limit
                with np.errstate(divide="ignore", invalid="ignore"):
                    s_dot_per_joint = np.where(
                        abs_dq_ds > 1e-9,
                        v_max_joint / abs_dq_ds,
                        np.inf,
                    )

                # The binding joint limit determines max achievable s_dot
                s_dot_from_joints = float(np.min(s_dot_per_joint))

                # If Cartesian constraint is more restrictive, scale down all limits
                if max_sdot < s_dot_from_joints and s_dot_from_joints > 0:
                    scale = max_sdot / s_dot_from_joints
                    q_dot_max = v_max_joint * scale
                else:
                    # Cartesian constraint is not binding, use joint limits
                    q_dot_max = v_max_joint

                vlim_buffer[:, 0] = -q_dot_max
                vlim_buffer[:, 1] = q_dot_max

                return vlim_buffer

            return constraint.JointVelocityConstraintVarying(vlim_func)

        except Exception as e:
            logger.warning("Failed to build Cartesian velocity constraint: %s", e)
            return None

    def _build_ruckig_trajectory(self) -> Trajectory:
        """
        Build trajectory using Ruckig for jerk-limited point-to-point motion.

        Note: This does NOT follow the path waypoints - it goes directly from
        start to end. Use TOPP-RA profiles for path-following motion.
        """
        n_dofs = 6
        gen = Ruckig(n_dofs, self.dt)
        inp = InputParameter(n_dofs)
        out = OutputParameter(n_dofs)

        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        # Ruckig requires Python lists for input parameters
        inp.current_position = start_pos.tolist()
        inp.current_velocity = [0.0] * n_dofs
        inp.current_acceleration = [0.0] * n_dofs
        inp.target_position = end_pos.tolist()
        inp.target_velocity = [0.0] * n_dofs
        inp.target_acceleration = [0.0] * n_dofs
        inp.max_velocity = self.v_max.tolist()
        inp.max_acceleration = self.a_max.tolist()
        inp.max_jerk = self.j_max.tolist()

        # Pre-allocate buffer (estimate max iterations from simple duration + margin)
        est_duration = self._estimate_simple_duration()
        max_iters = int(est_duration / self.dt) + 500  # generous margin
        trajectory_rad = np.empty((max_iters, n_dofs), dtype=np.float64)

        count = 0
        result = Result.Working

        while result == Result.Working:
            result = gen.update(inp, out)
            if count < max_iters:
                trajectory_rad[count] = out.new_position
            count += 1
            out.pass_to_input(inp)

        if result == Result.Error:
            raise RuntimeError("Ruckig failed to compute trajectory")

        actual_duration = out.trajectory.duration

        # Trim to actual size
        trajectory_rad = trajectory_rad[:count]

        # Convert to motor steps (vectorized)
        steps = _rad_to_steps_alloc(trajectory_rad)

        return Trajectory(steps=steps, duration=actual_duration)

    def _estimate_simple_duration(self) -> float:
        """Estimate minimum duration based on joint velocity limits.

        With adaptive time distribution, each segment gets time proportional
        to its joint movement, so total duration is sum of per-segment times.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        # Compute per-segment time based on max joint movement
        deltas = np.diff(positions, axis=0)  # (N-1, 6)
        segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)

        return max(float(np.sum(segment_times)), self.dt * 2)
