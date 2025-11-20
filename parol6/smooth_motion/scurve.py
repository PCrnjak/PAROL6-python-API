"""
S-curve profile generator and multi-axis S-curve trajectory.
"""

from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    # For type checkers and to satisfy linters referencing QuinticPolynomial in annotations
    from .quintic import QuinticPolynomial

from .motion_constraints import MotionConstraints


class FeasibilityInfo(TypedDict, total=False):
    status: str
    warnings: list[str]
    achievable_a: float
    achievable_v: float


class SCurveProfile:
    """
    Seven-segment S-curve velocity profile generator.

    Creates jerk-limited trajectories with smooth acceleration transitions.
    The seven segments are:
    1. Acceleration buildup (constant positive jerk)
    2. Constant acceleration
    3. Acceleration ramp-down (constant negative jerk)
    4. Constant velocity (cruise)
    5. Deceleration buildup (constant negative jerk)
    6. Constant deceleration
    7. Deceleration ramp-down (constant positive jerk)
    """

    def __init__(
        self,
        distance: float,
        v_max: float,
        a_max: float,
        j_max: float,
        v_start: float = 0,
        v_end: float = 0,
    ):
        """
        Calculate S-curve profile for given motion parameters.

        Args:
            distance: Total distance to travel
            v_max: Maximum velocity constraint
            a_max: Maximum acceleration constraint
            j_max: Maximum jerk constraint
            v_start: Initial velocity (default 0)
            v_end: Final velocity (default 0)
        """
        self.distance = distance
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.j_max = float(j_max)
        self.v_start = float(v_start)
        self.v_end = float(v_end)

        # Initialize typed containers for type checkers
        self.profile_type: str = ""
        self.segments: dict[str, float] = {}
        self.segment_boundaries: list[dict[str, float]] = []

        # Check feasibility and adjust parameters if needed
        self.feasibility_status = self._check_feasibility()

        # Calculate profile type and segment durations
        self.profile_type, self.segments = self._calculate_profile()

        # Pre-calculate segment boundaries for proper evaluation
        self._calculate_segment_boundaries()

    def _calculate_profile(self) -> tuple[str, dict[str, float]]:
        """
        Calculate the S-curve profile type and segment durations.

        Returns:
            profile_type: 'full', 'triangular', 'reduced', or 'asymmetric'
            segments: Dictionary with segment durations
        """
        # For symmetric profiles with v_start = v_end = 0
        if self.v_start == 0 and self.v_end == 0:
            return self._calculate_symmetric_profile()
        # Asymmetric profiles for non-zero boundary velocities
        return self._calculate_asymmetric_profile()

    def _calculate_symmetric_profile(self) -> tuple[str, dict[str, float]]:
        """Calculate symmetric S-curve profile (v_start = v_end = 0)."""
        # Time to reach maximum acceleration from zero (jerk phase)
        T_j = self.a_max / self.j_max if self.j_max > 0 else 0.0

        # Distance covered during jerk phases
        d_jerk = (self.a_max**3) / (self.j_max**2) if self.j_max > 0 else 0.0

        # Check if we can reach maximum velocity
        d_to_vmax = (
            (self.v_max**2 / self.a_max + self.v_max * self.a_max / self.j_max)
            if self.a_max > 0 and self.j_max > 0
            else float("inf")
        )

        if self.distance < 2 * d_jerk:
            # Case 1: Reduced acceleration profile (never reach a_max)
            profile_type = "reduced"
            a_reached = (
                (abs(self.distance) * self.j_max**2 / 2) ** (1 / 3)
                if self.j_max > 0
                else 0.0
            )
            T_j_actual = a_reached / self.j_max if self.j_max > 0 else 0.0

            segments = {
                "T_j1": T_j_actual,  # Jerk up
                "T_a": 0.0,  # No constant acceleration
                "T_j2": T_j_actual,  # Jerk down
                "T_v": 0.0,  # No cruise
                "T_j3": T_j_actual,  # Jerk down (decel)
                "T_d": 0.0,  # No constant deceleration
                "T_j4": T_j_actual,  # Jerk up (decel end)
                "a_reached": a_reached,
                "v_reached": a_reached * T_j_actual,
            }
        elif self.distance < 2 * d_to_vmax:
            # Case 2: Triangular velocity profile (never reach v_max)
            profile_type = "triangular"

            v_reached = np.sqrt(
                max(self.distance * self.a_max - 2 * self.a_max**3 / self.j_max**2, 0.0)
            )
            T_a = (
                (v_reached - self.a_max**2 / self.j_max) / self.a_max
                if self.a_max > 0 and self.j_max > 0
                else 0.0
            )

            segments = {
                "T_j1": T_j,
                "T_a": T_a,
                "T_j2": T_j,
                "T_v": 0.0,  # No cruise phase
                "T_j3": T_j,
                "T_d": T_a,
                "T_j4": T_j,
                "v_reached": v_reached,
            }
        else:
            # Case 3: Full S-curve with cruise phase
            profile_type = "full"

            # Time at constant acceleration (after jerk phases)
            T_a = (
                (self.v_max - self.a_max**2 / self.j_max) / self.a_max
                if self.a_max > 0 and self.j_max > 0
                else 0.0
            )

            # Distance covered during acceleration/deceleration
            d_accel = (
                self.v_max**2 / self.a_max + self.v_max * self.a_max / self.j_max
                if self.a_max > 0 and self.j_max > 0
                else 0.0
            )

            # Distance at cruise velocity
            d_cruise = self.distance - 2 * d_accel

            # Time at cruise velocity
            T_v = d_cruise / self.v_max if self.v_max > 0 else 0.0

            segments = {
                "T_j1": T_j,
                "T_a": max(T_a, 0.0),
                "T_j2": T_j,
                "T_v": max(T_v, 0.0),
                "T_j3": T_j,
                "T_d": max(T_a, 0.0),
                "T_j4": T_j,
                "v_reached": self.v_max,
            }

        # Calculate total time
        total_time = (
            segments.get("T_j1", 0.0)
            + segments.get("T_a", 0.0)
            + segments.get("T_j2", 0.0)
            + segments.get("T_v", 0.0)
            + segments.get("T_j3", 0.0)
            + segments.get("T_d", 0.0)
            + segments.get("T_j4", 0.0)
        )
        segments["T_total"] = total_time

        return profile_type, segments

    def _calculate_asymmetric_profile(self) -> tuple[str, dict[str, float]]:
        """Calculate asymmetric S-curve for non-zero boundary velocities."""
        v_diff = abs(self.v_end - self.v_start)
        v_avg = (self.v_start + self.v_end) / 2

        # Time to change between velocities at max acceleration
        T_vel_change = v_diff / self.a_max if self.a_max > 0 else 0.0

        # Jerk time (time to reach max acceleration)
        T_j = self.a_max / self.j_max if self.j_max > 0 else 0.0

        if self.v_start < self.v_max and self.v_end < self.v_max:
            v_peak = min(
                self.v_max, v_avg + np.sqrt(max(self.distance * self.a_max, 0.0))
            )

            T_accel = (v_peak - self.v_start) / self.a_max if self.a_max > 0 else 0.0
            T_a = max(0.0, T_accel - 2 * T_j)

            T_decel = (v_peak - self.v_end) / self.a_max if self.a_max > 0 else 0.0
            T_d = max(0.0, T_decel - 2 * T_j)

            d_accel = (
                self.v_start * (T_a + 2 * T_j) + 0.5 * self.a_max * (T_a + T_j) ** 2
            )
            d_decel = self.v_end * (T_d + 2 * T_j) + 0.5 * self.a_max * (T_d + T_j) ** 2
            d_cruise = self.distance - d_accel - d_decel

            if d_cruise > 0 and v_peak > 0:
                T_v = d_cruise / v_peak
            else:
                T_v = 0.0
                v_peak = np.sqrt(
                    max(
                        (
                            2 * self.distance * self.a_max
                            + self.v_start**2
                            + self.v_end**2
                        )
                        / 2,
                        0.0,
                    )
                )
        else:
            # Simple case - just decelerate
            T_a = 0.0
            T_v = 0.0
            T_d = T_vel_change
            v_peak = max(self.v_start, self.v_end)

        segments = {
            "T_j1": T_j,
            "T_a": T_a,
            "T_j2": T_j,
            "T_v": T_v,
            "T_j3": T_j,
            "T_d": T_d,
            "T_j4": T_j,
            "v_reached": v_peak,
            "T_total": 2 * T_j + T_a + T_v + 2 * T_j + T_d,
        }
        return "asymmetric", segments

    def _check_feasibility(self) -> FeasibilityInfo:
        """
        Check if S-curve profile is achievable with given constraints.

        Returns:
            Dictionary with feasibility status and adjusted parameters
        """
        # Minimum distance to reach maximum acceleration
        d_min_jerk = (self.a_max**3) / (self.j_max**2) if self.j_max > 0 else 0.0

        # Minimum distance to reach maximum velocity
        d_min_vel = (
            (self.v_max**2 / self.a_max + self.v_max * self.a_max / self.j_max)
            if self.a_max > 0 and self.j_max > 0
            else float("inf")
        )

        warnings: list[str] = []
        feasibility: FeasibilityInfo = {"status": "feasible", "warnings": warnings}

        if self.distance < 2 * d_min_jerk:
            achievable_a = (
                (abs(self.distance) * self.j_max**2 / 2) ** (1 / 3)
                if self.j_max > 0
                else 0.0
            )
            feasibility["status"] = "reduced_acceleration"
            feasibility["achievable_a"] = achievable_a
            feasibility["warnings"].append(
                f"Distance too short to reach full acceleration. Max achievable: {achievable_a:.2f}"
            )
        elif self.distance < 2 * d_min_vel:
            achievable_v = (
                np.sqrt(self.distance * self.a_max) if self.a_max > 0 else 0.0
            )
            feasibility["status"] = "triangular_velocity"
            feasibility["achievable_v"] = achievable_v
            feasibility["warnings"].append(
                f"Distance too short to reach full velocity. Max achievable: {achievable_v:.2f}"
            )

        if self.distance > 0 and self.a_max > 0:
            time_estimate = 2 * np.sqrt(self.distance / self.a_max)
            if time_estimate > 100:
                feasibility["warnings"].append(
                    f"Very long trajectory time ({time_estimate:.1f}s) may cause numerical issues"
                )

        if self.v_max <= 0 or self.a_max <= 0 or self.j_max <= 0:
            feasibility["status"] = "invalid_constraints"
            feasibility["warnings"].append(
                "Invalid constraints: v_max, a_max, and j_max must all be positive"
            )

        return feasibility

    def _calculate_segment_boundaries(self) -> None:
        """
        Pre-calculate position, velocity, and acceleration at segment boundaries.
        This ensures proper continuity across segments.
        """
        boundary_list: list[dict[str, float]] = []

        # Initial state
        pos = 0.0
        vel = self.v_start
        acc = 0.0

        # Segment 1: Jerk up (acceleration buildup)
        T_j1 = self.segments["T_j1"]
        if T_j1 > 0:
            j = self.j_max
            acc_end = acc + j * T_j1
            vel_end = vel + acc * T_j1 + 0.5 * j * T_j1**2
            pos_end = pos + vel * T_j1 + 0.5 * acc * T_j1**2 + (1 / 6) * j * T_j1**3
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_j1,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end

        # Segment 2: Constant acceleration
        T_a = self.segments["T_a"]
        if T_a > 0:
            j = 0.0
            acc_end = acc
            vel_end = vel + acc * T_a
            pos_end = pos + vel * T_a + 0.5 * acc * T_a**2
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_a,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end

        # Segment 3: Jerk down (acceleration ramp-down)
        T_j2 = self.segments["T_j2"]
        if T_j2 > 0:
            j = -self.j_max
            acc_end = acc + j * T_j2
            vel_end = vel + acc * T_j2 + 0.5 * j * T_j2**2
            pos_end = pos + vel * T_j2 + 0.5 * acc * T_j2**2 + (1 / 6) * j * T_j2**3
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_j2,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end

        # Segment 4: Constant velocity (cruise)
        T_v = self.segments["T_v"]
        if T_v > 0:
            j = 0.0
            acc_end = 0.0
            vel_end = vel
            pos_end = pos + vel * T_v
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_v,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end

        # Segment 5: Jerk down (deceleration buildup)
        T_j3 = self.segments["T_j3"]
        if T_j3 > 0:
            j = -self.j_max
            acc_end = j * T_j3
            vel_end = vel + 0.5 * j * T_j3**2
            pos_end = pos + vel * T_j3 + (1 / 6) * j * T_j3**3
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_j3,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end

        # Segment 6: Constant deceleration
        T_d = self.segments["T_d"]
        if T_d > 0:
            j = 0.0
            acc_end = acc
            vel_end = vel + acc * T_d
            pos_end = pos + vel * T_d + 0.5 * acc * T_d**2
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_d,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end

        # Segment 7: Jerk up (deceleration ramp-down)
        T_j4 = self.segments["T_j4"]
        if T_j4 > 0:
            j = self.j_max
            acc_end = acc + j * T_j4
            vel_end = vel + acc * T_j4 + 0.5 * j * T_j4**2
            pos_end = pos + vel * T_j4 + 0.5 * acc * T_j4**2 + (1 / 6) * j * T_j4**3
            boundary_list.append(
                {
                    "pos_start": pos,
                    "vel_start": vel,
                    "acc_start": acc,
                    "pos_end": pos_end,
                    "vel_end": vel_end,
                    "acc_end": acc_end,
                    "jerk": j,
                    "duration": T_j4,
                }
            )
            pos, vel, acc = pos_end, vel_end, acc_end
        # finalize
        self.segment_boundaries = boundary_list

    def generate_trajectory_quintics(self) -> list["QuinticPolynomial"]:
        """
        Generate quintic polynomials for each segment of the S-curve.

        Returns:
            List of QuinticPolynomial objects representing each segment
        """
        from .quintic import QuinticPolynomial  # Local import to avoid cycle

        quintics: list[QuinticPolynomial] = []
        for seg in self.segment_boundaries:
            if seg["duration"] > 0:
                q = QuinticPolynomial(
                    seg["pos_start"],
                    seg["pos_end"],
                    seg["vel_start"],
                    seg["vel_end"],
                    seg["acc_start"],
                    seg["acc_end"],
                    seg["duration"],
                )
                quintics.append(q)

        return quintics

    def get_total_time(self) -> float:
        """Get total time for the S-curve trajectory."""
        return float(self.segments["T_total"])

    def evaluate_at_time(self, t: float) -> dict[str, float]:
        """
        Evaluate S-curve at specific time.

        Returns:
            Dictionary with position, velocity, acceleration, jerk
        """
        if t <= 0:
            return {
                "position": 0.0,
                "velocity": self.v_start,
                "acceleration": 0.0,
                "jerk": 0.0,
            }

        if t >= self.segments["T_total"]:
            if self.segment_boundaries:
                last = self.segment_boundaries[-1]
                return {
                    "position": last["pos_end"],
                    "velocity": last["vel_end"],
                    "acceleration": 0.0,
                    "jerk": 0.0,
                }
            return {
                "position": self.distance,
                "velocity": self.v_end,
                "acceleration": 0.0,
                "jerk": 0.0,
            }

        # Find which segment we're in
        cumulative_time = 0.0
        for seg in self.segment_boundaries:
            if t <= cumulative_time + seg["duration"]:
                t_in_segment = t - cumulative_time
                return self._evaluate_in_segment(seg, t_in_segment)
            cumulative_time += seg["duration"]

        # Fallback
        return {
            "position": self.distance,
            "velocity": self.v_end,
            "acceleration": 0.0,
            "jerk": 0.0,
        }

    def _evaluate_in_segment(
        self, segment: dict[str, float], t: float
    ) -> dict[str, float]:
        """
        Evaluate motion within a specific segment using proper kinematics.
        """
        p0 = segment["pos_start"]
        v0 = segment["vel_start"]
        a0 = segment["acc_start"]
        j = segment["jerk"]

        # Kinematics within segment
        acc = a0 + j * t
        vel = v0 + a0 * t + 0.5 * j * t**2
        pos = p0 + v0 * t + 0.5 * a0 * t**2 + (1 / 6) * j * t**3

        return {"position": pos, "velocity": vel, "acceleration": acc, "jerk": j}


class MultiAxisSCurveTrajectory:
    """
    Multi-axis synchronized S-curve trajectory generator.

    Coordinates multiple S-curve profiles (one per axis) and synchronizes them
    to ensure all axes complete their motion simultaneously while respecting
    individual axis constraints.
    """

    def __init__(
        self,
        start_pose: list[float],
        end_pose: list[float],
        v0: list[float] | None = None,
        vf: list[float] | None = None,
        T: float | None = None,
        jerk_limit: float | None = None,
    ):
        """
        Create synchronized S-curve trajectories for multiple axes.

        Args:
            start_pose: Starting position for each axis
            end_pose: Target position for each axis
            v0: Initial velocities (defaults to zeros)
            vf: Final velocities (defaults to zeros)
            T: Desired duration (if None, calculates minimum time)
            jerk_limit: Maximum jerk limit to apply to all axes
        """
        self.start_pose = np.array(start_pose, dtype=float)
        self.end_pose = np.array(end_pose, dtype=float)
        self.num_axes = len(start_pose)

        self.v0 = (
            np.array(v0, dtype=float)
            if v0 is not None
            else np.zeros(self.num_axes, dtype=float)
        )
        self.vf = (
            np.array(vf, dtype=float)
            if vf is not None
            else np.zeros(self.num_axes, dtype=float)
        )

        self.constraints = MotionConstraints()

        self.axis_profiles: list[SCurveProfile | None] = []
        self.max_time = 0.0

        for i in range(self.num_axes):
            distance = self.end_pose[i] - self.start_pose[i]

            if abs(distance) < 1e-6:
                self.axis_profiles.append(None)
                continue

            joint_constraints = self.constraints.get_joint_constraints(i)
            if joint_constraints is None:
                joint_constraints = {
                    "v_max": 10000.0,
                    "a_max": 20000.0,
                    "j_max": jerk_limit if jerk_limit else 50000.0,
                }

            j_max = jerk_limit if jerk_limit is not None else joint_constraints["j_max"]

            axis_profile = SCurveProfile(
                float(distance),
                float(joint_constraints["v_max"]),
                float(joint_constraints["a_max"]),
                float(j_max),
                v_start=float(self.v0[i]),
                v_end=float(self.vf[i]),
            )
            self.axis_profiles.append(axis_profile)
            self.max_time = max(self.max_time, axis_profile.get_total_time())

        self.T = float(T) if T is not None else float(self.max_time)

        # Calculate time scaling factors for synchronization
        self.time_scales: list[float] = []
        for maybe_profile in self.axis_profiles:
            if maybe_profile is not None and self.T > 0:
                scale = maybe_profile.get_total_time() / self.T
                self.time_scales.append(scale)
            else:
                self.time_scales.append(1.0)

    def get_trajectory_points(self, dt: float = 0.01) -> dict[str, np.ndarray]:
        """
        Generate synchronized trajectory points for all axes.

        Args:
            dt: Time step for sampling
        """
        num_points = int(self.T / dt) + 1 if self.T > 0 else 1
        timestamps = np.linspace(0, self.T, num_points)

        positions = np.zeros((num_points, self.num_axes))
        velocities = np.zeros((num_points, self.num_axes))
        accelerations = np.zeros((num_points, self.num_axes))

        for idx, t in enumerate(timestamps):
            for axis in range(self.num_axes):
                axis_profile = self.axis_profiles[axis]
                if axis_profile is None:
                    positions[idx, axis] = self.start_pose[axis]
                    velocities[idx, axis] = 0.0
                    accelerations[idx, axis] = 0.0
                else:
                    t_scaled = t * self.time_scales[axis]
                    values = axis_profile.evaluate_at_time(t_scaled)
                    positions[idx, axis] = self.start_pose[axis] + values["position"]
                    velocities[idx, axis] = values["velocity"]
                    accelerations[idx, axis] = values["acceleration"]

        return {
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "timestamps": timestamps,
        }

    def evaluate_at_time(self, t: float) -> dict[str, np.ndarray]:
        """
        Evaluate trajectory at specific time.

        Args:
            t: Time point to evaluate
        """
        position = np.zeros(self.num_axes)
        velocity = np.zeros(self.num_axes)
        acceleration = np.zeros(self.num_axes)

        for axis in range(self.num_axes):
            axis_profile = self.axis_profiles[axis]
            if axis_profile is None:
                position[axis] = self.start_pose[axis]
                velocity[axis] = 0.0
                acceleration[axis] = 0.0
            else:
                t_scaled = min(
                    t * self.time_scales[axis], axis_profile.get_total_time()
                )
                values = axis_profile.evaluate_at_time(t_scaled)
                position[axis] = self.start_pose[axis] + values["position"]
                velocity[axis] = values["velocity"]
                acceleration[axis] = values["acceleration"]

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
        }
