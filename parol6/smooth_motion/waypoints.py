"""
Waypoint trajectory planner.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class WaypointTrajectoryPlanner:
    """
    Trajectory planner for smooth motion through waypoints with corner cutting.

    Implements mstraj-style parabolic blending at waypoints to avoid acceleration
    spikes and ensure smooth motion through complex paths.
    """

    def __init__(self, waypoints: List[List[float]], constraints: Optional[Dict] = None, sample_rate: float = 100.0):
        """
        Initialize waypoint trajectory planner.

        Args:
            waypoints: List of waypoint poses [x, y, z, rx, ry, rz]
            constraints: Motion constraints (max_velocity, max_acceleration, max_jerk)
            sample_rate: Trajectory sampling rate in Hz
        """
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.num_waypoints = len(waypoints)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate

        # Default constraints
        default_constraints = {
            "max_velocity": 100.0,  # mm/s
            "max_acceleration": 500.0,  # mm/s²
            "max_jerk": 5000.0,  # mm/s³
        }
        self.constraints = constraints if constraints else default_constraints

        # Blend planning data
        self.blend_radii: List[float] = []
        self.blend_regions: List[Optional[Dict[str, np.ndarray]]] = []
        self.segment_velocities: List[float] = []
        self.via_modes = ["via"] * self.num_waypoints  # Default: pass through all

    def calculate_corner_angle(self, idx: int) -> float:
        """
        Calculate the angle between incoming and outgoing path segments.

        Args:
            idx: Waypoint index

        Returns:
            Angle in radians (0 to π)
        """
        if idx <= 0 or idx >= self.num_waypoints - 1:
            return 0.0  # No corner at start/end

        # Vectors for path segments
        v_in = self.waypoints[idx] - self.waypoints[idx - 1]
        v_out = self.waypoints[idx + 1] - self.waypoints[idx]

        # Normalize (use only position components)
        v_in_norm = v_in[:3] / (np.linalg.norm(v_in[:3]) + 1e-9)
        v_out_norm = v_out[:3] / (np.linalg.norm(v_out[:3]) + 1e-9)

        # Calculate angle
        cos_angle = np.clip(np.dot(v_in_norm, v_out_norm), -1, 1)
        angle = np.arccos(cos_angle)

        return angle

    def calculate_safe_blend_radius(self, idx: int, approach_velocity: float) -> float:
        """
        Calculate maximum safe blend radius for a waypoint.

        Args:
            idx: Waypoint index
            approach_velocity: Velocity approaching the waypoint

        Returns:
            Safe blend radius in mm
        """
        angle = self.calculate_corner_angle(idx)

        if angle < 0.01:  # Nearly straight (< 0.5 degrees)
            return 0.0

        # Dynamic blend radius based on velocity and angle
        a_max = self.constraints["max_acceleration"]

        # Centripetal acceleration constraint
        # r = v² / (a_max * sin(θ/2))
        sin_half_angle = np.sin(angle / 2)
        if sin_half_angle > 0:
            r_dynamic = (approach_velocity**2) / (a_max * sin_half_angle)
        else:
            r_dynamic = 0.0

        # Geometric constraint - don't exceed segment lengths
        r_geometric = self.get_max_geometric_radius(idx)

        # Apply safety factor and limits
        r_safe = min(r_dynamic, r_geometric) * 0.7  # 70% safety factor
        r_safe = max(0, min(r_safe, 100))  # Cap at 100mm

        return r_safe

    def get_max_geometric_radius(self, idx: int) -> float:
        """
        Get maximum blend radius based on segment geometry.

        Args:
            idx: Waypoint index

        Returns:
            Maximum geometric radius in mm
        """
        if idx <= 0 or idx >= self.num_waypoints - 1:
            return 0.0

        # Distance to previous waypoint
        dist_prev = np.linalg.norm(self.waypoints[idx][:3] - self.waypoints[idx - 1][:3])

        # Distance to next waypoint
        dist_next = np.linalg.norm(self.waypoints[idx + 1][:3] - self.waypoints[idx][:3])

        # Maximum radius is 40% of shortest segment
        max_radius = 0.4 * min(dist_prev, dist_next)

        return max_radius

    def calculate_auto_blend_radii(self):
        """
        Automatically calculate blend radius for each waypoint.
        """
        self.blend_radii = []

        for i in range(self.num_waypoints):
            if i == 0 or i == self.num_waypoints - 1:
                # No blending at start/end
                self.blend_radii.append(0.0)
            else:
                # Estimate approach velocity
                segment_length = np.linalg.norm(self.waypoints[i][:3] - self.waypoints[i - 1][:3])

                # Simple velocity estimation
                if segment_length > 0:
                    approach_velocity = min(
                        self.constraints["max_velocity"],
                        np.sqrt(self.constraints["max_acceleration"] * segment_length),
                    )
                else:
                    approach_velocity = 0

                # Calculate safe radius
                radius = self.calculate_safe_blend_radius(i, approach_velocity)
                self.blend_radii.append(radius)

    def compute_blend_points(self, idx: int, blend_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate blend entry and exit points for a waypoint.

        Args:
            idx: Waypoint index
            blend_radius: Blend radius in mm

        Returns:
            Tuple of (entry_point, exit_point)
        """
        if blend_radius <= 0 or idx <= 0 or idx >= self.num_waypoints - 1:
            return self.waypoints[idx], self.waypoints[idx]

        # Get path vectors
        v_in = self.waypoints[idx] - self.waypoints[idx - 1]
        v_out = self.waypoints[idx + 1] - self.waypoints[idx]

        # Normalize position components
        v_in_norm = np.zeros_like(v_in)
        v_out_norm = np.zeros_like(v_out)

        v_in_norm[:3] = v_in[:3] / (np.linalg.norm(v_in[:3]) + 1e-9)
        v_out_norm[:3] = v_out[:3] / (np.linalg.norm(v_out[:3]) + 1e-9)

        # Calculate blend entry point (along incoming path)
        blend_entry = self.waypoints[idx].copy()
        blend_entry[:3] -= v_in_norm[:3] * blend_radius

        # Calculate blend exit point (along outgoing path)
        blend_exit = self.waypoints[idx].copy()
        blend_exit[:3] += v_out_norm[:3] * blend_radius

        # Interpolate orientations
        angle = self.calculate_corner_angle(idx)
        if angle > 0:
            # Weighted average for orientations at blend points
            alpha_entry = 1.0 - (blend_radius / (np.linalg.norm(v_in[:3]) + 1e-9))
            alpha_exit = blend_radius / (np.linalg.norm(v_out[:3]) + 1e-9)

            blend_entry[3:] = (self.waypoints[idx - 1][3:] * (1 - alpha_entry) + self.waypoints[idx][3:] * alpha_entry)
            blend_exit[3:] = (self.waypoints[idx][3:] * (1 - alpha_exit) + self.waypoints[idx + 1][3:] * alpha_exit)

        return blend_entry, blend_exit

    def generate_parabolic_blend(
        self,
        entry_point: np.ndarray,
        exit_point: np.ndarray,
        v_entry: np.ndarray,
        v_exit: np.ndarray,
        blend_time: Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Generate parabolic trajectory for corner blend with acceleration limits.

        Parabolic blends have constant acceleration, providing smooth motion.

        Args:
            entry_point: Blend entry position
            exit_point: Blend exit position
            v_entry: Velocity at blend entry
            v_exit: Velocity at blend exit
            blend_time: Blend duration (auto-calculated if None)

        Returns:
            List of trajectory points through the blend
        """
        # Calculate blend parameters
        delta_p = exit_point - entry_point
        distance = np.linalg.norm(delta_p[:3])

        if distance < 1e-6:
            return [entry_point]  # No blend needed

        # Calculate velocity change needed
        delta_v = v_exit - v_entry
        delta_v_mag = np.linalg.norm(delta_v[:3])

        # Minimum blend time from acceleration constraint
        min_blend_time = delta_v_mag / (self.constraints["max_acceleration"] + 1e-9)

        # Calculate blend time if not specified
        if blend_time is None:
            v_avg = (np.linalg.norm(v_entry[:3]) + np.linalg.norm(v_exit[:3])) / 2
            if v_avg > 0:
                time_from_velocity = distance / v_avg
            else:
                time_from_velocity = np.sqrt(2 * distance / (self.constraints["max_acceleration"] + 1e-9))
            blend_time = max(min_blend_time, time_from_velocity)
        else:
            blend_time = max(blend_time, min_blend_time)

        blend_time = max(blend_time, 0.01)  # Numerical stability

        # Acceleration (bounded)
        a_blend = delta_v / blend_time
        a_mag = np.linalg.norm(a_blend[:3])
        if a_mag > self.constraints["max_acceleration"] * 1.1:  # 10% tolerance
            scale = self.constraints["max_acceleration"] / (a_mag + 1e-9)
            a_blend = a_blend * scale
            blend_time = blend_time / (scale + 1e-9)

        # Generate trajectory using cubic Hermite interpolation (C1 continuity)
        num_points = max(5, int(blend_time * self.sample_rate))  # At least 5 points
        blend_traj = []

        for i in range(num_points):
            # Normalized time from 0 to 1
            s = i / (num_points - 1) if num_points > 1 else 0.0

            # Cubic Hermite basis functions
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2

            # Interpolate position using hermite spline
            # Scale velocities by blend_time to get tangents
            pos = h00 * entry_point + h10 * (v_entry * blend_time) + h01 * exit_point + h11 * (v_exit * blend_time)

            blend_traj.append(pos)

        return blend_traj

    def generate_linear_segment(self, start: np.ndarray, end: np.ndarray, velocity: Optional[float] = None) -> List[np.ndarray]:
        """
        Generate linear trajectory segment between two points.

        Args:
            start: Start position
            end: End position
            velocity: Desired velocity (uses max if None)

        Returns:
            List of trajectory points
        """
        distance = np.linalg.norm(end[:3] - start[:3])

        if distance < 1e-6:
            return [start]

        # Determine velocity
        if velocity is None:
            velocity = self.constraints["max_velocity"]

        # Calculate duration and number of points
        duration = distance / velocity
        num_points = max(2, int(duration * self.sample_rate))

        # Generate trajectory
        segment = []
        for i in range(num_points):
            alpha = i / (num_points - 1) if num_points > 1 else 0.0
            pos = start * (1 - alpha) + end * alpha
            segment.append(pos)

        return segment

    def compute_blend_regions(self):
        """
        Compute all blend regions for the trajectory.
        """
        self.blend_regions = []

        for i in range(1, self.num_waypoints - 1):
            if self.blend_radii[i] > 0 and self.via_modes[i] == "via":
                entry, exit = self.compute_blend_points(i, self.blend_radii[i])

                # Calculate velocities at blend points
                v_entry_dir = self.waypoints[i] - self.waypoints[i - 1]
                v_entry = v_entry_dir[:3] / (np.linalg.norm(v_entry_dir[:3]) + 1e-9) * self.segment_velocities[i - 1]
                v_entry_full = np.zeros(len(entry))
                v_entry_full[:3] = v_entry

                v_exit_dir = self.waypoints[i + 1] - self.waypoints[i]
                v_exit = v_exit_dir[:3] / (np.linalg.norm(v_exit_dir[:3]) + 1e-9) * self.segment_velocities[i]
                v_exit_full = np.zeros(len(exit))
                v_exit_full[:3] = v_exit

                self.blend_regions.append(
                    {
                        "waypoint_idx": i,
                        "entry": entry,
                        "exit": exit,
                        "radius": self.blend_radii[i],
                        "v_entry": v_entry_full,
                        "v_exit": v_exit_full,
                    }
                )
            else:
                # No blend or stop point
                self.blend_regions.append(None)

    def plan_trajectory(
        self,
        blend_mode: str = "auto",
        blend_radii: Optional[List[float]] = None,
        via_modes: Optional[List[str]] = None,
        trajectory_type: str = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Plan complete trajectory through waypoints with blending.

        Args:
            blend_mode: 'auto', 'manual', or 'none'
            blend_radii: Manual blend radii (if blend_mode='manual')
            via_modes: 'via' or 'stop' for each waypoint
            trajectory_type: 'cubic', 'quintic', or 's_curve' velocity profile
            jerk_limit: Maximum jerk for s_curve profile (mm/s^3)

        Returns:
            Complete trajectory as numpy array
        """
        if self.num_waypoints < 2:
            return self.waypoints

        # Set blend radii
        if blend_mode in ["auto", "parabolic", "circular"]:
            self.calculate_auto_blend_radii()
        elif blend_mode == "manual" and blend_radii is not None:
            self.blend_radii = blend_radii
        elif blend_mode == "none":
            self.blend_radii = [0.0] * self.num_waypoints
        else:
            # Default to auto for unrecognized modes
            self.calculate_auto_blend_radii()

        # Set via modes
        if via_modes is not None:
            self.via_modes = via_modes

        # Calculate segment velocities
        self.segment_velocities = []
        for i in range(self.num_waypoints - 1):
            segment_length = np.linalg.norm(self.waypoints[i + 1][:3] - self.waypoints[i][:3])
            # Simple velocity planning
            if self.via_modes[i] == "stop" or self.via_modes[i + 1] == "stop":
                # Trapezoid profile with acceleration
                v_max = min(self.constraints["max_velocity"], np.sqrt(self.constraints["max_acceleration"] * segment_length))
            else:
                v_max = self.constraints["max_velocity"]

            self.segment_velocities.append(v_max)

        # Compute blend regions
        self.compute_blend_regions()

        # Generate full trajectory
        full_trajectory: List[np.ndarray] = []

        for i in range(self.num_waypoints - 1):
            # Determine segment start and end
            if i == 0:
                segment_start = self.waypoints[0]
            else:
                # Check for blend at current waypoint
                blend_region_prev = self.blend_regions[i - 1] if i - 1 < len(self.blend_regions) else None
                if blend_region_prev:
                    segment_start = blend_region_prev["exit"]
                else:
                    segment_start = self.waypoints[i]

            if i < self.num_waypoints - 2:
                # Check for blend at next waypoint
                blend_region_next = self.blend_regions[i] if i < len(self.blend_regions) else None
                if blend_region_next:
                    segment_end = blend_region_next["entry"]
                else:
                    segment_end = self.waypoints[i + 1]
            else:
                segment_end = self.waypoints[i + 1]

            # Generate linear segment
            segment = self.generate_linear_segment(segment_start, segment_end, self.segment_velocities[i])

            # Add segment to trajectory
            if i == 0:
                full_trajectory.extend(segment)
            else:
                # Skip first point to avoid duplication
                full_trajectory.extend(segment[1:])

            # Add blend if needed
            if i < len(self.blend_regions) and self.blend_regions[i]:
                blend_region = self.blend_regions[i]
                blend_traj = self.generate_parabolic_blend(
                    blend_region["entry"],
                    blend_region["exit"],
                    blend_region["v_entry"],
                    blend_region["v_exit"],
                )
                # Skip first point to avoid duplication
                full_trajectory.extend(blend_traj[1:])

        trajectory_array = np.array(full_trajectory)

        # Apply profile to the generated trajectory if not cubic
        if trajectory_type != "cubic":
            trajectory_array = self.apply_velocity_profile(trajectory_array, trajectory_type, jerk_limit)

        return trajectory_array

    def apply_velocity_profile(
        self, trajectory: np.ndarray, profile_type: str = "quintic", jerk_limit: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply velocity profile to existing trajectory points.

        Instead of re-interpolating with sparse waypoints, this method
        applies the velocity profile to ALL trajectory points, preserving
        the geometric path while adjusting the timing.

        Args:
            trajectory: Input trajectory points
            profile_type: 'quintic' or 's_curve'
            jerk_limit: Maximum jerk for s_curve (mm/s^3)

        Returns:
            Trajectory with velocity profile applied
        """
        if len(trajectory) < 2:
            return trajectory

        # Calculate cumulative arc length
        arc_lengths = [0.0]
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i][:3] - trajectory[i - 1][:3])
            arc_lengths.append(arc_lengths[-1] + dist)

        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return trajectory

        # Generate new time mapping based on profile
        num_points = len(trajectory)
        new_trajectory = np.zeros_like(trajectory)

        for i in range(num_points):
            # Get normalized position along path
            s = i / (num_points - 1) if num_points > 1 else 0.0

            # Apply velocity profile to get new arc length position
            if profile_type == "quintic":
                # Quintic polynomial: smooth acceleration/deceleration
                s_new = 10 * s**3 - 15 * s**4 + 6 * s**5
            elif profile_type == "s_curve":
                # S-curve (smoothstep): smooth jerk-limited motion
                if s <= 0.0:
                    s_new = 0.0
                elif s >= 1.0:
                    s_new = 1.0
                else:
                    # Smoothstep function for smooth acceleration
                    s_new = s * s * (3.0 - 2.0 * s)
            else:
                # Default to linear (no change)
                s_new = s

            # Find the corresponding point on original trajectory
            # Use linear interpolation between trajectory points
            target_arc_length = s_new * total_length

            # Find the segment containing this arc length
            for j in range(len(arc_lengths) - 1):
                if arc_lengths[j] <= target_arc_length <= arc_lengths[j + 1]:
                    # Interpolate within this segment
                    segment_length = arc_lengths[j + 1] - arc_lengths[j]
                    if segment_length > 1e-6:
                        alpha = (target_arc_length - arc_lengths[j]) / segment_length
                    else:
                        alpha = 0.0

                    # Linear interpolation between points
                    new_trajectory[i] = (1 - alpha) * trajectory[j] + alpha * trajectory[j + 1]
                    break
            else:
                # If we didn't find it (shouldn't happen), use the last point
                new_trajectory[i] = trajectory[-1]

        return new_trajectory

    def validate_trajectory(self, trajectory: np.ndarray) -> Dict[str, bool]:
        """
        Validate that trajectory respects all constraints.

        Args:
            trajectory: Trajectory to validate

        Returns:
            Dictionary of validation results with detailed information
        """
        results: Dict[str, float | bool] = {
            "velocity_ok": True,
            "acceleration_ok": True,
            "jerk_ok": True,
            "continuity_ok": True,
            "max_velocity": 0.0,
            "max_acceleration": 0.0,
            "max_jerk": 0.0,
            "max_step": 0.0,
        }

        if len(trajectory) < 2:
            return results

        dt = self.dt
        n_dims = min(trajectory.shape[1], 6)

        # Check continuity - maximum step size between points
        position_diffs = np.diff(trajectory[:, :3], axis=0)
        step_sizes = np.linalg.norm(position_diffs, axis=1)
        max_step = float(np.max(step_sizes)) if step_sizes.size else 0.0
        results["max_step"] = max_step

        expected_max_step = self.constraints["max_velocity"] * dt * 1.5  # 50% tolerance
        results["continuity_ok"] = max_step <= expected_max_step

        # Velocity - use all relevant dimensions
        velocities = np.diff(trajectory[:, :n_dims], axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities[:, :3], axis=1) if velocities.shape[0] else np.array([0.0])
        max_vel = float(np.max(velocity_magnitudes)) if velocity_magnitudes.size else 0.0
        results["max_velocity"] = max_vel
        results["velocity_ok"] = max_vel <= self.constraints["max_velocity"] * 1.1

        # Acceleration
        if len(trajectory) > 2:
            accelerations = np.diff(velocities[:, :3], axis=0) / dt if velocities.shape[0] > 1 else np.zeros((0, 3))
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1) if accelerations.shape[0] else np.array([0.0])
            max_acc = float(np.max(acceleration_magnitudes)) if acceleration_magnitudes.size else 0.0
            results["max_acceleration"] = max_acc
            results["acceleration_ok"] = max_acc <= self.constraints["max_acceleration"] * 1.2

            # Jerk
            if len(trajectory) > 3:
                jerks = np.diff(accelerations, axis=0) / dt if accelerations.shape[0] > 1 else np.zeros((0, 3))
                jerk_magnitudes = np.linalg.norm(jerks, axis=1) if jerks.shape[0] else np.array([0.0])
                max_jerk = float(np.max(jerk_magnitudes)) if jerk_magnitudes.size else 0.0
                results["max_jerk"] = max_jerk
                results["jerk_ok"] = max_jerk <= self.constraints["max_jerk"] * 1.5

        return results
