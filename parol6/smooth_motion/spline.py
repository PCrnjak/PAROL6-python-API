"""
Spline trajectory generator.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .base import TrajectoryGenerator
from .quintic import MultiAxisQuinticTrajectory
from .scurve import SCurveProfile


class SplineMotion(TrajectoryGenerator):
    """Generate smooth spline trajectories through waypoints"""

    def generate_quintic_spline_true(
        self,
        waypoints: List[List[float]],
        waypoint_behavior: str = "stop",
        profile_type: str = "s_curve",
        optimization: str = "jerk",
        time_optimal: bool = False,
        jerk_limit: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate true quintic spline trajectory with optional S-curve profiles.

        This is the enhanced version using our quintic polynomial implementation
        instead of the cubic-based version.

        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            waypoint_behavior: 'stop' or 'continuous' at waypoints
            profile_type: 's_curve', 'quintic', or 'cubic'
            optimization: 'time', 'jerk', or 'energy'
            time_optimal: Calculate minimum time if True
        """
        if profile_type == "s_curve":
            return self._generate_scurve_waypoints(waypoints, waypoint_behavior, optimization, jerk_limit)
        if profile_type == "quintic":
            return self._generate_pure_quintic_waypoints(waypoints, waypoint_behavior)
        # Fall back to cubic implementation
        return self.generate_cubic_spline(waypoints)

    def _generate_pure_quintic_waypoints(self, waypoints, behavior):
        """Generate quintic trajectories between waypoints."""
        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)

        if num_waypoints < 2:
            return waypoints

        # Calculate timing for each segment
        segment_times = []
        for i in range(num_waypoints - 1):
            distance = np.linalg.norm(waypoints[i + 1, :3] - waypoints[i, :3])
            time = max(1.0, distance / 50.0)  # 50 mm/s average
            segment_times.append(time)

        # Generate trajectory segments
        full_trajectory = []
        prev_vf: Optional[List[float]] = None

        for i in range(num_waypoints - 1):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]
            T = segment_times[i]

            # Determine boundary velocities based on behavior
            if behavior == "stop":
                v0 = [0.0] * 6
                vf = [0.0] * 6
            else:  # continuous
                if i == 0:
                    v0 = [0.0] * 6
                else:
                    v0 = prev_vf if prev_vf is not None else [0.0] * 6

                if i == num_waypoints - 2:
                    vf = [0.0] * 6
                else:
                    next_direction = waypoints[i + 2] - waypoints[i + 1]
                    next_segment_time = segment_times[i + 1] if (i + 1) < len(segment_times) else segment_times[i]
                    current_direction = waypoints[i + 1] - waypoints[i]
                    avg_direction = (current_direction / segment_times[i] + next_direction / next_segment_time) * 0.5
                    vf = list(avg_direction[:6] * 0.7)  # Scale factor for stability

                prev_vf = vf

            # Create multi-axis quintic trajectory
            segment_traj = MultiAxisQuinticTrajectory(list(start_pose), list(end_pose), v0, vf, T=T)

            # Sample the segment
            segment_points = segment_traj.get_trajectory_points(self.dt)

            # Add to full trajectory (avoid duplicating waypoints)
            if i == 0:
                full_trajectory.extend(segment_points["position"])
            else:
                full_trajectory.extend(segment_points["position"][1:])

        return np.array(full_trajectory)

    def _generate_scurve_waypoints(self, waypoints, behavior, optimization, jerk_limit=None):
        """Generate S-curve trajectories between waypoints."""
        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)

        if num_waypoints < 2:
            return waypoints

        full_trajectory: List[List[float]] = []

        for i in range(num_waypoints - 1):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]

            # For each joint, calculate S-curve profile
            segment_trajectories = []
            max_time = 0.0

            for j in range(6):  # 6 joints
                distance = end_pose[j] - start_pose[j]

                # Get joint constraints
                constraints = self.constraints.get_joint_constraints(j)  # type: ignore[attr-defined]

                # Use provided jerk limit if available, otherwise use constraints
                j_max = jerk_limit if jerk_limit is not None else constraints["j_max"]

                # Create S-curve profile
                scurve = SCurveProfile(distance, constraints["v_max"], constraints["a_max"], j_max)

                max_time = max(max_time, scurve.get_total_time())
                segment_trajectories.append(scurve)

            # Synchronize all joints to slowest
            if optimization == "time":
                sync_time = max_time
            else:
                sync_time = max_time * 1.2  # margin

            # Generate synchronized points
            timestamps = self.generate_timestamps(sync_time)

            for t in timestamps:
                pose = []
                for j in range(6):
                    joint_scurve = segment_trajectories[j]
                    t_normalized = t / sync_time  # Normalize to [0, 1]
                    t_joint = t_normalized * joint_scurve.get_total_time()

                    values = joint_scurve.evaluate_at_time(float(t_joint))
                    pose.append(float(start_pose[j] + values["position"]))
                full_trajectory.append(pose)

        return np.array(full_trajectory)

    def generate_cubic_spline(
        self,
        waypoints: List[List[float]],
        timestamps: Optional[List[float]] = None,
        velocity_start: Optional[List[float]] = None,
        velocity_end: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Generate cubic spline trajectory through waypoints

        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            timestamps: Time for each waypoint (auto-generated if None)
            velocity_start: Initial velocity (zero if None)
            velocity_end: Final velocity (zero if None)

        Returns:
            Array of interpolated poses
        """
        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)

        # Auto-generate timestamps if not provided
        if timestamps is None:
            total_dist = 0.0
            for i in range(1, num_waypoints):
                dist = np.linalg.norm(waypoints[i, :3] - waypoints[i - 1, :3])
                total_dist += dist

            # Assume average speed of 50 mm/s
            total_time = total_dist / 50.0 if total_dist > 0 else 0.0
            timestamps = np.linspace(0, total_time, num_waypoints)

        # Validate array dimensions before creating splines
        if len(timestamps) != len(waypoints):
            raise ValueError(f"Timestamps length ({len(timestamps)}) must match waypoints length ({len(waypoints)})")

        # Position trajectory splines
        pos_splines = []
        for i in range(3):
            bc_type = "not-a-knot"  # Default boundary condition
            if velocity_start is not None and velocity_end is not None:
                bc_type = ((1, velocity_start[i]), (1, velocity_end[i]))  # type: ignore[assignment]
            spline = CubicSpline(timestamps, waypoints[:, i], bc_type=bc_type)
            pos_splines.append(spline)

        # Orientation trajectory splines
        rotations = [Rotation.from_euler("xyz", wp[3:], degrees=True) for wp in waypoints]
        quats = np.array([r.as_quat() for r in rotations])
        key_rots = Rotation.from_quat(quats)
        slerp = Slerp(timestamps, key_rots)

        # Generate dense trajectory
        t_eval = self.generate_timestamps(float(timestamps[-1] if len(timestamps) else 0.0))
        trajectory: List[NDArray[np.floating]] = []

        for t in t_eval:
            pos = [spline(float(t)) for spline in pos_splines]
            rot = slerp(float(t))
            orient = rot.as_euler("xyz", degrees=True)
            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)

    def generate_quintic_spline(self, waypoints: List[List[float]], timestamps: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate quintic (5th order) spline with zero velocity and acceleration at endpoints

        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            timestamps: Time for each waypoint
        """
        # Quintic spline boundary conditions at the endpoints
        return self.generate_cubic_spline(waypoints, timestamps, velocity_start=[0, 0, 0], velocity_end=[0, 0, 0])

    def generate_scurve_spline(
        self,
        waypoints: List[List[float]],
        duration: Optional[float] = None,
        jerk_limit: float = 1000.0,
        timestamps: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Generate S-curve spline with jerk-limited motion profile

        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            duration: Total duration for the trajectory (optional)
            jerk_limit: Maximum jerk value in mm/s^3
            timestamps: Time for each waypoint (optional)

        Returns:
            Array of interpolated poses with S-curve velocity profile
        """
        basic_trajectory = self.generate_cubic_spline(waypoints, timestamps, velocity_start=[0, 0, 0], velocity_end=[0, 0, 0])

        if len(basic_trajectory) < 2:
            return basic_trajectory

        # Calculate total path length
        path_length = 0.0
        for i in range(1, len(basic_trajectory)):
            segment_length = np.linalg.norm(np.array(basic_trajectory[i][:3]) - np.array(basic_trajectory[i - 1][:3]))
            path_length += float(segment_length)

        if path_length < 0.001:
            return basic_trajectory

        # Determine duration if not provided
        if duration is None:
            avg_speed = 50.0  # mm/s conservative speed
            duration = path_length / avg_speed

        # Generate S-curve profile for the motion
        num_points = int(duration * self.control_rate)
        if num_points < 2:
            num_points = 2

        # Create S-curve time parameterization
        time_points = np.linspace(0, duration, num_points)
        s_curve_params: List[float] = []

        for t in time_points:
            tau = t / duration
            if tau <= 0.0:
                s = 0.0
            elif tau >= 1.0:
                s = 1.0
            else:
                s = tau * tau * (3.0 - 2.0 * tau)  # smoothstep
            s_curve_params.append(float(s))

        # Re-sample the trajectory according to S-curve profile
        new_indices = np.array(s_curve_params) * (len(basic_trajectory) - 1)

        resampled_trajectory: List[List[float]] = []
        for new_idx in new_indices:
            if new_idx <= 0:
                resampled_trajectory.append(basic_trajectory[0].tolist())
            elif new_idx >= len(basic_trajectory) - 1:
                resampled_trajectory.append(basic_trajectory[-1].tolist())
            else:
                lower_idx = int(new_idx)
                upper_idx = min(lower_idx + 1, len(basic_trajectory) - 1)
                alpha = float(new_idx - lower_idx)

                lower_point = np.array(basic_trajectory[lower_idx])
                upper_point = np.array(basic_trajectory[upper_idx])
                interpolated = lower_point + alpha * (upper_point - lower_point)
                resampled_trajectory.append(interpolated.tolist())

        return np.array(resampled_trajectory)
