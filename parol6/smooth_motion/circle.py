"""
Circle trajectory generator.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from .base import TrajectoryGenerator


class CircularMotion(TrajectoryGenerator):
    """Generate circular and arc trajectories in 3D space"""

    def generate_arc_3d(
        self,
        start_pose: Sequence[float],
        end_pose: Sequence[float],
        center: Sequence[float] | NDArray,
        normal: Sequence[float] | NDArray | None = None,
        clockwise: bool = True,
        duration: float | np.floating = 2.0,
    ) -> np.ndarray:
        """
        Generate a 3D circular arc trajectory

        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz] (mm and degrees)
            end_pose: Ending pose [x, y, z, rx, ry, rz] (mm and degrees)
            center: Center point of arc [x, y, z] (mm)
            normal: Normal vector to arc plane (default: z-axis)
            clockwise: Direction of rotation
            duration: Time to complete arc (seconds)

        Returns:
            Array of poses along the arc trajectory
        """
        # Convert to numpy arrays
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])
        center_pt = np.array(center)

        # Arc geometry vectors
        r1 = start_pos - center_pt
        r2 = end_pos - center_pt

        # Arc plane normal computation
        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:  # Points are collinear
                normal = np.array([0, 0, 1])  # Default to XY plane
        normal_np: np.ndarray = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)

        # Arc sweep angle calculation
        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)

        # Arc direction validation
        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, normal_np) < 0:
            arc_angle = 2 * np.pi - arc_angle

        if clockwise:
            arc_angle = -arc_angle

        # Generate trajectory points
        timestamps = self.generate_timestamps(duration)
        trajectory = []

        for i, t in enumerate(timestamps):
            # Interpolation factor
            s = t / duration

            # Exact start position for trajectory begin
            if i == 0:
                current_pos = start_pos
            else:
                # Rotate radius vector
                angle = s * arc_angle
                rot_matrix = self._rotation_matrix_from_axis_angle(normal_np, angle)
                current_pos = center_pt + rot_matrix @ r1

            # Interpolate orientation (SLERP)
            current_orient = self._slerp_orientation(
                np.asarray(start_pose[3:], dtype=float),
                np.asarray(end_pose[3:], dtype=float),
                float(s),
            )

            # Combine position and orientation
            pose = np.concatenate([current_pos, current_orient])
            trajectory.append(pose)

        return np.array(trajectory)

    def generate_arc_with_profile(
        self,
        start_pose: Sequence[float],
        end_pose: Sequence[float],
        center: Sequence[float] | NDArray,
        normal: Sequence[float] | NDArray | None = None,
        clockwise: bool = True,
        duration: float | np.floating = 2.0,
        trajectory_type: str = "cubic",
        jerk_limit: float | None = None,
    ) -> np.ndarray:
        """
        Generate arc trajectory with specified velocity profile.

        This method generates arcs with direct velocity profiles instead of
        re-interpolating, ensuring smooth motion without jerking.

        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz]
            end_pose: Ending pose [x, y, z, rx, ry, rz]
            center: Arc center point [x, y, z]
            normal: Normal vector to arc plane
            clockwise: Direction of rotation
            duration: Time to complete arc (seconds)
            trajectory_type: 'cubic', 'quintic', or 's_curve'
            jerk_limit: Maximum jerk for s_curve (mm/s³)

        Returns:
            Array of poses with velocity profile applied
        """
        if trajectory_type == "cubic":
            # Use existing cubic implementation
            return self.generate_arc_3d(
                start_pose, end_pose, center, normal, clockwise, duration
            )

        # Convert to numpy arrays
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])
        center_pt = np.array(center)

        # Arc geometry
        r1 = start_pos - center_pt
        r2 = end_pos - center_pt

        # Arc plane normal
        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:
                normal = np.array([0, 0, 1])
        normal_np: np.ndarray = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)

        # Calculate arc angle
        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)

        # Determine arc direction
        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, normal_np) < 0:
            arc_angle = 2 * np.pi - arc_angle
        if clockwise:
            arc_angle = -arc_angle

        # Generate trajectory points with profile
        num_points = int(duration * self.control_rate)
        trajectory = []

        for i in range(num_points):
            # Normalized time [0, 1]
            t = i / (num_points - 1) if num_points > 1 else 0.0

            # Apply velocity profile
            if trajectory_type == "quintic":
                # Quintic polynomial for smooth acceleration
                s = 10 * t**3 - 15 * t**4 + 6 * t**5
            elif trajectory_type == "s_curve":
                # S-curve (smoothstep) for jerk-limited motion
                if t <= 0.0:
                    s = 0.0
                elif t >= 1.0:
                    s = 1.0
                else:
                    s = t * t * (3.0 - 2.0 * t)
            else:
                s = t  # Linear/cubic fallback

            # Current angle along arc
            angle = s * arc_angle

            # Rotation matrix for arc
            rot_matrix = self._rotation_matrix_from_axis_angle(normal_np, angle)
            current_pos = center_pt + rot_matrix @ r1

            # Interpolate orientation (SLERP)
            current_orient = self._slerp_orientation(
                np.asarray(start_pose[3:], dtype=float),
                np.asarray(end_pose[3:], dtype=float),
                float(s),
            )

            # Combine position and orientation
            pose = np.concatenate([current_pos, current_orient])
            trajectory.append(pose)

        return np.array(trajectory)

    def generate_circle_3d(
        self,
        center: Sequence[float] | NDArray,
        radius: float,
        normal: Sequence[float] | NDArray = [0, 0, 1],
        start_angle: float | None = None,
        duration: float | np.floating = 4.0,
        start_point: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Generate a complete circle trajectory that starts at start_point
        """
        timestamps = self.generate_timestamps(duration)
        trajectory = []

        # Circle coordinate system
        normal_np: np.ndarray = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal_np)
        v = np.cross(normal_np, u)

        center_np = np.array(center, dtype=float)

        # CRITICAL FIX: Validate and handle geometry
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            # Project start point onto the circle plane
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal_np) * normal_np

            # Get distance from center in the plane
            dist_in_plane = np.linalg.norm(to_start_plane)

            if dist_in_plane < 0.001:
                # Center start point - undefined angle
                print(
                    "    WARNING: Start point is at circle center, using default position"
                )
                start_angle = 0
                actual_start = center_np + radius * u
            else:
                # Start point angular position
                to_start_normalized = to_start_plane / dist_in_plane
                u_comp = np.dot(to_start_normalized, u)
                v_comp = np.dot(to_start_normalized, v)
                start_angle = np.arctan2(v_comp, u_comp)

                # Check if entry trajectory might be needed
                radius_error = abs(dist_in_plane - radius)
                if radius_error > radius * 0.05:  # More than 5% off
                    print(
                        f"    INFO: Starting {dist_in_plane:.1f}mm from center (radius: {radius}mm)"
                    )
                    if radius_error > radius * 0.3:  # More than 30% off
                        print(
                            "    WARNING: Large distance from circle - consider using entry trajectory"
                        )

                actual_start = start_pos
        else:
            start_angle = 0 if start_angle is None else start_angle
            actual_start = None

        # Generate the circle
        for i, t in enumerate(timestamps):
            if i == 0 and actual_start is not None:
                # First point MUST be exactly the start point
                pos = actual_start
            else:
                # Generate circle points
                angle = float(start_angle if start_angle is not None else 0.0) + (
                    2 * np.pi * t / duration
                )
                pos = center_np + radius * (np.cos(angle) * u + np.sin(angle) * v)

            # Placeholder orientation (will be overridden)
            orient = [0, 0, 0]
            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)

    def generate_circle_with_profile(
        self,
        center: Sequence[float] | NDArray,
        radius: float,
        normal: Sequence[float] | NDArray = [0, 0, 1],
        duration: float | np.floating = 4.0,
        trajectory_type: str = "cubic",
        jerk_limit: float | None = None,
        start_angle: float | None = None,
        start_point: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Generate circle with specified trajectory profile.

        Args:
            center: Center of circle [x, y, z]
            radius: Circle radius in mm
            normal: Normal vector to circle plane
            duration: Time to complete circle (seconds)
            trajectory_type: 'cubic', 'quintic', or 's_curve'
            jerk_limit: Maximum jerk for s_curve profile (mm/s^3)
            start_angle: Starting angle in radians
            start_point: Starting position [x, y, z, rx, ry, rz]

        Returns:
            Array of waypoints with shape (N, 6)
        """
        # Calculate adaptive control rate for small circles
        circumference = 2 * np.pi * radius
        min_arc_length = 2.0  # Minimum 2mm between points for smoothness
        min_points = int(circumference / min_arc_length)

        # Calculate control rate (100-200Hz range)
        base_rate = self.control_rate
        required_rate = min_points / duration
        adaptive_rate = float(min(200, max(base_rate, required_rate)))

        # Temporarily override control rate for small circles
        if radius < 30 and adaptive_rate > base_rate:
            original_rate = self.control_rate
            original_dt = self.dt
            self.control_rate = adaptive_rate
            self.dt = 1.0 / adaptive_rate
            # Use print for debug info since logger not imported here
            print(
                f"    [ADAPTIVE] Using {adaptive_rate:.0f}Hz control rate for {radius:.0f}mm radius circle"
            )
        else:
            original_rate = None
            original_dt = None

        try:
            if trajectory_type == "cubic":
                # Use existing implementation
                result = self.generate_circle_3d(
                    center, radius, normal, start_angle, duration, start_point
                )
            elif trajectory_type == "quintic":
                result = self.generate_quintic_circle(
                    center, radius, normal, duration, start_angle, start_point
                )
            elif trajectory_type == "s_curve":
                result = self.generate_scurve_circle(
                    center,
                    radius,
                    normal,
                    duration,
                    jerk_limit,
                    start_angle,
                    start_point,
                )
            else:
                raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        finally:
            # Restore original control rate if we changed it
            if original_rate is not None and original_dt is not None:
                self.control_rate = original_rate
                self.dt = original_dt

        return result

    def generate_quintic_circle(
        self,
        center: Sequence[float] | NDArray,
        radius: float,
        normal: Sequence[float] | NDArray = [0, 0, 1],
        duration: float | np.floating = 4.0,
        start_angle: float | None = None,
        start_point: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Generate circle trajectory using quintic polynomial velocity profile.
        Provides smooth acceleration and deceleration in Cartesian space.
        """
        # First generate uniform angular points
        num_points = int(duration * self.control_rate)

        # Setup coordinate system
        normal_np: np.ndarray = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal_np)
        v = np.cross(normal_np, u)
        center_np = np.array(center, dtype=float)

        # Handle start point if provided
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal_np) * normal_np
            dist_in_plane = np.linalg.norm(to_start_plane)

            if dist_in_plane < 0.001:
                start_angle = 0
            else:
                to_start_normalized = to_start_plane / dist_in_plane
                u_comp = np.dot(to_start_normalized, u)
                v_comp = np.dot(to_start_normalized, v)
                start_angle = np.arctan2(v_comp, u_comp)

                # Check if entry trajectory might be needed
                radius_error = abs(dist_in_plane - radius)
                if radius_error > radius * 0.05:  # More than 5% off
                    print(
                        f"    INFO: Starting {dist_in_plane:.1f}mm from center (radius: {radius}mm)"
                    )
                    if radius_error > radius * 0.2:  # More than 20% off
                        print(
                            "    WARNING: Large distance from circle - consider using entry trajectory"
                        )
        else:
            start_angle = 0 if start_angle is None else start_angle

        # Step 1: Generate uniformly spaced angular points
        angles = np.linspace(
            float(start_angle if start_angle is not None else 0.0),
            float((start_angle if start_angle is not None else 0.0) + 2 * np.pi),
            num_points,
        )
        uniform_positions = []

        for angle in angles:
            pos = center_np + radius * (np.cos(angle) * u + np.sin(angle) * v)
            uniform_positions.append(pos)

        # Step 2: Apply quintic velocity profile to Cartesian motion
        trajectory = []
        timestamps = np.linspace(0, duration, num_points)

        for i, t in enumerate(timestamps):
            tau = t / duration  # Normalized time [0, 1]

            # Quintic profile for smooth acceleration/deceleration
            # Applied to arc length parameterization, not angular
            s_normalized = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

            # Map to position index
            position_index = s_normalized * (num_points - 1)

            # Interpolate between positions
            if position_index <= 0:
                pos = uniform_positions[0]
            elif position_index >= num_points - 1:
                pos = uniform_positions[-1]
            else:
                lower_idx = int(position_index)
                upper_idx = min(lower_idx + 1, num_points - 1)
                alpha = position_index - lower_idx
                pos = uniform_positions[lower_idx] + alpha * (
                    uniform_positions[upper_idx] - uniform_positions[lower_idx]
                )

            # Placeholder orientation
            orient = [0, 0, 0]
            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)

    def generate_scurve_circle(
        self,
        center: Sequence[float] | NDArray,
        radius: float,
        normal: Sequence[float] | NDArray = [0, 0, 1],
        duration: float | np.floating = 4.0,
        jerk_limit: float | None = 5000.0,
        start_angle: float | None = None,
        start_point: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Generate circle trajectory using S-curve velocity profile.
        Provides jerk-limited motion in Cartesian space for maximum smoothness.
        """
        if jerk_limit is None:
            jerk_limit = 5000.0  # Default jerk limit in mm/s^3

        # Generate timestamps at control rate
        num_points = int(duration * self.control_rate)

        # Setup coordinate system
        normal_np: np.ndarray = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal_np)
        v = np.cross(normal_np, u)
        center_np = np.array(center, dtype=float)

        # Handle start point if provided
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal_np) * normal_np
            dist_in_plane = np.linalg.norm(to_start_plane)

            if dist_in_plane < 0.001:
                start_angle = 0
            else:
                to_start_normalized = to_start_plane / dist_in_plane
                u_comp = np.dot(to_start_normalized, u)
                v_comp = np.dot(to_start_normalized, v)
                start_angle = np.arctan2(v_comp, u_comp)

                # Check if entry trajectory might be needed
                radius_error = abs(dist_in_plane - radius)
                if radius_error > radius * 0.05:  # More than 5% off
                    print(
                        f"    INFO: Starting {dist_in_plane:.1f}mm from center (radius: {radius}mm)"
                    )
                    if radius_error > radius * 0.2:  # More than 20% off
                        print(
                            "    WARNING: Large distance from circle - consider using entry trajectory"
                        )
        else:
            start_angle = 0 if start_angle is None else start_angle

        # Step 1: Generate uniformly spaced angular points
        angles = np.linspace(
            float(start_angle if start_angle is not None else 0.0),
            float((start_angle if start_angle is not None else 0.0) + 2 * np.pi),
            num_points,
        )
        uniform_positions = []

        for angle in angles:
            pos = center_np + radius * (np.cos(angle) * u + np.sin(angle) * v)
            uniform_positions.append(pos)

        # Step 2: Apply S-curve velocity profile to Cartesian motion
        trajectory = []
        timestamps = np.linspace(0, duration, num_points)

        for i, t in enumerate(timestamps):
            tau = t / duration  # Normalized time [0, 1]

            # S-curve (smoothstep) profile for smooth acceleration
            # Applied to arc length, not angle
            if tau <= 0.0:
                s_normalized = 0.0
            elif tau >= 1.0:
                s_normalized = 1.0
            else:
                # Smoothstep: 3t² - 2t³ for smooth acceleration and deceleration
                s_normalized = tau * tau * (3.0 - 2.0 * tau)

            # Map to position index
            position_index = s_normalized * (num_points - 1)

            # Interpolate between positions
            if position_index <= 0:
                pos = uniform_positions[0]
            elif position_index >= num_points - 1:
                pos = uniform_positions[-1]
            else:
                lower_idx = int(position_index)
                upper_idx = min(lower_idx + 1, num_points - 1)
                alpha = position_index - lower_idx
                pos = uniform_positions[lower_idx] + alpha * (
                    uniform_positions[upper_idx] - uniform_positions[lower_idx]
                )

            # Placeholder orientation
            orient = [0, 0, 0]
            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)

    def _rotation_matrix_from_axis_angle(
        self, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """Generate rotation matrix using Rodrigues' formula"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Cross-product matrix
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        # Rodrigues' formula
        R = np.eye(3) + sin_a * K + (1 - cos_a) * K @ K
        return R

    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to the given vector"""
        v = np.array(v, dtype=float)  # Ensure it's a numpy array
        if abs(v[0]) < 0.9:
            cross = np.cross(v, [1, 0, 0])
            return cross / np.linalg.norm(cross)
        else:
            cross = np.cross(v, [0, 1, 0])
            return cross / np.linalg.norm(cross)

    def _slerp_orientation(
        self,
        start_orient: NDArray[np.floating],
        end_orient: NDArray[np.floating],
        t: float,
    ) -> np.ndarray:
        """Spherical linear interpolation for orientation"""
        # Convert to quaternions
        r1 = Rotation.from_euler("xyz", start_orient, degrees=True)
        r2 = Rotation.from_euler("xyz", end_orient, degrees=True)

        # Quaternion interpolation setup
        key_rots = Rotation.from_quat(np.stack([r1.as_quat(), r2.as_quat()]))
        slerp = Slerp(np.array([0.0, 1.0], dtype=float), key_rots)

        # Interpolate at a single time by passing a 1D array
        interp_rot = slerp(np.array([t], dtype=float))
        return interp_rot.as_euler("xyz", degrees=True)[0]
