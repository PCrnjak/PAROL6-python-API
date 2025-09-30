"""
Helix trajectory generator.
"""

from typing import Sequence, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .base import TrajectoryGenerator


class HelixMotion(TrajectoryGenerator):
    """Generate helical trajectories with various velocity profiles"""

    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to the given vector"""
        v = np.array(v, dtype=float)  # Ensure it's a numpy array
        if abs(v[0]) < 0.9:
            return np.cross(v, [1, 0, 0]) / np.linalg.norm(np.cross(v, [1, 0, 0]))
        else:
            return np.cross(v, [0, 1, 0]) / np.linalg.norm(np.cross(v, [0, 1, 0]))

    def generate_helix_with_profile(
        self,
        center: Union[Sequence[float], NDArray],
        radius: float,
        pitch: float,
        height: float,
        axis: Union[Sequence[float], NDArray] = [0, 0, 1],
        duration: Union[float, np.floating] = 4.0,
        trajectory_type: str = "cubic",
        jerk_limit: Optional[float] = None,
        start_point: Optional[Sequence[float]] = None,
        clockwise: bool = False,
    ) -> np.ndarray:
        """
        Generate helix with specified trajectory profile.

        Args:
            center: Helix center point [x, y, z]
            radius: Helix radius in mm
            pitch: Vertical distance per revolution in mm
            height: Total height of helix in mm
            axis: Helix axis vector (normalized internally)
            duration: Time to complete helix in seconds
            trajectory_type: 'cubic', 'quintic', or 's_curve'
            jerk_limit: Maximum jerk for s_curve profile
            start_point: Starting position [x, y, z, rx, ry, rz]
            clockwise: Rotation direction

        Returns:
            Array of waypoints with shape (N, 6) for [x, y, z, rx, ry, rz]
        """
        if trajectory_type == "cubic":
            return self.generate_cubic_helix(
                center, radius, pitch, height, axis, duration, start_point, clockwise
            )
        if trajectory_type == "quintic":
            return self.generate_quintic_helix(
                center, radius, pitch, height, axis, duration, start_point, clockwise
            )
        if trajectory_type == "s_curve":
            return self.generate_scurve_helix(
                center, radius, pitch, height, axis, duration, jerk_limit, start_point, clockwise
            )

        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    def generate_cubic_helix(
        self,
        center: Union[Sequence[float], NDArray],
        radius: float,
        pitch: float,
        height: float,
        axis: Union[Sequence[float], NDArray] = [0, 0, 1],
        duration: Union[float, np.floating] = 4.0,
        start_point: Optional[Sequence[float]] = None,
        clockwise: bool = False,
    ) -> np.ndarray:
        """
        Generate helix with cubic (linear) interpolation.
        This is the simplest profile with constant angular velocity.
        """
        # Calculate number of revolutions
        num_revolutions = height / pitch if pitch > 0 else 1
        total_angle = 2 * np.pi * num_revolutions

        # Setup coordinate system
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        u = self._get_perpendicular_vector(axis)
        v = np.cross(axis, u)
        center_np = np.array(center, dtype=float)

        # Determine starting angle if start_point provided
        start_angle = 0.0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, axis) * axis

            if np.linalg.norm(to_start_plane) > 0.001:
                to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
                start_angle = np.arctan2(np.dot(to_start_normalized, v), np.dot(to_start_normalized, u))

        # Generate trajectory points
        num_points = int(duration * self.control_rate)
        timestamps = np.linspace(0, duration, num_points)
        trajectory = []

        for t in timestamps:
            # Linear interpolation for simple cubic profile
            progress = t / duration

            # Angular position (constant velocity)
            if clockwise:
                theta = start_angle - total_angle * progress
            else:
                theta = start_angle + total_angle * progress

            # Vertical position (constant velocity)
            z_offset = height * progress

            # Calculate 3D position
            pos = center_np + radius * (np.cos(theta) * u + np.sin(theta) * v) + z_offset * axis

            # Placeholder orientation (could be enhanced)
            orient = [0, 0, 0] if start_point is None else start_point[3:6]

            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)

    def generate_quintic_helix(
        self,
        center: Union[Sequence[float], NDArray],
        radius: float,
        pitch: float,
        height: float,
        axis: Union[Sequence[float], NDArray] = [0, 0, 1],
        duration: Union[float, np.floating] = 4.0,
        start_point: Optional[Sequence[float]] = None,
        clockwise: bool = False,
    ) -> np.ndarray:
        """
        Generate helix with quintic polynomial profile.
        Provides smooth acceleration and deceleration.
        """
        # Calculate total angle
        num_revolutions = height / pitch if pitch > 0 else 1
        total_angle = 2 * np.pi * num_revolutions

        # Setup coordinate system
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        u = self._get_perpendicular_vector(axis)
        v = np.cross(axis, u)
        center_np = np.array(center, dtype=float)

        # Determine starting angle
        start_angle = 0.0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, axis) * axis

            if np.linalg.norm(to_start_plane) > 0.001:
                to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
                start_angle = np.arctan2(np.dot(to_start_normalized, v), np.dot(to_start_normalized, u))

        # Generate trajectory with quintic profile
        num_points = int(duration * self.control_rate)
        timestamps = np.linspace(0, duration, num_points)
        trajectory = []

        for t in timestamps:
            # Quintic polynomial: 10τ³ - 15τ⁴ + 6τ⁵
            tau = t / duration
            progress = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

            # Apply to both angular and vertical motion
            if clockwise:
                theta = start_angle - total_angle * progress
            else:
                theta = start_angle + total_angle * progress

            z_offset = height * progress

            # Calculate position
            pos = center_np + radius * (np.cos(theta) * u + np.sin(theta) * v) + z_offset * axis

            # Orientation
            orient = [0, 0, 0] if start_point is None else start_point[3:6]

            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)

    def generate_scurve_helix(
        self,
        center: Union[Sequence[float], NDArray],
        radius: float,
        pitch: float,
        height: float,
        axis: Union[Sequence[float], NDArray] = [0, 0, 1],
        duration: Union[float, np.floating] = 4.0,
        jerk_limit: Optional[float] = None,
        start_point: Optional[Sequence[float]] = None,
        clockwise: bool = False,
    ) -> np.ndarray:
        """
        Generate helix with S-curve (smoothstep) profile.
        Provides jerk-limited motion.
        """
        # Calculate total angle
        num_revolutions = height / pitch if pitch > 0 else 1
        total_angle = 2 * np.pi * num_revolutions

        # Setup coordinate system
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        u = self._get_perpendicular_vector(axis)
        v = np.cross(axis, u)
        center_np = np.array(center, dtype=float)

        # Determine starting angle
        start_angle = 0.0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, axis) * axis

            if np.linalg.norm(to_start_plane) > 0.001:
                to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
                start_angle = np.arctan2(np.dot(to_start_normalized, v), np.dot(to_start_normalized, u))

        # Generate trajectory with S-curve profile
        num_points = int(duration * self.control_rate)
        timestamps = np.linspace(0, duration, num_points)
        trajectory = []

        for t in timestamps:
            # S-curve (smoothstep): 3τ² - 2τ³
            tau = t / duration
            if tau <= 0.0:
                progress = 0.0
            elif tau >= 1.0:
                progress = 1.0
            else:
                progress = tau * tau * (3.0 - 2.0 * tau)

            # Apply to motion
            if clockwise:
                theta = start_angle - total_angle * progress
            else:
                theta = start_angle + total_angle * progress

            z_offset = height * progress

            # Calculate position
            pos = center_np + radius * (np.cos(theta) * u + np.sin(theta) * v) + z_offset * axis

            # Orientation
            orient = [0, 0, 0] if start_point is None else start_point[3:6]

            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)
