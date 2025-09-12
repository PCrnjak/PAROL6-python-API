"""
Base trajectory generator.

Provides common timing utilities and constraints for derived generators.
"""

from typing import Union

import numpy as np

from .motion_constraints import MotionConstraints


class TrajectoryGenerator:
    """Base class for trajectory generation with caching support"""

    def __init__(self, control_rate: float = 100.0):
        """
        Initialize trajectory generator

        Args:
            control_rate: Control loop frequency in Hz (default 100Hz for PAROL6)
        """
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        self.trajectory_cache = {}
        self.constraints = MotionConstraints()  # Add constraints

    def generate_timestamps(self, duration: Union[float, np.floating]) -> np.ndarray:
        """Generate evenly spaced timestamps for trajectory"""
        num_points = int(duration * self.control_rate)
        return np.linspace(0, duration, num_points)
