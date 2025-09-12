"""
Advanced trajectory blending utilities (C2 continuity, minimum-jerk, etc.).
"""

from typing import Optional, Tuple

import numpy as np


class AdvancedMotionBlender:
    """
    Advanced trajectory blender with C2 continuity support.

    Provides multiple blending methods including quintic polynomials for true
    smooth motion with continuous position, velocity, and acceleration.
    """

    def __init__(self, sample_rate: float = 100.0):
        """
        Initialize advanced motion blender.

        Args:
            sample_rate: Trajectory sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate

        # Blend method registry
        self.blend_methods = {
            "quintic": self._blend_quintic,
            "s_curve": self._blend_scurve,
            "smoothstep": self._blend_smoothstep,  # Legacy compatibility
            "minimum_jerk": self._blend_minimum_jerk,
            "cubic": self._blend_cubic,
        }

    def extract_trajectory_state(self, trajectory: np.ndarray, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract position, velocity, and acceleration at a trajectory point.

        Args:
            trajectory: Trajectory array
            index: Point index (-1 for last point)

        Returns:
            Tuple of (position, velocity, acceleration)
        """
        if len(trajectory) < 3:
            pos = trajectory[index]
            vel = np.zeros_like(pos)
            acc = np.zeros_like(pos)
            return pos, vel, acc

        # Position
        pos = trajectory[index].copy()

        # Velocity
        if index == 0 or (index == -1 and len(trajectory) == 1):
            vel = (trajectory[1] - trajectory[0]) / self.dt
        elif index == -1 or index == len(trajectory) - 1:
            vel = (trajectory[-1] - trajectory[-2]) / self.dt
        else:
            vel = (trajectory[index + 1] - trajectory[index - 1]) / (2 * self.dt)

        # Acceleration
        if len(trajectory) < 3:
            acc = np.zeros_like(pos)
        elif index == 0:
            v1 = (trajectory[1] - trajectory[0]) / self.dt
            v2 = (trajectory[2] - trajectory[1]) / self.dt
            acc = (v2 - v1) / self.dt
        elif index == -1 or index == len(trajectory) - 1:
            v1 = (trajectory[-2] - trajectory[-3]) / self.dt
            v2 = (trajectory[-1] - trajectory[-2]) / self.dt
            acc = (v2 - v1) / self.dt
        else:
            acc = (trajectory[index + 1] - 2 * trajectory[index] + trajectory[index - 1]) / (self.dt**2)

        return pos, vel, acc

    def calculate_blend_region_size(self, traj1: np.ndarray, traj2: np.ndarray, max_accel: float = 1000.0) -> int:
        """
        Calculate optimal blend region size based on velocity mismatch.

        Args:
            traj1: First trajectory
            traj2: Second trajectory
            max_accel: Maximum allowed acceleration (mm/s² or deg/s²)

        Returns:
            Number of samples for blend region
        """
        _, v1, _ = self.extract_trajectory_state(traj1, -1)
        _, v2, _ = self.extract_trajectory_state(traj2, 0)

        delta_v = np.linalg.norm(v2[:3] - v1[:3])  # Focus on position components

        if delta_v < 1.0:
            return 20  # Minimal blend

        t_blend = delta_v / max_accel
        t_blend *= 1.5  # safety

        blend_samples = int(t_blend * self.sample_rate)
        blend_samples = max(20, min(blend_samples, 200))
        return blend_samples

    def solve_quintic_coefficients(
        self, p0: np.ndarray, pf: np.ndarray, v0: np.ndarray, vf: np.ndarray, a0: np.ndarray, af: np.ndarray, T: float
    ) -> np.ndarray:
        """
        Solve for quintic polynomial coefficients given boundary conditions.

        Returns:
            6xN array of coefficients [a0, a1, a2, a3, a4, a5] for each dimension
        """
        M = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # p(0)
                [1, T, T**2, T**3, T**4, T**5],  # p(T)
                [0, 1, 0, 0, 0, 0],  # v(0)
                [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],  # v(T)
                [0, 0, 2, 0, 0, 0],  # a(0)
                [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],  # a(T)
            ]
        )

        num_dims = len(p0)
        coeffs = np.zeros((6, num_dims))

        for i in range(num_dims):
            b = np.array([p0[i], pf[i], v0[i], vf[i], a0[i], af[i]])
            coeffs[:, i] = np.linalg.solve(M, b)

        return coeffs

    def evaluate_quintic(self, coeffs: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate quintic polynomial at time t.

        Returns position, velocity, and acceleration.
        """
        pos = coeffs[0] + coeffs[1] * t + coeffs[2] * t**2 + coeffs[3] * t**3 + coeffs[4] * t**4 + coeffs[5] * t**5
        vel = coeffs[1] + 2 * coeffs[2] * t + 3 * coeffs[3] * t**2 + 4 * coeffs[4] * t**3 + 5 * coeffs[5] * t**4
        acc = 2 * coeffs[2] + 6 * coeffs[3] * t + 12 * coeffs[4] * t**2 + 20 * coeffs[5] * t**3
        return pos, vel, acc

    def _blend_quintic(self, traj1: np.ndarray, traj2: np.ndarray, blend_samples: int) -> np.ndarray:
        """
        Generate quintic polynomial blend with C2 continuity.

        This ensures smooth position, velocity, and acceleration transitions.
        """
        p0, v0, a0 = self.extract_trajectory_state(traj1, -1)
        pf, vf, af = self.extract_trajectory_state(traj2, 0)

        T = blend_samples * self.dt
        coeffs = self.solve_quintic_coefficients(p0, pf, v0, vf, a0, af, T)

        blend_traj = []
        for i in range(blend_samples):
            t = i * T / (blend_samples - 1) if blend_samples > 1 else 0.0
            pos, _, _ = self.evaluate_quintic(coeffs, t)
            blend_traj.append(pos)

        return np.array(blend_traj)

    def _blend_scurve(self, traj1: np.ndarray, traj2: np.ndarray, blend_samples: int) -> np.ndarray:
        """
        Generate S-curve blend with jerk limiting.
        """
        p0, v0, _ = self.extract_trajectory_state(traj1, -1)
        pf, vf, _ = self.extract_trajectory_state(traj2, 0)

        try:
            from .scurve import MultiAxisSCurveTrajectory

            scurve = MultiAxisSCurveTrajectory(
                p0[:6],  # assume first 6 are XYZRPY
                pf[:6],
                v0=v0[:6],
                vf=vf[:6],
                T=blend_samples * self.dt,
                jerk_limit=5000,
            )
            points = scurve.get_trajectory_points(self.dt)

            blend_traj = []
            for i in range(len(points["position"])):
                pose = np.zeros_like(p0)
                pose[:6] = points["position"][i]
                if len(p0) > 6:
                    alpha = i / (len(points["position"]) - 1) if len(points["position"]) > 1 else 1.0
                    pose[6:] = p0[6:] * (1 - alpha) + pf[6:] * alpha
                blend_traj.append(pose)

            return np.array(blend_traj)
        except Exception:
            return self._blend_quintic(traj1, traj2, blend_samples)

    def _blend_smoothstep(self, traj1: np.ndarray, traj2: np.ndarray, blend_samples: int) -> np.ndarray:
        """
        Legacy smoothstep blend for backward compatibility.
        """
        p0 = traj1[-1] if len(traj1) > 0 else traj2[0]
        pf = traj2[0] if len(traj2) > 0 else traj1[-1]

        blend_traj = []
        for i in range(blend_samples):
            t = i / (blend_samples - 1) if blend_samples > 1 else 0.0
            s = t * t * (3 - 2 * t)
            pos = p0 * (1 - s) + pf * s
            blend_traj.append(pos)

        return np.array(blend_traj)

    def _blend_minimum_jerk(self, traj1: np.ndarray, traj2: np.ndarray, blend_samples: int) -> np.ndarray:
        """
        Minimum jerk trajectory blend.

        Uses the minimum jerk trajectory equation for smooth motion.
        """
        p0, _, _ = self.extract_trajectory_state(traj1, -1)
        pf, _, _ = self.extract_trajectory_state(traj2, 0)

        blend_traj = []
        for i in range(blend_samples):
            tau = i / (blend_samples - 1) if blend_samples > 1 else 0.0
            pos = p0 + (pf - p0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            blend_traj.append(pos)

        return np.array(blend_traj)

    def _blend_cubic(self, traj1: np.ndarray, traj2: np.ndarray, blend_samples: int) -> np.ndarray:
        """
        Cubic spline blend with C1 continuity.
        """
        p0, v0, _ = self.extract_trajectory_state(traj1, -1)
        pf, vf, _ = self.extract_trajectory_state(traj2, 0)

        T = blend_samples * self.dt
        blend_traj = []

        for i in range(blend_samples):
            t = i * T / (blend_samples - 1) if blend_samples > 1 else 0.0
            tau = t / T if T > 0 else 0.0

            h00 = 2 * tau**3 - 3 * tau**2 + 1
            h10 = tau**3 - 2 * tau**2 + tau
            h01 = -2 * tau**3 + 3 * tau**2
            h11 = tau**3 - tau**2

            pos = h00 * p0 + h10 * T * v0 + h01 * pf + h11 * T * vf
            blend_traj.append(pos)

        return np.array(blend_traj)

    def blend_trajectories(
        self,
        traj1: np.ndarray,
        traj2: np.ndarray,
        method: str = "quintic",
        blend_samples: Optional[int] = None,
        auto_size: bool = True,
    ) -> np.ndarray:
        """
        Blend two trajectory segments with specified method.

        Args:
            traj1: First trajectory segment
            traj2: Second trajectory segment
            method: Blend method ('quintic', 's_curve', 'smoothstep', 'minimum_jerk', 'cubic')
            blend_samples: Number of blend samples (auto-calculated if None)
            auto_size: Automatically calculate blend region size

        Returns:
            Combined trajectory with smooth blend
        """
        if len(traj1) == 0 and len(traj2) == 0:
            return np.array([])
        if len(traj1) == 0:
            return traj2
        if len(traj2) == 0:
            return traj1

        if blend_samples is None or auto_size:
            blend_samples = self.calculate_blend_region_size(traj1, traj2)

        blend_samples = max(4, blend_samples)

        if method not in self.blend_methods:
            print(f"[WARNING] Unknown blend method '{method}', using quintic")
            method = "quintic"

        blend_func = self.blend_methods[method]
        blend_traj = blend_func(traj1, traj2, blend_samples)

        overlap_start = max(0, len(traj1) - 1)
        overlap_end = min(1, len(traj2))

        result = np.vstack([traj1[:overlap_start], blend_traj, traj2[overlap_end:]])
        return result
