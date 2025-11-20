"""
Motion blending utilities.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class MotionBlender:
    """Blend between different motion segments for smooth transitions"""

    def __init__(self, blend_time: float = 0.5):
        self.blend_time = blend_time

    def blend_trajectories(
        self, traj1: np.ndarray, traj2: np.ndarray, blend_samples: int = 50
    ) -> np.ndarray:
        """Blend two trajectory segments with improved velocity continuity"""

        if blend_samples < 4:
            return np.vstack([traj1, traj2])

        # Use more samples for smoother blending
        blend_samples = max(blend_samples, 20)  # Minimum 20 samples for smooth blend

        # Trajectory overlap region analysis
        overlap_start = max(0, len(traj1) - blend_samples // 3)
        overlap_end = min(len(traj2), blend_samples // 3)

        # Extract blend region
        blend_start_pose = (
            traj1[overlap_start] if overlap_start < len(traj1) else traj1[-1]
        )
        blend_end_pose = traj2[overlap_end] if overlap_end < len(traj2) else traj2[0]

        # Generate smooth transition using S-curve
        blended: list[np.ndarray] = []
        for i in range(blend_samples):
            t = i / (blend_samples - 1)
            # Use smoothstep function for smoother acceleration
            s = t * t * (3 - 2 * t)  # Smoothstep

            # Blend position
            pos_blend = blend_start_pose * (1 - s) + blend_end_pose * s

            # Orientation interpolation via SLERP (pass array-like time)
            r1 = Rotation.from_euler("xyz", blend_start_pose[3:], degrees=True)
            r2 = Rotation.from_euler("xyz", blend_end_pose[3:], degrees=True)
            key_rots = Rotation.from_quat(np.stack([r1.as_quat(), r2.as_quat()]))
            slerp = Slerp(np.array([0.0, 1.0], dtype=float), key_rots)
            orient_blend = slerp(np.array([float(s)], dtype=float)).as_euler(
                "xyz", degrees=True
            )[0]

            pos_blend[3:] = orient_blend
            blended.append(pos_blend)

        # Combine with better overlap handling
        result = np.vstack(
            [traj1[:overlap_start], np.array(blended), traj2[overlap_end:]]
        )

        return result
