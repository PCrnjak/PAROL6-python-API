"""
Motion constraints for PAROL6 robot.

Defines per-joint limits for velocity, acceleration, and jerk
based on gear ratios and mechanical properties.
"""

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT


class MotionConstraints:
    """
    Motion constraints for PAROL6 robot.

    Defines per-joint limits for velocity, acceleration, and jerk
    based on gear ratios and mechanical properties.
    """

    def __init__(self):
        """Initialize with PAROL6-specific constraints."""
        # Use gear ratios from PAROL6_ROBOT.py
        self.gear_ratios = [float(x) for x in PAROL6_ROBOT.joint.ratio.tolist()]

        # Use jerk limits from PAROL6_ROBOT.py
        self.max_jerk = [float(x) for x in PAROL6_ROBOT.joint.jerk.max.tolist()]

        # Use maximum velocities from PAROL6_ROBOT.py
        self.max_velocity = [float(x) for x in PAROL6_ROBOT.joint.speed.max.tolist()]

        # Use max acceleration from PAROL6_ROBOT.py (convert from RAD/S² to STEP/S²)
        # steps/s² = (rad/s²) * (steps/rad) , and steps/rad = ratio / radian_per_step
        max_acc_rad = float(PAROL6_ROBOT.joint.acc.max_rad)
        steps_per_rad_base = 1.0 / float(PAROL6_ROBOT.conv.radian_per_step)
        self.max_acceleration = [
            max_acc_rad * steps_per_rad_base * float(ratio) for ratio in PAROL6_ROBOT.joint.ratio
        ]

    def get_joint_constraints(self, joint_idx: int) -> dict[str, float] | None:
        """Get constraints for specific joint."""
        if joint_idx >= len(self.gear_ratios):
            return None

        return {
            "gear_ratio": self.gear_ratios[joint_idx],
            "v_max": self.max_velocity[joint_idx],
            "a_max": self.max_acceleration[joint_idx],
            "j_max": self.max_jerk[joint_idx],
        }

    def scale_for_gear_ratio(
        self, joint_idx: int, base_limits: dict[str, float]
    ) -> dict[str, float]:
        """Scale motion limits based on gear ratio."""
        ratio = self.gear_ratios[joint_idx]

        # Higher gear ratio = lower speed but higher precision
        scaled = {
            "v_max": base_limits["v_max"] / ratio,
            "a_max": base_limits["a_max"] / ratio,
            "j_max": base_limits["j_max"] / ratio,
        }

        return scaled

    def validate_trajectory(
        self, trajectory: np.ndarray, joint_idx: int, dt: float = 0.01
    ) -> dict[str, float | bool]:
        """
        Validate that trajectory respects constraints.

        Returns:
            Dictionary with validation results
        """
        constraints = self.get_joint_constraints(joint_idx)
        if constraints is None or len(trajectory) < 3:
            return {
                "velocity_ok": True,
                "acceleration_ok": True,
                "jerk_ok": True,
                "max_velocity": 0.0,
                "max_acceleration": 0.0,
                "max_jerk": 0.0,
            }

        # Calculate derivatives numerically
        velocities = np.diff(trajectory) / dt
        accelerations = np.diff(velocities) / dt
        jerks = np.diff(accelerations) / dt

        validation: dict[str, float | bool] = {
            "velocity_ok": bool(np.all(np.abs(velocities) <= constraints["v_max"])),
            "acceleration_ok": bool(np.all(np.abs(accelerations) <= constraints["a_max"])),
            "jerk_ok": bool(np.all(np.abs(jerks) <= constraints["j_max"])),
            "max_velocity": float(np.max(np.abs(velocities))) if velocities.size else 0.0,
            "max_acceleration": float(np.max(np.abs(accelerations))) if accelerations.size else 0.0,
            "max_jerk": float(np.max(np.abs(jerks))) if jerks.size else 0.0,
        }

        return validation
