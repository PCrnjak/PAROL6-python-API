"""
Motion constraints for PAROL6 robot.

Defines per-joint limits for velocity, acceleration, and jerk
based on gear ratios and mechanical properties.
"""

from typing import Dict, Optional, List

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
        self.gear_ratios: List[float] = PAROL6_ROBOT.Joint_reduction_ratio
        
        # Use jerk limits from PAROL6_ROBOT.py
        self.max_jerk: List[float] = PAROL6_ROBOT.Joint_max_jerk
        
        # Use maximum velocities from PAROL6_ROBOT.py
        self.max_velocity: List[float] = PAROL6_ROBOT.Joint_max_speed
        
        # Use max acceleration from PAROL6_ROBOT.py (convert from RAD/S² to STEP/S²)
        # Calculate max accelerations for each joint
        self.max_acceleration: List[float] = []
        for i in range(len(self.gear_ratios)):
            # Convert from RAD/S² to STEP/S² using gear ratio
            acc_rad_s2 = PAROL6_ROBOT.Joint_max_acc
            acc_step_s2 = PAROL6_ROBOT.SPEED_RAD2STEP(acc_rad_s2, i)
            self.max_acceleration.append(acc_step_s2)
        
    def get_joint_constraints(self, joint_idx: int) -> Optional[Dict[str, float]]:
        """Get constraints for specific joint."""
        if joint_idx >= len(self.gear_ratios):
            return None
            
        return {
            'gear_ratio': self.gear_ratios[joint_idx],
            'v_max': self.max_velocity[joint_idx],
            'a_max': self.max_acceleration[joint_idx],
            'j_max': self.max_jerk[joint_idx]
        }
    
    def scale_for_gear_ratio(self, joint_idx: int, base_limits: Dict[str, float]) -> Dict[str, float]:
        """Scale motion limits based on gear ratio."""
        ratio = self.gear_ratios[joint_idx]
        
        # Higher gear ratio = lower speed but higher precision
        scaled = {
            'v_max': base_limits['v_max'] / ratio,
            'a_max': base_limits['a_max'] / ratio,
            'j_max': base_limits['j_max'] / ratio
        }
        
        return scaled
    
    def validate_trajectory(self, trajectory: np.ndarray, 
                           joint_idx: int, dt: float = 0.01) -> Dict[str, float | bool]:
        """
        Validate that trajectory respects constraints.
        
        Returns:
            Dictionary with validation results
        """
        constraints = self.get_joint_constraints(joint_idx)
        if constraints is None or len(trajectory) < 3:
            return {
                'velocity_ok': True,
                'acceleration_ok': True,
                'jerk_ok': True,
                'max_velocity': 0.0,
                'max_acceleration': 0.0,
                'max_jerk': 0.0
            }
        
        # Calculate derivatives numerically
        velocities = np.diff(trajectory) / dt
        accelerations = np.diff(velocities) / dt
        jerks = np.diff(accelerations) / dt
        
        validation = {
            'velocity_ok': np.all(np.abs(velocities) <= constraints['v_max']),
            'acceleration_ok': np.all(np.abs(accelerations) <= constraints['a_max']),
            'jerk_ok': np.all(np.abs(jerks) <= constraints['j_max']),
            'max_velocity': float(np.max(np.abs(velocities))) if velocities.size else 0.0,
            'max_acceleration': float(np.max(np.abs(accelerations))) if accelerations.size else 0.0,
            'max_jerk': float(np.max(np.abs(jerks))) if jerks.size else 0.0
        }
        
        return validation
