"""
IK Helper Functions and Utilities
Shared functions used by multiple command classes for inverse kinematics calculations.
"""

import numpy as np
import logging
from collections import namedtuple
from roboticstoolbox import DHRobot
from spatialmath import SE3
import parol6.PAROL6_ROBOT as PAROL6_ROBOT

logger = logging.getLogger(__name__)

# This dictionary maps descriptive axis names to movement vectors
# Format: ([x, y, z], [rx, ry, rz])
AXIS_MAP = {
    'X+': ([1, 0, 0], [0, 0, 0]), 'X-': ([-1, 0, 0], [0, 0, 0]),
    'Y+': ([0, 1, 0], [0, 0, 0]), 'Y-': ([0, -1, 0], [0, 0, 0]),
    'Z+': ([0, 0, 1], [0, 0, 0]), 'Z-': ([0, 0, -1], [0, 0, 0]),
    'RX+': ([0, 0, 0], [1, 0, 0]), 'RX-': ([0, 0, 0], [-1, 0, 0]),
    'RY+': ([0, 0, 0], [0, 1, 0]), 'RY-': ([0, 0, 0], [0, -1, 0]),
    'RZ+': ([0, 0, 0], [0, 0, 1]), 'RZ-': ([0, 0, 0], [0, 0, -1]),
}

def unwrap_angles(q_solution, q_current):
    """
    Vectorized unwrap: bring solution angles near current by adding/subtracting 2*pi.
    This minimizes joint motion between consecutive configurations.
    """
    qs = np.asarray(q_solution, dtype=float)
    qc = np.asarray(q_current, dtype=float)
    diff = qs - qc
    q_unwrapped = qs.copy()
    q_unwrapped[diff > np.pi] -= 2 * np.pi
    q_unwrapped[diff < -np.pi] += 2 * np.pi
    return q_unwrapped

IKResult = namedtuple('IKResult', 'success q iterations residual violations')

def solve_ik(
    robot: DHRobot,
    target_pose: SE3,
    current_q,
    jogging: bool = False,
    safety_margin_rad: float = 0.03
):
    """
    IK solver

    Parameters
    ----------
    robot : DHRobot
        Robot model
    target_pose : SE3
        Target pose to reach
    current_q : array_like
        Current joint configuration in radians
    jogging : bool, optional
        If True, use very strict tolerance for jogging (default: False)
    safety_margin_rad : float, optional
        Buffer distance (radians) from joint limits (default: 0.03)

    Returns
    -------
    IKResult
        success - True if solution found
        q - Joint configuration in radians (or None if failed)
        iterations - Number of iterations used
        residual - Final error value
        tolerance_used - Tolerance used for convergence
        violations - Error message if failed, None if successful
    """
    result = robot.ets().ik_LM(
        target_pose,
        q0=current_q,
        tol=1e-10,
        joint_limits=True,
        k=0.0,
        method="sugihara"
    )
    q = result[0]
    success = result[1] > 0
    iterations = result[2]
    remaining = result[3]
    
    violations = None
    
    if success and jogging:
        # Vectorized safety validation with recovery support
        qlim = robot.qlim
        buffered_min = qlim[0, :] + safety_margin_rad
        buffered_max = qlim[1, :] - safety_margin_rad
        
        # Check which joints were in danger zone (beyond buffer)
        in_danger_old = (current_q < buffered_min) | (current_q > buffered_max)
        
        # Compute distance from nearest limit for all joints
        dist_old_lower = np.abs(current_q - qlim[0, :])
        dist_old_upper = np.abs(current_q - qlim[1, :])
        dist_old = np.minimum(dist_old_lower, dist_old_upper)
        
        dist_new_lower = np.abs(q - qlim[0, :])
        dist_new_upper = np.abs(q - qlim[1, :])
        dist_new = np.minimum(dist_new_lower, dist_new_upper)
        
        # Check for recovery violations (was in danger, moved closer to limit)
        recovery_violations = in_danger_old & (dist_new < dist_old)
        
        # Check for safety violations (was safe, left buffer zone)
        in_danger_new = (q < buffered_min) | (q > buffered_max)
        safety_violations = (~in_danger_old) & in_danger_new
        
        # Report first violation found
        if np.any(recovery_violations):
            idx = np.argmax(recovery_violations)
            success = False
            violations = f"J{idx+1} moving further into danger zone (recovery blocked)"
            logger.warning(violations)
        elif np.any(safety_violations):
            idx = np.argmax(safety_violations)
            success = False
            violations = f"J{idx+1} would leave safe zone (buffer violated)"
            logger.warning(violations)
        
    if success:
        # Valid solution - apply unwrapping to minimize joint motion
        q_unwrapped = unwrap_angles(q, current_q)
        
        # Verify unwrapped solution still within actual limits
        within_limits = PAROL6_ROBOT.check_limits(
            current_q, q_unwrapped, allow_recovery=True, log=True
        )
        
        if within_limits:
            q = q_unwrapped
        # else: use original result.q which already passed library's limit check
    else:
        violations = f"IK failed to solve."
    return IKResult(
        success=success,
        q=q if success else None,
        iterations=iterations,
        residual=remaining,
        violations=violations
    )

def quintic_scaling(s: float) -> float:
    """
    Calculates a smooth 0-to-1 scaling factor for progress 's'
    using a quintic polynomial, ensuring smooth start/end accelerations.
    """
    return 6 * (s**5) - 15 * (s**4) + 10 * (s**3)
