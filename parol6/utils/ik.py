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

IKResult = namedtuple('IKResult', 'success q iterations residual tolerance_used violations')

def solve_ik_simple(
    robot: DHRobot,
    target_pose: SE3,
    current_q,
    ilimit: int = 100,
    tol: float = 1e-12,
    jogging: bool = False
):
    """
    Simplified IK solver using roboticstoolbox's built-in capabilities.
    
    Removes brittle heuristics:
    - No adaptive tolerance based on manipulability
    - No configuration-dependent workspace limits
    - No recovery mode detection
    - No complex subdivision logic
    
    Instead, relies on:
    - Proper joint limits defined in robot.qlim
    - Fixed, consistent damping
    - Library's built-in joint limit validation with smart wrapping
    
    Parameters
    ----------
    robot : DHRobot
        Robot model (with qlim properly set on each link)
    target_pose : SE3
        Target pose to reach
    current_q : array_like
        Current joint configuration in radians
    ilimit : int, optional
        Maximum iterations for IK solver (default: 100)
    tol : float, optional
        Convergence tolerance (default: 1e-6)
    jogging : bool, optional
        If True, use very strict tolerance for jogging (default: False)

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
    if success:
        q_unwrapped = unwrap_angles(q, current_q)
        
        # Verify unwrapped solution still within limits
        within_limits = PAROL6_ROBOT.check_limits(
            current_q, q_unwrapped, allow_recovery=True, log=True
        )
        
        if within_limits:
            q = q_unwrapped
        # else: use original result.q which already passed library's limit check
    
    violations = None if success else f"IK failed to solve."
    
    return IKResult(
        success=success,
        q=q if success else None,
        iterations=iterations,
        residual=remaining,
        tolerance_used=tol,
        violations=violations
    )

def quintic_scaling(s: float) -> float:
    """
    Calculates a smooth 0-to-1 scaling factor for progress 's'
    using a quintic polynomial, ensuring smooth start/end accelerations.
    """
    return 6 * (s**5) - 15 * (s**4) + 10 * (s**3)
