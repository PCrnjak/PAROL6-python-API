"""
IK Helper Functions and Utilities
Shared functions used by multiple command classes for inverse kinematics calculations.
"""

import numpy as np
import logging
from collections import namedtuple
from roboticstoolbox import DHRobot
from spatialmath import SE3
from spatialmath.base import trinterp
import PAROL6_ROBOT

logger = logging.getLogger(__name__)

# Global variable to track previous tolerance for logging changes
_prev_tolerance = None

# --- Wrapper class to make integers mutable when passed to functions ---
class CommandValue:
    def __init__(self, value):
        self.value = value

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

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range to handle angle wrapping"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def unwrap_angles(q_solution, q_current):
    """
    Unwrap angles in the solution to be closest to current position.
    This handles the angle wrapping issue where -179° and 181° are close but appear far.
    """
    q_unwrapped = q_solution.copy()
    
    for i in range(len(q_solution)):
        # Angle difference for unwrapping
        diff = q_solution[i] - q_current[i]
        
        # Unwrap angles beyond pi boundary
        if diff > np.pi:
            # Solution is too far in positive direction, subtract 2*pi
            q_unwrapped[i] = q_solution[i] - 2 * np.pi
        elif diff < -np.pi:
            # Solution is too far in negative direction, add 2*pi
            q_unwrapped[i] = q_solution[i] + 2 * np.pi
    
    return q_unwrapped

IKResult = namedtuple('IKResult', 'success q iterations residual tolerance_used violations')

def calculate_adaptive_tolerance(robot, q, strict_tol=1e-10, loose_tol=1e-7):
    """
    Calculate adaptive tolerance based on proximity to singularities.
    Near singularities: looser tolerance for easier convergence.
    Away from singularities: stricter tolerance for precise solutions.
    
    Parameters
    ----------
    robot : DHRobot
        Robot model
    q : array_like
        Joint configuration
    strict_tol : float
        Strict tolerance away from singularities (default: 1e-10)
    loose_tol : float
        Loose tolerance near singularities (1e-7)
        
    Returns
    -------
    float
        Adaptive tolerance value
    """
    global _prev_tolerance
    
    q_array = np.array(q, dtype=float)
    
    # Manipulability for singularity detection
    manip = robot.manipulability(q_array)
    singularity_threshold = 0.001
    
    sing_normalized = np.clip(manip / singularity_threshold, 0.0, 1.0)
    adaptive_tol = loose_tol + (strict_tol - loose_tol) * sing_normalized
    
    # Log tolerance changes (only log significant changes to avoid spam)
    if _prev_tolerance is None or abs(adaptive_tol - _prev_tolerance) / _prev_tolerance > 0.5:
        tol_category = "LOOSE" if adaptive_tol > 1e-7 else "MODERATE" if adaptive_tol > 5e-10 else "STRICT"
        logger.debug(f"Adaptive IK tolerance: {adaptive_tol:.2e} ({tol_category}) - Manipulability: {manip:.8f} (threshold: {singularity_threshold})")
        _prev_tolerance = adaptive_tol
    
    return adaptive_tol

def calculate_configuration_dependent_max_reach(q_seed):
    """
    Calculate maximum reach based on joint configuration, particularly joint 5.
    When joint 5 is at 90 degrees, the effective reach is reduced by approximately 0.045.
    
    Parameters
    ----------
    q_seed : array_like
        Current joint configuration in radians
        
    Returns
    -------
    float
        Configuration-dependent maximum reach threshold
    """
    base_max_reach = 0.44  # Base maximum reach from experimentation
    
    j5_angle = q_seed[4] if len(q_seed) > 4 else 0.0
    j5_normalized = normalize_angle(j5_angle)
    angle_90_deg = np.pi / 2
    angle_neg_90_deg = -np.pi / 2
    dist_from_90 = abs(j5_normalized - angle_90_deg)
    dist_from_neg_90 = abs(j5_normalized - angle_neg_90_deg)
    dist_from_90_deg = min(dist_from_90, dist_from_neg_90)
    reduction_range = np.pi / 4  # 45 degrees
    if dist_from_90_deg <= reduction_range:
        # Reach reduction near J5 90° positions
        proximity_factor = 1.0 - (dist_from_90_deg / reduction_range)
        reach_reduction = 0.045 * proximity_factor
        effective_max_reach = base_max_reach - reach_reduction
        
        return effective_max_reach
    else:
        return base_max_reach

def solve_ik_with_adaptive_tol_subdivision(
        robot: DHRobot,
        target_pose: SE3,
        current_q,
        current_pose: SE3 | None = None,
        max_depth: int = 4,
        ilimit: int = 100,
        jogging: bool = False
):
    """
    Uses adaptive tolerance based on proximity to singularities:
    - Near singularities: looser tolerance for easier convergence
    - Away from singularities: stricter tolerance for precise solutions
    If necessary, recursively subdivide the motion until ikine_LMS converges
    on every segment. Finally check that solution respects joint limits. From experimentation,
    jogging with lower tolerances often produces q_paths that do not differ from current_q,
    essentially freezing the robot.

    Parameters
    ----------
    robot : DHRobot
        Robot model
    target_pose : SE3
        Target pose to reach
    current_q : array_like
        Current joint configuration
    current_pose : SE3, optional
        Current pose (computed if None)
    max_depth : int, optional
        Maximum subdivision depth (default: 8)
    ilimit : int, optional
        Maximum iterations for IK solver (default: 100)

    Returns
    -------
    IKResult
        success  - True/False
        q_path   - (mxn) array of the final joint configuration 
        iterations, residual  - aggregated diagnostics
        tolerance_used - which tolerance was used
        violations - joint limit violations (if any)
    """
    if current_pose is None:
        current_pose = robot.fkine(current_q)

    # ── inner recursive solver───────────────────
    def _solve(Ta: SE3, Tb: SE3, q_seed, depth, tol):
        """Return (path_list, success_flag, iterations, residual)."""
        # Workspace reach analysis
        current_reach = np.linalg.norm(Ta.t)
        target_reach = np.linalg.norm(Tb.t)
        
        # Inward motion detection for recovery mode
        is_recovery = target_reach < current_reach
        
        # J5-dependent maximum reach threshold
        max_reach_threshold = calculate_configuration_dependent_max_reach(q_seed)
        
        # Adaptive damping for IK convergence
        if is_recovery:
            # Recovery mode - always use low damping
            damping = 0.0000001
        else:
            # Workspace limit validation
            # print(f"current_reach:{current_reach:.3f}, max_reach_threshold:{max_reach_threshold:.3f}")
            if not is_recovery and target_reach > max_reach_threshold:
                logger.warning(f"Target reach limit exceeded: {target_reach:.3f} > {max_reach_threshold:.3f}")
                return [], False, depth, 0
            else:
                damping = 0.0000001  # Normal low damping
        
        res = robot.ikine_LMS(Tb, q0=q_seed, ilimit=ilimit, tol=tol, wN=damping)
        if res.success:
            q_good = unwrap_angles(res.q, q_seed)      # << unwrap vs *previous*
            return [q_good], True, res.iterations, res.residual

        if depth >= max_depth:
            return [], False, res.iterations, res.residual
        # split the segment and recurse
        Tc = SE3(trinterp(Ta.A, Tb.A, 0.5))            # mid-pose (screw interp)

        left_path,  ok_L, it_L, r_L = _solve(Ta, Tc, q_seed, depth+1, tol)
        if not ok_L:
            return [], False, it_L, r_L

        q_mid = left_path[-1]                          # last solved joint set
        right_path, ok_R, it_R, r_R = _solve(Tc, Tb, q_mid, depth+1, tol)

        return (
            left_path + right_path,
            ok_R,
            it_L + it_R,
            r_R
        )

    # ── kick-off with adaptive tolerance ──────────────────────────────────
    if jogging:
        adaptive_tol = 1e-10
    else:
        adaptive_tol = calculate_adaptive_tolerance(robot, current_q)
    path, ok, its, resid = _solve(current_pose, target_pose, current_q, 0, adaptive_tol)
    # Joint limit constraint validation
    target_q = path[-1] if len(path) != 0 else None
    solution_valid, violations = PAROL6_ROBOT.check_joint_limits(current_q, target_q)
    if ok and solution_valid:
        return IKResult(True, path[-1], its, resid, adaptive_tol, violations)
    else:
        return IKResult(False, None, its, resid, adaptive_tol, violations)

def quintic_scaling(s: float) -> float:
    """
    Calculates a smooth 0-to-1 scaling factor for progress 's'
    using a quintic polynomial, ensuring smooth start/end accelerations.
    """
    return 6 * (s**5) - 15 * (s**4) + 10 * (s**3)