"""
Smooth Motion Module for PAROL6 Robotic Arm
============================================
This module provides advanced trajectory generation capabilities including:
- Circular and arc movements
- Cubic spline trajectories
- Motion blending
- Pre-computed and real-time trajectory generation

Compatible with:
- numpy==1.23.4
- scipy==1.11.4
- roboticstoolbox-python==1.0.3
"""

import sys
import warnings
from collections import namedtuple
from typing import Tuple, Optional, Dict, List, Union
from roboticstoolbox import DHRobot
from spatialmath.base import trinterp

# Version compatibility check
try:
    import numpy as np
    # Numpy version validation
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    if np_version < (1, 23):
        warnings.warn(f"NumPy version {np.__version__} detected. Recommended: 1.23.4")
    
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp
    import scipy
    # Scipy version validation
    scipy_version = tuple(map(int, scipy.__version__.split('.')[:2]))
    if scipy_version < (1, 11):
        warnings.warn(f"SciPy version {scipy.__version__} detected. Recommended: 1.11.4")
        
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip3 install numpy==1.23.4 scipy==1.11.4")
    sys.exit(1)

from spatialmath import SE3
import time
from typing import List, Tuple, Optional, Dict, Union
from collections import deque

# Import PAROL6 specific modules (these should be in your path)
try:
    import PAROL6_ROBOT
except ImportError:
    print("Warning: PAROL6 modules not found. Some functions may not work.")
    PAROL6_ROBOT = None

# Global variable to track previous tolerance for logging changes
_prev_tolerance = None

# IK Result structure
IKResult = namedtuple('IKResult', 'success q iterations residual tolerance_used violations')

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

def calculate_adaptive_tolerance(robot, q, strict_tol=1e-10, loose_tol=1e-7):
    """
    Calculate adaptive tolerance based on proximity to singularities.
    Near singularities: looser tolerance for easier convergence.
    Away from singularities: stricter tolerance for precise solutions.
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
        print(f"Adaptive IK tolerance: {adaptive_tol:.2e} ({tol_category}) - Manipulability: {manip:.8f} (threshold: {singularity_threshold})")
        _prev_tolerance = adaptive_tol
    
    return adaptive_tol

def calculate_configuration_dependent_max_reach(q_seed):
    """
    Calculate maximum reach based on joint configuration, particularly joint 5.
    When joint 5 is at 90 degrees, the effective reach is reduced by approximately 0.045.
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
        current_pose: SE3 = None,
        max_depth: int = 4,
        ilimit: int = 100,
        jogging: bool = False
):
    """
    Uses adaptive tolerance based on proximity to singularities.
    If necessary, recursively subdivide the motion until ikine_LMS converges
    on every segment. Finally check that solution respects joint limits.
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
            if current_reach >= max_reach_threshold:
                print(f"Reach limit exceeded: {current_reach:.3f} >= {max_reach_threshold:.3f}")
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
    if PAROL6_ROBOT is not None:
        target_q = path[-1] if len(path) != 0 else None
        solution_valid, violations = PAROL6_ROBOT.check_joint_limits(current_q, target_q)
        if ok and solution_valid:
            return IKResult(True, path[-1], its, resid, adaptive_tol, violations)
        else:
            return IKResult(False, None, its, resid, adaptive_tol, violations)
    else:
        # Skip joint limits if robot model unavailable
        if ok and len(path) > 0:
            return IKResult(True, path[-1], its, resid, adaptive_tol, None)
        else:
            return IKResult(False, None, its, resid, adaptive_tol, None)

# ============================================================================
# END OF IK SOLVER FUNCTIONS
# ============================================================================

# ============================================================================
# QUINTIC POLYNOMIAL PRIMITIVES - Foundation for S-Curve Trajectories
# ============================================================================

class QuinticPolynomial:
    """
    Single-axis quintic polynomial trajectory primitive.
    
    Provides C² continuous trajectories (continuous position, velocity, acceleration)
    with zero jerk at boundaries. This is the building block for S-curve profiles
    and advanced multi-segment trajectories.
    
    Based on the analytical solution approach from the implementation plan,
    using normalized time domain [0,1] for numerical stability.
    """
    
    def __init__(self, q0: float, qf: float, 
                 v0: float = 0, vf: float = 0,
                 a0: float = 0, af: float = 0, 
                 T: float = 1.0):
        """
        Generate quintic polynomial trajectory.
        
        Args:
            q0: Initial position
            qf: Final position
            v0: Initial velocity (default 0)
            vf: Final velocity (default 0)
            a0: Initial acceleration (default 0)
            af: Final acceleration (default 0)
            T: Duration of trajectory (must be > 0)
        """
        if T <= 0:
            raise ValueError(f"Duration must be positive, got T={T}")
            
        self.T = T
        self.q0 = q0
        self.qf = qf
        
        # Store boundary conditions
        self.boundary_conditions = {
            'q0': q0, 'qf': qf,
            'v0': v0, 'vf': vf,
            'a0': a0, 'af': af
        }
        
        # Solve for polynomial coefficients using analytical method
        self.coeffs = self._solve_coefficients_analytical(q0, qf, v0, vf, a0, af, T)
        
        # Pre-compute coefficient derivatives for faster evaluation
        self._prepare_derivative_coeffs()
        
    def _solve_coefficients_analytical(self, q0, qf, v0, vf, a0, af, T):
        """
        Analytical solution for quintic polynomial coefficients.
        
        Uses closed-form solution to avoid numerical issues with matrix inversion.
        Works in normalized time [0,1] then scales back to actual time.
        
        The quintic polynomial is: q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        
        Returns:
            numpy array of coefficients [a0, a1, a2, a3, a4, a5]
        """
        # CRITICAL FIX: Corrected time scaling as per S-Curve_Concerns.md
        # We solve in normalized time τ ∈ [0,1] where τ = t/T
        # This requires scaling the boundary velocities and accelerations
        
        # Scale boundary conditions for normalized time
        # When τ = t/T, then dq/dτ = (dq/dt) * (dt/dτ) = (dq/dt) * T = v * T
        # Similarly, d²q/dτ² = a * T²
        v0_norm = v0 * T
        vf_norm = vf * T
        a0_norm = a0 * T**2
        af_norm = af * T**2
        
        # Analytical solution for normalized quintic coefficients (τ ∈ [0,1])
        # These formulas are derived from solving the linear system with boundary conditions
        a0 = q0
        a1 = v0_norm
        a2 = a0_norm / 2.0
        
        # Closed-form solution for remaining coefficients
        a3 = 10*(qf - q0) - 6*v0_norm - 4*vf_norm - (3*a0_norm - af_norm) / 2.0
        a4 = -15*(qf - q0) + 8*v0_norm + 7*vf_norm + (3*a0_norm - 2*af_norm) / 2.0
        a5 = 6*(qf - q0) - 3*(v0_norm + vf_norm) - (a0_norm - af_norm) / 2.0
        
        # Now convert back to actual time domain
        # The polynomial in actual time is q(t) where t = τ * T
        # We want q(t) = b0 + b1*t + b2*t² + ... where t is actual time
        # Since τ = t/T, we have q(t) = a0 + a1*(t/T) + a2*(t/T)² + ...
        # Therefore: b_n = a_n / T^n
        
        coeffs = np.array([
            a0,           # b0 = a0
            a1 / T,       # b1 = a1 / T
            a2 / T**2,    # b2 = a2 / T²
            a3 / T**3,    # b3 = a3 / T³
            a4 / T**4,    # b4 = a4 / T⁴
            a5 / T**5     # b5 = a5 / T⁵
        ])
        
        return coeffs
    
    def _prepare_derivative_coeffs(self):
        """Pre-compute coefficients for velocity, acceleration, and jerk."""
        # Velocity coefficients: derivative of position polynomial
        self.vel_coeffs = np.array([
            self.coeffs[1],
            2 * self.coeffs[2],
            3 * self.coeffs[3],
            4 * self.coeffs[4],
            5 * self.coeffs[5]
        ])
        
        # Acceleration coefficients: derivative of velocity polynomial
        self.acc_coeffs = np.array([
            2 * self.coeffs[2],
            6 * self.coeffs[3],
            12 * self.coeffs[4],
            20 * self.coeffs[5]
        ])
        
        # Jerk coefficients: derivative of acceleration polynomial
        self.jerk_coeffs = np.array([
            6 * self.coeffs[3],
            24 * self.coeffs[4],
            60 * self.coeffs[5]
        ])
    
    def position(self, t: float) -> float:
        """
        Evaluate position at time t using Horner's method.
        
        Horner's method is numerically stable and computationally efficient,
        reducing the number of multiplications from O(n²) to O(n).
        """
        if t < 0:
            return self.q0
        if t > self.T:
            return self.qf
            
        # Horner's method: a5*t^5 + a4*t^4 + ... + a0
        # Rewritten as: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t + a0
        result = self.coeffs[5]
        for i in range(4, -1, -1):
            result = result * t + self.coeffs[i]
        return result
    
    def velocity(self, t: float) -> float:
        """Evaluate velocity at time t using Horner's method."""
        if t < 0:
            return self.boundary_conditions['v0']
        if t > self.T:
            return self.boundary_conditions['vf']
            
        result = self.vel_coeffs[4] if len(self.vel_coeffs) > 4 else 0
        for i in range(min(3, len(self.vel_coeffs) - 1), -1, -1):
            result = result * t + self.vel_coeffs[i]
        return result
    
    def acceleration(self, t: float) -> float:
        """Evaluate acceleration at time t using Horner's method."""
        if t < 0:
            return self.boundary_conditions['a0']
        if t > self.T:
            return self.boundary_conditions['af']
            
        result = self.acc_coeffs[3] if len(self.acc_coeffs) > 3 else 0
        for i in range(min(2, len(self.acc_coeffs) - 1), -1, -1):
            result = result * t + self.acc_coeffs[i]
        return result
    
    def jerk(self, t: float) -> float:
        """Evaluate jerk at time t using Horner's method."""
        if t < 0 or t > self.T:
            return 0  # Jerk is zero at boundaries by design
            
        result = self.jerk_coeffs[2] if len(self.jerk_coeffs) > 2 else 0
        for i in range(min(1, len(self.jerk_coeffs) - 1), -1, -1):
            result = result * t + self.jerk_coeffs[i]
        return result
    
    def evaluate(self, t: float, derivative: int = 0) -> float:
        """
        Unified evaluation function for any derivative order.
        
        Args:
            t: Time point to evaluate
            derivative: 0=position, 1=velocity, 2=acceleration, 3=jerk
            
        Returns:
            Value of the specified derivative at time t
        """
        if derivative == 0:
            return self.position(t)
        elif derivative == 1:
            return self.velocity(t)
        elif derivative == 2:
            return self.acceleration(t)
        elif derivative == 3:
            return self.jerk(t)
        else:
            raise ValueError(f"Derivative order {derivative} not supported (max is 3)")
    
    def get_trajectory_points(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Generate trajectory points at specified time interval.
        
        Args:
            dt: Time step (default 0.01 for 100Hz)
            
        Returns:
            Dictionary with 'time', 'position', 'velocity', 'acceleration', 'jerk' arrays
        """
        time_points = np.arange(0, self.T + dt, dt)
        
        trajectory = {
            'time': time_points,
            'position': np.array([self.position(t) for t in time_points]),
            'velocity': np.array([self.velocity(t) for t in time_points]),
            'acceleration': np.array([self.acceleration(t) for t in time_points]),
            'jerk': np.array([self.jerk(t) for t in time_points])
        }
        
        return trajectory
    
    def validate_continuity(self, tolerance: float = 1e-10) -> Dict[str, bool]:
        """
        Validate that boundary conditions are satisfied.
        
        Returns:
            Dictionary with validation results for each boundary condition
        """
        validation = {
            'q0': abs(self.position(0) - self.boundary_conditions['q0']) < tolerance,
            'qf': abs(self.position(self.T) - self.boundary_conditions['qf']) < tolerance,
            'v0': abs(self.velocity(0) - self.boundary_conditions['v0']) < tolerance,
            'vf': abs(self.velocity(self.T) - self.boundary_conditions['vf']) < tolerance,
            'a0': abs(self.acceleration(0) - self.boundary_conditions['a0']) < tolerance,
            'af': abs(self.acceleration(self.T) - self.boundary_conditions['af']) < tolerance,
        }
        
        return validation
    
    def validate_numerical_stability(self) -> Dict[str, any]:
        """
        Check for potential numerical stability issues.
        
        Returns:
            Dictionary with stability metrics and warnings
        """
        stability = {'is_stable': True, 'warnings': [], 'metrics': {}}
        
        # Check condition number (ratio of time to distance)
        distance = abs(self.qf - self.q0)
        if distance > 1e-6:
            time_distance_ratio = self.T / distance
            stability['metrics']['time_distance_ratio'] = time_distance_ratio
            
            if time_distance_ratio > 100:
                stability['is_stable'] = False
                stability['warnings'].append(f"Poor conditioning: T/d ratio = {time_distance_ratio:.1f}")
        
        # Check coefficient magnitudes
        coeff_magnitudes = [abs(c) for c in self.coeffs]
        max_coeff = max(coeff_magnitudes)
        min_nonzero_coeff = min([m for m in coeff_magnitudes if m > 1e-10])
        
        if min_nonzero_coeff > 0:
            coeff_ratio = max_coeff / min_nonzero_coeff
            stability['metrics']['coefficient_ratio'] = coeff_ratio
            
            if coeff_ratio > 1e6:
                stability['warnings'].append(f"Large coefficient ratio: {coeff_ratio:.2e}")
        
        # Check for very small time durations
        if self.T < 0.001:
            stability['warnings'].append(f"Very small duration T={self.T} may cause numerical issues")
        
        # Check for very large accelerations/jerks
        max_jerk = max(abs(self.jerk(t)) for t in np.linspace(0, self.T, 10))
        if max_jerk > 1e6:
            stability['warnings'].append(f"Very large jerk values: {max_jerk:.2e}")
        
        return stability


class MultiAxisQuinticTrajectory:
    """
    Multi-axis synchronized quintic trajectory generator.
    
    Ensures all axes complete their motion simultaneously using the 
    time-scaling approach from mstraj (Peter Corke's implementation).
    """
    
    def __init__(self, q0: List[float], qf: List[float],
                 v0: Optional[List[float]] = None,
                 vf: Optional[List[float]] = None,
                 a0: Optional[List[float]] = None,
                 af: Optional[List[float]] = None,
                 T: Optional[float] = None,
                 constraints: Optional['MotionConstraints'] = None):
        """
        Generate synchronized quintic trajectories for multiple axes.
        
        Args:
            q0: Initial positions for each axis
            qf: Final positions for each axis
            v0: Initial velocities (default zeros)
            vf: Final velocities (default zeros)
            a0: Initial accelerations (default zeros)
            af: Final accelerations (default zeros)
            T: Duration (if None, calculated from constraints)
            constraints: Motion constraints for time calculation
        """
        self.num_axes = len(q0)
        
        # Default boundary conditions
        v0 = v0 if v0 is not None else [0] * self.num_axes
        vf = vf if vf is not None else [0] * self.num_axes
        a0 = a0 if a0 is not None else [0] * self.num_axes
        af = af if af is not None else [0] * self.num_axes
        
        # Calculate minimum time if not specified
        if T is None:
            T = self._calculate_minimum_time(q0, qf, v0, vf, constraints)
        
        self.T = T
        
        # Generate quintic polynomial for each axis
        self.axis_trajectories = []
        for i in range(self.num_axes):
            quintic = QuinticPolynomial(
                q0[i], qf[i], v0[i], vf[i], a0[i], af[i], T
            )
            self.axis_trajectories.append(quintic)
    
    def _calculate_minimum_time(self, q0, qf, v0, vf, constraints):
        """
        Calculate minimum time based on velocity and acceleration constraints.
        Uses the approach from the research document.
        """
        if constraints is None:
            # Default time based on distance
            max_distance = max(abs(qf[i] - q0[i]) for i in range(self.num_axes))
            return max(2.0, max_distance / 50.0)  # Assume 50 units/s default
        
        min_times = []
        for i in range(self.num_axes):
            distance = abs(qf[i] - q0[i])
            
            # Time based on velocity limit
            if constraints.max_velocity and i < len(constraints.max_velocity):
                t_vel = distance / constraints.max_velocity[i]
                min_times.append(t_vel)
            
            # Time based on acceleration limit (triangular profile)
            if constraints.max_acceleration and i < len(constraints.max_acceleration):
                t_acc = 2 * np.sqrt(distance / constraints.max_acceleration[i])
                min_times.append(t_acc)
        
        # Use maximum time to ensure all constraints are satisfied
        return max(min_times) * 1.2  # 20% safety margin
    
    def evaluate_all(self, t: float) -> Dict[str, List[float]]:
        """
        Evaluate all axes at time t.
        
        Returns:
            Dictionary with 'position', 'velocity', 'acceleration', 'jerk' lists
        """
        result = {
            'position': [],
            'velocity': [],
            'acceleration': [],
            'jerk': []
        }
        
        for traj in self.axis_trajectories:
            result['position'].append(traj.position(t))
            result['velocity'].append(traj.velocity(t))
            result['acceleration'].append(traj.acceleration(t))
            result['jerk'].append(traj.jerk(t))
        
        return result
    
    def get_trajectory_points(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """Generate trajectory points for all axes."""
        time_points = np.arange(0, self.T + dt, dt)
        num_points = len(time_points)
        
        positions = np.zeros((num_points, self.num_axes))
        velocities = np.zeros((num_points, self.num_axes))
        accelerations = np.zeros((num_points, self.num_axes))
        jerks = np.zeros((num_points, self.num_axes))
        
        for i, t in enumerate(time_points):
            values = self.evaluate_all(t)
            positions[i] = values['position']
            velocities[i] = values['velocity']
            accelerations[i] = values['acceleration']
            jerks[i] = values['jerk']
        
        return {
            'time': time_points,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
            'jerk': jerks
        }


# ============================================================================
# S-CURVE PROFILE GENERATOR - Seven-Segment Jerk-Limited Trajectories
# ============================================================================

class SCurveProfile:
    """
    Seven-segment S-curve velocity profile generator.
    
    Creates jerk-limited trajectories with smooth acceleration transitions.
    Based on the research from S_curve_research.md and implementation plan.
    
    The seven segments are:
    1. Acceleration buildup (constant positive jerk)
    2. Constant acceleration 
    3. Acceleration ramp-down (constant negative jerk)
    4. Constant velocity (cruise)
    5. Deceleration buildup (constant negative jerk)
    6. Constant deceleration
    7. Deceleration ramp-down (constant positive jerk)
    """
    
    def __init__(self, distance: float, v_max: float, a_max: float, j_max: float,
                 v_start: float = 0, v_end: float = 0):
        """
        Calculate S-curve profile for given motion parameters.
        
        Args:
            distance: Total distance to travel
            v_max: Maximum velocity constraint
            a_max: Maximum acceleration constraint  
            j_max: Maximum jerk constraint
            v_start: Initial velocity (default 0)
            v_end: Final velocity (default 0)
        """
        self.distance = distance
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.v_start = v_start
        self.v_end = v_end
        
        # Check feasibility and adjust parameters if needed
        self.feasibility_status = self._check_feasibility()
        
        # Calculate profile type and segment durations
        self.profile_type, self.segments = self._calculate_profile()
        
        # Pre-calculate segment boundaries for proper evaluation
        self._calculate_segment_boundaries()
        
    def _calculate_profile(self):
        """
        Calculate the S-curve profile type and segment durations.
        
        Returns:
            profile_type: 'full', 'triangular', or 'reduced'
            segments: Dictionary with segment durations
        """
        # For symmetric profiles with v_start = v_end = 0
        if self.v_start == 0 and self.v_end == 0:
            return self._calculate_symmetric_profile()
        else:
            # Asymmetric profiles for non-zero boundary velocities
            return self._calculate_asymmetric_profile()
    
    def _calculate_symmetric_profile(self):
        """Calculate symmetric S-curve profile (v_start = v_end = 0)."""
        
        # Time to reach maximum acceleration from zero (jerk phase)
        T_j = self.a_max / self.j_max
        
        # Distance covered during jerk phases
        d_jerk = self.a_max**3 / (self.j_max**2)
        
        # Check if we can reach maximum velocity
        d_to_vmax = self.v_max**2 / self.a_max + self.v_max * self.a_max / self.j_max
        
        if self.distance < 2 * d_jerk:
            # Case 1: Reduced acceleration profile (never reach a_max)
            # Pure jerk-limited motion
            profile_type = 'reduced'
            
            # Solve for actual acceleration reached
            # Use abs to handle numerical issues near zero
            a_reached = (abs(self.distance) * self.j_max**2 / 2)**(1/3)
            T_j_actual = a_reached / self.j_max
            
            segments = {
                'T_j1': T_j_actual,  # Jerk up
                'T_a': 0,            # No constant acceleration
                'T_j2': T_j_actual,  # Jerk down
                'T_v': 0,            # No cruise
                'T_j3': T_j_actual,  # Jerk down (decel)
                'T_d': 0,            # No constant deceleration
                'T_j4': T_j_actual,  # Jerk up (decel end)
                'a_reached': a_reached,
                'v_reached': a_reached * T_j_actual
            }
            
        elif self.distance < 2 * d_to_vmax:
            # Case 2: Triangular velocity profile (never reach v_max)
            profile_type = 'triangular'
            
            # Maximum velocity reached
            v_reached = np.sqrt(self.distance * self.a_max - 
                              2 * self.a_max**3 / self.j_max**2)
            
            # Time at constant acceleration
            T_a = (v_reached - self.a_max**2 / self.j_max) / self.a_max
            
            segments = {
                'T_j1': T_j,
                'T_a': T_a,
                'T_j2': T_j,
                'T_v': 0,  # No cruise phase
                'T_j3': T_j,
                'T_d': T_a,
                'T_j4': T_j,
                'v_reached': v_reached
            }
            
        else:
            # Case 3: Full S-curve with cruise phase
            profile_type = 'full'
            
            # Time at constant acceleration (after jerk phases)
            T_a = (self.v_max - self.a_max**2 / self.j_max) / self.a_max
            
            # Distance covered during acceleration/deceleration
            d_accel = self.v_max**2 / self.a_max + self.v_max * self.a_max / self.j_max
            
            # Distance at cruise velocity
            d_cruise = self.distance - 2 * d_accel
            
            # Time at cruise velocity
            T_v = d_cruise / self.v_max
            
            segments = {
                'T_j1': T_j,
                'T_a': T_a,
                'T_j2': T_j,
                'T_v': T_v,
                'T_j3': T_j,
                'T_d': T_a,
                'T_j4': T_j,
                'v_reached': self.v_max
            }
        
        # Calculate total time
        total_time = sum([segments[f'T_j{i}'] for i in range(1, 5)]) + \
                    segments['T_a'] + segments['T_d'] + segments['T_v']
        segments['T_total'] = total_time
        
        return profile_type, segments
    
    def _calculate_asymmetric_profile(self):
        """Calculate asymmetric S-curve for non-zero boundary velocities."""
        # Handle asymmetric case with non-zero start/end velocities
        v_diff = abs(self.v_end - self.v_start)
        v_avg = (self.v_start + self.v_end) / 2
        
        # Time to change between velocities at max acceleration
        T_vel_change = v_diff / self.a_max if self.a_max > 0 else 0
        
        # Jerk time (time to reach max acceleration)
        T_j = self.a_max / self.j_max if self.j_max > 0 else 0
        
        # Check if we need acceleration phase
        if self.v_start < self.v_max and self.v_end < self.v_max:
            # Need to accelerate to some velocity
            v_peak = min(self.v_max, v_avg + np.sqrt(self.distance * self.a_max))
            
            # Time to accelerate from v_start to v_peak
            T_accel = (v_peak - self.v_start) / self.a_max if self.a_max > 0 else 0
            T_a = max(0, T_accel - 2 * T_j)
            
            # Time to decelerate from v_peak to v_end
            T_decel = (v_peak - self.v_end) / self.a_max if self.a_max > 0 else 0
            T_d = max(0, T_decel - 2 * T_j)
            
            # Check if we have a cruise phase
            d_accel = self.v_start * (T_a + 2*T_j) + 0.5 * self.a_max * (T_a + T_j)**2
            d_decel = self.v_end * (T_d + 2*T_j) + 0.5 * self.a_max * (T_d + T_j)**2
            d_cruise = self.distance - d_accel - d_decel
            
            if d_cruise > 0 and v_peak > 0:
                T_v = d_cruise / v_peak
            else:
                T_v = 0
                # Recalculate without cruise
                v_peak = np.sqrt((2 * self.distance * self.a_max + self.v_start**2 + self.v_end**2) / 2)
        else:
            # Simple case - just decelerate
            T_a = 0
            T_v = 0
            T_d = T_vel_change
            v_peak = max(self.v_start, self.v_end)
        
        segments = {
            'T_j1': T_j,
            'T_a': T_a,
            'T_j2': T_j,
            'T_v': T_v,
            'T_j3': T_j,
            'T_d': T_d,
            'T_j4': T_j,
            'v_reached': v_peak,
            'T_total': 2 * T_j + T_a + T_v + 2 * T_j + T_d
        }
        
        return 'asymmetric', segments
    
    def _check_feasibility(self) -> Dict[str, any]:
        """
        Check if S-curve profile is achievable with given constraints.
        
        Returns:
            Dictionary with feasibility status and adjusted parameters
        """
        # Minimum distance to reach maximum acceleration
        d_min_jerk = (self.a_max**3) / (self.j_max**2) if self.j_max > 0 else 0
        
        # Minimum distance to reach maximum velocity
        d_min_vel = (self.v_max**2 / self.a_max + self.v_max * self.a_max / self.j_max) if self.a_max > 0 and self.j_max > 0 else float('inf')
        
        feasibility = {'status': 'feasible', 'warnings': []}
        
        if self.distance < 2 * d_min_jerk:
            # Cannot reach full acceleration
            achievable_a = (abs(self.distance) * self.j_max**2 / 2)**(1/3) if self.j_max > 0 else 0
            feasibility['status'] = 'reduced_acceleration'
            feasibility['achievable_a'] = achievable_a
            feasibility['warnings'].append(f"Distance too short to reach full acceleration. Max achievable: {achievable_a:.2f}")
            
        elif self.distance < 2 * d_min_vel:
            # Cannot reach full velocity
            achievable_v = np.sqrt(self.distance * self.a_max) if self.a_max > 0 else 0
            feasibility['status'] = 'triangular_velocity'
            feasibility['achievable_v'] = achievable_v
            feasibility['warnings'].append(f"Distance too short to reach full velocity. Max achievable: {achievable_v:.2f}")
        
        # Check for numerical stability
        if self.distance > 0:
            time_estimate = 2 * np.sqrt(self.distance / self.a_max) if self.a_max > 0 else 0
            if time_estimate > 100:
                feasibility['warnings'].append(f"Very long trajectory time ({time_estimate:.1f}s) may cause numerical issues")
        
        # Check constraint validity
        if self.v_max <= 0 or self.a_max <= 0 or self.j_max <= 0:
            feasibility['status'] = 'invalid_constraints'
            feasibility['warnings'].append("Invalid constraints: v_max, a_max, and j_max must all be positive")
        
        return feasibility
    
    def _calculate_segment_boundaries(self):
        """
        Pre-calculate position, velocity, and acceleration at segment boundaries.
        This ensures proper continuity across segments.
        """
        self.segment_boundaries = []
        
        # Initial state
        pos = 0
        vel = self.v_start
        acc = 0
        
        # Segment 1: Jerk up (acceleration buildup)
        T_j1 = self.segments['T_j1']
        if T_j1 > 0:
            j = self.j_max
            acc_end = acc + j * T_j1  # acc increases from 0 to a_max
            vel_end = vel + acc * T_j1 + 0.5 * j * T_j1**2
            pos_end = pos + vel * T_j1 + 0.5 * acc * T_j1**2 + (1/6) * j * T_j1**3
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_j1
            })
            pos, vel, acc = pos_end, vel_end, acc_end
        
        # Segment 2: Constant acceleration
        T_a = self.segments['T_a']
        if T_a > 0:
            j = 0
            acc_end = acc  # Constant
            vel_end = vel + acc * T_a
            pos_end = pos + vel * T_a + 0.5 * acc * T_a**2
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_a
            })
            pos, vel, acc = pos_end, vel_end, acc_end
        
        # Segment 3: Jerk down (acceleration ramp-down)
        T_j2 = self.segments['T_j2']
        if T_j2 > 0:
            j = -self.j_max
            acc_end = acc + j * T_j2  # Should go to zero
            vel_end = vel + acc * T_j2 + 0.5 * j * T_j2**2
            pos_end = pos + vel * T_j2 + 0.5 * acc * T_j2**2 + (1/6) * j * T_j2**3
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_j2
            })
            pos, vel, acc = pos_end, vel_end, acc_end
        
        # Segment 4: Constant velocity (cruise)
        T_v = self.segments['T_v']
        if T_v > 0:
            j = 0
            acc_end = 0
            vel_end = vel  # Constant
            pos_end = pos + vel * T_v
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_v
            })
            pos, vel, acc = pos_end, vel_end, acc_end
        
        # Segment 5: Jerk down (deceleration buildup)
        T_j3 = self.segments['T_j3']
        if T_j3 > 0:
            j = -self.j_max
            acc_end = j * T_j3  # Negative acceleration
            vel_end = vel + 0.5 * j * T_j3**2
            pos_end = pos + vel * T_j3 + (1/6) * j * T_j3**3
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_j3
            })
            pos, vel, acc = pos_end, vel_end, acc_end
        
        # Segment 6: Constant deceleration
        T_d = self.segments['T_d']
        if T_d > 0:
            j = 0
            acc_end = acc  # Constant (negative)
            vel_end = vel + acc * T_d
            pos_end = pos + vel * T_d + 0.5 * acc * T_d**2
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_d
            })
            pos, vel, acc = pos_end, vel_end, acc_end
        
        # Segment 7: Jerk up (deceleration ramp-down)
        T_j4 = self.segments['T_j4']
        if T_j4 > 0:
            j = self.j_max
            acc_end = acc + j * T_j4  # Should go to zero
            vel_end = vel + acc * T_j4 + 0.5 * j * T_j4**2
            pos_end = pos + vel * T_j4 + 0.5 * acc * T_j4**2 + (1/6) * j * T_j4**3
            self.segment_boundaries.append({
                'pos_start': pos, 'vel_start': vel, 'acc_start': acc,
                'pos_end': pos_end, 'vel_end': vel_end, 'acc_end': acc_end,
                'jerk': j, 'duration': T_j4
            })
            pos, vel, acc = pos_end, vel_end, acc_end
    
    def generate_trajectory_quintics(self) -> List[QuinticPolynomial]:
        """
        Generate quintic polynomials for each segment of the S-curve.
        
        Returns:
            List of QuinticPolynomial objects representing each segment
        """
        quintics = []
        
        # Use pre-calculated segment boundaries for consistency
        for i, seg in enumerate(self.segment_boundaries):
            if seg['duration'] > 0:
                q = QuinticPolynomial(
                    seg['pos_start'], seg['pos_end'],
                    seg['vel_start'], seg['vel_end'],
                    seg['acc_start'], seg['acc_end'],
                    seg['duration']
                )
                quintics.append(q)
        
        return quintics
    
    def get_total_time(self) -> float:
        """Get total time for the S-curve trajectory."""
        return self.segments['T_total']
    
    def evaluate_at_time(self, t: float) -> Dict[str, float]:
        """
        Evaluate S-curve at specific time.
        
        Returns:
            Dictionary with position, velocity, acceleration, jerk
        """
        if t <= 0:
            return {'position': 0, 'velocity': self.v_start, 
                   'acceleration': 0, 'jerk': 0}
        
        if t >= self.segments['T_total']:
            # Return final values
            if self.segment_boundaries:
                last = self.segment_boundaries[-1]
                return {'position': last['pos_end'], 
                       'velocity': last['vel_end'],
                       'acceleration': 0, 'jerk': 0}
            else:
                return {'position': self.distance, 'velocity': self.v_end,
                       'acceleration': 0, 'jerk': 0}
        
        # Find which segment we're in
        cumulative_time = 0
        for seg in self.segment_boundaries:
            if t <= cumulative_time + seg['duration']:
                # We're in this segment
                t_in_segment = t - cumulative_time
                return self._evaluate_in_segment(seg, t_in_segment)
            cumulative_time += seg['duration']
        
        # Should not reach here
        return {'position': self.distance, 'velocity': self.v_end,
                'acceleration': 0, 'jerk': 0}
    
    def _evaluate_in_segment(self, segment: Dict, t: float) -> Dict[str, float]:
        """
        Evaluate motion within a specific segment using proper kinematics.
        """
        # Extract segment parameters
        p0 = segment['pos_start']
        v0 = segment['vel_start']
        a0 = segment['acc_start']
        j = segment['jerk']
        
        # Calculate values at time t within segment
        # Position: p(t) = p0 + v0*t + 0.5*a0*t^2 + (1/6)*j*t^3
        # Velocity: v(t) = v0 + a0*t + 0.5*j*t^2
        # Acceleration: a(t) = a0 + j*t
        
        acc = a0 + j * t
        vel = v0 + a0 * t + 0.5 * j * t**2
        pos = p0 + v0 * t + 0.5 * a0 * t**2 + (1/6) * j * t**3
        
        return {'position': pos, 'velocity': vel, 'acceleration': acc, 'jerk': j}


class MultiAxisSCurveTrajectory:
    """
    Multi-axis synchronized S-curve trajectory generator.
    
    Coordinates multiple S-curve profiles (one per axis) and synchronizes them
    to ensure all axes complete their motion simultaneously while respecting
    individual axis constraints.
    """
    
    def __init__(self, start_pose: List[float], end_pose: List[float], 
                 v0: Optional[List[float]] = None, vf: Optional[List[float]] = None,
                 T: Optional[float] = None, jerk_limit: Optional[float] = None):
        """
        Create synchronized S-curve trajectories for multiple axes.
        
        Args:
            start_pose: Starting position for each axis
            end_pose: Target position for each axis
            v0: Initial velocities (defaults to zeros)
            vf: Final velocities (defaults to zeros)
            T: Desired duration (if None, calculates minimum time)
            jerk_limit: Maximum jerk limit to apply to all axes
        """
        self.start_pose = np.array(start_pose)
        self.end_pose = np.array(end_pose)
        self.num_axes = len(start_pose)
        
        # Initialize velocities
        self.v0 = np.array(v0) if v0 is not None else np.zeros(self.num_axes)
        self.vf = np.array(vf) if vf is not None else np.zeros(self.num_axes)
        
        # Get motion constraints
        self.constraints = MotionConstraints()
        
        # Create individual S-curve profiles for each axis
        self.axis_profiles = []
        self.max_time = 0
        
        for i in range(self.num_axes):
            distance = self.end_pose[i] - self.start_pose[i]
            
            # Skip axes with no motion
            if abs(distance) < 1e-6:
                self.axis_profiles.append(None)
                continue
            
            # Get per-axis constraints
            joint_constraints = self.constraints.get_joint_constraints(i)
            if joint_constraints is None:
                # Default constraints if joint index out of range
                joint_constraints = {
                    'v_max': 10000,
                    'a_max': 20000,
                    'j_max': jerk_limit if jerk_limit else 50000
                }
            
            # Use provided jerk limit if specified
            j_max = jerk_limit if jerk_limit is not None else joint_constraints['j_max']
            
            # Create S-curve profile for this axis
            profile = SCurveProfile(
                distance,
                joint_constraints['v_max'],
                joint_constraints['a_max'],
                j_max,
                v_start=self.v0[i],
                v_end=self.vf[i]
            )
            
            self.axis_profiles.append(profile)
            self.max_time = max(self.max_time, profile.get_total_time())
        
        # Set synchronized time
        self.T = T if T is not None else self.max_time
        
        # Calculate time scaling factors for synchronization
        self.time_scales = []
        for profile in self.axis_profiles:
            if profile is not None:
                # Scale time so all axes finish together
                scale = profile.get_total_time() / self.T if self.T > 0 else 1.0
                self.time_scales.append(scale)
            else:
                self.time_scales.append(1.0)
    
    def get_trajectory_points(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Generate synchronized trajectory points for all axes.
        
        Args:
            dt: Time step for sampling
            
        Returns:
            Dictionary with 'position', 'velocity', 'acceleration' arrays
        """
        num_points = int(self.T / dt) + 1
        timestamps = np.linspace(0, self.T, num_points)
        
        positions = np.zeros((num_points, self.num_axes))
        velocities = np.zeros((num_points, self.num_axes))
        accelerations = np.zeros((num_points, self.num_axes))
        
        for idx, t in enumerate(timestamps):
            for axis in range(self.num_axes):
                if self.axis_profiles[axis] is None:
                    # No motion for this axis
                    positions[idx, axis] = self.start_pose[axis]
                    velocities[idx, axis] = 0
                    accelerations[idx, axis] = 0
                else:
                    # Scale time for this axis's profile
                    t_scaled = t * self.time_scales[axis]
                    
                    # Get values from S-curve profile
                    values = self.axis_profiles[axis].evaluate_at_time(t_scaled)
                    
                    # Add to start position
                    positions[idx, axis] = self.start_pose[axis] + values['position']
                    velocities[idx, axis] = values['velocity']
                    accelerations[idx, axis] = values['acceleration']
        
        return {
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
            'timestamps': timestamps
        }
    
    def evaluate_at_time(self, t: float) -> Dict[str, np.ndarray]:
        """
        Evaluate trajectory at specific time.
        
        Args:
            t: Time point to evaluate
            
        Returns:
            Dictionary with position, velocity, acceleration arrays for all axes
        """
        position = np.zeros(self.num_axes)
        velocity = np.zeros(self.num_axes)
        acceleration = np.zeros(self.num_axes)
        
        for axis in range(self.num_axes):
            if self.axis_profiles[axis] is None:
                position[axis] = self.start_pose[axis]
                velocity[axis] = 0
                acceleration[axis] = 0
            else:
                # Scale time for this axis
                t_scaled = min(t * self.time_scales[axis], 
                              self.axis_profiles[axis].get_total_time())
                
                values = self.axis_profiles[axis].evaluate_at_time(t_scaled)
                position[axis] = self.start_pose[axis] + values['position']
                velocity[axis] = values['velocity']
                acceleration[axis] = values['acceleration']
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration
        }


class MotionConstraints:
    """
    Motion constraints for PAROL6 robot.
    
    Defines per-joint limits for velocity, acceleration, and jerk
    based on gear ratios and mechanical properties.
    """
    
    def __init__(self):
        """Initialize with PAROL6-specific constraints."""
        # Gear ratios from research
        self.gear_ratios = [6.4, 20, 18.1, 4, 4, 10]
        
        # Maximum jerk limits (steps/s³) from implementation plan
        self.max_jerk = [1600, 1000, 1100, 3000, 3000, 2000]
        
        # Maximum velocities (steps/s) - from PAROL6_ROBOT.py
        self.max_velocity = [8000, 18000, 10000, 22000, 22000, 48000]
        
        # Calculate max accelerations based on response time
        # Assuming 50ms response time for steppers
        response_time = 0.05
        self.max_acceleration = [v / (10 * response_time) for v in self.max_velocity]
        
    def get_joint_constraints(self, joint_idx: int) -> Dict[str, float]:
        """Get constraints for specific joint."""
        if joint_idx >= len(self.gear_ratios):
            return None
            
        return {
            'gear_ratio': self.gear_ratios[joint_idx],
            'v_max': self.max_velocity[joint_idx],
            'a_max': self.max_acceleration[joint_idx],
            'j_max': self.max_jerk[joint_idx]
        }
    
    def scale_for_gear_ratio(self, joint_idx: int, base_limits: Dict) -> Dict:
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
                           joint_idx: int, dt: float = 0.01) -> Dict[str, bool]:
        """
        Validate that trajectory respects constraints.
        
        Returns:
            Dictionary with validation results
        """
        constraints = self.get_joint_constraints(joint_idx)
        
        # Calculate derivatives numerically
        velocities = np.diff(trajectory) / dt
        accelerations = np.diff(velocities) / dt
        jerks = np.diff(accelerations) / dt
        
        validation = {
            'velocity_ok': np.all(np.abs(velocities) <= constraints['v_max']),
            'acceleration_ok': np.all(np.abs(accelerations) <= constraints['a_max']),
            'jerk_ok': np.all(np.abs(jerks) <= constraints['j_max']),
            'max_velocity': np.max(np.abs(velocities)),
            'max_acceleration': np.max(np.abs(accelerations)),
            'max_jerk': np.max(np.abs(jerks))
        }
        
        return validation


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
        
    def generate_timestamps(self, duration: float) -> np.ndarray:
        """Generate evenly spaced timestamps for trajectory"""
        num_points = int(duration * self.control_rate)
        return np.linspace(0, duration, num_points)

class CircularMotion(TrajectoryGenerator):
    """Generate circular and arc trajectories in 3D space"""
    
    def generate_arc_3d(self, 
                       start_pose: List[float], 
                       end_pose: List[float], 
                       center: List[float], 
                       normal: Optional[List[float]] = None,
                       clockwise: bool = True,
                       duration: float = 2.0) -> np.ndarray:
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
        radius = np.linalg.norm(r1)
        
        # Arc plane normal computation
        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:  # Points are collinear
                normal = np.array([0, 0, 1])  # Default to XY plane
        normal = normal / np.linalg.norm(normal)
        
        # Arc sweep angle calculation
        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)
        
        # Arc direction validation
        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, normal) < 0:
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
                rot_matrix = self._rotation_matrix_from_axis_angle(normal, angle)
                current_pos = center_pt + rot_matrix @ r1
            
            # Interpolate orientation (SLERP)
            current_orient = self._slerp_orientation(start_pose[3:], end_pose[3:], s)
            
            # Combine position and orientation
            pose = np.concatenate([current_pos, current_orient])
            trajectory.append(pose)
            
        return np.array(trajectory)
    
    def generate_arc_with_profile(self,
                                 start_pose: List[float],
                                 end_pose: List[float],
                                 center: List[float],
                                 normal: Optional[List[float]] = None,
                                 clockwise: bool = True,
                                 duration: float = 2.0,
                                 trajectory_type: str = 'cubic',
                                 jerk_limit: Optional[float] = None) -> np.ndarray:
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
        if trajectory_type == 'cubic':
            # Use existing cubic implementation
            return self.generate_arc_3d(start_pose, end_pose, center, normal, clockwise, duration)
        
        # Convert to numpy arrays
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])
        center_pt = np.array(center)
        
        # Arc geometry
        r1 = start_pos - center_pt
        r2 = end_pos - center_pt
        radius = np.linalg.norm(r1)
        
        # Arc plane normal
        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:
                normal = np.array([0, 0, 1])
        normal = normal / np.linalg.norm(normal)
        
        # Calculate arc angle
        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)
        
        # Determine arc direction
        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, normal) < 0:
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
            if trajectory_type == 'quintic':
                # Quintic polynomial for smooth acceleration
                s = 10 * t**3 - 15 * t**4 + 6 * t**5
            elif trajectory_type == 's_curve':
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
            rot_matrix = self._rotation_matrix_from_axis_angle(normal, angle)
            current_pos = center_pt + rot_matrix @ r1
            
            # Interpolate orientation (SLERP)
            current_orient = self._slerp_orientation(start_pose[3:], end_pose[3:], s)
            
            # Combine position and orientation
            pose = np.concatenate([current_pos, current_orient])
            trajectory.append(pose)
        
        return np.array(trajectory)
    
    def generate_circle_3d(self,
                      center: List[float],
                      radius: float,
                      normal: List[float] = [0, 0, 1],
                      start_angle: float = None,
                      duration: float = 4.0,
                      start_point: List[float] = None) -> np.ndarray:
        """
        Generate a complete circle trajectory that starts at start_point
        """
        timestamps = self.generate_timestamps(duration)
        trajectory = []
        
        # Circle coordinate system
        normal_np = np.array(normal)
        normal = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal)
        v = np.cross(normal, u)
        
        center_np = np.array(center)
        
        # CRITICAL FIX: Validate and handle geometry
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            
            # Project start point onto the circle plane
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal) * normal
            
            # Get distance from center in the plane
            dist_in_plane = np.linalg.norm(to_start_plane)
            
            if dist_in_plane < 0.001:
                # Center start point - undefined angle
                print(f"    WARNING: Start point is at circle center, using default position")
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
                    print(f"    INFO: Starting {dist_in_plane:.1f}mm from center (radius: {radius}mm)")
                    if radius_error > radius * 0.3:  # More than 30% off
                        print(f"    WARNING: Large distance from circle - consider using entry trajectory")
                    # Note: We do NOT adjust the center - this ensures repeatability
                    # The same command will always produce the same geometric circle
                
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
                angle = start_angle + (2 * np.pi * t / duration)
                pos = center_np + radius * (np.cos(angle) * u + np.sin(angle) * v)
            
            # Placeholder orientation (will be overridden)
            orient = [0, 0, 0]
            trajectory.append(np.concatenate([pos, orient]))
        
        return np.array(trajectory)
    
    def generate_circle_with_profile(self,
                                    center: List[float],
                                    radius: float,
                                    normal: List[float] = [0, 0, 1],
                                    duration: float = 4.0,
                                    trajectory_type: str = 'cubic',
                                    jerk_limit: Optional[float] = None,
                                    start_angle: float = None,
                                    start_point: List[float] = None) -> np.ndarray:
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
        adaptive_rate = min(200, max(base_rate, required_rate))
        
        # Temporarily override control rate for small circles
        if radius < 30 and adaptive_rate > base_rate:
            original_rate = self.control_rate
            original_dt = self.dt
            self.control_rate = adaptive_rate
            self.dt = 1.0 / adaptive_rate
            # Use print for debug info since logger not imported here
            print(f"    [ADAPTIVE] Using {adaptive_rate:.0f}Hz control rate for {radius:.0f}mm radius circle")
        else:
            original_rate = None
            original_dt = None
        
        try:
            if trajectory_type == 'cubic':
                # Use existing implementation
                result = self.generate_circle_3d(center, radius, normal, start_angle, duration, start_point)
            elif trajectory_type == 'quintic':
                result = self.generate_quintic_circle(center, radius, normal, duration, start_angle, start_point)
            elif trajectory_type == 's_curve':
                result = self.generate_scurve_circle(center, radius, normal, duration, jerk_limit, start_angle, start_point)
            else:
                raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        finally:
            # Restore original control rate if we changed it
            if original_rate is not None:
                self.control_rate = original_rate
                self.dt = original_dt
        
        return result
    
    def generate_quintic_circle(self,
                               center: List[float],
                               radius: float,
                               normal: List[float] = [0, 0, 1],
                               duration: float = 4.0,
                               start_angle: float = None,
                               start_point: List[float] = None) -> np.ndarray:
        """
        Generate circle trajectory using quintic polynomial velocity profile.
        Provides smooth acceleration and deceleration in Cartesian space.
        """
        # First generate uniform angular points
        num_points = int(duration * self.control_rate)
        
        # Setup coordinate system
        normal_np = np.array(normal)
        normal = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal)
        v = np.cross(normal, u)
        center_np = np.array(center)
        
        # Handle start point if provided
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal) * normal
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
                    print(f"    INFO: Starting {dist_in_plane:.1f}mm from center (radius: {radius}mm)")
                    if radius_error > radius * 0.2:  # More than 20% off
                        print(f"    WARNING: Large distance from circle - consider using entry trajectory")
        else:
            start_angle = 0 if start_angle is None else start_angle
        
        # Step 1: Generate uniformly spaced angular points
        angles = np.linspace(start_angle, start_angle + 2 * np.pi, num_points)
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
                pos = uniform_positions[lower_idx] + alpha * (uniform_positions[upper_idx] - uniform_positions[lower_idx])
            
            # Placeholder orientation
            orient = [0, 0, 0]
            trajectory.append(np.concatenate([pos, orient]))
        
        return np.array(trajectory)
    
    def generate_scurve_circle(self,
                              center: List[float],
                              radius: float,
                              normal: List[float] = [0, 0, 1],
                              duration: float = 4.0,
                              jerk_limit: Optional[float] = 5000.0,
                              start_angle: float = None,
                              start_point: List[float] = None) -> np.ndarray:
        """
        Generate circle trajectory using S-curve velocity profile.
        Provides jerk-limited motion in Cartesian space for maximum smoothness.
        """
        if jerk_limit is None:
            jerk_limit = 5000.0  # Default jerk limit in mm/s^3
        
        # Generate timestamps at control rate
        num_points = int(duration * self.control_rate)
        
        # Setup coordinate system
        normal_np = np.array(normal)
        normal = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal)
        v = np.cross(normal, u)
        center_np = np.array(center)
        
        # Handle start point if provided
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal) * normal
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
                    print(f"    INFO: Starting {dist_in_plane:.1f}mm from center (radius: {radius}mm)")
                    if radius_error > radius * 0.2:  # More than 20% off
                        print(f"    WARNING: Large distance from circle - consider using entry trajectory")
        else:
            start_angle = 0 if start_angle is None else start_angle
        
        # Step 1: Generate uniformly spaced angular points
        angles = np.linspace(start_angle, start_angle + 2 * np.pi, num_points)
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
            # Applied to arc length parameterization, not angular
            if tau <= 0.0:
                s_normalized = 0.0
            elif tau >= 1.0:
                s_normalized = 1.0
            else:
                # Smoothstep: 3t² - 2t³ for smooth acceleration and deceleration
                # Applied to arc length, not angle
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
                pos = uniform_positions[lower_idx] + alpha * (uniform_positions[upper_idx] - uniform_positions[lower_idx])
            
            # Placeholder orientation
            orient = [0, 0, 0]
            trajectory.append(np.concatenate([pos, orient]))
        
        return np.array(trajectory)
    
    def _rotation_matrix_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Generate rotation matrix using Rodrigues' formula"""
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Cross-product matrix
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        # Rodrigues' formula
        R = np.eye(3) + sin_a * K + (1 - cos_a) * K @ K
        return R
    
    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to the given vector"""
        v = np.array(v)  # Ensure it's a numpy array
        if abs(v[0]) < 0.9:
            return np.cross(v, [1, 0, 0]) / np.linalg.norm(np.cross(v, [1, 0, 0]))
        else:
            return np.cross(v, [0, 1, 0]) / np.linalg.norm(np.cross(v, [0, 1, 0]))
    
    def generate_circle_entry(self,
                            current_pos: np.ndarray,
                            circle_center: np.ndarray,
                            radius: float,
                            normal: np.ndarray,
                            duration: float = 1.0,
                            profile_type: str = 'quintic',
                            control_rate: float = None) -> np.ndarray:
        """
        Generate smooth entry trajectory to circle starting point.
        
        Args:
            current_pos: Current robot position [x,y,z] or full pose [x,y,z,rx,ry,rz]
            circle_center: Center of target circle
            radius: Circle radius
            normal: Circle plane normal vector
            duration: Time for entry motion
            profile_type: 'cubic', 'quintic', or 's_curve'
            control_rate: Points per second
        
        Returns:
            Array of waypoints for entry trajectory with full 6D poses
        """
        # Extract position and orientation
        current_position = current_pos[:3] if len(current_pos) > 3 else current_pos
        current_orientation = current_pos[3:] if len(current_pos) > 3 else np.array([0, 0, 0])
        
        # Calculate target point on circle
        to_current = current_position - circle_center
        to_current_plane = to_current - np.dot(to_current, normal) * normal
        
        if np.linalg.norm(to_current_plane) < 0.001:
            # At center, move to +X direction on circle
            u = self._get_perpendicular_vector(normal)
            target = circle_center + radius * u
        else:
            # Move to nearest point on circle
            direction = to_current_plane / np.linalg.norm(to_current_plane)
            target = circle_center + radius * direction
        
        # Generate trajectory based on profile type
        # Use provided control_rate or default to self.control_rate
        rate = control_rate if control_rate is not None else self.control_rate
        num_points = int(duration * rate)
        timestamps = np.linspace(0, duration, num_points)
        trajectory = []
        
        if profile_type == 'quintic':
            # Quintic polynomial for smooth acceleration
            for t in timestamps:
                tau = t / duration
                # Quintic: 10τ³ - 15τ⁴ + 6τ⁵
                s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                pos = current_position + s * (target - current_position)
                # Combine position with orientation for full 6D pose
                full_pose = np.concatenate([pos, current_orientation])
                trajectory.append(full_pose)
        elif profile_type == 's_curve':
            # S-curve (smoothstep) for jerk-limited motion
            for t in timestamps:
                tau = t / duration
                # Smoothstep: 3τ² - 2τ³
                s = tau * tau * (3.0 - 2.0 * tau)
                pos = current_position + s * (target - current_position)
                # Combine position with orientation for full 6D pose
                full_pose = np.concatenate([pos, current_orientation])
                trajectory.append(full_pose)
        else:  # cubic or default
            # Cubic spline interpolation
            for t in timestamps:
                tau = t / duration
                # Simple cubic: 3τ² - 2τ³
                s = 3 * tau**2 - 2 * tau**3
                pos = current_position + s * (target - current_position)
                # Combine position with orientation for full 6D pose
                full_pose = np.concatenate([pos, current_orientation])
                trajectory.append(full_pose)
        
        return np.array(trajectory)
    
    def _slerp_orientation(self, start_orient: List[float], 
                          end_orient: List[float], 
                          t: float) -> np.ndarray:
        """Spherical linear interpolation for orientation"""
        # Convert to quaternions
        r1 = Rotation.from_euler('xyz', start_orient, degrees=True)
        r2 = Rotation.from_euler('xyz', end_orient, degrees=True)
        
        # Quaternion interpolation setup
        # Stack rotations into a single Rotation object
        key_rots = Rotation.from_quat([r1.as_quat(), r2.as_quat()])
        slerp = Slerp([0, 1], key_rots)
        
        # Interpolate
        interp_rot = slerp(t)
        return interp_rot.as_euler('xyz', degrees=True)


class HelixMotion(TrajectoryGenerator):
    """Generate helical trajectories with various velocity profiles"""
    
    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to the given vector"""
        v = np.array(v)  # Ensure it's a numpy array
        if abs(v[0]) < 0.9:
            return np.cross(v, [1, 0, 0]) / np.linalg.norm(np.cross(v, [1, 0, 0]))
        else:
            return np.cross(v, [0, 1, 0]) / np.linalg.norm(np.cross(v, [0, 1, 0]))
    
    def generate_helix_with_profile(self,
                                   center: List[float],
                                   radius: float,
                                   pitch: float,
                                   height: float,
                                   axis: List[float] = [0, 0, 1],
                                   duration: float = 4.0,
                                   trajectory_type: str = 'cubic',
                                   jerk_limit: Optional[float] = None,
                                   start_point: Optional[List[float]] = None,
                                   clockwise: bool = False) -> np.ndarray:
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
        if trajectory_type == 'cubic':
            return self.generate_cubic_helix(
                center, radius, pitch, height, axis, 
                duration, start_point, clockwise
            )
        elif trajectory_type == 'quintic':
            return self.generate_quintic_helix(
                center, radius, pitch, height, axis,
                duration, start_point, clockwise
            )
        elif trajectory_type == 's_curve':
            return self.generate_scurve_helix(
                center, radius, pitch, height, axis,
                duration, jerk_limit, start_point, clockwise
            )
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    def generate_cubic_helix(self,
                           center: List[float],
                           radius: float,
                           pitch: float,
                           height: float,
                           axis: List[float] = [0, 0, 1],
                           duration: float = 4.0,
                           start_point: Optional[List[float]] = None,
                           clockwise: bool = False) -> np.ndarray:
        """
        Generate helix with cubic (linear) interpolation.
        This is the simplest profile with constant angular velocity.
        """
        # Calculate number of revolutions
        num_revolutions = height / pitch if pitch > 0 else 1
        total_angle = 2 * np.pi * num_revolutions
        
        # Setup coordinate system
        axis_np = np.array(axis)
        axis = axis_np / np.linalg.norm(axis_np)
        u = self._get_perpendicular_vector(axis)
        v = np.cross(axis, u)
        center_np = np.array(center)
        
        # Determine starting angle if start_point provided
        start_angle = 0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, axis) * axis
            
            if np.linalg.norm(to_start_plane) > 0.001:
                to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
                start_angle = np.arctan2(
                    np.dot(to_start_normalized, v),
                    np.dot(to_start_normalized, u)
                )
        
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
    
    def generate_quintic_helix(self,
                              center: List[float],
                              radius: float,
                              pitch: float,
                              height: float,
                              axis: List[float] = [0, 0, 1],
                              duration: float = 4.0,
                              start_point: Optional[List[float]] = None,
                              clockwise: bool = False) -> np.ndarray:
        """
        Generate helix with quintic polynomial profile.
        Provides smooth acceleration and deceleration.
        """
        # Calculate total angle
        num_revolutions = height / pitch if pitch > 0 else 1
        total_angle = 2 * np.pi * num_revolutions
        
        # Setup coordinate system
        axis_np = np.array(axis)
        axis = axis_np / np.linalg.norm(axis_np)
        u = self._get_perpendicular_vector(axis)
        v = np.cross(axis, u)
        center_np = np.array(center)
        
        # Determine starting angle
        start_angle = 0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, axis) * axis
            
            if np.linalg.norm(to_start_plane) > 0.001:
                to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
                start_angle = np.arctan2(
                    np.dot(to_start_normalized, v),
                    np.dot(to_start_normalized, u)
                )
        
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
    
    def generate_scurve_helix(self,
                             center: List[float],
                             radius: float,
                             pitch: float,
                             height: float,
                             axis: List[float] = [0, 0, 1],
                             duration: float = 4.0,
                             jerk_limit: Optional[float] = None,
                             start_point: Optional[List[float]] = None,
                             clockwise: bool = False) -> np.ndarray:
        """
        Generate helix with S-curve (smoothstep) profile.
        Provides jerk-limited motion.
        """
        # Calculate total angle
        num_revolutions = height / pitch if pitch > 0 else 1
        total_angle = 2 * np.pi * num_revolutions
        
        # Setup coordinate system
        axis_np = np.array(axis)
        axis = axis_np / np.linalg.norm(axis_np)
        u = self._get_perpendicular_vector(axis)
        v = np.cross(axis, u)
        center_np = np.array(center)
        
        # Determine starting angle
        start_angle = 0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, axis) * axis
            
            if np.linalg.norm(to_start_plane) > 0.001:
                to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
                start_angle = np.arctan2(
                    np.dot(to_start_normalized, v),
                    np.dot(to_start_normalized, u)
                )
        
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


class SplineMotion(TrajectoryGenerator):
    """Generate smooth spline trajectories through waypoints"""
    
    def generate_quintic_spline_true(self,
                                     waypoints: List[List[float]],
                                     waypoint_behavior: str = 'stop',
                                     profile_type: str = 's_curve',
                                     optimization: str = 'jerk',
                                     time_optimal: bool = False,
                                     jerk_limit: Optional[float] = None) -> np.ndarray:
        """
        Generate true quintic spline trajectory with optional S-curve profiles.
        
        This is the enhanced version using our new quintic polynomial implementation
        instead of the misleading cubic-based version.
        
        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            waypoint_behavior: 'stop' or 'continuous' at waypoints
            profile_type: 's_curve', 'quintic', or 'cubic'
            optimization: 'time', 'jerk', or 'energy'
            time_optimal: Calculate minimum time if True
            
        Returns:
            Array of interpolated poses with smooth acceleration
        """
        if profile_type == 's_curve':
            return self._generate_scurve_waypoints(waypoints, waypoint_behavior, optimization, jerk_limit)
        elif profile_type == 'quintic':
            return self._generate_pure_quintic_waypoints(waypoints, waypoint_behavior)
        else:
            # Fall back to existing cubic implementation
            return self.generate_cubic_spline(waypoints)
    
    def _generate_pure_quintic_waypoints(self, waypoints, behavior):
        """Generate quintic trajectories between waypoints."""
        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)
        
        if num_waypoints < 2:
            return waypoints
        
        # Calculate timing for each segment
        segment_times = []
        for i in range(num_waypoints - 1):
            distance = np.linalg.norm(waypoints[i+1, :3] - waypoints[i, :3])
            # Estimate time based on average velocity
            time = max(1.0, distance / 50.0)  # 50 mm/s average
            segment_times.append(time)
        
        # Generate trajectory segments
        full_trajectory = []
        prev_vf = None
        
        for i in range(num_waypoints - 1):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]
            T = segment_times[i]
            
            # Determine boundary velocities based on behavior
            if behavior == 'stop':
                v0 = [0] * 6
                vf = [0] * 6
            else:  # continuous
                # Calculate velocities for smooth transition
                if i == 0:
                    v0 = [0] * 6
                else:
                    # Use previous segment's final velocity
                    v0 = prev_vf if prev_vf is not None else [0] * 6
                
                if i == num_waypoints - 2:
                    vf = [0] * 6
                else:
                    # Calculate velocity toward next waypoint using correct segment timing
                    # Use the NEXT segment's time, not current segment time
                    next_direction = waypoints[i+2] - waypoints[i+1]
                    next_segment_time = segment_times[i+1] if (i+1) < len(segment_times) else segment_times[i]
                    # Use tangent-based velocity for smoother continuity
                    # Average the incoming and outgoing directions for smooth transition
                    current_direction = waypoints[i+1] - waypoints[i]
                    avg_direction = (current_direction / segment_times[i] + next_direction / next_segment_time) * 0.5
                    vf = list(avg_direction[:6] * 0.7)  # Scale factor for stability
                
                prev_vf = vf
            
            # Create multi-axis quintic trajectory
            segment_traj = MultiAxisQuinticTrajectory(
                list(start_pose), list(end_pose), v0, vf, T=T
            )
            
            # Sample the segment
            segment_points = segment_traj.get_trajectory_points(self.dt)
            
            # Add to full trajectory (avoid duplicating waypoints)
            if i == 0:
                full_trajectory.extend(segment_points['position'])
            else:
                full_trajectory.extend(segment_points['position'][1:])
        
        return np.array(full_trajectory)
    
    def _generate_scurve_waypoints(self, waypoints, behavior, optimization, jerk_limit=None):
        """Generate S-curve trajectories between waypoints."""
        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)
        
        if num_waypoints < 2:
            return waypoints
        
        full_trajectory = []
        
        for i in range(num_waypoints - 1):
            # Calculate segment parameters
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]
            
            # For each joint, calculate S-curve profile
            segment_trajectories = []
            max_time = 0
            
            for j in range(6):  # 6 joints
                distance = end_pose[j] - start_pose[j]
                
                # Get joint constraints
                constraints = self.constraints.get_joint_constraints(j)
                
                # Use provided jerk_limit if available, otherwise use constraints
                j_max = jerk_limit if jerk_limit is not None else constraints['j_max']
                
                # Create S-curve profile
                scurve = SCurveProfile(
                    distance,
                    constraints['v_max'],
                    constraints['a_max'],
                    j_max
                )
                
                max_time = max(max_time, scurve.get_total_time())
                segment_trajectories.append(scurve)
            
            # Synchronize all joints to slowest
            if optimization == 'time':
                # Use calculated minimum time
                sync_time = max_time
            else:
                # Add margin for smoother motion
                sync_time = max_time * 1.2
            
            # Generate synchronized points
            timestamps = self.generate_timestamps(sync_time)
            
            for t in timestamps:
                pose = []
                for j in range(6):
                    # Each joint should complete at sync_time
                    # Scale time proportionally for proper synchronization
                    joint_scurve = segment_trajectories[j]
                    # Ensure we don't exceed the joint's actual profile duration
                    t_normalized = t / sync_time  # Normalize to [0, 1]
                    t_joint = t_normalized * joint_scurve.get_total_time()
                    
                    values = joint_scurve.evaluate_at_time(t_joint)
                    pose.append(start_pose[j] + values['position'])
                
                # Add orientation components (simplified for now)
                if len(start_pose) > 6:
                    for k in range(3, 6):
                        # Linear interpolation for orientation
                        alpha = t / sync_time
                        pose.append(start_pose[k+3] * (1-alpha) + end_pose[k+3] * alpha)
                
                full_trajectory.append(pose)
        
        return np.array(full_trajectory)
    
    def generate_cubic_spline(self,
                             waypoints: List[List[float]],
                             timestamps: Optional[List[float]] = None,
                             velocity_start: Optional[List[float]] = None,
                             velocity_end: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate cubic spline trajectory through waypoints
        
        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            timestamps: Time for each waypoint (auto-generated if None)
            velocity_start: Initial velocity (zero if None)
            velocity_end: Final velocity (zero if None)
            
        Returns:
            Array of interpolated poses
        """
        waypoints = np.array(waypoints)
        num_waypoints = len(waypoints)
        
        # Auto-generate timestamps if not provided
        if timestamps is None:
            # Estimate based on distance
            total_dist = 0
            for i in range(1, num_waypoints):
                dist = np.linalg.norm(waypoints[i, :3] - waypoints[i-1, :3])
                total_dist += dist
            
            # Assume average speed of 50 mm/s
            total_time = total_dist / 50.0
            timestamps = np.linspace(0, total_time, num_waypoints)
        
        # Position trajectory splines
        # Validate array dimensions before creating splines
        if len(timestamps) != len(waypoints):
            raise ValueError(f"Timestamps length ({len(timestamps)}) must match waypoints length ({len(waypoints)})")
        
        pos_splines = []
        for i in range(3):
            bc_type = 'not-a-knot'  # Default boundary condition
            
            # Apply velocity boundary conditions if specified
            if velocity_start is not None and velocity_end is not None:
                bc_type = ((1, velocity_start[i]), (1, velocity_end[i]))
            
            spline = CubicSpline(timestamps, waypoints[:, i], bc_type=bc_type)
            pos_splines.append(spline)
        
        # Orientation trajectory splines
        rotations = [Rotation.from_euler('xyz', wp[3:], degrees=True) for wp in waypoints]
        # Stack quaternions for scipy 1.11.4 compatibility
        quats = np.array([r.as_quat() for r in rotations])
        key_rots = Rotation.from_quat(quats)
        slerp = Slerp(timestamps, key_rots)
        
        # Generate dense trajectory
        t_eval = self.generate_timestamps(timestamps[-1])
        trajectory = []
        
        for t in t_eval:
            # Evaluate position splines
            pos = [spline(t) for spline in pos_splines]
            
            # Evaluate orientation
            rot = slerp(t)
            orient = rot.as_euler('xyz', degrees=True)
            
            trajectory.append(np.concatenate([pos, orient]))
        
        return np.array(trajectory)
    
    def generate_quintic_spline(self,
                               waypoints: List[List[float]],
                               timestamps: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate quintic (5th order) spline with zero velocity and acceleration at endpoints
        
        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            timestamps: Time for each waypoint
            
        Returns:
            Array of interpolated poses
        """
        # Quintic spline boundary conditions
        # at the endpoints for smooth motion
        return self.generate_cubic_spline(
            waypoints, 
            timestamps,
            velocity_start=[0, 0, 0],
            velocity_end=[0, 0, 0]
        )
    
    def generate_scurve_spline(self,
                               waypoints: List[List[float]],
                               duration: Optional[float] = None,
                               jerk_limit: float = 1000.0,
                               timestamps: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate S-curve spline with jerk-limited motion profile
        
        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz]
            duration: Total duration for the trajectory (optional)
            jerk_limit: Maximum jerk value in mm/s^3
            timestamps: Time for each waypoint (optional)
            
        Returns:
            Array of interpolated poses with S-curve velocity profile
        """
        # First generate a cubic spline through the waypoints
        basic_trajectory = self.generate_cubic_spline(
            waypoints,
            timestamps,
            velocity_start=[0, 0, 0],
            velocity_end=[0, 0, 0]
        )
        
        if len(basic_trajectory) < 2:
            return basic_trajectory
        
        # Calculate total path length
        path_length = 0
        for i in range(1, len(basic_trajectory)):
            segment_length = np.linalg.norm(
                np.array(basic_trajectory[i][:3]) - np.array(basic_trajectory[i-1][:3])
            )
            path_length += segment_length
        
        if path_length < 0.001:  # Path too short
            return basic_trajectory
        
        # Determine duration if not provided
        if duration is None:
            # Estimate based on path length and conservative speed
            avg_speed = 50.0  # mm/s conservative speed
            duration = path_length / avg_speed
        
        # Generate S-curve profile for the motion
        num_points = int(duration * self.control_rate)
        if num_points < 2:
            num_points = 2
        
        # Create S-curve time parameterization
        time_points = np.linspace(0, duration, num_points)
        s_curve_params = []
        
        for t in time_points:
            # Simple S-curve profile (smoothstep)
            tau = t / duration
            # Smooth acceleration and deceleration
            s = tau * tau * (3.0 - 2.0 * tau)
            s_curve_params.append(s)
        
        # Re-sample the trajectory according to S-curve profile
        original_indices = np.linspace(0, len(basic_trajectory) - 1, len(basic_trajectory))
        new_indices = np.array(s_curve_params) * (len(basic_trajectory) - 1)
        
        # Interpolate each dimension
        resampled_trajectory = []
        for new_idx in new_indices:
            if new_idx <= 0:
                resampled_trajectory.append(basic_trajectory[0])
            elif new_idx >= len(basic_trajectory) - 1:
                resampled_trajectory.append(basic_trajectory[-1])
            else:
                # Linear interpolation between points
                lower_idx = int(new_idx)
                upper_idx = min(lower_idx + 1, len(basic_trajectory) - 1)
                alpha = new_idx - lower_idx
                
                lower_point = np.array(basic_trajectory[lower_idx])
                upper_point = np.array(basic_trajectory[upper_idx])
                interpolated = lower_point + alpha * (upper_point - lower_point)
                resampled_trajectory.append(interpolated.tolist())
        
        return np.array(resampled_trajectory)

class MotionBlender:
    """Blend between different motion segments for smooth transitions"""
    
    def __init__(self, blend_time: float = 0.5):
        self.blend_time = blend_time
        
    def blend_trajectories(self, traj1, traj2, blend_samples=50):
        """Blend two trajectory segments with improved velocity continuity"""
        
        if blend_samples < 4:
            return np.vstack([traj1, traj2])
        
        # Use more samples for smoother blending
        blend_samples = max(blend_samples, 20)  # Minimum 20 samples for smooth blend
        
        # Trajectory overlap region analysis
        overlap_start = max(0, len(traj1) - blend_samples // 3)
        overlap_end = min(len(traj2), blend_samples // 3)
        
        # Extract blend region
        blend_start_pose = traj1[overlap_start] if overlap_start < len(traj1) else traj1[-1]
        blend_end_pose = traj2[overlap_end] if overlap_end < len(traj2) else traj2[0]
        
        # Generate smooth transition using S-curve
        blended = []
        for i in range(blend_samples):
            t = i / (blend_samples - 1)
            # Use smoothstep function for smoother acceleration
            s = t * t * (3 - 2 * t)  # Smoothstep
            
            # Blend position
            pos_blend = blend_start_pose * (1 - s) + blend_end_pose * s
            
            # Orientation interpolation via SLERP
            r1 = Rotation.from_euler('xyz', blend_start_pose[3:], degrees=True)
            r2 = Rotation.from_euler('xyz', blend_end_pose[3:], degrees=True)
            key_rots = Rotation.from_quat([r1.as_quat(), r2.as_quat()])
            slerp = Slerp([0, 1], key_rots)
            orient_blend = slerp(s).as_euler('xyz', degrees=True)
            
            pos_blend[3:] = orient_blend
            blended.append(pos_blend)
        
        # Combine with better overlap handling
        result = np.vstack([
            traj1[:overlap_start],
            np.array(blended),
            traj2[overlap_end:]
        ])
        
        return result


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
            'quintic': self._blend_quintic,
            's_curve': self._blend_scurve,
            'smoothstep': self._blend_smoothstep,  # Legacy compatibility
            'minimum_jerk': self._blend_minimum_jerk,
            'cubic': self._blend_cubic
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
            # Not enough points for derivative calculation
            pos = trajectory[index]
            vel = np.zeros_like(pos)
            acc = np.zeros_like(pos)
            return pos, vel, acc
        
        # Get position
        pos = trajectory[index].copy()
        
        # Calculate velocity using finite differences
        if index == 0 or (index == -1 and len(trajectory) == 1):
            # Forward difference at start
            vel = (trajectory[1] - trajectory[0]) / self.dt
        elif index == -1 or index == len(trajectory) - 1:
            # Backward difference at end
            vel = (trajectory[-1] - trajectory[-2]) / self.dt
        else:
            # Central difference in middle
            vel = (trajectory[index + 1] - trajectory[index - 1]) / (2 * self.dt)
        
        # Calculate acceleration
        if len(trajectory) < 3:
            acc = np.zeros_like(pos)
        elif index == 0:
            # Forward difference
            v1 = (trajectory[1] - trajectory[0]) / self.dt
            v2 = (trajectory[2] - trajectory[1]) / self.dt
            acc = (v2 - v1) / self.dt
        elif index == -1 or index == len(trajectory) - 1:
            # Backward difference
            v1 = (trajectory[-2] - trajectory[-3]) / self.dt
            v2 = (trajectory[-1] - trajectory[-2]) / self.dt
            acc = (v2 - v1) / self.dt
        else:
            # Central difference
            acc = (trajectory[index + 1] - 2 * trajectory[index] + trajectory[index - 1]) / (self.dt ** 2)
        
        return pos, vel, acc
    
    def calculate_blend_region_size(self, traj1: np.ndarray, traj2: np.ndarray, 
                                   max_accel: float = 1000.0) -> int:
        """
        Calculate optimal blend region size based on velocity mismatch.
        
        Args:
            traj1: First trajectory
            traj2: Second trajectory
            max_accel: Maximum allowed acceleration (mm/s² or deg/s²)
            
        Returns:
            Number of samples for blend region
        """
        # Extract end state of first trajectory
        _, v1, _ = self.extract_trajectory_state(traj1, -1)
        
        # Extract start state of second trajectory
        _, v2, _ = self.extract_trajectory_state(traj2, 0)
        
        # Calculate velocity difference
        delta_v = np.linalg.norm(v2[:3] - v1[:3])  # Focus on position components
        
        if delta_v < 1.0:  # Nearly matching velocities
            return 20  # Minimal blend
        
        # Calculate required time for velocity change
        t_blend = delta_v / max_accel
        
        # Add safety factor
        t_blend *= 1.5
        
        # Convert to samples
        blend_samples = int(t_blend * self.sample_rate)
        
        # Apply limits
        blend_samples = max(20, min(blend_samples, 200))
        
        return blend_samples
    
    def solve_quintic_coefficients(self, p0: np.ndarray, pf: np.ndarray,
                                  v0: np.ndarray, vf: np.ndarray,
                                  a0: np.ndarray, af: np.ndarray,
                                  T: float) -> np.ndarray:
        """
        Solve for quintic polynomial coefficients given boundary conditions.
        
        The quintic polynomial is: p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        
        Args:
            p0, pf: Initial and final positions
            v0, vf: Initial and final velocities
            a0, af: Initial and final accelerations
            T: Trajectory duration
            
        Returns:
            6xN array of coefficients [a0, a1, a2, a3, a4, a5] for each dimension
        """
        # Build constraint matrix
        # [p(0), p(T), v(0), v(T), a(0), a(T)] = M * [a0, a1, a2, a3, a4, a5]
        M = np.array([
            [1, 0,   0,      0,       0,        0],       # p(0)
            [1, T,   T**2,   T**3,    T**4,     T**5],    # p(T)
            [0, 1,   0,      0,       0,        0],       # v(0)
            [0, 1,   2*T,    3*T**2,  4*T**3,   5*T**4],  # v(T)
            [0, 0,   2,      0,       0,        0],       # a(0)
            [0, 0,   2,      6*T,     12*T**2,  20*T**3]  # a(T)
        ])
        
        # Solve for each dimension
        num_dims = len(p0)
        coeffs = np.zeros((6, num_dims))
        
        for i in range(num_dims):
            # Build constraint vector for this dimension
            b = np.array([p0[i], pf[i], v0[i], vf[i], a0[i], af[i]])
            
            # Solve linear system
            coeffs[:, i] = np.linalg.solve(M, b)
        
        return coeffs
    
    def evaluate_quintic(self, coeffs: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate quintic polynomial at time t.
        
        Returns position, velocity, and acceleration.
        """
        # Position: p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        pos = (coeffs[0] + coeffs[1] * t + coeffs[2] * t**2 + 
               coeffs[3] * t**3 + coeffs[4] * t**4 + coeffs[5] * t**5)
        
        # Velocity: v(t) = a1 + 2*a2*t + 3*a3*t² + 4*a4*t³ + 5*a5*t⁴
        vel = (coeffs[1] + 2 * coeffs[2] * t + 3 * coeffs[3] * t**2 + 
               4 * coeffs[4] * t**3 + 5 * coeffs[5] * t**4)
        
        # Acceleration: a(t) = 2*a2 + 6*a3*t + 12*a4*t² + 20*a5*t³
        acc = (2 * coeffs[2] + 6 * coeffs[3] * t + 
               12 * coeffs[4] * t**2 + 20 * coeffs[5] * t**3)
        
        return pos, vel, acc
    
    def _blend_quintic(self, traj1: np.ndarray, traj2: np.ndarray, 
                      blend_samples: int) -> np.ndarray:
        """
        Generate quintic polynomial blend with C2 continuity.
        
        This ensures smooth position, velocity, and acceleration transitions.
        """
        # Extract boundary conditions
        p0, v0, a0 = self.extract_trajectory_state(traj1, -1)
        pf, vf, af = self.extract_trajectory_state(traj2, 0)
        
        # Blend duration
        T = blend_samples * self.dt
        
        # Solve for quintic coefficients
        coeffs = self.solve_quintic_coefficients(p0, pf, v0, vf, a0, af, T)
        
        # Generate blend trajectory
        blend_traj = []
        for i in range(blend_samples):
            t = i * T / (blend_samples - 1) if blend_samples > 1 else 0
            pos, _, _ = self.evaluate_quintic(coeffs, t)
            blend_traj.append(pos)
        
        return np.array(blend_traj)
    
    def _blend_scurve(self, traj1: np.ndarray, traj2: np.ndarray,
                     blend_samples: int) -> np.ndarray:
        """
        Generate S-curve blend with jerk limiting.
        """
        # Extract boundary conditions
        p0, v0, _ = self.extract_trajectory_state(traj1, -1)
        pf, vf, _ = self.extract_trajectory_state(traj2, 0)
        
        # Use MultiAxisSCurveTrajectory if available
        try:
            scurve = MultiAxisSCurveTrajectory(
                p0[:6], pf[:6],  # Use first 6 DOF
                v0=v0[:6], vf=vf[:6],
                T=blend_samples * self.dt,
                jerk_limit=5000  # Default jerk limit
            )
            points = scurve.get_trajectory_points(self.dt)
            
            # Convert to full pose format
            blend_traj = []
            for i in range(len(points['position'])):
                pose = np.zeros(len(p0))
                pose[:6] = points['position'][i]
                if len(p0) > 6:
                    # Linear interpolation for additional DOF
                    alpha = i / (len(points['position']) - 1)
                    pose[6:] = p0[6:] * (1 - alpha) + pf[6:] * alpha
                blend_traj.append(pose)
            
            return np.array(blend_traj)
        except:
            # Fallback to quintic if S-curve not available
            return self._blend_quintic(traj1, traj2, blend_samples)
    
    def _blend_smoothstep(self, traj1: np.ndarray, traj2: np.ndarray,
                         blend_samples: int) -> np.ndarray:
        """
        Legacy smoothstep blend for backward compatibility.
        """
        # Extract blend points
        p0 = traj1[-1] if len(traj1) > 0 else traj2[0]
        pf = traj2[0] if len(traj2) > 0 else traj1[-1]
        
        blend_traj = []
        for i in range(blend_samples):
            t = i / (blend_samples - 1) if blend_samples > 1 else 0
            # Smoothstep function
            s = t * t * (3 - 2 * t)
            
            # Interpolate position
            pos = p0 * (1 - s) + pf * s
            blend_traj.append(pos)
        
        return np.array(blend_traj)
    
    def _blend_minimum_jerk(self, traj1: np.ndarray, traj2: np.ndarray,
                           blend_samples: int) -> np.ndarray:
        """
        Minimum jerk trajectory blend.
        
        Uses the minimum jerk trajectory equation for smooth motion.
        """
        # Extract boundary conditions
        p0, v0, a0 = self.extract_trajectory_state(traj1, -1)
        pf, vf, af = self.extract_trajectory_state(traj2, 0)
        
        T = blend_samples * self.dt
        blend_traj = []
        
        for i in range(blend_samples):
            tau = i / (blend_samples - 1) if blend_samples > 1 else 0
            
            # Minimum jerk trajectory equation
            pos = p0 + (pf - p0) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            
            blend_traj.append(pos)
        
        return np.array(blend_traj)
    
    def _blend_cubic(self, traj1: np.ndarray, traj2: np.ndarray,
                    blend_samples: int) -> np.ndarray:
        """
        Cubic spline blend with C1 continuity.
        """
        # Extract boundary conditions
        p0, v0, _ = self.extract_trajectory_state(traj1, -1)
        pf, vf, _ = self.extract_trajectory_state(traj2, 0)
        
        T = blend_samples * self.dt
        
        # Cubic coefficients (4 constraints: p0, pf, v0, vf)
        # p(t) = a0 + a1*t + a2*t² + a3*t³
        num_dims = len(p0)
        blend_traj = []
        
        for i in range(blend_samples):
            t = i * T / (blend_samples - 1) if blend_samples > 1 else 0
            tau = t / T if T > 0 else 0
            
            # Hermite cubic interpolation
            h00 = 2 * tau**3 - 3 * tau**2 + 1
            h10 = tau**3 - 2 * tau**2 + tau
            h01 = -2 * tau**3 + 3 * tau**2
            h11 = tau**3 - tau**2
            
            pos = h00 * p0 + h10 * T * v0 + h01 * pf + h11 * T * vf
            blend_traj.append(pos)
        
        return np.array(blend_traj)
    
    def blend_trajectories(self, traj1: np.ndarray, traj2: np.ndarray,
                          method: str = 'quintic',
                          blend_samples: Optional[int] = None,
                          auto_size: bool = True) -> np.ndarray:
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
        # Validate inputs
        if len(traj1) == 0 and len(traj2) == 0:
            return np.array([])
        if len(traj1) == 0:
            return traj2
        if len(traj2) == 0:
            return traj1
        
        # Calculate blend region size
        if blend_samples is None or auto_size:
            blend_samples = self.calculate_blend_region_size(traj1, traj2)
        
        blend_samples = max(4, blend_samples)  # Minimum 4 samples
        
        # Select blend method
        if method not in self.blend_methods:
            print(f"[WARNING] Unknown blend method '{method}', using quintic")
            method = 'quintic'
        
        blend_func = self.blend_methods[method]
        
        # Generate blend trajectory
        blend_traj = blend_func(traj1, traj2, blend_samples)
        
        # Calculate overlap regions to avoid duplication
        overlap_start = max(0, len(traj1) - 1)  # Skip last point of traj1
        overlap_end = min(1, len(traj2))  # Skip first point of traj2
        
        # Combine trajectories
        result = np.vstack([
            traj1[:overlap_start],
            blend_traj,
            traj2[overlap_end:]
        ])
        
        return result


class WaypointTrajectoryPlanner:
    """
    Trajectory planner for smooth motion through waypoints with corner cutting.
    
    Implements mstraj-style parabolic blending at waypoints to avoid acceleration
    spikes and ensure smooth motion through complex paths.
    """
    
    def __init__(self, waypoints: List[List[float]], constraints: Optional[Dict] = None,
                 sample_rate: float = 100.0):
        """
        Initialize waypoint trajectory planner.
        
        Args:
            waypoints: List of waypoint poses [x, y, z, rx, ry, rz]
            constraints: Motion constraints (max_velocity, max_acceleration, max_jerk)
            sample_rate: Trajectory sampling rate in Hz
        """
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.num_waypoints = len(waypoints)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Default constraints
        default_constraints = {
            'max_velocity': 100.0,      # mm/s
            'max_acceleration': 500.0,   # mm/s²
            'max_jerk': 5000.0          # mm/s³
        }
        self.constraints = constraints if constraints else default_constraints
        
        # Blend planning data
        self.blend_radii = []
        self.blend_regions = []
        self.segment_velocities = []
        self.via_modes = ['via'] * self.num_waypoints  # Default: pass through all
        
    def calculate_corner_angle(self, idx: int) -> float:
        """
        Calculate the angle between incoming and outgoing path segments.
        
        Args:
            idx: Waypoint index
            
        Returns:
            Angle in radians (0 to π)
        """
        if idx <= 0 or idx >= self.num_waypoints - 1:
            return 0.0  # No corner at start/end
        
        # Vectors for path segments
        v_in = self.waypoints[idx] - self.waypoints[idx - 1]
        v_out = self.waypoints[idx + 1] - self.waypoints[idx]
        
        # Normalize (use only position components)
        v_in_norm = v_in[:3] / (np.linalg.norm(v_in[:3]) + 1e-9)
        v_out_norm = v_out[:3] / (np.linalg.norm(v_out[:3]) + 1e-9)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(v_in_norm, v_out_norm), -1, 1)
        angle = np.arccos(cos_angle)
        
        return angle
    
    def calculate_safe_blend_radius(self, idx: int, approach_velocity: float) -> float:
        """
        Calculate maximum safe blend radius for a waypoint.
        
        Args:
            idx: Waypoint index
            approach_velocity: Velocity approaching the waypoint
            
        Returns:
            Safe blend radius in mm
        """
        angle = self.calculate_corner_angle(idx)
        
        if angle < 0.01:  # Nearly straight (< 0.5 degrees)
            return 0.0
        
        # Dynamic blend radius based on velocity and angle
        a_max = self.constraints['max_acceleration']
        
        # Centripetal acceleration constraint
        # r = v² / (a_max * sin(θ/2))
        sin_half_angle = np.sin(angle / 2)
        if sin_half_angle > 0:
            r_dynamic = (approach_velocity ** 2) / (a_max * sin_half_angle)
        else:
            r_dynamic = 0.0
        
        # Geometric constraint - don't exceed segment lengths
        r_geometric = self.get_max_geometric_radius(idx)
        
        # Apply safety factor and limits
        r_safe = min(r_dynamic, r_geometric) * 0.7  # 70% safety factor
        r_safe = max(0, min(r_safe, 100))  # Cap at 100mm
        
        return r_safe
    
    def get_max_geometric_radius(self, idx: int) -> float:
        """
        Get maximum blend radius based on segment geometry.
        
        Args:
            idx: Waypoint index
            
        Returns:
            Maximum geometric radius in mm
        """
        if idx <= 0 or idx >= self.num_waypoints - 1:
            return 0.0
        
        # Distance to previous waypoint
        dist_prev = np.linalg.norm(
            self.waypoints[idx][:3] - self.waypoints[idx - 1][:3]
        )
        
        # Distance to next waypoint
        dist_next = np.linalg.norm(
            self.waypoints[idx + 1][:3] - self.waypoints[idx][:3]
        )
        
        # Maximum radius is 40% of shortest segment
        max_radius = 0.4 * min(dist_prev, dist_next)
        
        return max_radius
    
    def calculate_auto_blend_radii(self):
        """
        Automatically calculate blend radius for each waypoint.
        """
        self.blend_radii = []
        
        for i in range(self.num_waypoints):
            if i == 0 or i == self.num_waypoints - 1:
                # No blending at start/end
                self.blend_radii.append(0.0)
            else:
                # Estimate approach velocity
                segment_length = np.linalg.norm(
                    self.waypoints[i][:3] - self.waypoints[i - 1][:3]
                )
                
                # Simple velocity estimation
                if segment_length > 0:
                    approach_velocity = min(
                        self.constraints['max_velocity'],
                        np.sqrt(2 * self.constraints['max_acceleration'] * segment_length)
                    )
                else:
                    approach_velocity = 0
                
                # Calculate safe radius
                radius = self.calculate_safe_blend_radius(i, approach_velocity)
                self.blend_radii.append(radius)
    
    def compute_blend_points(self, idx: int, blend_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate blend entry and exit points for a waypoint.
        
        Args:
            idx: Waypoint index
            blend_radius: Blend radius in mm
            
        Returns:
            Tuple of (entry_point, exit_point)
        """
        if blend_radius <= 0 or idx <= 0 or idx >= self.num_waypoints - 1:
            return self.waypoints[idx], self.waypoints[idx]
        
        # Get path vectors
        v_in = self.waypoints[idx] - self.waypoints[idx - 1]
        v_out = self.waypoints[idx + 1] - self.waypoints[idx]
        
        # Normalize position components
        v_in_norm = np.zeros_like(v_in)
        v_out_norm = np.zeros_like(v_out)
        
        v_in_norm[:3] = v_in[:3] / (np.linalg.norm(v_in[:3]) + 1e-9)
        v_out_norm[:3] = v_out[:3] / (np.linalg.norm(v_out[:3]) + 1e-9)
        
        # Calculate blend entry point (along incoming path)
        blend_entry = self.waypoints[idx].copy()
        blend_entry[:3] -= v_in_norm[:3] * blend_radius
        
        # Calculate blend exit point (along outgoing path)
        blend_exit = self.waypoints[idx].copy()
        blend_exit[:3] += v_out_norm[:3] * blend_radius
        
        # Interpolate orientations
        angle = self.calculate_corner_angle(idx)
        if angle > 0:
            # Weighted average for orientations at blend points
            alpha_entry = 1.0 - (blend_radius / np.linalg.norm(v_in[:3]))
            alpha_exit = blend_radius / np.linalg.norm(v_out[:3])
            
            blend_entry[3:] = (self.waypoints[idx - 1][3:] * (1 - alpha_entry) + 
                              self.waypoints[idx][3:] * alpha_entry)
            blend_exit[3:] = (self.waypoints[idx][3:] * (1 - alpha_exit) + 
                             self.waypoints[idx + 1][3:] * alpha_exit)
        
        return blend_entry, blend_exit
    
    def generate_parabolic_blend(self, entry_point: np.ndarray, exit_point: np.ndarray,
                                v_entry: np.ndarray, v_exit: np.ndarray,
                                blend_time: Optional[float] = None) -> List[np.ndarray]:
        """
        Generate parabolic trajectory for corner blend with acceleration limits.
        
        Parabolic blends have constant acceleration, providing smooth motion.
        
        Args:
            entry_point: Blend entry position
            exit_point: Blend exit position
            v_entry: Velocity at blend entry
            v_exit: Velocity at blend exit
            blend_time: Blend duration (auto-calculated if None)
            
        Returns:
            List of trajectory points through the blend
        """
        # Calculate blend parameters
        delta_p = exit_point - entry_point
        distance = np.linalg.norm(delta_p[:3])
        
        if distance < 1e-6:
            return [entry_point]  # No blend needed
        
        # Calculate velocity change needed
        delta_v = v_exit - v_entry
        delta_v_mag = np.linalg.norm(delta_v[:3])
        
        # Calculate minimum blend time based on acceleration constraint
        # We need: |a| = |delta_v|/t <= max_acceleration
        # Therefore: t >= |delta_v|/max_acceleration
        min_blend_time = delta_v_mag / self.constraints['max_acceleration']
        
        # Calculate blend time if not specified
        if blend_time is None:
            # Option 1: Time based on average velocity
            v_avg = (np.linalg.norm(v_entry[:3]) + np.linalg.norm(v_exit[:3])) / 2
            if v_avg > 0:
                time_from_velocity = distance / v_avg
            else:
                time_from_velocity = np.sqrt(2 * distance / self.constraints['max_acceleration'])
            
            # Use the larger of the two times to respect acceleration limits
            blend_time = max(min_blend_time, time_from_velocity)
        else:
            # Enforce minimum time even if specified
            blend_time = max(blend_time, min_blend_time)
        
        # Ensure minimum blend time for numerical stability
        blend_time = max(blend_time, 0.01)  # At least 10ms
        
        # Calculate acceleration (now guaranteed within limits)
        a_blend = delta_v / blend_time
        
        # Verify acceleration is within limits (should be by construction)
        a_mag = np.linalg.norm(a_blend[:3])
        if a_mag > self.constraints['max_acceleration'] * 1.1:  # 10% tolerance
            # Scale down acceleration if needed
            scale = self.constraints['max_acceleration'] / a_mag
            a_blend = a_blend * scale
            blend_time = blend_time / scale  # Adjust time accordingly
        
        # Generate trajectory using cubic hermite interpolation
        # This guarantees position and velocity boundary conditions
        num_points = max(5, int(blend_time * self.sample_rate))  # At least 5 points
        blend_traj = []
        
        # Use cubic hermite spline which guarantees C1 continuity
        for i in range(num_points):
            # Normalized time from 0 to 1
            s = i / (num_points - 1) if num_points > 1 else 0
            
            # Cubic hermite basis functions
            h00 = 2*s**3 - 3*s**2 + 1  # Blend function for start position
            h10 = s**3 - 2*s**2 + s     # Blend function for start velocity
            h01 = -2*s**3 + 3*s**2      # Blend function for end position
            h11 = s**3 - s**2            # Blend function for end velocity
            
            # Interpolate position using hermite spline
            # Scale velocities by blend_time to get tangents
            pos = (h00 * entry_point + 
                   h10 * (v_entry * blend_time) + 
                   h01 * exit_point + 
                   h11 * (v_exit * blend_time))
            
            blend_traj.append(pos)
        
        return blend_traj
    
    def generate_linear_segment(self, start: np.ndarray, end: np.ndarray,
                               velocity: Optional[float] = None) -> List[np.ndarray]:
        """
        Generate linear trajectory segment between two points.
        
        Args:
            start: Start position
            end: End position
            velocity: Desired velocity (uses max if None)
            
        Returns:
            List of trajectory points
        """
        distance = np.linalg.norm(end[:3] - start[:3])
        
        if distance < 1e-6:
            return [start]
        
        # Determine velocity
        if velocity is None:
            velocity = self.constraints['max_velocity']
        
        # Calculate duration and number of points
        duration = distance / velocity
        num_points = max(2, int(duration * self.sample_rate))
        
        # Generate trajectory
        segment = []
        for i in range(num_points):
            alpha = i / (num_points - 1) if num_points > 1 else 0
            pos = start * (1 - alpha) + end * alpha
            segment.append(pos)
        
        return segment
    
    def compute_blend_regions(self):
        """
        Compute all blend regions for the trajectory.
        """
        self.blend_regions = []
        
        for i in range(1, self.num_waypoints - 1):
            if self.blend_radii[i] > 0 and self.via_modes[i] == 'via':
                entry, exit = self.compute_blend_points(i, self.blend_radii[i])
                
                # Calculate velocities at blend points
                v_entry_dir = self.waypoints[i] - self.waypoints[i - 1]
                v_entry = v_entry_dir[:3] / np.linalg.norm(v_entry_dir[:3]) * self.segment_velocities[i - 1]
                v_entry_full = np.zeros(len(entry))
                v_entry_full[:3] = v_entry
                
                v_exit_dir = self.waypoints[i + 1] - self.waypoints[i]
                v_exit = v_exit_dir[:3] / np.linalg.norm(v_exit_dir[:3]) * self.segment_velocities[i]
                v_exit_full = np.zeros(len(exit))
                v_exit_full[:3] = v_exit
                
                self.blend_regions.append({
                    'waypoint_idx': i,
                    'entry': entry,
                    'exit': exit,
                    'radius': self.blend_radii[i],
                    'v_entry': v_entry_full,
                    'v_exit': v_exit_full
                })
            else:
                # No blend or stop point
                self.blend_regions.append(None)
    
    def plan_trajectory(self, blend_mode: str = 'auto',
                       blend_radii: Optional[List[float]] = None,
                       via_modes: Optional[List[str]] = None,
                       trajectory_type: str = 'cubic',
                       jerk_limit: Optional[float] = None) -> np.ndarray:
        """
        Plan complete trajectory through waypoints with blending.
        
        Args:
            blend_mode: 'auto', 'manual', or 'none'
            blend_radii: Manual blend radii (if blend_mode='manual')
            via_modes: 'via' or 'stop' for each waypoint
            trajectory_type: 'cubic', 'quintic', or 's_curve' velocity profile
            jerk_limit: Maximum jerk for s_curve profile (mm/s^3)
            
        Returns:
            Complete trajectory as numpy array
        """
        if self.num_waypoints < 2:
            return self.waypoints
        
        # Set blend radii
        if blend_mode in ['auto', 'parabolic', 'circular']:
            self.calculate_auto_blend_radii()
        elif blend_mode == 'manual' and blend_radii is not None:
            self.blend_radii = blend_radii
        elif blend_mode == 'none':
            self.blend_radii = [0.0] * self.num_waypoints
        else:
            # Default to auto for unrecognized modes
            self.calculate_auto_blend_radii()
        
        # Set via modes
        if via_modes is not None:
            self.via_modes = via_modes
        
        # Calculate segment velocities
        self.segment_velocities = []
        for i in range(self.num_waypoints - 1):
            segment_length = np.linalg.norm(
                self.waypoints[i + 1][:3] - self.waypoints[i][:3]
            )
            # Simple velocity planning
            if self.via_modes[i] == 'stop' or self.via_modes[i + 1] == 'stop':
                # Trapezoid profile with acceleration
                v_max = min(
                    self.constraints['max_velocity'],
                    np.sqrt(self.constraints['max_acceleration'] * segment_length)
                )
            else:
                v_max = self.constraints['max_velocity']
            
            self.segment_velocities.append(v_max)
        
        # Compute blend regions
        self.compute_blend_regions()
        
        # Generate full trajectory
        full_trajectory = []
        
        for i in range(self.num_waypoints - 1):
            # Determine segment start and end
            if i == 0:
                segment_start = self.waypoints[0]
            else:
                # Check for blend at current waypoint
                blend_region = self.blend_regions[i - 1] if i > 0 and i - 1 < len(self.blend_regions) else None
                if blend_region:
                    segment_start = blend_region['exit']
                else:
                    segment_start = self.waypoints[i]
            
            if i < self.num_waypoints - 2:
                # Check for blend at next waypoint
                blend_region = self.blend_regions[i] if i < len(self.blend_regions) else None
                if blend_region:
                    segment_end = blend_region['entry']
                else:
                    segment_end = self.waypoints[i + 1]
            else:
                segment_end = self.waypoints[i + 1]
            
            # Generate linear segment
            segment = self.generate_linear_segment(
                segment_start, segment_end, self.segment_velocities[i]
            )
            
            # Add segment to trajectory
            if i == 0:
                full_trajectory.extend(segment)
            else:
                # Skip first point to avoid duplication
                full_trajectory.extend(segment[1:])
            
            # Add blend if needed
            if i < len(self.blend_regions) and self.blend_regions[i]:
                blend_region = self.blend_regions[i]
                blend_traj = self.generate_parabolic_blend(
                    blend_region['entry'],
                    blend_region['exit'],
                    blend_region['v_entry'],
                    blend_region['v_exit']
                )
                # Skip first point to avoid duplication
                full_trajectory.extend(blend_traj[1:])
        
        # Apply trajectory profile if not cubic
        trajectory_array = np.array(full_trajectory)
        
        if trajectory_type != 'cubic':
            # Apply velocity profile to the generated trajectory
            trajectory_array = self.apply_velocity_profile(
                trajectory_array, trajectory_type, jerk_limit
            )
        
        return trajectory_array
    
    def apply_velocity_profile(self, trajectory: np.ndarray, 
                              profile_type: str = 'quintic',
                              jerk_limit: Optional[float] = None) -> np.ndarray:
        """
        Apply velocity profile to existing trajectory points.
        
        Instead of re-interpolating with sparse waypoints, this method
        applies the velocity profile to ALL trajectory points, preserving
        the geometric path while adjusting the timing.
        
        Args:
            trajectory: Input trajectory points
            profile_type: 'quintic' or 's_curve' 
            jerk_limit: Maximum jerk for s_curve (mm/s^3)
            
        Returns:
            Trajectory with velocity profile applied
        """
        if len(trajectory) < 2:
            return trajectory
        
        # Calculate cumulative arc length
        arc_lengths = [0.0]
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i][:3] - trajectory[i-1][:3])
            arc_lengths.append(arc_lengths[-1] + dist)
        
        total_length = arc_lengths[-1]
        if total_length < 1e-6:
            return trajectory
        
        # Normalize arc lengths to [0, 1]
        s_values = np.array(arc_lengths) / total_length
        
        # Generate new time mapping based on profile
        num_points = len(trajectory)
        new_trajectory = np.zeros_like(trajectory)
        
        for i in range(num_points):
            # Get normalized position along path
            s = i / (num_points - 1) if num_points > 1 else 0.0
            
            # Apply velocity profile to get new arc length position
            if profile_type == 'quintic':
                # Quintic polynomial: smooth acceleration/deceleration
                s_new = 10 * s**3 - 15 * s**4 + 6 * s**5
            elif profile_type == 's_curve':
                # S-curve (smoothstep): smooth jerk-limited motion
                if s <= 0.0:
                    s_new = 0.0
                elif s >= 1.0:
                    s_new = 1.0
                else:
                    # Smoothstep function for smooth acceleration
                    s_new = s * s * (3.0 - 2.0 * s)
            else:
                # Default to linear (no change)
                s_new = s
            
            # Find the corresponding point on original trajectory
            # Use linear interpolation between trajectory points
            target_arc_length = s_new * total_length
            
            # Find the segment containing this arc length
            for j in range(len(arc_lengths) - 1):
                if arc_lengths[j] <= target_arc_length <= arc_lengths[j + 1]:
                    # Interpolate within this segment
                    segment_length = arc_lengths[j + 1] - arc_lengths[j]
                    if segment_length > 1e-6:
                        alpha = (target_arc_length - arc_lengths[j]) / segment_length
                    else:
                        alpha = 0.0
                    
                    # Linear interpolation between points
                    new_trajectory[i] = (1 - alpha) * trajectory[j] + alpha * trajectory[j + 1]
                    break
            else:
                # If we didn't find it (shouldn't happen), use the last point
                new_trajectory[i] = trajectory[-1]
        
        return new_trajectory
    
    def validate_trajectory(self, trajectory: np.ndarray) -> Dict[str, bool]:
        """
        Validate that trajectory respects all constraints.
        
        Args:
            trajectory: Trajectory to validate
            
        Returns:
            Dictionary of validation results with detailed information
        """
        results = {
            'velocity_ok': True,
            'acceleration_ok': True,
            'jerk_ok': True,
            'continuity_ok': True,
            'max_velocity': 0.0,
            'max_acceleration': 0.0,
            'max_jerk': 0.0,
            'max_step': 0.0
        }
        
        if len(trajectory) < 2:
            return results
        
        # Calculate derivatives
        dt = self.dt
        
        # Determine trajectory dimensions (3 for position only, 6 for pose)
        n_dims = min(trajectory.shape[1], 6)
        
        # Check continuity - maximum step size between points
        position_diffs = np.diff(trajectory[:, :3], axis=0)
        step_sizes = np.linalg.norm(position_diffs, axis=1)
        max_step = np.max(step_sizes)
        results['max_step'] = max_step
        
        # Expected max step based on velocity and dt
        expected_max_step = self.constraints['max_velocity'] * dt * 1.5  # 50% tolerance
        results['continuity_ok'] = max_step <= expected_max_step
        
        # Velocity - use all relevant dimensions
        velocities = np.diff(trajectory[:, :n_dims], axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities[:, :3], axis=1)  # Only position for velocity
        max_vel = np.max(velocity_magnitudes)
        results['max_velocity'] = max_vel
        results['velocity_ok'] = max_vel <= self.constraints['max_velocity'] * 1.1
        
        # Acceleration
        if len(trajectory) > 2:
            accelerations = np.diff(velocities[:, :3], axis=0) / dt  # Position accelerations
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            max_acc = np.max(acceleration_magnitudes)
            results['max_acceleration'] = max_acc
            results['acceleration_ok'] = max_acc <= self.constraints['max_acceleration'] * 1.2
            
            # Jerk
            if len(trajectory) > 3:
                jerks = np.diff(accelerations, axis=0) / dt
                jerk_magnitudes = np.linalg.norm(jerks, axis=1)
                max_jerk = np.max(jerk_magnitudes)
                results['max_jerk'] = max_jerk
                results['jerk_ok'] = max_jerk <= self.constraints['max_jerk'] * 1.5
        
        return results


class SmoothMotionCommand:
    """Command class for executing smooth motions on PAROL6"""
    
    def __init__(self, trajectory: np.ndarray, speed_factor: float = 1.0):
        """
        Initialize smooth motion command
        
        Args:
            trajectory: Pre-computed trajectory array
            speed_factor: Speed scaling factor (1.0 = normal speed)
        """
        self.trajectory = trajectory
        self.speed_factor = speed_factor
        self.current_index = 0
        self.is_finished = False
        self.is_valid = True
        
    def prepare_for_execution(self, current_position_in):
        """Validate trajectory is reachable from current position"""
        # IK solver availability check
        if solve_ik_with_adaptive_tol_subdivision is None:
            print("Warning: IK solver not available, skipping validation")
            self.is_valid = True
            return True
            
        try:
            # Convert current position to radians
            current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                                 for i, p in enumerate(current_position_in)])
            
            # Initial waypoint reachability
            first_pose = self.trajectory[0]
            target_se3 = SE3(first_pose[0]/1000, first_pose[1]/1000, first_pose[2]/1000) * \
                        SE3.RPY(first_pose[3:], unit='deg', order='xyz')
            
            ik_result = solve_ik_with_adaptive_tol_subdivision(
                PAROL6_ROBOT.robot, target_se3, current_q, ilimit=20
            )
            
            if not ik_result.success:
                print(f"Smooth motion validation failed: Cannot reach first waypoint")
                self.is_valid = False
                return False
                
            print(f"Smooth motion prepared with {len(self.trajectory)} waypoints")
            return True
            
        except Exception as e:
            print(f"Smooth motion preparation error: {e}")
            self.is_valid = False
            return False
    
    def execute_step(self, Position_in, Speed_out, Command_out, **kwargs):
        """Execute one step of the smooth motion"""
        if self.is_finished or not self.is_valid:
            return True
        
        # Module dependency validation
        if PAROL6_ROBOT is None or solve_ik_with_adaptive_tol_subdivision is None:
            print("Error: Required PAROL6 modules not available")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Apply speed scaling
        step_increment = max(1, int(self.speed_factor))
        self.current_index += step_increment
        
        if self.current_index >= len(self.trajectory):
            print("Smooth motion completed")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Get current target pose
        target_pose = self.trajectory[self.current_index]
        
        # Convert to SE3
        target_se3 = SE3(target_pose[0]/1000, target_pose[1]/1000, target_pose[2]/1000) * \
                    SE3.RPY(target_pose[3:], unit='deg', order='xyz')
        
        # Get current joint configuration
        current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                             for i, p in enumerate(Position_in)])
        
        # Solve IK
        ik_result = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, target_se3, current_q, ilimit=20
        )
        
        if not ik_result.success:
            print(f"IK failed at trajectory point {self.current_index}")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Convert to steps and send
        target_steps = [int(PAROL6_ROBOT.RAD2STEPS(q, i)) 
                       for i, q in enumerate(ik_result.q)]
        
        # Following motion velocity profile
        for i in range(6):
            Speed_out[i] = int((target_steps[i] - Position_in[i]) * 10)  # P-control factor
        
        Command_out.value = 156  # Smooth motion command
        return False

# Robot API integration utilities

def execute_circle(center: List[float], 
                  radius: float, 
                  duration: float = 4.0,
                  normal: List[float] = [0, 0, 1]) -> str:
    """
    Execute a circular motion on PAROL6
    
    Args:
        center: Center point [x, y, z] in mm
        radius: Circle radius in mm
        duration: Time to complete circle
        normal: Normal vector to circle plane
        
    Returns:
        Command string for robot_api
    """
    motion_gen = CircularMotion()
    trajectory = motion_gen.generate_circle_3d(center, radius, normal, 0, duration)
    
    # Convert to command string format
    traj_str = "|".join([",".join(map(str, pose)) for pose in trajectory])
    command = f"SMOOTH_MOTION|CIRCLE|{traj_str}"
    
    return command

def execute_arc(start_pose: List[float],
               end_pose: List[float],
               center: List[float],
               clockwise: bool = True,
               duration: float = 2.0) -> str:
    """
    Execute an arc motion on PAROL6
    
    Args:
        start_pose: Starting pose [x, y, z, rx, ry, rz]
        end_pose: Ending pose [x, y, z, rx, ry, rz]
        center: Arc center point [x, y, z]
        clockwise: Direction of rotation
        duration: Time to complete arc
        
    Returns:
        Command string for robot_api
    """
    motion_gen = CircularMotion()
    trajectory = motion_gen.generate_arc_3d(start_pose, end_pose, center, 
                                           clockwise=clockwise, duration=duration)
    
    # Convert to command string format
    traj_str = "|".join([",".join(map(str, pose)) for pose in trajectory])
    command = f"SMOOTH_MOTION|ARC|{traj_str}"
    
    return command

def execute_spline(waypoints: List[List[float]], 
                  total_time: Optional[float] = None) -> str:
    """
    Execute a spline motion through waypoints
    
    Args:
        waypoints: List of poses [x, y, z, rx, ry, rz]
        total_time: Total time for motion (auto-calculated if None)
        
    Returns:
        Command string for robot_api
    """
    motion_gen = SplineMotion()
    
    # Generate timestamps if total_time is provided
    timestamps = None
    if total_time:
        timestamps = np.linspace(0, total_time, len(waypoints))
    
    trajectory = motion_gen.generate_cubic_spline(waypoints, timestamps)
    
    # Convert to command string format
    traj_str = "|".join([",".join(map(str, pose)) for pose in trajectory])
    command = f"SMOOTH_MOTION|SPLINE|{traj_str}"
    
    return command

# Example usage
if __name__ == "__main__":
    # Example: Generate a circle trajectory
    circle_gen = CircularMotion()
    circle_traj = circle_gen.generate_circle_3d(
        center=[200, 0, 200],  # mm
        radius=50,  # mm
        duration=4.0  # seconds
    )
    print(f"Generated circle with {len(circle_traj)} points")
    
    # Example: Generate arc trajectory
    arc_traj = circle_gen.generate_arc_3d(
        start_pose=[250, 0, 200, 0, 0, 0],
        end_pose=[200, 50, 200, 0, 0, 90],
        center=[200, 0, 200],
        duration=2.0
    )
    print(f"Generated arc with {len(arc_traj)} points")
    
    # Example: Generate spline through waypoints
    spline_gen = SplineMotion()
    waypoints = [
        [200, 0, 100, 0, 0, 0],
        [250, 50, 150, 0, 15, 45],
        [200, 100, 200, 0, 30, 90],
        [150, 50, 150, 0, 15, 45],
        [200, 0, 100, 0, 0, 0]
    ]
    spline_traj = spline_gen.generate_cubic_spline(waypoints)
    print(f"Generated spline with {len(spline_traj)} points")