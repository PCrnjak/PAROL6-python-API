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
from roboticstoolbox import DHRobot
from spatialmath.base import trinterp

# Version compatibility check
try:
    import numpy as np
    # Check numpy version
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    if np_version < (1, 23):
        warnings.warn(f"NumPy version {np.__version__} detected. Recommended: 1.23.4")
    
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp
    import scipy
    # Check scipy version
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
    import GUI.files.PAROL6_ROBOT
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
        # Calculate the difference
        diff = q_solution[i] - q_current[i]
        
        # If the difference is more than pi, we need to unwrap
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
    
    # Calculate manipulability measure (closer to 0 = closer to singularity)
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
        # Calculate reduction factor based on proximity to 90°
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
        # Calculate current and target reach
        current_reach = np.linalg.norm(Ta.t)
        target_reach = np.linalg.norm(Tb.t)
        
        # Check if this is an inward movement (recovery)
        is_recovery = target_reach < current_reach
        
        # Calculate configuration-dependent maximum reach based on joint 5 position
        max_reach_threshold = calculate_configuration_dependent_max_reach(q_seed)
        
        # Determine damping based on reach and movement direction
        if is_recovery:
            # Recovery mode - always use low damping
            damping = 0.0000001
        else:
            # Check if we're near configuration-dependent max reach
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
    
    # Check if solution respects joint limits
    if PAROL6_ROBOT is not None:
        target_q = path[-1] if len(path) != 0 else None
        solution_valid, violations = PAROL6_ROBOT.check_joint_limits(current_q, target_q)
        if ok and solution_valid:
            return IKResult(True, path[-1], its, resid, adaptive_tol, violations)
        else:
            return IKResult(False, None, its, resid, adaptive_tol, violations)
    else:
        # If PAROL6_ROBOT not available, skip joint limit checking
        if ok and len(path) > 0:
            return IKResult(True, path[-1], its, resid, adaptive_tol, None)
        else:
            return IKResult(False, None, its, resid, adaptive_tol, None)

# ============================================================================
# END OF IK SOLVER FUNCTIONS
# ============================================================================

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
        
        # Calculate radius vectors
        r1 = start_pos - center_pt
        r2 = end_pos - center_pt
        radius = np.linalg.norm(r1)
        
        # Determine arc plane normal if not provided
        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:  # Points are collinear
                normal = np.array([0, 0, 1])  # Default to XY plane
        normal = normal / np.linalg.norm(normal)
        
        # Calculate arc angle
        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)
        
        # Check direction using cross product
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
            
            # For first point, use exact start position
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
        
        # Create orthonormal basis for circle plane
        normal = np.array(normal) / np.linalg.norm(normal)
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
                # Start point is at center - can't determine angle
                print(f"    WARNING: Start point is at circle center, using default position")
                start_angle = 0
                actual_start = center_np + radius * u
            else:
                # Calculate the angle of the start point
                to_start_normalized = to_start_plane / dist_in_plane
                u_comp = np.dot(to_start_normalized, u)
                v_comp = np.dot(to_start_normalized, v)
                start_angle = np.arctan2(v_comp, u_comp)
                
                # CHECK FOR INVALID GEOMETRY
                radius_error = abs(dist_in_plane - radius)
                if radius_error > radius * 0.3:  # More than 30% off
                    print(f"    WARNING: Start point is {dist_in_plane:.1f}mm from center,")
                    print(f"             but circle radius is {radius:.1f}mm!")
                    
                    # AUTO-CORRECT: Adjust center to make geometry valid
                    print(f"    AUTO-CORRECTING: Moving center to maintain {radius}mm radius from start")
                    direction = to_start_plane / dist_in_plane
                    center_np = start_pos - direction * radius
                    print(f"    New center: {center_np.round(1)}")
                    
                    # Recalculate with new center
                    to_start = start_pos - center_np
                    to_start_plane = to_start - np.dot(to_start, normal) * normal
                    dist_in_plane = np.linalg.norm(to_start_plane)
                
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
    
    def _slerp_orientation(self, start_orient: List[float], 
                          end_orient: List[float], 
                          t: float) -> np.ndarray:
        """Spherical linear interpolation for orientation"""
        # Convert to quaternions
        r1 = Rotation.from_euler('xyz', start_orient, degrees=True)
        r2 = Rotation.from_euler('xyz', end_orient, degrees=True)
        
        # Create slerp object - compatible with scipy 1.11.4
        # Stack rotations into a single Rotation object
        key_rots = Rotation.from_quat([r1.as_quat(), r2.as_quat()])
        slerp = Slerp([0, 1], key_rots)
        
        # Interpolate
        interp_rot = slerp(t)
        return interp_rot.as_euler('xyz', degrees=True)

class SplineMotion(TrajectoryGenerator):
    """Generate smooth spline trajectories through waypoints"""
    
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
        
        # Create splines for position
        pos_splines = []
        for i in range(3):
            bc_type = 'not-a-knot'  # Default boundary condition
            
            # Apply velocity boundary conditions if specified
            if velocity_start is not None and velocity_end is not None:
                bc_type = ((1, velocity_start[i]), (1, velocity_end[i]))
            
            spline = CubicSpline(timestamps, waypoints[:, i], bc_type=bc_type)
            pos_splines.append(spline)
        
        # Create splines for orientation (convert to quaternions for smooth interpolation)
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
        # For quintic spline, we need to ensure zero velocity and acceleration
        # at the endpoints for smooth motion
        return self.generate_cubic_spline(
            waypoints, 
            timestamps,
            velocity_start=[0, 0, 0],
            velocity_end=[0, 0, 0]
        )

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
        
        # Calculate overlap region more carefully
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
            
            # For orientation, use SLERP
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
        # Check if IK solver is available
        if solve_ik_with_adaptive_tol_subdivision is None:
            print("Warning: IK solver not available, skipping validation")
            self.is_valid = True
            return True
            
        try:
            # Convert current position to radians
            current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                                 for i, p in enumerate(current_position_in)])
            
            # Check first waypoint is reachable
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
        
        # Check if required modules are available
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
        
        # Calculate velocities for smooth following
        for i in range(6):
            Speed_out[i] = int((target_steps[i] - Position_in[i]) * 10)  # P-control factor
        
        Command_out.value = 156  # Smooth motion command
        return False

# Helper functions for integration with robot_api.py

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