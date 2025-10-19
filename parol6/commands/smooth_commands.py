"""
Smooth Motion Commands
Contains all smooth trajectory generation commands for advanced robot movements
"""
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Any, TYPE_CHECKING, Sequence, Union

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
import json
from spatialmath import SE3
from parol6.smooth_motion import (
    CircularMotion, SplineMotion, HelixMotion, WaypointTrajectoryPlanner
)
from parol6.commands.base import CommandBase, ExecutionStatus, ExecutionStatusCode, MotionCommand
from parol6.utils.errors import IKError
from parol6.utils.ik import solve_ik
from parol6.utils.frames import (
    point_trf_to_wrf_mm,
    pose6_trf_to_wrf,
    se3_to_pose6_mm_deg,
    transform_center_trf_to_wrf,
    transform_start_pose_if_needed,
    transform_command_params_to_wrf,
)
from parol6.config import INTERVAL_S, NEAR_MM_TOL_MM, ENTRY_MM_TOL_MM
from parol6.commands.cartesian_commands import MovePoseCommand
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_se3
from parol6.protocol.wire import CommandCode
from parol6.smooth_motion.advanced import AdvancedMotionBlender

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class BaseSmoothMotionCommand(MotionCommand):
    """
    Base class for all smooth motion commands with proper error tracking.
    """
    __slots__ = (
        "description",
        "trajectory",
        "trajectory_command",
        "transition_command",
        "specified_start_pose",
        "transition_complete",
        "trajectory_prepared",
        "trajectory_generated",
    )
    
    def __init__(self, description="smooth motion"):
        super().__init__()
        self.description = description
        self.trajectory: Optional[np.ndarray] = None
        self.trajectory_command: Optional["SmoothTrajectoryCommand"] = None
        self.transition_command: Optional["MovePoseCommand"] = None
        self.specified_start_pose: Optional[NDArray[np.floating]] = None
        self.transition_complete = False
        self.trajectory_prepared = False
        self.trajectory_generated = False
        
    # Parsing utility methods
    @staticmethod
    def parse_start_pose(start_str: str) -> Optional[NDArray[np.floating]]:
        """
        Parse start pose from string.
        
        Args:
            start_str: Either 'CURRENT', 'NONE', or comma-separated pose values
            
        Returns:
            None for CURRENT/NONE, or numpy array of floats for specified pose
        """
        if start_str.upper() in ('CURRENT', 'NONE'):
            return None
        else:
            try:
                return np.asarray(list(map(float, start_str.split(','))), dtype=np.float64)
            except Exception:
                raise ValueError(f"Invalid start pose format: {start_str}")
    
    @staticmethod
    def parse_timing(timing_type: str, timing_value: float, path_length: float) -> float:
        """
        Convert timing specification to duration.
        
        Args:
            timing_type: 'DURATION' or 'SPEED'
            timing_value: Duration in seconds or speed in mm/s
            path_length: Estimated path length in mm
            
        Returns:
            Duration in seconds
        """
        if timing_type.upper() == 'DURATION':
            return timing_value
        elif timing_type.upper() == 'SPEED':
            if timing_value <= 0:
                raise ValueError(f"Speed must be positive, got {timing_value}")
            return path_length / timing_value
        else:
            raise ValueError(f"Unknown timing type: {timing_type}")
    
    @staticmethod
    def calculate_path_length(command_type: str, params: dict) -> float:
        """
        Estimate trajectory path length for timing calculations.
        
        Args:
            command_type: Type of smooth motion command
            params: Command parameters
            
        Returns:
            Estimated path length in mm
        """
        if command_type == 'SMOOTH_CIRCLE':
            # Full circle circumference
            return 2 * np.pi * params.get('radius', 100)
        elif command_type in ['SMOOTH_ARC_CENTER', 'SMOOTH_ARC_PARAM']:
            # Estimate arc length (approximate)
            radius = params.get('radius', 100)
            angle = params.get('arc_angle', 90)  # degrees
            return radius * np.radians(angle)
        elif command_type == 'SMOOTH_HELIX':
            # Helix path length
            radius = params.get('radius', 100)
            height = params.get('height', 100)
            turns = height / params.get('pitch', 10) if params.get('pitch', 10) > 0 else 1
            return np.sqrt((2 * np.pi * radius * turns)**2 + height**2)
        else:
            # Default estimate
            return 300  # mm
    
    @staticmethod
    def parse_trajectory_type(parts: List[str], index: int) -> Tuple[str, Optional[float], int]:
        """
        Parse trajectory type and optional jerk limit.
        
        Args:
            parts: Command parts
            index: Current index in parts
            
        Returns:
            Tuple of (trajectory_type, jerk_limit, next_index)
        """
        if index >= len(parts):
            return 'cubic', None, index
        
        traj_type = parts[index].lower()
        index += 1
        
        if traj_type not in ['cubic', 'quintic', 's_curve']:
            # Not a valid trajectory type, use default
            return 'cubic', None, index - 1
        
        # Check for jerk limit if s_curve
        jerk_limit = None
        if traj_type == 's_curve' and index < len(parts):
            try:
                jerk_limit = float(parts[index])
                index += 1
            except (ValueError, IndexError):
                jerk_limit = 1000  # Default jerk limit
        
        return traj_type, jerk_limit, index
    
    def create_transition_command(self, current_pose: np.ndarray, target_pose: NDArray[np.floating]) -> Optional["MovePoseCommand"]:
        """Create a MovePose command for smooth transition to start position."""
        pos_error = np.linalg.norm(target_pose[:3] - current_pose[:3])
        
        if pos_error < NEAR_MM_TOL_MM:  # proximity threshold
            self.log_info("  -> Already near start position (error: %.1fmm)", pos_error)
            return None
        
        self.log_info("  -> Creating smooth transition to start (%.1fmm away)", pos_error)
        
        # Calculate transition speed based on distance
        if pos_error < 10:
            transition_speed = 20.0  # mm/s for short distances
        elif pos_error < 30:
            transition_speed = 30.0  # mm/s for medium distances
        else:
            transition_speed = 40.0  # mm/s for long distances
        
        transition_duration = max(pos_error / transition_speed, 0.5)  # Minimum 0.5s
        
        # MovePoseCommand expects a list, so convert array to list here
        transition_cmd: MovePoseCommand = MovePoseCommand(target_pose.tolist(), transition_duration)
        
        return transition_cmd
    
    def get_current_pose_from_position(self, position_in):
        """Convert current position to pose [x,y,z,rx,ry,rz]"""
        current_pose_se3 = get_fkine_se3()
        
        current_xyz = current_pose_se3.t * 1000  # Convert to mm
        current_rpy = current_pose_se3.rpy(unit='deg', order='xyz')
        return np.concatenate([current_xyz, current_rpy])
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Minimal preparation - just check if we need a transition."""
        self.log_debug("  -> Preparing %s...", self.description)
        
        # If there's a specified start pose, prepare transition
        if self.specified_start_pose is not None:
            actual_current_pose = self.get_current_pose_from_position(state.Position_in)
            self.transition_command = self.create_transition_command(
                actual_current_pose, self.specified_start_pose
            )
            
            if self.transition_command:
                self.transition_command.setup(state)
                if not self.transition_command.is_valid:
                    self.log_error("  -> ERROR: Cannot reach specified start position")
                    self.fail("Cannot reach specified start position")
                    return
        else:
            self.transition_command = None
            
        # DON'T generate trajectory yet - wait until execution
        self.trajectory_generated = False
        self.trajectory_prepared = False
        self.log_debug("  -> %s preparation complete (trajectory will be generated at execution)", self.description)
    
    def generate_main_trajectory(self, effective_start_pose):
        """Override this in subclasses to generate the specific motion trajectory."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")
    
    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute transition first if needed, then generate and execute trajectory."""
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid smooth command")

        # Execute transition first if needed
        if self.transition_command and not self.transition_complete:
            status = self.transition_command.execute_step(state)
            if status.code == ExecutionStatusCode.COMPLETED:
                self.log_info("  -> Transition complete")
                self.transition_complete = True
                # Continue to main trajectory generation next tick
                return ExecutionStatus.executing("Transition completed")
            elif status.code == ExecutionStatusCode.FAILED:
                self.fail(getattr(self.transition_command, 'error_message', 'Transition failed'))
                self.finish()
                MotionCommand.stop_and_idle(state)
                return ExecutionStatus.failed("Transition failed")
            else:
                return ExecutionStatus.executing("Transitioning")

        # Generate trajectory on first execution step (not during preparation!)
        if not self.trajectory_generated:
            # Get ACTUAL current position NOW
            actual_current_pose = self.get_current_pose_from_position(state.Position_in)
            self.log_info("  -> Generating %s from ACTUAL position: %s", self.description, [round(p, 1) for p in actual_current_pose[:3]])

            # Generate trajectory from where we ACTUALLY are
            self.trajectory = self.generate_main_trajectory(actual_current_pose)
            self.trajectory_command = SmoothTrajectoryCommand(self.trajectory, self.description)

            # Quick validation of first point only
            current_q = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
            first_pose = self.trajectory[0]
            target_se3 = SE3(first_pose[0] / 1000, first_pose[1] / 1000, first_pose[2] / 1000) * SE3.RPY(first_pose[3:], unit='deg', order='xyz')

            ik_result = solve_ik(
                PAROL6_ROBOT.robot, target_se3, current_q, jogging=False
            )

            if not ik_result.success:
                self.log_error("  -> ERROR: Cannot reach first trajectory point")
                self.finish()
                self.fail("Cannot reach trajectory start")
                MotionCommand.stop_and_idle(state)
                return ExecutionStatus.failed("Cannot reach trajectory start", error=IKError("Cannot reach trajectory start"))

            self.trajectory_generated = True
            self.trajectory_prepared = True

            # Verify first point is close to current
            distance = np.linalg.norm(first_pose[:3] - np.array(actual_current_pose[:3]))
            if distance > 5.0:
                self.log_warning("  -> WARNING: First trajectory point %.1fmm from current!", distance)

        # Execute main trajectory
        if self.trajectory_command and self.trajectory_prepared:
            status = self.trajectory_command.execute_step(state)

            # Check for errors in trajectory execution
            if hasattr(self.trajectory_command, 'error_state') and self.trajectory_command.error_state:
                self.fail(self.trajectory_command.error_message)

            if status.code == ExecutionStatusCode.COMPLETED:
                self.finish()
                return ExecutionStatus.completed(f"Smooth {self.description} complete")
            elif status.code == ExecutionStatusCode.FAILED:
                self.finish()
                return status
            else:
                return ExecutionStatus.executing(f"Smooth {self.description}")

        self.finish()
        return ExecutionStatus.completed(f"Smooth {self.description} complete")

    def _generate_radial_entry(self, start_pose, center, normal, radius, duration, sample_rate: float = 100.0):
        """
        Generate a simple radial entry trajectory from the current pose to the circle/helix perimeter.
        Produces a straight-line interpolation in Cartesian space.
        """
        start_pose = np.array(start_pose, dtype=float)
        center = np.array(center, dtype=float)
        normal = np.array(normal, dtype=float)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)

        start_pos = start_pose[:3]
        to_start = start_pos - center
        # Project onto plane
        to_plane = to_start - np.dot(to_start, normal) * normal
        dist = float(np.linalg.norm(to_plane))

        if dist < 1e-6:
            # Choose arbitrary direction perpendicular to normal
            axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(axis, normal)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            direction = np.cross(normal, axis)
            direction = direction / np.linalg.norm(direction)
        else:
            direction = to_plane / dist

        target_pos = center + direction * float(radius)
        # Keep orientation constant
        target_orient = start_pose[3:]

        # Number of samples
        n = max(2, int(max(0.05, float(duration)) * float(sample_rate)))
        ts = np.linspace(0.0, 1.0, n)
        traj = []
        for s in ts:
            pos = (1.0 - s) * start_pos + s * target_pos
            traj.append(np.concatenate([pos, target_orient]))
        return np.array(traj)

class SmoothTrajectoryCommand:
    """Command class for executing pre-generated smooth trajectories."""
    
    def __init__(self, trajectory, description="smooth motion"):
        self.trajectory = np.array(trajectory)
        self.description = description
        self.trajectory_index = 0
        self.is_valid = True
        self.is_finished = False
        self.first_step = True
        self.error_state = False
        self.error_message = ""
        
        logger.debug(f"Initializing smooth {description} with {len(trajectory)} points")
    
    def setup(self, state: "ControllerState"):
        """Skip validation - trajectory is already generated from correct position"""
        self.is_valid = True
        return
    
    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute one step of the smooth trajectory"""
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid smooth trajectory")

        if self.trajectory_index >= len(self.trajectory):
            logger.info(f"Smooth {self.description} finished.")
            self.is_finished = True
            MotionCommand.stop_and_idle(state)
            return ExecutionStatus.completed(f"Smooth {self.description} complete")
        
        # Get target pose for this step
        target_pose = self.trajectory[self.trajectory_index]
        
        # Convert to SE3
        target_se3 = SE3(target_pose[0]/1000, target_pose[1]/1000, target_pose[2]/1000) * SE3.RPY(target_pose[3:], unit='deg', order='xyz')
        
        # Get current joint configuration
        current_q = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        
        # Solve IK
        ik_result = solve_ik(
            PAROL6_ROBOT.robot, target_se3, current_q, jogging=False
        )
        
        if not ik_result.success:
            logger.error(f"  -> IK failed at trajectory point {self.trajectory_index}")
            self.is_finished = True
            self.error_state = True
            self.error_message = f"IK failed at point {self.trajectory_index}/{len(self.trajectory)}"
            MotionCommand.stop_and_idle(state)
            return ExecutionStatus.failed(self.error_message, error=IKError(self.error_message))
        
        # Convert to steps
        target_steps = PAROL6_ROBOT.ops.rad_to_steps(ik_result.q)
        
        # ADD VELOCITY LIMITING - This prevents violent movements
        if self.trajectory_index > 0:
            # Vectorized per-tick clamping with 1.2x safety margin
            target_steps = PAROL6_ROBOT.ops.clamp_steps_delta(
                state.Position_in,
                np.asarray(target_steps, dtype=np.int32),
                dt=INTERVAL_S,
                safety=1.2
            )
        
        # Send position command (inline to avoid instance-method binding)
        np.copyto(state.Position_out, np.asarray(target_steps, dtype=np.int32), casting='no')
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.MOVE
        
        # Advance to next point
        self.trajectory_index += 1
        
        return ExecutionStatus.executing(f"Smooth {self.description}")


@register_command("SMOOTH_CIRCLE")
class SmoothCircleCommand(BaseSmoothMotionCommand):
    """Execute smooth circular motion."""
    
    __slots__ = (
        "center",
        "radius",
        "plane",
        "duration",
        "clockwise",
        "frame",
        "trajectory_type",
        "jerk_limit",
        "center_mode",
        "entry_mode",
        "normal_vector",
        "current_position_in",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth circle")
        self.center: Optional[NDArray[np.floating]] = None
        self.radius: float = 100.0
        self.plane: str = 'XY'
        self.duration: float = 5.0
        self.clockwise: bool = False
        self.frame: str = 'WRF'
        self.trajectory_type: str = 'cubic'
        self.jerk_limit: Optional[float] = None
        self.center_mode: str = 'ABSOLUTE'
        self.entry_mode: str = 'NONE'
        self.normal_vector: Optional[NDArray] = None
        self.current_position_in: Optional[NDArray[np.int32]] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_CIRCLE command.
        Format: SMOOTH_CIRCLE|center_x,y,z|radius|plane|frame|start_pose|timing_type|timing_value|[trajectory_type]|[jerk_limit]|[clockwise]
        """
        if parts[0].upper() != "SMOOTH_CIRCLE":
            return False, None
        
        if len(parts) < 8:
            return False, "SMOOTH_CIRCLE requires at least 8 parameters"
        
        try:
            # Parse center
            center_list = list(map(float, parts[1].split(',')))
            if len(center_list) != 3:
                return False, "Center must have 3 coordinates"
            self.center = np.asarray(center_list, dtype=np.float64)
            
            # Parse radius
            self.radius = float(parts[2])
            
            # Parse plane
            self.plane = parts[3].upper()
            if self.plane not in ['XY', 'XZ', 'YZ']:
                return False, f"Invalid plane: {self.plane}"
            
            # Parse frame
            self.frame = parts[4].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse start pose
            self.specified_start_pose = self.parse_start_pose(parts[5])
            
            # Parse timing
            timing_type = parts[6].upper()
            timing_value = float(parts[7])
            path_length = 2 * np.pi * self.radius
            self.duration = self.parse_timing(timing_type, timing_value, path_length)
            
            # Parse optional trajectory type and jerk limit
            idx = 8
            if idx < len(parts):
                self.trajectory_type, self.jerk_limit, idx = self.parse_trajectory_type(parts, idx)
            
            # Parse optional clockwise flag
            if idx < len(parts) and parts[idx].upper() in ['CW', 'CLOCKWISE', 'TRUE']:
                self.clockwise = True
                idx += 1
            
            # Parse optional center mode
            if idx < len(parts):
                self.center_mode = parts[idx].upper()
                if self.center_mode not in ['ABSOLUTE', 'TOOL', 'RELATIVE']:
                    self.center_mode = 'ABSOLUTE'
                idx += 1
            
            # Parse optional entry mode
            if idx < len(parts):
                self.entry_mode = parts[idx].upper()
                if self.entry_mode not in ['AUTO', 'TANGENT', 'DIRECT', 'NONE']:
                    self.entry_mode = 'NONE'
            
            # Initialize description
            self.description = f"circle (r={self.radius}mm, {self.frame}, {self.trajectory_type})"
            
            return True, None
            
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_CIRCLE parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform parameters if in TRF, then prepare normally."""

        # Store current position for potential use in generate_main_trajectory
        self.current_position_in = np.asarray(state.Position_in, dtype=np.int32)
        
        if self.frame == 'TRF':
            # Transform parameters to WRF
            params = {
                'center': self.center,
                'plane': self.plane
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_CIRCLE', params, 'TRF', state.Position_in
            )
            
            # Update with transformed values
            self.center = transformed['center']
            self.normal_vector = transformed.get('normal_vector')
            
            logger.info(f"  -> TRF Circle: center {self.center[:3].tolist()} (WRF), normal {self.normal_vector}")
            
            # Transform start_pose if specified - convert array to list for the API
            if self.specified_start_pose is not None:
                result = transform_start_pose_if_needed(
                    self.specified_start_pose.tolist(), self.frame, state.Position_in
                )
                if result is not None:
                    self.specified_start_pose = np.asarray(result, dtype=np.float64)
        
        # Now do normal preparation with transformed parameters
        return super().do_setup(state)
    
    def generate_main_trajectory(self, effective_start_pose):
        """Generate circle starting from the actual start position."""
        motion_gen = CircularMotion()
        
        # Use transformed normal for TRF, or standard for WRF
        if self.normal_vector is not None:
            # TRF - use the transformed normal vector
            normal = np.array(self.normal_vector)
            logger.info(f"    Using transformed normal: {normal.round(3)}")
        else:
            # WRF - use standard plane definition
            plane_normals = {'XY': [0, 0, 1], 'XZ': [0, 1, 0], 'YZ': [1, 0, 0]}
            normal = np.array(plane_normals.get(self.plane, [0, 0, 1]))
            logger.info(f"    Using WRF plane {self.plane} with normal: {normal}")
        
        logger.info(f"    Generating circle from position: {[round(p, 1) for p in effective_start_pose[:3]]}")
        if self.center is not None:
            logger.info(f"    Circle center: {[round(c, 1) for c in self.center]}")
        
        # Handle center_mode
        if self.center is not None:
            actual_center = self.center.copy()
        else:
            actual_center = np.array([0.0, 0.0, 0.0])
        if self.center_mode == 'TOOL':
            # Center at current tool position
            actual_center = np.array(effective_start_pose[:3])
            logger.info(f"    Center mode: TOOL - centering at current position {actual_center}")
        elif self.center_mode == 'RELATIVE':
            # Center relative to current position
            actual_center = np.array([effective_start_pose[i] + self.center[i] for i in range(3)])
            logger.info(f"    Center mode: RELATIVE - center offset from current position to {actual_center}")
        else:
            # ABSOLUTE mode uses provided center as-is
            actual_center = np.array(actual_center)
        
        # Check if entry trajectory might be needed
        distance_to_center = np.linalg.norm(np.array(effective_start_pose[:3]) - np.array(actual_center))
        distance_from_perimeter = float(abs(distance_to_center - self.radius))
        
        # Automatically generate entry trajectory if needed
        entry_trajectory = None
        if distance_from_perimeter > ENTRY_MM_TOL_MM:  # perimeter tolerance
            effective_entry_mode = self.entry_mode
            
            # Auto-detect need for entry if not specified
            if self.entry_mode == 'NONE' and distance_from_perimeter > 5.0:  # Auto-enable for > 5mm
                logger.warning(f"    Robot is {distance_from_perimeter:.1f}mm from circle perimeter - auto-enabling entry trajectory")
                effective_entry_mode = 'AUTO'
            
            if effective_entry_mode != 'NONE':
                logger.info(f"    Generating {effective_entry_mode} entry trajectory (distance: {distance_from_perimeter:.1f}mm)")
                
                # Calculate entry duration based on distance (0.5s min, 2.0s max)
                entry_duration = min(2.0, max(0.5, distance_from_perimeter / 50.0))
                
                # Generate entry trajectory (radial approach)
                entry_trajectory = self._generate_radial_entry(
                    effective_start_pose, actual_center, normal, self.radius, entry_duration
                )
                
                if entry_trajectory is not None and len(entry_trajectory) > 0:
                    logger.info(f"    Entry trajectory generated: {len(entry_trajectory)} points over {entry_duration:.1f}s")
        
        # Generate circle with specified trajectory profile
        trajectory = motion_gen.generate_circle_with_profile(
            center=actual_center,  # Use adjusted center
            radius=self.radius,
            normal=normal,  # This normal now correctly represents the plane
            start_point=effective_start_pose,  # Pass full pose including orientation
            duration=self.duration,
            trajectory_type=self.trajectory_type,
            jerk_limit=self.jerk_limit
        )
        
        if self.clockwise:
            trajectory = trajectory[::-1]
        
        # Update orientations to match start pose
        for i in range(len(trajectory)):
            trajectory[i][3:] = effective_start_pose[3:]
        
        # Concatenate entry trajectory if it exists
        if entry_trajectory is not None and len(entry_trajectory) > 0:
            full_trajectory = np.concatenate([entry_trajectory, trajectory])
            return full_trajectory
        else:
            return trajectory


@register_command("SMOOTH_ARC_CENTER")
class SmoothArcCenterCommand(BaseSmoothMotionCommand):
    """Execute smooth arc motion defined by center point."""
    
    __slots__ = (
        "end_pose",
        "center",
        "duration",
        "clockwise",
        "frame",
        "trajectory_type",
        "jerk_limit",
        "normal_vector",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth arc (center)")
        self.end_pose: Optional[List[float]] = None
        self.center: Optional[List[float]] = None
        self.duration: float = 5.0
        self.clockwise: bool = False
        self.frame: str = 'WRF'
        self.trajectory_type: str = 'cubic'
        self.jerk_limit: Optional[float] = None
        self.normal_vector: Optional[NDArray] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_ARC_CENTER command.
        Format: SMOOTH_ARC_CENTER|end_x,y,z,rx,ry,rz|center_x,y,z|frame|start_pose|timing_type|timing_value|[trajectory_type]|[jerk_limit]|[clockwise]
        """
        if parts[0].upper() != "SMOOTH_ARC_CENTER":
            return False, None
        
        if len(parts) < 7:
            return False, "SMOOTH_ARC_CENTER requires at least 7 parameters"
        
        try:
            # Parse end pose
            self.end_pose = list(map(float, parts[1].split(',')))
            if len(self.end_pose) != 6:
                return False, "End pose must have 6 values (x,y,z,rx,ry,rz)"
            
            # Parse center
            self.center = list(map(float, parts[2].split(',')))
            if len(self.center) != 3:
                return False, "Center must have 3 coordinates"
            
            # Parse frame
            self.frame = parts[3].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse start pose
            self.specified_start_pose = self.parse_start_pose(parts[4])
            
            # Parse timing
            timing_type = parts[5].upper()
            timing_value = float(parts[6])
            # Estimate arc length
            path_length = 300  # Default estimate, could be improved
            self.duration = self.parse_timing(timing_type, timing_value, path_length)
            
            # Parse optional trajectory type and jerk limit
            idx = 7
            if idx < len(parts):
                self.trajectory_type, self.jerk_limit, idx = self.parse_trajectory_type(parts, idx)
            
            # Parse optional clockwise flag
            if idx < len(parts) and parts[idx].upper() in ['CW', 'CLOCKWISE', 'TRUE']:
                self.clockwise = True
            
            # Initialize description
            self.description = f"arc (center-based, {self.frame}, {self.trajectory_type})"
            
            return True, None
            
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_ARC_CENTER parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform parameters if in TRF."""

        if self.frame == 'TRF':
            params = {
                'end_pose': self.end_pose,
                'center': self.center
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_ARC_CENTER', params, 'TRF', state.Position_in
            )
            self.end_pose = transformed['end_pose']
            self.center = transformed['center']
            self.normal_vector = transformed.get('normal_vector')
            
            # Transform start_pose if specified
            self.specified_start_pose = transform_start_pose_if_needed(
                self.specified_start_pose, self.frame, state.Position_in
            )
        
        return super().do_setup(state)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc from actual start to end with direct velocity profile."""
        motion_gen = CircularMotion()
        
        # Use new direct profile generation method
        trajectory = motion_gen.generate_arc_with_profile(
            effective_start_pose, self.end_pose, self.center,
            normal=self.normal_vector,  # Use transformed normal if TRF
            clockwise=self.clockwise, 
            duration=self.duration,
            trajectory_type=self.trajectory_type,
            jerk_limit=self.jerk_limit
        )
        
        return trajectory


@register_command("SMOOTH_ARC_PARAM")
class SmoothArcParamCommand(BaseSmoothMotionCommand):
    """Execute smooth arc motion defined by radius and angle."""
    
    __slots__ = (
        "end_pose",
        "radius",
        "arc_angle",
        "duration",
        "clockwise",
        "frame",
        "trajectory_type",
        "jerk_limit",
        "normal_vector",
        "current_position_in",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth arc (param)")
        self.end_pose: Optional[List[float]] = None
        self.radius: float = 100.0
        self.arc_angle: float = 90.0
        self.duration: float = 5.0
        self.clockwise: bool = False
        self.frame: str = 'WRF'
        self.trajectory_type: str = 'cubic'
        self.jerk_limit: Optional[float] = None
        self.normal_vector: Optional[NDArray] = None
        self.current_position_in: Optional[Sequence[int]] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_ARC_PARAM command.
        Format: SMOOTH_ARC_PARAM|end_x,y,z,rx,ry,rz|radius|arc_angle|frame|start_pose|timing_type|timing_value|[trajectory_type]|[jerk_limit]|[clockwise]
        """
        if parts[0].upper() != "SMOOTH_ARC_PARAM":
            return False, None
        
        if len(parts) < 8:
            return False, "SMOOTH_ARC_PARAM requires at least 8 parameters"
        
        try:
            # Parse end pose
            self.end_pose = list(map(float, parts[1].split(',')))
            if len(self.end_pose) != 6:
                return False, "End pose must have 6 values (x,y,z,rx,ry,rz)"
            
            # Parse radius and arc angle
            self.radius = float(parts[2])
            self.arc_angle = float(parts[3])
            
            # Parse frame
            self.frame = parts[4].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse start pose
            self.specified_start_pose = self.parse_start_pose(parts[5])
            
            # Parse timing
            timing_type = parts[6].upper()
            timing_value = float(parts[7])
            path_length = self.radius * np.radians(self.arc_angle)
            self.duration = self.parse_timing(timing_type, timing_value, path_length)
            
            # Parse optional trajectory type and jerk limit
            idx = 8
            if idx < len(parts):
                self.trajectory_type, self.jerk_limit, idx = self.parse_trajectory_type(parts, idx)
            
            # Parse optional clockwise flag
            if idx < len(parts) and parts[idx].upper() in ['CW', 'CLOCKWISE', 'TRUE']:
                self.clockwise = True
            
            # Initialize description
            self.description = f"parametric arc (r={self.radius}mm, θ={self.arc_angle}°, {self.frame}, {self.trajectory_type})"
            
            return True, None
            
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_ARC_PARAM parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform parameters if in TRF, then prepare normally."""

        self.current_position_in = state.Position_in
        
        if self.frame == 'TRF':
            # Transform parameters to WRF
            params = {
                'end_pose': self.end_pose,
                'plane': 'XY'  # Default plane for parametric arc
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_ARC_PARAM', params, 'TRF', state.Position_in
            )
            
            # Update with transformed values
            self.end_pose = transformed['end_pose']
            self.normal_vector = transformed.get('normal_vector')
            
            logger.info(f"  -> TRF Parametric Arc: end {self.end_pose[:3]} (WRF)")
            
            # Transform start_pose if specified
            self.specified_start_pose = transform_start_pose_if_needed(
                self.specified_start_pose, self.frame, state.Position_in
            )
        
        return super().do_setup(state)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc based on radius and angle from actual start."""
        # Get start and end positions
        start_xyz = np.array(effective_start_pose[:3])
        end_xyz = np.array(self.end_pose[:3])
        
        # If we have a transformed normal (TRF), use it to define the arc plane
        if self.normal_vector is not None:
            normal = np.array(self.normal_vector)
            
            # Find center of arc using radius and angle
            chord_vec = end_xyz - start_xyz
            chord_length = np.linalg.norm(chord_vec)
            
            if chord_length > 2 * self.radius:
                logger.warning(f"  -> Warning: Points too far apart ({chord_length:.1f}mm) for radius {self.radius}mm")
                self.radius = chord_length / 2 + 1
            
            # Calculate center position using the normal vector
            chord_mid = (start_xyz + end_xyz) / 2
            
            # Height from chord midpoint to arc center
            if chord_length > 0:
                h = np.sqrt(max(0, self.radius**2 - (chord_length/2)**2))
                
                # Find perpendicular in the plane defined by normal
                chord_dir = chord_vec / chord_length
                perp_in_plane = np.cross(normal, chord_dir)
                if np.linalg.norm(perp_in_plane) > 0.001:
                    perp_in_plane = perp_in_plane / np.linalg.norm(perp_in_plane)
                else:
                    # Chord is parallel to normal (shouldn't happen)
                    perp_in_plane = np.array([1, 0, 0])
                
                # Arc center
                if self.clockwise:
                    center_3d = chord_mid - h * perp_in_plane
                else:
                    center_3d = chord_mid + h * perp_in_plane
            else:
                center_3d = start_xyz
            
        else:
            # WRF - use XY plane (standard behavior)
            normal = np.array([0, 0, 1])
            
            # Calculate arc center in XY plane
            start_xy = start_xyz[:2]
            end_xy = end_xyz[:2]
            
            # Midpoint
            mid = (start_xy + end_xy) / 2
            
            # Distance between points
            d = np.linalg.norm(end_xy - start_xy)
            
            if d > 2 * self.radius:
                logger.warning(f"  -> Warning: Points too far apart ({d:.1f}mm) for radius {self.radius}mm")
                self.radius = d / 2 + 1
            
            # Height of arc center from midpoint
            h = np.sqrt(max(0, self.radius**2 - (d/2)**2))
            
            # Perpendicular direction
            if d > 0:
                perp = np.array([-(end_xy[1] - start_xy[1]), end_xy[0] - start_xy[0]])
                perp = perp / np.linalg.norm(perp)
            else:
                perp = np.array([1, 0])
            
            # Arc center (choose based on clockwise)
            if self.clockwise:
                center_2d = mid - h * perp
            else:
                center_2d = mid + h * perp
            
            # Use average Z for center
            center_3d = [center_2d[0], center_2d[1], (start_xyz[2] + end_xyz[2]) / 2]
        
        # Generate arc trajectory with direct velocity profile
        motion_gen = CircularMotion()
        
        # Use new direct profile generation method
        trajectory = motion_gen.generate_arc_with_profile(
            effective_start_pose, self.end_pose, center_3d,
            normal=normal if self.normal_vector is not None else None,
            clockwise=self.clockwise, 
            duration=self.duration,
            trajectory_type=self.trajectory_type,
            jerk_limit=self.jerk_limit
        )
        
        return trajectory


@register_command("SMOOTH_HELIX")
class SmoothHelixCommand(BaseSmoothMotionCommand):
    """Execute smooth helical motion."""
    
    __slots__ = (
        "center",
        "radius",
        "pitch",
        "height",
        "duration",
        "clockwise",
        "frame",
        "trajectory_type",
        "jerk_limit",
        "helix_axis",
        "up_vector",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth helix")
        self.center: Optional[List[float]] = None
        self.radius: float = 100.0
        self.pitch: float = 10.0
        self.height: float = 100.0
        self.duration: float = 5.0
        self.clockwise: bool = False
        self.frame: str = 'WRF'
        self.trajectory_type: str = 'cubic'
        self.jerk_limit: Optional[float] = None
        self.helix_axis: Optional[NDArray] = None
        self.up_vector: Optional[NDArray] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_HELIX command.
        Format: SMOOTH_HELIX|center_x,y,z|radius|pitch|height|frame|start_pose|timing_type|timing_value|[trajectory_type]|[jerk_limit]|[clockwise]
        """
        if parts[0].upper() != "SMOOTH_HELIX":
            return False, None
        
        if len(parts) < 9:
            return False, "SMOOTH_HELIX requires at least 9 parameters"
        
        try:
            # Parse center
            self.center = list(map(float, parts[1].split(',')))
            if len(self.center) != 3:
                return False, "Center must have 3 coordinates"
            
            # Parse radius, pitch, height
            self.radius = float(parts[2])
            self.pitch = float(parts[3])
            self.height = float(parts[4])
            
            # Parse frame
            self.frame = parts[5].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse start pose
            self.specified_start_pose = self.parse_start_pose(parts[6])
            
            # Parse timing
            timing_type = parts[7].upper()
            timing_value = float(parts[8])
            turns = self.height / self.pitch if self.pitch > 0 else 1
            path_length = np.sqrt((2 * np.pi * self.radius * turns)**2 + self.height**2)
            self.duration = self.parse_timing(timing_type, timing_value, path_length)
            
            # Parse optional trajectory type and jerk limit
            idx = 9
            if idx < len(parts):
                self.trajectory_type, self.jerk_limit, idx = self.parse_trajectory_type(parts, idx)
            
            # Parse optional clockwise flag
            if idx < len(parts) and parts[idx].upper() in ['CW', 'CLOCKWISE', 'TRUE']:
                self.clockwise = True
            
            # Initialize description
            self.description = f"helix (h={self.height}mm, {self.frame}, {self.trajectory_type})"
            
            return True, None
            
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_HELIX parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform parameters if in TRF."""

        if self.frame == 'TRF':
            params = {'center': self.center}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_HELIX', params, 'TRF', state.Position_in
            )
            self.center = transformed['center']
            self.helix_axis = transformed.get('helix_axis', [0, 0, 1])
            self.up_vector = transformed.get('up_vector', [0, 1, 0])
            
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_HELIX', params, 'TRF', state.Position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().do_setup(state)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate helix with entry trajectory if needed and proper trajectory profile."""
        helix_gen = HelixMotion()
        
        # Get helix axis (default Z for WRF, transformed for TRF)
        if self.helix_axis is not None:
            axis = self.helix_axis
        else:
            axis = [0, 0, 1]  # Default vertical
        
        # Calculate distance from start position to helix start point
        start_pos = np.array(effective_start_pose[:3])
        center_np = np.array(self.center)
        
        # Project start position onto the helix plane (perpendicular to axis)
        axis_np = np.array(axis)
        axis_np = axis_np / np.linalg.norm(axis_np)
        to_start = start_pos - center_np
        to_start_plane = to_start - np.dot(to_start, axis_np) * axis_np
        dist_to_center = np.linalg.norm(to_start_plane)
        
        # Check if entry trajectory is needed
        entry_trajectory = None
        distance_from_perimeter = abs(dist_to_center - self.radius)
        
        if distance_from_perimeter > self.radius * 0.05:  # More than 5% off the perimeter
            logger.info(f"    Generating helix entry trajectory (distance from perimeter: {distance_from_perimeter:.1f}mm)")
            
            # Calculate entry duration based on distance (0.5s min, 2.0s max)
            entry_duration = min(2.0, max(0.5, distance_from_perimeter / 50.0))
            
            # Generate entry trajectory to helix start position
            motion_gen = CircularMotion()
            
            # Calculate the target position on the helix perimeter
            if dist_to_center > 0.001:
                direction = to_start_plane / dist_to_center
            else:
                # If exactly at center, move to any point on perimeter
                u = np.array([1, 0, 0]) if abs(axis_np[0]) < 0.9 else np.array([0, 1, 0])
                u = u - np.dot(u, axis_np) * axis_np
                direction = u / np.linalg.norm(u)
            
            target_on_perimeter = center_np + direction * self.radius
            # For helix, we want to start at the correct Z level
            target_on_perimeter[2] = start_pos[2]  # Keep same Z as start
            
            # Generate smooth approach trajectory
            entry_trajectory = self._generate_radial_entry(
                effective_start_pose, center_np, axis_np, self.radius, entry_duration
            )
            
            if entry_trajectory is not None and len(entry_trajectory) > 0:
                logger.info(f"    Helix entry trajectory generated: {len(entry_trajectory)} points over {entry_duration:.1f}s")
        
        # Generate main helix trajectory
        trajectory = helix_gen.generate_helix_with_profile(
            center=self.center,
            radius=self.radius,
            pitch=self.pitch,
            height=self.height,
            axis=axis_np,
            duration=self.duration,
            trajectory_type=self.trajectory_type,
            jerk_limit=self.jerk_limit,
            start_point=effective_start_pose,
            clockwise=self.clockwise
        )
        
        # Update orientations to match effective start
        for i in range(len(trajectory)):
            trajectory[i][3:] = effective_start_pose[3:]
        
        # Concatenate entry trajectory if it exists
        if entry_trajectory is not None and len(entry_trajectory) > 0:
            full_trajectory = np.concatenate([entry_trajectory, trajectory])
            return full_trajectory
        else:
            return np.array(trajectory)


@register_command("SMOOTH_SPLINE")
class SmoothSplineCommand(BaseSmoothMotionCommand):
    """Execute smooth spline motion through waypoints."""
    
    __slots__ = (
        "waypoints",
        "duration",
        "frame",
        "trajectory_type",
        "jerk_limit",
        "current_position_in",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth spline")
        self.waypoints: Optional[List[List[float]]] = None
        self.duration: float = 5.0
        self.frame: str = 'WRF'
        self.trajectory_type: str = 'cubic'
        self.jerk_limit: Optional[float] = None
        self.current_position_in: Optional[Sequence[int]] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_SPLINE command.
        Format: SMOOTH_SPLINE|wp1_x,y,z,rx,ry,rz;wp2_x,y,z,rx,ry,rz;...|frame|start_pose|timing_type|timing_value|[trajectory_type]|[jerk_limit]
        """
        if parts[0].upper() != "SMOOTH_SPLINE":
            return False, None
        
        if len(parts) < 6:
            return False, "SMOOTH_SPLINE requires at least 6 parameters"
        
        # Support alternate wire format:
        # SMOOTH_SPLINE|<count>|<frame>|<start_pose>|<DURATION|SPEED>|<value>|[trajectory_type]|[jerk?]|<flattened waypoints...>
        if len(parts) >= 7 and parts[1].isdigit():
            try:
                num = int(parts[1])
                self.frame = parts[2].upper()
                if self.frame not in ['WRF', 'TRF']:
                    return False, f"Invalid frame: {self.frame}"
                self.specified_start_pose = self.parse_start_pose(parts[3])
                timing_type = parts[4].upper()
                timing_value = float(parts[5])
                idx = 6
                if idx < len(parts) and parts[idx].lower() in ['cubic', 'quintic', 's_curve']:
                    self.trajectory_type = parts[idx].lower()
                    idx += 1
                    if self.trajectory_type == 's_curve' and idx < len(parts):
                        self.jerk_limit = float(parts[idx])
                        idx += 1
                needed = num * 6
                if len(parts) - idx < needed:
                    return False, "Insufficient waypoint values"
                vals = list(map(float, parts[idx:idx + needed]))
                self.waypoints = [vals[i:i + 6] for i in range(0, needed, 6)]
                # Estimate path length
                path_length = 0.0
                for i in range(1, len(self.waypoints)):
                    path_length += float(np.linalg.norm(np.array(self.waypoints[i][:3]) - np.array(self.waypoints[i - 1][:3])))
                self.duration = self.parse_timing(timing_type, timing_value, path_length)
                self.description = f"spline ({len(self.waypoints)} points, {self.frame}, {self.trajectory_type})"
                return True, None
            except Exception as e:
                return False, f"Invalid SMOOTH_SPLINE parameters: {e}"
        
        try:
            # Parse waypoints (semicolon separated)
            waypoint_strs = parts[1].split(';')
            self.waypoints = []
            for wp_str in waypoint_strs:
                wp = list(map(float, wp_str.split(',')))
                if len(wp) != 6:
                    return False, f"Each waypoint must have 6 values (x,y,z,rx,ry,rz)"
                self.waypoints.append(wp)
            
            if len(self.waypoints) < 2:
                return False, "SMOOTH_SPLINE requires at least 2 waypoints"
            
            # Parse frame
            self.frame = parts[2].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse start pose
            self.specified_start_pose = self.parse_start_pose(parts[3])
            
            # Parse timing
            timing_type = parts[4].upper()
            timing_value = float(parts[5])
            # Estimate path length from waypoints
            path_length = 0
            for i in range(1, len(self.waypoints)):
                path_length += np.linalg.norm(
                    np.array(self.waypoints[i][:3]) - np.array(self.waypoints[i-1][:3])
                )
            self.duration = self.parse_timing(timing_type, timing_value, path_length)
            
            # Parse optional trajectory type and jerk limit
            idx = 6
            if idx < len(parts):
                self.trajectory_type, self.jerk_limit, idx = self.parse_trajectory_type(parts, idx)
            
            # Initialize description
            self.description = f"spline ({len(self.waypoints)} points, {self.frame}, {self.trajectory_type})"
            
            return True, None
            
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_SPLINE parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform parameters if in TRF, then prepare normally."""

        self.current_position_in = state.Position_in
        
        if self.frame == 'TRF':
            # Transform waypoints to WRF
            params = {'waypoints': self.waypoints}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_SPLINE', params, 'TRF', state.Position_in
            )
            
            # Update with transformed values
            self.waypoints = transformed['waypoints']
            
            logger.info(f"  -> TRF Spline: transformed {len(self.waypoints)} waypoints to WRF")
            
            # Transform start_pose if specified
            self.specified_start_pose = transform_start_pose_if_needed(
                self.specified_start_pose, self.frame, state.Position_in
            )
        
        return super().do_setup(state)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate spline starting from actual position."""
        motion_gen = SplineMotion()
        
        # Always start from the effective start pose
        first_wp_error = np.linalg.norm(
            np.array(self.waypoints[0][:3]) - np.array(effective_start_pose[:3])
        )
        
        if first_wp_error > 5.0:
            # First waypoint is far, prepend the start position
            modified_waypoints = [effective_start_pose] + self.waypoints
            logger.info(f"    Added start position as first waypoint (distance: {first_wp_error:.1f}mm)")
        else:
            # Replace first waypoint with actual start to ensure continuity
            modified_waypoints = [effective_start_pose] + self.waypoints[1:]
            logger.info("    Replaced first waypoint with actual start position")
        
        timestamps = np.linspace(0, self.duration, len(modified_waypoints))
        
        # Generate the spline trajectory based on type
        if self.trajectory_type == 'quintic':
            trajectory = motion_gen.generate_quintic_spline(
                modified_waypoints, timestamps=None
            )
        elif self.trajectory_type == 's_curve':
            trajectory = motion_gen.generate_scurve_spline(
                modified_waypoints, duration=self.duration,
                jerk_limit=self.jerk_limit if self.jerk_limit else 1000
            )
        else:  # cubic (default)
            trajectory = motion_gen.generate_cubic_spline(modified_waypoints, timestamps)
        
        logger.debug(f"    Generated spline with {len(trajectory)} points")
        
        return trajectory


@register_command("SMOOTH_BLEND")
class SmoothBlendCommand(BaseSmoothMotionCommand):
    """Execute smooth blended trajectory through multiple segments."""
    
    __slots__ = (
        "segment_definitions",
        "blend_time",
        "frame",
        "trajectory_type",
        "jerk_limit",
        "current_position_in",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth blend")
        self.segment_definitions: Optional[List[dict]] = None
        self.blend_time: float = 0.5
        self.frame: str = 'WRF'
        self.trajectory_type: str = 'cubic'
        self.jerk_limit: Optional[float] = None
        self.current_position_in: Optional[Sequence[int]] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_BLEND command.
        Format: SMOOTH_BLEND|segments_json|blend_time|frame|start_pose|[trajectory_type]|[jerk_limit]
        """
        if parts[0].upper() != "SMOOTH_BLEND":
            return False, None
        
        if len(parts) < 5:
            return False, "SMOOTH_BLEND requires at least 5 parameters"
        
        # New wire format: SMOOTH_BLEND|num_segments|blend_time|frame|start_pose|timing|SEG1||SEG2||...
        if parts[1].isdigit():
            try:
                num_segments = int(parts[1])
                self.blend_time = float(parts[2])
                self.frame = parts[3].upper()
                if self.frame not in ['WRF', 'TRF']:
                    return False, f"Invalid frame: {self.frame}"
                self.specified_start_pose = self.parse_start_pose(parts[4])
                # parts[5] timing token (DEFAULT/DURATION/SPEED) not strictly needed for segments
                # Reconstruct remainder and split by '||' to obtain segments
                remainder = "|".join(parts[6:])
                raw_segments = [s for s in remainder.split("||") if s.strip() != ""]
                seg_defs = []
                for seg_str in raw_segments:
                    tokens = [t for t in seg_str.split("|") if t != ""]
                    if not tokens:
                        continue
                    seg_type = tokens[0].upper()
                    if seg_type == "LINE":
                        if len(tokens) < 3:
                            return False, "LINE segment requires end pose and duration"
                        end = list(map(float, tokens[1].split(",")))
                        duration = float(tokens[2])
                        seg_defs.append({"type": "LINE", "end": end, "duration": duration})
                    elif seg_type == "CIRCLE":
                        if len(tokens) < 6:
                            return False, "CIRCLE segment requires center,radius,plane,duration,clockwise"
                        center = list(map(float, tokens[1].split(",")))
                        radius = float(tokens[2])
                        plane = tokens[3].upper()
                        duration = float(tokens[4])
                        clockwise = tokens[5] in ("1", "TRUE", "True", "true", "CW", "CLOCKWISE")
                        seg_defs.append({"type": "CIRCLE", "center": center, "radius": radius, "plane": plane, "duration": duration, "clockwise": clockwise})
                    elif seg_type == "ARC":
                        if len(tokens) < 5:
                            return False, "ARC segment requires end,center,duration,clockwise"
                        end = list(map(float, tokens[1].split(",")))
                        center = list(map(float, tokens[2].split(",")))
                        duration = float(tokens[3])
                        clockwise = tokens[4] in ("1", "TRUE", "True", "true", "CW", "CLOCKWISE")
                        seg_defs.append({"type": "ARC", "end": end, "center": center, "duration": duration, "clockwise": clockwise})
                    elif seg_type == "SPLINE":
                        if len(tokens) < 4:
                            return False, "SPLINE segment requires count,waypoints,duration"
                        count = int(tokens[1])
                        wp_tokens = tokens[2].split(";")
                        if len(wp_tokens) != count:
                            return False, "SPLINE waypoint count mismatch"
                        waypoints = [list(map(float, wp.split(","))) for wp in wp_tokens]
                        duration = float(tokens[3])
                        seg_defs.append({"type": "SPLINE", "waypoints": waypoints, "duration": duration})
                    else:
                        return False, f"Invalid segment type: {seg_type}"
                self.segment_definitions = seg_defs
                self.description = f"blended ({len(self.segment_definitions)} segments, {self.frame}, {self.trajectory_type})"
                return True, None
            except Exception as e:
                return False, f"Invalid SMOOTH_BLEND parameters: {e}"
        
        try:
            # Parse segment definitions (JSON format)
            self.segment_definitions = json.loads(parts[1])
            
            # Validate segment definitions
            if not isinstance(self.segment_definitions, list):
                return False, "Segments must be a list"
            
            for seg in self.segment_definitions:
                if 'type' not in seg:
                    return False, "Each segment must have a 'type' field"
                if seg['type'] not in ['LINE', 'ARC', 'CIRCLE', 'SPLINE']:
                    return False, f"Invalid segment type: {seg['type']}"
            
            # Parse blend time
            self.blend_time = float(parts[2])
            
            # Parse frame
            self.frame = parts[3].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse start pose
            self.specified_start_pose = self.parse_start_pose(parts[4])
            
            # Parse optional trajectory type and jerk limit
            idx = 5
            if idx < len(parts):
                self.trajectory_type, self.jerk_limit, idx = self.parse_trajectory_type(parts, idx)
            
            # Initialize description
            self.description = f"blended ({len(self.segment_definitions)} segments, {self.frame}, {self.trajectory_type})"
            
            return True, None
            
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            return False, f"Invalid SMOOTH_BLEND parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform parameters if in TRF, then prepare normally."""

        self.current_position_in = state.Position_in
        
        if self.frame == 'TRF':
            # Transform all segment definitions to WRF
            params = {'segments': self.segment_definitions}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_BLEND', params, 'TRF', state.Position_in
            )
            
            # Update with transformed values
            self.segment_definitions = transformed['segments']
            
            logger.info(f"  -> TRF Blend: transformed {len(self.segment_definitions)} segments to WRF")
            
            # Transform start_pose if specified
            self.specified_start_pose = transform_start_pose_if_needed(
                self.specified_start_pose, self.frame, state.Position_in
            )
        
        return super().do_setup(state)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate blended trajectory starting from actual position."""
        trajectories = []
        motion_gen_circle = CircularMotion()
        motion_gen_spline = SplineMotion()
        
        # Always start from effective start pose
        last_end_pose = effective_start_pose
        
        for i, seg_def in enumerate(self.segment_definitions):
            seg_type = seg_def['type']
            
            # First segment always starts from effective_start_pose
            segment_start = effective_start_pose if i == 0 else last_end_pose
            
            if seg_type == 'LINE':
                end = seg_def['end']
                duration = seg_def['duration']
                
                # Generate line segment from actual position
                num_points = int(duration * 100)
                timestamps = np.linspace(0, duration, num_points)
                
                traj = []
                for t in timestamps:
                    s = t / duration if duration > 0 else 1
                    # Interpolate position
                    pos = [
                        segment_start[j] * (1-s) + end[j] * s
                        for j in range(3)
                    ]
                    # Interpolate orientation
                    orient = [
                        segment_start[j+3] * (1-s) + end[j+3] * s
                        for j in range(3)
                    ]
                    traj.append(pos + orient)
                
                trajectories.append(np.array(traj))
                last_end_pose = end
                
            elif seg_type == 'ARC':
                end = seg_def['end']
                center = seg_def['center']
                duration = seg_def['duration']
                clockwise = seg_def.get('clockwise', False)
                
                # Check if we have a transformed normal (from TRF)
                normal = seg_def.get('normal_vector', None)
                
                traj = motion_gen_circle.generate_arc_3d(
                    segment_start, end, center, 
                    normal=normal,  # Use transformed normal if available
                    clockwise=clockwise, duration=duration
                )
                trajectories.append(traj)
                last_end_pose = end
                
            elif seg_type == 'CIRCLE':
                center = seg_def['center']
                radius = seg_def['radius']
                plane = seg_def.get('plane', 'XY')
                duration = seg_def['duration']
                clockwise = seg_def.get('clockwise', False)
                
                # Use transformed normal if available (from TRF)
                if 'normal_vector' in seg_def:
                    normal = seg_def['normal_vector']
                else:
                    plane_normals = {'XY': [0, 0, 1], 'XZ': [0, 1, 0], 'YZ': [1, 0, 0]}
                    normal = plane_normals.get(plane, [0, 0, 1])
                
                traj = motion_gen_circle.generate_circle_3d(
                    center, radius, normal, 
                    start_point=segment_start[:3],
                    duration=duration
                )
                
                if clockwise:
                    traj = traj[::-1]
                    
                # Update orientations
                for j in range(len(traj)):
                    traj[j][3:] = segment_start[3:]
                    
                trajectories.append(traj)
                # Circle returns to start, so last pose is last point of trajectory
                last_end_pose = traj[-1]
                
            elif seg_type == 'SPLINE':
                waypoints = seg_def['waypoints']
                duration = seg_def['duration']
                
                # Check if first waypoint is close to segment start
                wp_error = np.linalg.norm(
                    np.array(waypoints[0][:3]) - np.array(segment_start[:3])
                )
                
                if wp_error > 5.0:
                    full_waypoints = [segment_start] + waypoints
                else:
                    full_waypoints = [segment_start] + waypoints[1:]
                
                timestamps = np.linspace(0, duration, len(full_waypoints))
                traj = motion_gen_spline.generate_cubic_spline(full_waypoints, timestamps)
                trajectories.append(traj)
                last_end_pose = waypoints[-1]
        
        # Blend all trajectories with advanced blending
        if len(trajectories) > 1:
            # Select blend method based on trajectory type
            if self.trajectory_type == 'quintic':
                blend_method = 'quintic'
            elif self.trajectory_type == 's_curve':
                blend_method = 's_curve'
            elif self.trajectory_type == 'cubic':
                blend_method = 'cubic'
            else:
                blend_method = 'smoothstep'  # Legacy compatibility
            
            # Create advanced blender
            advanced_blender = AdvancedMotionBlender(sample_rate=100.0)
            blended = trajectories[0]
            
            # Use auto sizing if blend_time is small, otherwise use specified time
            if self.blend_time < 0.1:
                auto_size = True
                blend_samples = None
            else:
                auto_size = False
                blend_samples = int(self.blend_time * 100)
            
            for i in range(1, len(trajectories)):
                blended = advanced_blender.blend_trajectories(
                    blended, trajectories[i], 
                    method=blend_method,
                    blend_samples=blend_samples,
                    auto_size=auto_size
                )
            
            logger.info(f"    Blended {len(trajectories)} segments into {len(blended)} points using {blend_method}")
            return blended
        elif trajectories:
            return trajectories[0]
        else:
            raise ValueError("No trajectories generated in blend")


@register_command("SMOOTH_WAYPOINTS")
class SmoothWaypointsCommand(BaseSmoothMotionCommand):
    """Execute waypoint trajectory with corner cutting."""
    
    __slots__ = (
        "waypoints",
        "blend_radii",
        "blend_mode",
        "via_modes",
        "max_velocity",
        "max_acceleration",
        "trajectory_type",
        "frame",
        "duration",
    )
    def __init__(self) -> None:
        super().__init__(description="smooth waypoints")
        self.waypoints: Optional[List[List[float]]] = None
        self.blend_radii: Any = 'auto'
        self.blend_mode: str = 'parabolic'
        self.via_modes: Optional[List[str]] = None
        self.max_velocity: float = 100.0
        self.max_acceleration: float = 500.0
        self.trajectory_type: str = 'quintic'
        self.frame: str = 'WRF'
        self.duration: Optional[float] = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SMOOTH_WAYPOINTS command.
        Format: SMOOTH_WAYPOINTS|wp1;wp2;...|blend_radii|blend_mode|via_modes|max_vel|max_accel|frame|[trajectory_type]|[duration]
        """
        if parts[0].upper() != "SMOOTH_WAYPOINTS":
            return False, None
        
        if len(parts) < 8:
            return False, "SMOOTH_WAYPOINTS requires at least 8 parameters"
        
        try:
            # Parse waypoints (semicolon separated)
            waypoint_strs = parts[1].split(';')
            self.waypoints = []
            for wp_str in waypoint_strs:
                wp = list(map(float, wp_str.split(',')))
                if len(wp) != 6:
                    return False, f"Each waypoint must have 6 values (x,y,z,rx,ry,rz)"
                self.waypoints.append(wp)
            
            if len(self.waypoints) < 2:
                return False, "SMOOTH_WAYPOINTS requires at least 2 waypoints"
            
            # Parse blend radii
            if parts[2].upper() == 'AUTO':
                self.blend_radii = 'auto'
            else:
                self.blend_radii = list(map(float, parts[2].split(',')))
                if len(self.blend_radii) != len(self.waypoints) - 2:
                    return False, f"Blend radii count must be {len(self.waypoints) - 2}"
            
            # Parse blend mode
            self.blend_mode = parts[3].lower()
            if self.blend_mode not in ['parabolic', 'circular', 'none']:
                return False, f"Invalid blend mode: {self.blend_mode}"
            
            # Parse via modes
            via_mode_strs = parts[4].split(',')
            self.via_modes = []
            for vm in via_mode_strs:
                vm = vm.lower()
                if vm not in ['via', 'stop']:
                    return False, f"Invalid via mode: {vm}"
                self.via_modes.append(vm)
            
            if len(self.via_modes) != len(self.waypoints):
                return False, f"Via modes count must match waypoint count"
            
            # Parse velocity and acceleration constraints
            self.max_velocity = float(parts[5])
            self.max_acceleration = float(parts[6])
            
            # Parse frame
            self.frame = parts[7].upper()
            if self.frame not in ['WRF', 'TRF']:
                return False, f"Invalid frame: {self.frame}"
            
            # Parse optional trajectory type
            idx = 8
            if idx < len(parts):
                self.trajectory_type = parts[idx].lower()
                if self.trajectory_type not in ['cubic', 'quintic', 's_curve']:
                    self.trajectory_type = 'quintic'
                idx += 1
            
            # Parse optional duration
            if idx < len(parts):
                self.duration = float(parts[idx])
            
            # Initialize description
            self.description = f"waypoints ({len(self.waypoints)} points, {self.frame}, {self.blend_mode})"
            
            return True, None
            
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_WAYPOINTS parameters: {e}"
    
    def do_setup(self, state: 'ControllerState') -> None:
        """Transform waypoints if in TRF."""

        if self.frame == 'TRF':
            # Transform all waypoints to WRF
            tool_pose = get_fkine_se3()
            transformed_waypoints = []
            for wp in self.waypoints:
                transformed_wp = pose6_trf_to_wrf(wp, tool_pose)
                transformed_waypoints.append(transformed_wp)
            
            self.waypoints = transformed_waypoints
            logger.info(f"  -> TRF Waypoints: transformed {len(self.waypoints)} points to WRF")
        
        # Basic validation
        if len(self.waypoints) < 2:
            self.fail("At least 2 waypoints required")
            return
        
        return super().do_setup(state)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate waypoint trajectory with corner cutting."""
        
        # Ensure first waypoint matches effective start pose
        first_wp_error = np.linalg.norm(
            np.array(self.waypoints[0][:3]) - np.array(effective_start_pose[:3])
        )
        
        if first_wp_error > 10.0:
            # Prepend effective start pose as first waypoint
            full_waypoints = [effective_start_pose] + self.waypoints
            if self.blend_radii != 'auto' and isinstance(self.blend_radii, list):
                # Prepend 0 blend radius for start
                full_blend_radii = [0] + self.blend_radii
            else:
                full_blend_radii = self.blend_radii
            full_via_modes = ['via'] + self.via_modes
        else:
            # Replace first waypoint with effective start pose
            full_waypoints = [effective_start_pose] + self.waypoints[1:]
            full_blend_radii = self.blend_radii
            full_via_modes = self.via_modes
        
        # Set up constraints
        constraints = {
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration,
            'max_jerk': 5000.0  # Default jerk limit
        }
        
        # Create planner
        planner = WaypointTrajectoryPlanner(
            full_waypoints,
            constraints=constraints,
            sample_rate=100.0
        )
        
        # Determine blend mode for planner
        if self.blend_mode == 'none':
            planner_blend_mode = 'none'
        elif self.blend_radii == 'auto':
            planner_blend_mode = 'auto'
        else:
            planner_blend_mode = 'manual'
        
        # Generate trajectory with direct profile support
        if planner_blend_mode == 'manual' and isinstance(full_blend_radii, list):
            opt_radii = [float(r) for r in full_blend_radii]
        else:
            opt_radii = None
        trajectory = planner.plan_trajectory(
            blend_mode=planner_blend_mode,
            blend_radii=opt_radii,
            via_modes=full_via_modes,
            trajectory_type=self.trajectory_type,
            jerk_limit=constraints['max_jerk'] if self.trajectory_type == 's_curve' else None
        )
        
        # Apply duration scaling if specified
        if self.duration and self.duration > 0:
            current_duration = len(trajectory) / 100.0
            if current_duration > 0:
                scale_factor = self.duration / current_duration
                if scale_factor != 1.0:
                    # Resample trajectory for desired duration
                    new_length = int(self.duration * 100)
                    old_indices = np.linspace(0, len(trajectory) - 1, new_length)
                    resampled = []
                    for idx in old_indices:
                        if idx < len(trajectory) - 1:
                            i = int(idx)
                            alpha = idx - i
                            pose = trajectory[i] * (1 - alpha) + trajectory[i + 1] * alpha
                        else:
                            pose = trajectory[-1]
                        resampled.append(pose)
                    trajectory = np.array(resampled)
        
        logger.info(f"    Generated waypoint trajectory with {len(trajectory)} points")
        return trajectory
