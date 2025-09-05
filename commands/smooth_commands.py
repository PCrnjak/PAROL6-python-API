"""
Smooth Motion Commands
Contains all smooth trajectory generation commands for advanced robot movements
"""

import logging
import numpy as np
import PAROL6_ROBOT
from spatialmath import SE3
from smooth_motion import (
    CircularMotion, SplineMotion, MotionBlender,
    HelixMotion, AdvancedMotionBlender, WaypointTrajectoryPlanner
)
from .ik_helpers import solve_ik_with_adaptive_tol_subdivision

logger = logging.getLogger(__name__)

# Import MovePoseCommand for transition commands
from .cartesian_commands import MovePoseCommand

def transform_command_params_to_wrf(command_type: str, params: dict, frame: str, current_position_in) -> dict:
    """
    Transform command parameters from TRF to WRF.
    Handles position, orientation, and directional vectors correctly.
    """
    if frame == 'WRF':
        return params
    
    # Get current tool pose
    current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                         for i, p in enumerate(current_position_in)])
    tool_pose = PAROL6_ROBOT.robot.fkine(current_q)
    
    transformed = params.copy()
    
    # SMOOTH_CIRCLE - Transform center and plane normal
    if command_type == 'SMOOTH_CIRCLE':
        if 'center' in params:
            center_trf = SE3(params['center'][0]/1000, 
                           params['center'][1]/1000, 
                           params['center'][2]/1000)
            center_wrf = tool_pose * center_trf
            transformed['center'] = (center_wrf.t * 1000).tolist()
        
        if 'plane' in params:
            plane_normals_trf = {
                'XY': [0, 0, 1],   # Tool's Z-axis
                'XZ': [0, 1, 0],   # Tool's Y-axis  
                'YZ': [1, 0, 0]    # Tool's X-axis
            }
            normal_trf = np.array(plane_normals_trf[params['plane']])
            normal_wrf = tool_pose.R @ normal_trf
            transformed['normal_vector'] = normal_wrf.tolist()
            logger.info(f"  -> TRF circle plane {params['plane']} transformed to WRF")
    
    # SMOOTH_ARC_CENTER - Transform center, end_pose, and implied plane
    elif command_type == 'SMOOTH_ARC_CENTER':
        if 'center' in params:
            center_trf = SE3(params['center'][0]/1000, 
                           params['center'][1]/1000, 
                           params['center'][2]/1000)
            center_wrf = tool_pose * center_trf
            transformed['center'] = (center_wrf.t * 1000).tolist()
        
        if 'end_pose' in params:
            end_trf = SE3(params['end_pose'][0]/1000, 
                         params['end_pose'][1]/1000, 
                         params['end_pose'][2]/1000) * \
                      SE3.RPY(params['end_pose'][3:], unit='deg', order='xyz')
            end_wrf = tool_pose * end_trf
            transformed['end_pose'] = np.concatenate([
                end_wrf.t * 1000,
                end_wrf.rpy(unit='deg', order='xyz')
            ]).tolist()
        
        # Arc plane is determined by start, end, and center points
        # But we should transform any specified plane normal
        if 'plane' in params:
            # Similar to circle plane transformation
            plane_normals_trf = {
                'XY': [0, 0, 1],
                'XZ': [0, 1, 0],
                'YZ': [1, 0, 0]
            }
            normal_trf = np.array(plane_normals_trf[params['plane']])
            normal_wrf = tool_pose.R @ normal_trf
            transformed['normal_vector'] = normal_wrf.tolist()
    
    # SMOOTH_ARC_PARAM - Transform end_pose and arc plane
    elif command_type == 'SMOOTH_ARC_PARAM':
        if 'end_pose' in params:
            end_trf = SE3(params['end_pose'][0]/1000, 
                         params['end_pose'][1]/1000, 
                         params['end_pose'][2]/1000) * \
                      SE3.RPY(params['end_pose'][3:], unit='deg', order='xyz')
            end_wrf = tool_pose * end_trf
            transformed['end_pose'] = np.concatenate([
                end_wrf.t * 1000,
                end_wrf.rpy(unit='deg', order='xyz')
            ]).tolist()
        
        # For parametric arc, the plane is usually XY of the tool
        # Transform the assumed plane normal
        if 'plane' not in params:
            params['plane'] = 'XY'  # Default to XY plane
        
        plane_normals_trf = {
            'XY': [0, 0, 1],
            'XZ': [0, 1, 0],
            'YZ': [1, 0, 0]
        }
        normal_trf = np.array(plane_normals_trf[params.get('plane', 'XY')])
        normal_wrf = tool_pose.R @ normal_trf
        transformed['normal_vector'] = normal_wrf.tolist()
    
    # SMOOTH_HELIX - Transform center and helix axis
    elif command_type == 'SMOOTH_HELIX':
        if 'center' in params:
            center_trf = SE3(params['center'][0]/1000, 
                           params['center'][1]/1000, 
                           params['center'][2]/1000)
            center_wrf = tool_pose * center_trf
            transformed['center'] = (center_wrf.t * 1000).tolist()
        
        # Transform helix axis (default is Z-axis of tool)
        axis_trf = np.array([0, 0, 1])  # Tool's Z-axis
        axis_wrf = tool_pose.R @ axis_trf
        transformed['helix_axis'] = axis_wrf.tolist()
        
        # Transform up vector (default is Y-axis of tool)
        up_trf = np.array([0, 1, 0])  # Tool's Y-axis
        up_wrf = tool_pose.R @ up_trf
        transformed['up_vector'] = up_wrf.tolist()
    
    # SMOOTH_SPLINE - Transform waypoints
    elif command_type == 'SMOOTH_SPLINE':
        if 'waypoints' in params:
            transformed_waypoints = []
            for wp in params['waypoints']:
                wp_trf = SE3(wp[0]/1000, wp[1]/1000, wp[2]/1000) * \
                        SE3.RPY(wp[3:], unit='deg', order='xyz')
                wp_wrf = tool_pose * wp_trf
                transformed_wp = np.concatenate([
                    wp_wrf.t * 1000,
                    wp_wrf.rpy(unit='deg', order='xyz')
                ]).tolist()
                transformed_waypoints.append(transformed_wp)
            transformed['waypoints'] = transformed_waypoints
    
    # SMOOTH_BLEND - Transform all segment definitions
    elif command_type == 'SMOOTH_BLEND':
        if 'segments' in params:
            transformed_segments = []
            for seg in params['segments']:
                seg_transformed = seg.copy()
                
                # Transform based on segment type
                if seg['type'] == 'LINE':
                    if 'end' in seg:
                        end_trf = SE3(seg['end'][0]/1000, seg['end'][1]/1000, seg['end'][2]/1000) * \
                                 SE3.RPY(seg['end'][3:], unit='deg', order='xyz')
                        end_wrf = tool_pose * end_trf
                        seg_transformed['end'] = np.concatenate([
                            end_wrf.t * 1000,
                            end_wrf.rpy(unit='deg', order='xyz')
                        ]).tolist()
                
                elif seg['type'] == 'ARC':
                    if 'end' in seg:
                        end_trf = SE3(seg['end'][0]/1000, seg['end'][1]/1000, seg['end'][2]/1000) * \
                                 SE3.RPY(seg['end'][3:], unit='deg', order='xyz')
                        end_wrf = tool_pose * end_trf
                        seg_transformed['end'] = np.concatenate([
                            end_wrf.t * 1000,
                            end_wrf.rpy(unit='deg', order='xyz')
                        ]).tolist()
                    
                    if 'center' in seg:
                        center_trf = SE3(seg['center'][0]/1000, seg['center'][1]/1000, seg['center'][2]/1000)
                        center_wrf = tool_pose * center_trf
                        seg_transformed['center'] = (center_wrf.t * 1000).tolist()
                    
                    # Transform plane normal if specified
                    if 'plane' in seg:
                        plane_normals_trf = {
                            'XY': [0, 0, 1],
                            'XZ': [0, 1, 0],
                            'YZ': [1, 0, 0]
                        }
                        normal_trf = np.array(plane_normals_trf[seg['plane']])
                        normal_wrf = tool_pose.R @ normal_trf
                        seg_transformed['normal_vector'] = normal_wrf.tolist()
                
                elif seg['type'] == 'CIRCLE':
                    if 'center' in seg:
                        center_trf = SE3(seg['center'][0]/1000, seg['center'][1]/1000, seg['center'][2]/1000)
                        center_wrf = tool_pose * center_trf
                        seg_transformed['center'] = (center_wrf.t * 1000).tolist()
                    
                    if 'plane' in seg:
                        plane_normals_trf = {
                            'XY': [0, 0, 1],
                            'XZ': [0, 1, 0],
                            'YZ': [1, 0, 0]
                        }
                        normal_trf = np.array(plane_normals_trf[seg['plane']])
                        normal_wrf = tool_pose.R @ normal_trf
                        seg_transformed['normal_vector'] = normal_wrf.tolist()
                
                elif seg['type'] == 'SPLINE':
                    if 'waypoints' in seg:
                        transformed_wps = []
                        for wp in seg['waypoints']:
                            wp_trf = SE3(wp[0]/1000, wp[1]/1000, wp[2]/1000) * \
                                    SE3.RPY(wp[3:], unit='deg', order='xyz')
                            wp_wrf = tool_pose * wp_trf
                            transformed_wp = np.concatenate([
                                wp_wrf.t * 1000,
                                wp_wrf.rpy(unit='deg', order='xyz')
                            ]).tolist()
                            transformed_wps.append(transformed_wp)
                        seg_transformed['waypoints'] = transformed_wps
                
                transformed_segments.append(seg_transformed)
            transformed['segments'] = transformed_segments
    
    # Generic transformations for any command with these parameters
    if 'start_pose' in params:
        start_trf = SE3(params['start_pose'][0]/1000, 
                       params['start_pose'][1]/1000, 
                       params['start_pose'][2]/1000) * \
                   SE3.RPY(params['start_pose'][3:], unit='deg', order='xyz')
        start_wrf = tool_pose * start_trf
        transformed['start_pose'] = np.concatenate([
            start_wrf.t * 1000,
            start_wrf.rpy(unit='deg', order='xyz')
        ]).tolist()
    
    return transformed


class BaseSmoothMotionCommand:
    """
    Base class for all smooth motion commands with proper error tracking.
    """
    
    def __init__(self, description="smooth motion"):
        self.description = description
        self.trajectory = None
        self.trajectory_command = None
        self.transition_command = None
        self.is_valid = True
        self.is_finished = False
        self.specified_start_pose = None
        self.transition_complete = False
        self.trajectory_prepared = False
        self.error_state = False
        self.error_message = ""
        self.trajectory_generated = False  # NEW: Track if trajectory is generated
        
    def create_transition_command(self, current_pose, target_pose):
        """Create a MovePose command for smooth transition to start position."""
        pos_error = np.linalg.norm(
            np.array(target_pose[:3]) - np.array(current_pose[:3])
        )
        
        # Lower threshold to 2mm for more aggressive transition generation
        if pos_error < 2.0:  # Changed from 5.0mm to 2.0mm
            logger.error(f"  -> Already near start position (error: {pos_error:.1f}mm)")
            return None
        
        logger.error(f"  -> Creating smooth transition to start ({pos_error:.1f}mm away)")
        
        # Calculate transition speed based on distance
        # Slower for short distances, faster for long distances
        if pos_error < 10:
            transition_speed = 20.0  # mm/s for short distances
        elif pos_error < 30:
            transition_speed = 30.0  # mm/s for medium distances
        else:
            transition_speed = 40.0  # mm/s for long distances
        
        transition_duration = max(pos_error / transition_speed, 0.5)  # Minimum 0.5s
        
        transition_cmd = MovePoseCommand(
            pose=target_pose,
            duration=transition_duration
        )
        
        return transition_cmd
    
    def get_current_pose_from_position(self, position_in):
        """Convert current position to pose [x,y,z,rx,ry,rz]"""
        current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                             for i, p in enumerate(position_in)])
        current_pose_se3 = PAROL6_ROBOT.robot.fkine(current_q)
        
        current_xyz = current_pose_se3.t * 1000  # Convert to mm
        current_rpy = current_pose_se3.rpy(unit='deg', order='xyz')
        return np.concatenate([current_xyz, current_rpy]).tolist()
        
    def prepare_for_execution(self, current_position_in):
        """Minimal preparation - just check if we need a transition."""
        logger.debug(f"  -> Preparing {self.description}...")
        
        # If there's a specified start pose, prepare transition
        if self.specified_start_pose:
            actual_current_pose = self.get_current_pose_from_position(current_position_in)
            self.transition_command = self.create_transition_command(
                actual_current_pose, self.specified_start_pose
            )
            
            if self.transition_command:
                self.transition_command.prepare_for_execution(current_position_in)
                if not self.transition_command.is_valid:
                    logger.error(f"  -> ERROR: Cannot reach specified start position")
                    self.is_valid = False
                    self.error_state = True
                    self.error_message = "Cannot reach specified start position"
                    return
        else:
            self.transition_command = None
            
        # DON'T generate trajectory yet - wait until execution
        self.trajectory_generated = False
        self.trajectory_prepared = False
        logger.debug(f"  -> {self.description} preparation complete (trajectory will be generated at execution)")
            
    def generate_main_trajectory(self, effective_start_pose):
        """Override this in subclasses to generate the specific motion trajectory."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")
        
    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """Execute transition first if needed, then generate and execute trajectory."""
        if self.is_finished or not self.is_valid:
            return True
        
        # Execute transition first if needed
        if self.transition_command and not self.transition_complete:
            is_done = self.transition_command.execute_step(
                Position_in, Homed_in, Speed_out, Command_out, **kwargs
            )
            
            if is_done:
                logger.info(f"  -> Transition complete")
                self.transition_complete = True
            return False
        
        # Generate trajectory on first execution step (not during preparation!)
        if not self.trajectory_generated:
            # Get ACTUAL current position NOW
            actual_current_pose = self.get_current_pose_from_position(Position_in)
            logger.info(f"  -> Generating {self.description} from ACTUAL position: {[round(p, 1) for p in actual_current_pose[:3]]}")
            
            # Generate trajectory from where we ACTUALLY are
            self.trajectory = self.generate_main_trajectory(actual_current_pose)
            self.trajectory_command = SmoothTrajectoryCommand(
                self.trajectory, self.description
            )
            
            # Quick validation of first point only
            current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                                 for i, p in enumerate(Position_in)])
            first_pose = self.trajectory[0]
            target_se3 = SE3(first_pose[0]/1000, first_pose[1]/1000, first_pose[2]/1000) * \
                        SE3.RPY(first_pose[3:], unit='deg', order='xyz')
            
            ik_result = solve_ik_with_adaptive_tol_subdivision(
                PAROL6_ROBOT.robot, target_se3, current_q, ilimit=50, jogging=False
            )
            
            if not ik_result.success:
                logger.error(f"  -> ERROR: Cannot reach first trajectory point")
                self.is_finished = True
                self.error_state = True
                self.error_message = "Cannot reach trajectory start"
                Speed_out[:] = [0] * 6
                Command_out.value = 255
                return True
                
            self.trajectory_generated = True
            self.trajectory_prepared = True
            
            # Verify first point is close to current
            distance = np.linalg.norm(first_pose[:3] - np.array(actual_current_pose[:3]))
            if distance > 5.0:
                logger.warning(f"  -> WARNING: First trajectory point {distance:.1f}mm from current!")
        
        # Execute main trajectory
        if self.trajectory_command and self.trajectory_prepared:
            is_done = self.trajectory_command.execute_step(
                Position_in, Homed_in, Speed_out, Command_out, **kwargs
            )
            
            # Check for errors in trajectory execution
            if hasattr(self.trajectory_command, 'error_state') and self.trajectory_command.error_state:
                self.error_state = True
                self.error_message = self.trajectory_command.error_message
            
            if is_done:
                self.is_finished = True
            
            return is_done
        else:
            self.is_finished = True
            return True

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
    
    def prepare_for_execution(self, current_position_in):
        """Skip validation - trajectory is already generated from correct position"""
        # No validation needed since trajectory was just generated from current position
        self.is_valid = True
        return
    
    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, Position_out=None, **kwargs):
        """Execute one step of the smooth trajectory"""
        if self.is_finished or not self.is_valid:
            return True
        
        # Get Position_out from kwargs if not provided
        if Position_out is None:
            Position_out = kwargs.get('Position_out', Position_in)
        
        if self.trajectory_index >= len(self.trajectory):
            logger.info(f"Smooth {self.description} finished.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Get target pose for this step
        target_pose = self.trajectory[self.trajectory_index]
        
        # Convert to SE3
        target_se3 = SE3(target_pose[0]/1000, target_pose[1]/1000, target_pose[2]/1000) * \
                    SE3.RPY(target_pose[3:], unit='deg', order='xyz')
        
        # Get current joint configuration
        current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                            for i, p in enumerate(Position_in)])
        
        # Solve IK
        ik_result = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, target_se3, current_q, ilimit=50, jogging=False
        )
        
        if not ik_result.success:
            logger.error(f"  -> IK failed at trajectory point {self.trajectory_index}")
            self.is_finished = True
            self.error_state = True
            self.error_message = f"IK failed at point {self.trajectory_index}/{len(self.trajectory)}"
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Convert to steps
        target_steps = [int(PAROL6_ROBOT.RAD2STEPS(q, i)) 
                    for i, q in enumerate(ik_result.q)]
        
        # ADD VELOCITY LIMITING - This prevents violent movements
        if self.trajectory_index > 0:
            for i in range(6):
                step_diff = abs(target_steps[i] - Position_in[i])
                max_step_diff = PAROL6_ROBOT.Joint_max_speed[i] * 0.01  # Max steps in 10ms
                
                # Use 1.2x safety margin (not 2x as before)
                if step_diff > max_step_diff * 1.2:
                    #print(f"  -> WARNING: Joint {i+1} velocity limit exceeded at point {self.trajectory_index}")
                    #print(f"     Step difference: {step_diff}, Max allowed: {max_step_diff * 1.2:.1f}")
                    
                    # Clamp the motion
                    sign = 1 if target_steps[i] > Position_in[i] else -1
                    target_steps[i] = Position_in[i] + sign * int(max_step_diff)
        
        # Send position command
        Position_out[:] = target_steps
        Speed_out[:] = [0] * 6
        Command_out.value = 156
        
        # Advance to next point
        self.trajectory_index += 1
        
        return False

class SmoothCircleCommand(BaseSmoothMotionCommand):
    def __init__(self, center, radius, plane, duration, clockwise, frame='WRF', start_pose=None, 
                 trajectory_type='cubic', jerk_limit=None, center_mode='ABSOLUTE', entry_mode='NONE'):
        super().__init__(f"circle (r={radius}mm, {frame}, {trajectory_type})")
        self.center = center
        self.radius = radius
        self.plane = plane
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame  # Store reference frame
        self.specified_start_pose = start_pose
        self.trajectory_type = trajectory_type
        self.jerk_limit = jerk_limit
        self.center_mode = center_mode  # ABSOLUTE, TOOL, RELATIVE
        self.entry_mode = entry_mode    # AUTO, TANGENT, DIRECT, NONE
        self.normal_vector = None  # Will be set if TRF
        self.current_position_in = None  # Store for TRF transformation
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        # Store current position for potential use in generate_main_trajectory
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform parameters to WRF
            params = {
                'center': self.center,
                'plane': self.plane
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_CIRCLE', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.center = transformed['center']
            self.normal_vector = transformed.get('normal_vector')
            
            logger.info(f"  -> TRF Circle: center {self.center[:3]} (WRF), normal {self.normal_vector}")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_CIRCLE', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        # Now do normal preparation with transformed parameters
        return super().prepare_for_execution(current_position_in)
        
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
            normal = np.array(plane_normals.get(self.plane, [0, 0, 1]))  # Convert to numpy array
            logger.info(f"    Using WRF plane {self.plane} with normal: {normal}")
        
        logger.info(f"    Generating circle from position: {[round(p, 1) for p in effective_start_pose[:3]]}")
        logger.info(f"    Circle center: {[round(c, 1) for c in self.center]}")
        
        # Add geometry validation
        center_np = np.array(self.center)
        start_np = np.array(effective_start_pose[:3])
        
        # Project start point onto circle plane to check distance
        to_start = start_np - center_np
        to_start_plane = to_start - np.dot(to_start, normal) * normal
        distance_to_center = np.linalg.norm(to_start_plane)
        
        # Handle center_mode
        actual_center = self.center.copy()
        if self.center_mode == 'TOOL':
            # Center at current tool position - ensure it's a numpy array
            actual_center = np.array(effective_start_pose[:3])
            logger.info(f"    Center mode: TOOL - centering at current position {actual_center}")
        elif self.center_mode == 'RELATIVE':
            # Center relative to current position
            actual_center = np.array([effective_start_pose[i] + self.center[i] for i in range(3)])
            logger.info(f"    Center mode: RELATIVE - center offset from current position to {actual_center}")
        else:
            # ABSOLUTE mode uses provided center as-is, ensure it's a numpy array
            actual_center = np.array(actual_center)
        
        # Check if entry trajectory might be needed
        distance_to_center = np.linalg.norm(np.array(effective_start_pose[:3]) - np.array(actual_center))
        distance_from_perimeter = abs(distance_to_center - self.radius)
        
        # Automatically generate entry trajectory if needed
        entry_trajectory = None
        if distance_from_perimeter > 2.0:  # More than 2mm off the perimeter
            effective_entry_mode = self.entry_mode
            
            # Auto-detect need for entry if not specified
            if self.entry_mode == 'NONE' and distance_from_perimeter > 5.0:  # Auto-enable for > 5mm
                logger.warning(f"    Robot is {distance_from_perimeter:.1f}mm from circle perimeter - auto-enabling entry trajectory")
                effective_entry_mode = 'AUTO'
            
            if effective_entry_mode != 'NONE':
                logger.info(f"    Generating {effective_entry_mode} entry trajectory (distance: {distance_from_perimeter:.1f}mm)")
                
                # Calculate entry duration based on distance (0.5s min, 2.0s max)
                entry_duration = min(2.0, max(0.5, distance_from_perimeter / 50.0))
                
                # Generate entry trajectory  
                entry_trajectory = motion_gen.generate_circle_entry(
                    current_pos=effective_start_pose,
                    circle_center=actual_center,
                    radius=self.radius,
                    normal=normal,
                    duration=entry_duration,
                    profile_type='quintic',  # Always use quintic for smooth entry
                    control_rate=100.0
                )
                
                # Entry trajectory now returns full 6D poses, no need to add orientation
                if entry_trajectory is not None and len(entry_trajectory) > 0:
                    logger.info(f"    Entry trajectory generated: {len(entry_trajectory)} points over {entry_duration:.1f}s")
        
        # Generate circle with specified trajectory profile
        # Use new direct trajectory generation method
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
            # Combine entry and main trajectories
            full_trajectory = np.concatenate([entry_trajectory, trajectory])
            return full_trajectory
        else:
            return trajectory
    
class SmoothArcCenterCommand(BaseSmoothMotionCommand):
    def __init__(self, end_pose, center, duration, clockwise, frame='WRF', start_pose=None, trajectory_type='cubic', jerk_limit=None):
        super().__init__(f"arc (center-based, {frame}, {trajectory_type})")
        self.end_pose = end_pose
        self.center = center
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame
        self.specified_start_pose = start_pose
        self.trajectory_type = trajectory_type
        self.jerk_limit = jerk_limit
        self.normal_vector = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF."""
        if self.frame == 'TRF':
            params = {
                'end_pose': self.end_pose,
                'center': self.center
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_ARC_CENTER', params, 'TRF', current_position_in
            )
            self.end_pose = transformed['end_pose']
            self.center = transformed['center']
            self.normal_vector = transformed.get('normal_vector')
            
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_ARC_CENTER', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
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
    
class SmoothArcParamCommand(BaseSmoothMotionCommand):
    def __init__(self, end_pose, radius, arc_angle, duration, clockwise, frame='WRF', start_pose=None, trajectory_type='cubic', jerk_limit=None):
        super().__init__(f"parametric arc (r={radius}mm, θ={arc_angle}°, {frame}, {trajectory_type})")
        self.end_pose = end_pose
        self.radius = radius
        self.arc_angle = arc_angle
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame
        self.specified_start_pose = start_pose
        self.trajectory_type = trajectory_type
        self.jerk_limit = jerk_limit
        self.normal_vector = None  # Will be set if TRF
        self.current_position_in = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform parameters to WRF
            params = {
                'end_pose': self.end_pose,
                'plane': 'XY'  # Default plane for parametric arc
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_ARC_PARAM', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.end_pose = transformed['end_pose']
            self.normal_vector = transformed.get('normal_vector')
            
            logger.info(f"  -> TRF Parametric Arc: end {self.end_pose[:3]} (WRF)")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_ARC_PARAM', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc based on radius and angle from actual start."""
        # Get start and end positions
        start_xyz = np.array(effective_start_pose[:3])
        end_xyz = np.array(self.end_pose[:3])
        
        # If we have a transformed normal (TRF), use it to define the arc plane
        if self.normal_vector is not None:
            normal = np.array(self.normal_vector)
            
            # Project start and end onto the plane perpendicular to normal
            # This ensures the arc stays in the correct plane for TRF
            
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
        # Ensure center_3d is a list (it might already be one)
        center_list = center_3d if isinstance(center_3d, list) else center_3d.tolist()
        
        # Use new direct profile generation method
        trajectory = motion_gen.generate_arc_with_profile(
            effective_start_pose, self.end_pose, center_list,
            normal=normal if self.normal_vector is not None else None,
            clockwise=self.clockwise, 
            duration=self.duration,
            trajectory_type=self.trajectory_type,
            jerk_limit=self.jerk_limit
        )
        
        return trajectory

class SmoothHelixCommand(BaseSmoothMotionCommand):
    def __init__(self, center, radius, pitch, height, duration, clockwise, frame='WRF', start_pose=None, trajectory_type='cubic', jerk_limit=None):
        super().__init__(f"helix (h={height}mm, {frame}, {trajectory_type})")
        self.center = center
        self.radius = radius
        self.pitch = pitch
        self.height = height
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame
        self.specified_start_pose = start_pose
        self.trajectory_type = trajectory_type
        self.jerk_limit = jerk_limit
        self.helix_axis = None
        self.up_vector = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF."""
        if self.frame == 'TRF':
            params = {'center': self.center}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_HELIX', params, 'TRF', current_position_in
            )
            self.center = transformed['center']
            self.helix_axis = transformed.get('helix_axis', [0, 0, 1])
            self.up_vector = transformed.get('up_vector', [0, 1, 0])
            
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_HELIX', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate helix with entry trajectory if needed and proper trajectory profile."""
        # Import here to avoid circular dependencies
        from smooth_motion import HelixMotion, CircularMotion
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
            # target_on_perimeter is already a numpy array, just update Z component
            target_on_perimeter[2] = start_pos[2]  # Keep same Z as start
            
            # Generate smooth approach trajectory
            entry_trajectory = motion_gen.generate_circle_entry(
                current_pos=effective_start_pose,
                circle_center=center_np,
                radius=self.radius,
                normal=axis_np,
                duration=entry_duration,
                profile_type='quintic',
                control_rate=100.0
            )
            
            # Entry trajectory now returns full 6D poses, no need to add orientation
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

class SmoothSplineCommand(BaseSmoothMotionCommand):
    def __init__(self, waypoints, duration, frame='WRF', start_pose=None, trajectory_type='cubic', jerk_limit=None):
        super().__init__(f"spline ({len(waypoints)} points, {frame}, {trajectory_type})")
        self.waypoints = waypoints
        self.duration = duration
        self.frame = frame
        self.specified_start_pose = start_pose
        self.trajectory_type = trajectory_type
        self.jerk_limit = jerk_limit
        self.current_position_in = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform waypoints to WRF
            params = {'waypoints': self.waypoints}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_SPLINE', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.waypoints = transformed['waypoints']
            
            logger.info(f"  -> TRF Spline: transformed {len(self.waypoints)} waypoints to WRF")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_SPLINE', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
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
            logger.info(f"    Replaced first waypoint with actual start position")
        
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

class SmoothBlendCommand(BaseSmoothMotionCommand):
    def __init__(self, segment_definitions, blend_time, frame='WRF', start_pose=None, trajectory_type='cubic', jerk_limit=None):
        super().__init__(f"blended ({len(segment_definitions)} segments, {frame}, {trajectory_type})")
        self.segment_definitions = segment_definitions
        self.blend_time = blend_time
        self.frame = frame
        self.specified_start_pose = start_pose
        self.trajectory_type = trajectory_type
        self.jerk_limit = jerk_limit
        self.current_position_in = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform all segment definitions to WRF
            params = {'segments': self.segment_definitions}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_BLEND', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.segment_definitions = transformed['segments']
            
            logger.info(f"  -> TRF Blend: transformed {len(self.segment_definitions)} segments to WRF")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_BLEND', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
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
                clockwise = seg_def['clockwise']
                
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
                clockwise = seg_def['clockwise']
                
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
                last_end_pose = traj[-1].tolist()
                
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
            # Use AdvancedMotionBlender for better continuity
            from smooth_motion import AdvancedMotionBlender
            
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


class SmoothWaypointsCommand(BaseSmoothMotionCommand):
    """Execute waypoint trajectory with corner cutting."""
    
    def __init__(self, waypoints, blend_radii, blend_mode, via_modes,
                 max_velocity, max_acceleration, trajectory_type,
                 frame='WRF', duration=None):
        """
        Initialize waypoint trajectory command.
        
        Args:
            waypoints: List of waypoint poses
            blend_radii: 'auto' or list of blend radii
            blend_mode: 'parabolic', 'circular', or 'none'
            via_modes: List of 'via' or 'stop' for each waypoint
            max_velocity: Maximum velocity constraint
            max_acceleration: Maximum acceleration constraint
            trajectory_type: 'quintic', 's_curve', or 'cubic'
            frame: Reference frame
            duration: Optional total duration
        """
        super().__init__(f"waypoints ({len(waypoints)} points, {frame}, {blend_mode})")
        
        self.waypoints = waypoints
        self.blend_radii = blend_radii
        self.blend_mode = blend_mode
        self.via_modes = via_modes
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.trajectory_type = trajectory_type
        self.frame = frame
        self.duration = duration
        
    def prepare_for_execution(self, current_position_in):
        """Transform waypoints if in TRF."""
        if self.frame == 'TRF':
            # Transform all waypoints to WRF
            transformed_waypoints = []
            for wp in self.waypoints:
                params = {'point': wp[:3], 'orientation': wp[3:]}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_WAYPOINTS', params, 'TRF', current_position_in
                )
                transformed_wp = transformed['point'] + transformed['orientation']
                transformed_waypoints.append(transformed_wp)
            
            self.waypoints = transformed_waypoints
            logger.info(f"  -> TRF Waypoints: transformed {len(self.waypoints)} points to WRF")
        
        # Basic validation
        if len(self.waypoints) < 2:
            self.is_valid = False
            self.error_message = "At least 2 waypoints required"
            return
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate waypoint trajectory with corner cutting."""
        from smooth_motion import WaypointTrajectoryPlanner
        
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
        constraints = {}
        if self.max_velocity:
            constraints['max_velocity'] = self.max_velocity
        else:
            constraints['max_velocity'] = 100.0  # Default 100 mm/s
            
        if self.max_acceleration:
            constraints['max_acceleration'] = self.max_acceleration
        else:
            constraints['max_acceleration'] = 500.0  # Default 500 mm/s²
        
        constraints['max_jerk'] = 5000.0  # Default jerk limit
        
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
        trajectory = planner.plan_trajectory(
            blend_mode=planner_blend_mode,
            blend_radii=full_blend_radii if planner_blend_mode == 'manual' else None,
            via_modes=full_via_modes,
            trajectory_type=self.trajectory_type,
            jerk_limit=constraints['max_jerk'] if self.trajectory_type == 's_curve' else None
        )
        
        # Apply duration scaling if specified
        if self.duration and self.duration > 0:
            current_duration = len(trajectory) / 100.0  # 100Hz sampling
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
