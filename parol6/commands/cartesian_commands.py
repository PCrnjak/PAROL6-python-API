"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, and MoveCart
"""

import logging
import numpy as np
import time
from typing import List, Tuple, Optional
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from spatialmath import SE3
import roboticstoolbox as rp
from parol6.utils.ik import solve_ik_with_adaptive_tol_subdivision, quintic_scaling, AXIS_MAP
from .base import CommandBase, ExecutionStatus, ExecutionStatusCode
from parol6.protocol.wire import CommandCode
from parol6.config import JOG_IK_ILIMIT, INTERVAL_S
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)


@register_command("CARTJOG")
class CartesianJogCommand(CommandBase):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    This is the final, refactored version using clean, standard spatial math
    operations now that the core unit bug has been fixed.
    """
    def __init__(self):
        """
        Initializes the Cartesian jog command.
        Parameters are parsed in match() method.
        """
        super().__init__()
        
        # Parameters (set in match())
        self.frame = None
        self.axis = None
        self.speed_percentage = 50
        self.duration = 1.5
        
        # Runtime state
        self.axis_vectors = None
        self.is_rotation = False
        self.end_time = None
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse CARTJOG command parameters.
        
        Format: CARTJOG|frame|axis|speed_pct|duration
        Example: CARTJOG|WRF|+X|50|2.0
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 5:
            return (False, "CARTJOG requires 4 parameters: frame, axis, speed, duration")
        
        try:
            # Parse parameters
            self.frame = parts[1].upper()
            self.axis = parts[2]
            self.speed_percentage = float(parts[3])
            self.duration = float(parts[4])
            
            # Validate frame
            if self.frame not in ['WRF', 'TRF']:
                return (False, f"Invalid frame: {self.frame}. Must be WRF or TRF")
            
            # Validate axis
            if self.axis not in AXIS_MAP:
                return (False, f"Invalid axis: {self.axis}")
            
            # Store axis vectors for execution
            self.axis_vectors = AXIS_MAP[self.axis]
            self.is_rotation = any(self.axis_vectors[1])
            
            logger.info(f"Parsed CartesianJog: Frame {self.frame}, Axis {self.axis}, Speed {self.speed_percentage}%, Duration {self.duration}s")
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid CARTJOG parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing CARTJOG: {str(e)}")
    
    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        """Set the end time when the command actually starts."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter
        self.end_time = time.time() + self.duration
        logger.debug("  -> CartesianJog command is ready.")

    def execute_step(self, state) -> ExecutionStatus:
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        # --- A. Check for completion ---
        if time.time() >= self.end_time:
            logger.info("Cartesian jog finished.")
            self.is_finished = True
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.IDLE
            return ExecutionStatus.completed("CARTJOG complete")

        # --- B. Calculate Target Pose using clean vector math ---
        state.Command_out = CommandCode.JOG
        
        q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
        T_current = PAROL6_ROBOT.robot.fkine(q_current)

        if not isinstance(T_current, SE3):
            return ExecutionStatus.executing("Waiting for valid pose")

        # Calculate speed and displacement for this cycle
        linear_speed_ms = float(np.interp(self.speed_percentage, [0, 100], [PAROL6_ROBOT.Cartesian_linear_velocity_min_JOG, PAROL6_ROBOT.Cartesian_linear_velocity_max_JOG]))
        angular_speed_degs = float(np.interp(self.speed_percentage, [0, 100], [PAROL6_ROBOT.Cartesian_angular_velocity_min, PAROL6_ROBOT.Cartesian_angular_velocity_max]))

        delta_linear = linear_speed_ms * INTERVAL_S
        delta_angular_rad = np.deg2rad(angular_speed_degs * INTERVAL_S)

        # Create the small incremental transformation (delta_pose)
        # Use explicit per-axis rotations to match original GUI behavior
        trans_vec = np.array(self.axis_vectors[0]) * delta_linear
        rot_vec = np.array(self.axis_vectors[1]) * delta_angular_rad
        
        # Build delta transformation using explicit rotation matrices
        if np.any(rot_vec != 0):
            # Find which axis has rotation (should be only one for single-axis jog)
            if rot_vec[0] != 0:  # RX rotation
                delta_pose = SE3.Rx(rot_vec[0]) * SE3(trans_vec)
            elif rot_vec[1] != 0:  # RY rotation
                delta_pose = SE3.Ry(rot_vec[1]) * SE3(trans_vec)
            elif rot_vec[2] != 0:  # RZ rotation
                delta_pose = SE3.Rz(rot_vec[2]) * SE3(trans_vec)
            else:
                delta_pose = SE3(trans_vec)
        else:
            delta_pose = SE3(trans_vec)

        # Apply the transformation in the correct reference frame
        if self.frame == 'WRF':
            # Pre-multiply to apply the change in the World Reference Frame
            target_pose = delta_pose * T_current
        else: # TRF
            # Post-multiply to apply the change in the Tool Reference Frame
            target_pose = T_current * delta_pose
        
        # --- C. Solve IK and Calculate Velocities ---
        var = solve_ik_with_adaptive_tol_subdivision(PAROL6_ROBOT.robot, target_pose, q_current, ilimit=JOG_IK_ILIMIT, jogging=True)

        if var.success:
            q_velocities = (var.q - q_current) / INTERVAL_S
            for i in range(6):
                state.Speed_out[i] = int(PAROL6_ROBOT.SPEED_RAD2STEP(q_velocities[i], i))
        else:
            logger.warning("IK Warning: Could not find solution for jog step. Stopping.")
            self.is_finished = True
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.IDLE
            return ExecutionStatus.failed("IK failed for jog step")

        # --- D. Speed Scaling ---
        max_scale_factor = 1.0
        for i in range(6):
            if abs(state.Speed_out[i]) > PAROL6_ROBOT.Joint_max_speed[i]:
                scale = abs(state.Speed_out[i]) / PAROL6_ROBOT.Joint_max_speed[i]
                if scale > max_scale_factor:
                    max_scale_factor = scale
        
        if max_scale_factor > 1.0:
            for i in range(6):
                state.Speed_out[i] = int(state.Speed_out[i] / max_scale_factor)

        return ExecutionStatus.executing("CARTJOG")


@register_command("MOVEPOSE")
class MovePoseCommand(CommandBase):
    """
    A non-blocking command to move the robot to a specific Cartesian pose.
    The movement itself is a joint-space interpolation.
    """
    def __init__(self):
        super().__init__()
        self.command_step = 0
        self.trajectory_steps = []
        
        # Parameters (set in match())
        self.pose = None
        self.duration = None
        self.velocity_percent = None
        self.accel_percent = 50
        self.trajectory_type = 'poly'
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse MOVEPOSE command parameters.
        
        Format: MOVEPOSE|x|y|z|rx|ry|rz|duration|speed
        Example: MOVEPOSE|100|200|300|0|0|0|None|50
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 9:
            return (False, "MOVEPOSE requires 8 parameters: x, y, z, rx, ry, rz, duration, speed")
        
        try:
            # Parse pose (6 values)
            self.pose = [float(parts[i]) for i in range(1, 7)]
            
            # Parse duration and speed
            self.duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            self.velocity_percent = None if parts[8].upper() == 'NONE' else float(parts[8])
            
            logger.info(f"Parsed MovePose to {self.pose}")
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid MOVEPOSE parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing MOVEPOSE: {str(e)}")
    
    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        """Calculates the full trajectory just-in-time before execution."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

        logger.debug(f"  -> Preparing trajectory for MovePose to {self.pose}...")

        initial_pos_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
        target_pose = SE3(self.pose[0] / 1000.0, self.pose[1] / 1000.0, self.pose[2] / 1000.0) * SE3.RPY(self.pose[3:6], unit='deg', order='xyz')
        
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, target_pose, initial_pos_rad, ilimit=100)

        if not ik_solution.success:
            logger.warning("  -> VALIDATION FAILED: Inverse kinematics failed at execution time.")
            self.is_valid = False
            return

        target_pos_rad = ik_solution.q

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                logger.info("  -> INFO: Both duration and velocity were provided. Using duration.")
            command_len = int(self.duration / INTERVAL_S)
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len) # type: ignore
            
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))

        elif self.velocity_percent is not None:
            try:
                accel_percent = self.accel_percent if self.accel_percent is not None else 50
                
                initial_pos_steps = np.array(state.Position_in)
                target_pos_steps = np.array([int(PAROL6_ROBOT.RAD2STEPS(rad, i)) for i, rad in enumerate(target_pos_rad)])

                all_joint_times = []
                for i in range(6):
                    path_to_travel = abs(target_pos_steps[i] - initial_pos_steps[i])
                    if path_to_travel == 0:
                        all_joint_times.append(0)
                        continue
                    
                    v_max_joint = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Joint_min_speed[i], PAROL6_ROBOT.Joint_max_speed[i]])
                    a_max_rad = np.interp(accel_percent, [0, 100], [PAROL6_ROBOT.Joint_min_acc, PAROL6_ROBOT.Joint_max_acc])
                    a_max_steps = PAROL6_ROBOT.SPEED_RAD2STEP(a_max_rad, i)

                    if v_max_joint <= 0 or a_max_steps <= 0:
                        raise ValueError(f"Invalid speed/acceleration for joint {i+1}. Must be positive.")

                    t_accel = v_max_joint / a_max_steps
                    if path_to_travel < v_max_joint * t_accel:
                        t_accel = np.sqrt(path_to_travel / a_max_steps)
                        joint_time = 2 * t_accel
                    else:
                        joint_time = path_to_travel / v_max_joint + t_accel
                    all_joint_times.append(joint_time)
            
                total_time = max(all_joint_times)

                if total_time <= 0:
                    self.is_finished = True
                    return

                if total_time < (2 * INTERVAL_S):
                    total_time = 2 * INTERVAL_S

                execution_time = np.arange(0, total_time, INTERVAL_S)
                
                all_q, all_qd = [], []
                for i in range(6):
                    if abs(target_pos_steps[i] - initial_pos_steps[i]) == 0:
                        all_q.append(np.full(len(execution_time), initial_pos_steps[i]))
                        all_qd.append(np.zeros(len(execution_time)))
                    else:
                        joint_traj = rp.trapezoidal(initial_pos_steps[i], target_pos_steps[i], execution_time)
                        all_q.append(joint_traj.q)
                        all_qd.append(joint_traj.qd)

                self.trajectory_steps = list(zip(np.array(all_q).T.astype(int), np.array(all_qd).T.astype(int)))
                logger.info(f"  -> Command is valid (duration calculated from speed: {total_time:.2f}s).")

            except Exception as e:
                logger.error(f"  -> VALIDATION FAILED: Could not calculate velocity-based trajectory. Error: {e}")
                self.is_valid = False
                return

        else:
            logger.debug("  -> Using conservative values for MovePose.")
            command_len = 200
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len)
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))
        
        if not self.trajectory_steps:
             logger.warning(" -> Trajectory calculation resulted in no steps. Command is invalid.")
             self.is_valid = False
        else:
             logger.debug(f" -> Trajectory prepared with {len(self.trajectory_steps)} steps.")

    def execute_step(self, state) -> ExecutionStatus:
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        if self.command_step >= len(self.trajectory_steps):
            logger.info(f"{type(self).__name__} finished.")
            self.is_finished = True
            state.Position_out[:] = state.Position_in[:]
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.MOVE
            return ExecutionStatus.completed("MOVEPOSE complete")
        else:
            pos_step, _ = self.trajectory_steps[self.command_step]
            state.Position_out[:] = pos_step
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.MOVE
            self.command_step += 1
            return ExecutionStatus.executing("MovePose")


@register_command("MOVECART")
class MoveCartCommand(CommandBase):
    """
    A non-blocking command to move the robot's end-effector in a straight line
    in Cartesian space, completing the move in an exact duration.

    It works by:
    1. Pre-validating the final target pose.
    2. Interpolating the pose in Cartesian space in real-time.
    3. Solving Inverse Kinematics for each intermediate step to ensure path validity.
    """
    def __init__(self):
        super().__init__()
        
        # Parameters (set in match())
        self.pose = None
        self.duration = None
        self.velocity_percent = None
        
        # Runtime state
        self.start_time = None
        self.initial_pose = None
        self.target_pose = None
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse MOVECART command parameters.
        
        Format: MOVECART|x|y|z|rx|ry|rz|duration|speed
        Example: MOVECART|100|200|300|0|0|0|2.0|None
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 9:
            return (False, "MOVECART requires 8 parameters: x, y, z, rx, ry, rz, duration, speed")
        
        try:
            # Parse pose (6 values)
            self.pose = [float(parts[i]) for i in range(1, 7)]
            
            # Parse duration and speed
            self.duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            self.velocity_percent = None if parts[8].upper() == 'NONE' else float(parts[8])
            
            # Validate that at least one timing parameter is given
            if self.duration is None and self.velocity_percent is None:
                return (False, "MOVECART requires either duration or velocity_percent")
            
            if self.duration is not None and self.velocity_percent is not None:
                logger.info("  -> INFO: Both duration and velocity_percent provided. Using duration.")
                self.velocity_percent = None  # Prioritize duration
            
            logger.info(f"Parsed MoveCart to {self.pose}")
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid MOVECART parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing MOVECART: {str(e)}")

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        """Captures the initial state and validates the path just before execution."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

        logger.debug(f"  -> Preparing for MoveCart to {self.pose}...")
        
        # Capture initial state from live data
        initial_q_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
        self.initial_pose = PAROL6_ROBOT.robot.fkine(initial_q_rad)
        self.target_pose = SE3(self.pose[0]/1000.0, self.pose[1]/1000.0, self.pose[2]/1000.0) * SE3.RPY(self.pose[3:6], unit='deg', order='xyz')

        logger.debug("  -> Pre-validating final target pose...")
        ik_check = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, self.target_pose, initial_q_rad
        )

        if not ik_check.success:
            logger.warning("  -> VALIDATION FAILED: The final target pose is unreachable.")
            if ik_check.violations:
                logger.warning(f"     Reason: Solution violates joint limits: {ik_check.violations}")
            self.is_valid = False # Mark as invalid if path fails
            return

        # --- NEW BLOCK: Calculate duration from velocity if needed ---
        if self.velocity_percent is not None:
            logger.debug(f"  -> Calculating duration for {self.velocity_percent}% speed...")
            # Calculate the total distance for translation and rotation
            linear_distance = np.linalg.norm(self.target_pose.t - self.initial_pose.t)
            angular_distance_rad = self.initial_pose.angdist(self.target_pose)

            # Interpolate the target speeds from percentages, assuming constants exist in PAROL6_ROBOT
            target_linear_speed = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Cartesian_linear_velocity_min, PAROL6_ROBOT.Cartesian_linear_velocity_max])
            target_angular_speed = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Cartesian_angular_velocity_min, PAROL6_ROBOT.Cartesian_angular_velocity_max])
            target_angular_speed_rad = np.deg2rad(target_angular_speed)

            # Calculate time required for each component of the movement
            time_linear = linear_distance / target_linear_speed if target_linear_speed > 0 else 0
            time_angular = angular_distance_rad / target_angular_speed_rad if target_angular_speed_rad > 0 else 0

            # The total duration is the longer of the two times to ensure synchronization
            calculated_duration = max(time_linear, time_angular)

            if calculated_duration <= 0:
                logger.info("  -> INFO: MoveCart has zero duration. Marking as finished.")
                self.is_finished = True
                self.is_valid = True # It's valid, just already done.
                return

            self.duration = calculated_duration
            logger.debug(f"  -> Calculated MoveCart duration: {self.duration:.2f}s")

        logger.debug("  -> Command is valid and ready for execution.")

    def execute_step(self, state) -> ExecutionStatus:
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        s = min(elapsed_time / self.duration, 1.0)
        s_scaled = quintic_scaling(s)

        assert self.initial_pose is not None
        current_target_pose = self.initial_pose.interp(self.target_pose, s_scaled)

        current_q_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, current_target_pose, current_q_rad
        )

        if not ik_solution.success:
            logger.error("  -> ERROR: MoveCart failed. An intermediate point on the path is unreachable.")
            if ik_solution.violations:
                 logger.warning(f"     Reason: Path violates joint limits: {ik_solution.violations}")
            self.is_finished = True
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.IDLE
            return ExecutionStatus.failed("MoveCart IK failure")

        current_pos_rad = ik_solution.q

        # Send only the target position and let the firmware's P-controller handle speed.
        state.Position_out[:] = [int(PAROL6_ROBOT.RAD2STEPS(p, i)) for i, p in enumerate(current_pos_rad)]
        state.Speed_out[:] = [0] * 6 # Set feed-forward velocity to zero for smooth P-control.
        state.Command_out = CommandCode.MOVE

        if s >= 1.0:
            logger.info(f"MoveCart finished in ~{elapsed_time:.2f}s.")
            self.is_finished = True
            return ExecutionStatus.completed("MOVECART complete")

        return ExecutionStatus.executing("MoveCart")
