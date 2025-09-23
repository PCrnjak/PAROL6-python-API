"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, and MoveCart
"""

import logging
import time
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, cast
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from spatialmath import SE3
from parol6.utils.ik import solve_ik_with_adaptive_tol_subdivision, quintic_scaling, AXIS_MAP
from .base import ExecutionStatus, MotionCommand, MotionProfile
from parol6.utils.errors import IKError
from parol6.protocol.wire import CommandCode
from parol6.config import JOG_IK_ILIMIT, INTERVAL_S, TRACE, DEFAULT_ACCEL_PERCENT
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)
# TODO: we really need to normalize and be consistent with the logging such that it lines of with the lifecycle and includes the commands name, etc. 
@register_command("CARTJOG")
class CartesianJogCommand(MotionCommand):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    """
    streamable = True

    __slots__ = (
        "frame",
        "axis",
        "speed_percentage",
        "duration",
        "axis_vectors",
        "is_rotation",
    )

    def __init__(self):
        """
        Initializes the Cartesian jog command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()
        
        # Parameters (set in do_match())
        self.frame = None
        self.axis = None
        self.speed_percentage = 50
        self.duration = 1.5
        
        # Runtime state
        self.axis_vectors = None
        self.is_rotation = False
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
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
        
        self.is_valid = True
        return (True, None)
    
    def do_setup(self, state):
        """Set the end time when the command actually starts."""
        self.start_timer(float(self.duration))

    def execute_step(self, state) -> ExecutionStatus:
        # --- A. Check for completion ---
        if self._t_end is None:
            # Initialize timer if missing (stream update or late init)
            self.start_timer(max(0.1, self.duration if self.duration is not None else 0.1))
        if self.timer_expired():
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("CARTJOG complete")

        # --- B. Calculate Target Pose using clean vector math ---
        state.Command_out = CommandCode.JOG
        
        q_current = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        T_current = PAROL6_ROBOT.robot.fkine(q_current)

        if not isinstance(T_current, SE3):
            return ExecutionStatus.executing("Waiting for valid pose")
        if self.axis_vectors is None:
            return ExecutionStatus.executing("Waiting for axis vectors")

        linear_speed_ms = self.linmap_pct(self.speed_percentage, self.CART_LIN_JOG_MIN, self.CART_LIN_JOG_MAX)
        angular_speed_degs = self.linmap_pct(self.speed_percentage, self.CART_ANG_JOG_MIN, self.CART_ANG_JOG_MAX)

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
            sps = PAROL6_ROBOT.ops.speed_rad_to_steps(q_velocities)
            np.copyto(state.Speed_out, np.asarray(sps), casting='no')
        else:
            raise IKError("IK Warning: Could not find solution for jog step. Stopping.")

        # --- D. Speed Scaling using base class helper ---
        scaled_speeds = self.scale_speeds_to_joint_max(state.Speed_out)
        np.copyto(state.Speed_out, scaled_speeds, casting='no')

        return ExecutionStatus.executing("CARTJOG")


@register_command("MOVEPOSE")
class MovePoseCommand(MotionCommand):
    """
    A non-blocking command to move the robot to a specific Cartesian pose.
    The movement itself is a joint-space interpolation.
    """
    __slots__ = (
        "command_step",
        "trajectory_steps",
        "pose",
        "duration",
        "velocity_percent",
        "accel_percent",
        "trajectory_type",
    )
    def __init__(self, pose=None, duration=None):
        super().__init__()
        self.command_step = 0
        self.trajectory_steps = []
        
        # Parameters (set in do_match())
        self.pose = pose
        self.duration = duration
        self.velocity_percent = None
        self.accel_percent = DEFAULT_ACCEL_PERCENT
        self.trajectory_type = 'trapezoid'
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
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
        
        # Parse pose (6 values)
        self.pose = [float(parts[i]) for i in range(1, 7)]
        
        # Parse duration and speed
        self.duration = None if parts[7].upper() == 'NONE' else float(parts[7])
        self.velocity_percent = None if parts[8].upper() == 'NONE' else float(parts[8])
        
        self.log_debug("Parsed MovePose: %s", self.pose)
        self.is_valid = True
        return (True, None)
    
    def do_setup(self, state):
        """Calculates the full trajectory just-in-time before execution."""
        self.log_trace("  -> Preparing trajectory for MovePose to %s...", self.pose)

        initial_pos_rad = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        pose = cast(List[float], self.pose)
        target_pose = SE3(pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0) * SE3.RPY(pose[3:6], unit='deg', order='xyz')
        
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, target_pose, initial_pos_rad, ilimit=100)

        if not ik_solution.success:
            error_str = "An intermediate point on the path is unreachable."
            if ik_solution.violations:
                 error_str += f" Reason: Path violates joint limits: {ik_solution.violations}"
            raise IKError(error_str)

        target_pos_rad = ik_solution.q

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                self.log_trace("  -> INFO: Both duration and velocity were provided. Using duration.")
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(PAROL6_ROBOT.ops.rad_to_steps(target_pos_rad), dtype=np.int32)
            dur = float(self.duration)
            self.trajectory_steps = MotionProfile.from_duration_steps(
                initial_pos_steps, target_pos_steps, dur, dt=INTERVAL_S
            )

        elif self.velocity_percent is not None:
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(PAROL6_ROBOT.ops.rad_to_steps(target_pos_rad), dtype=np.int32)
            accel_percent = float(self.accel_percent) if self.accel_percent is not None else float(DEFAULT_ACCEL_PERCENT)
            self.trajectory_steps = MotionProfile.from_velocity_percent(
                initial_pos_steps,
                target_pos_steps,
                float(self.velocity_percent),
                accel_percent,
                dt=INTERVAL_S,
            )
            self.log_trace("  -> Command is valid (velocity profile).")
        else:
            self.log_trace("  -> Using conservative values for MovePose.")
            command_len = 200
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(PAROL6_ROBOT.ops.rad_to_steps(target_pos_rad), dtype=np.int32)
            total_dur = float(command_len) * INTERVAL_S
            self.trajectory_steps = MotionProfile.from_duration_steps(
                initial_pos_steps, target_pos_steps, total_dur, dt=INTERVAL_S
            )
        
        if len(self.trajectory_steps) == 0:
            raise IKError("Trajectory calculation resulted in no steps. Command is invalid.")
        logger.log(TRACE, " -> Trajectory prepared with %s steps.", len(self.trajectory_steps))

    def execute_step(self, state) -> ExecutionStatus:
        if self.command_step >= len(self.trajectory_steps):
            logger.info(f"{type(self).__name__} finished.")
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MOVEPOSE complete")
        else:
            self.set_move_position(state, self.trajectory_steps[self.command_step])
            self.command_step += 1
            return ExecutionStatus.executing("MovePose")


@register_command("MOVECART")
class MoveCartCommand(MotionCommand):
    """
    A non-blocking command to move the robot's end-effector in a straight line
    in Cartesian space, completing the move in an exact duration.

    It works by:
    1. Pre-validating the final target pose.
    2. Interpolating the pose in Cartesian space in real-time.
    3. Solving Inverse Kinematics for each intermediate step to ensure path validity.
    """
    __slots__ = (
        "pose",
        "duration",
        "velocity_percent",
        "start_time",
        "initial_pose",
        "target_pose",
    )
    def __init__(self):
        super().__init__()
        
        # Parameters (set in do_match())
        self.pose = None
        self.duration = None
        self.velocity_percent = None
        
        # Runtime state
        self.start_time = None
        self.initial_pose = None
        self.target_pose = None
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
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
        
        self.log_debug("Parsed MoveCart: %s", self.pose)
        self.is_valid = True
        return (True, None)

    def do_setup(self, state):
        """Captures the initial state and validates the path just before execution."""
        # Capture initial state from live data
        initial_q_rad = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        self.initial_pose = PAROL6_ROBOT.robot.fkine(initial_q_rad)
        pose = cast(List[float], self.pose)
        self.target_pose = SE3(pose[0]/1000.0, pose[1]/1000.0, pose[2]/1000.0) * SE3.RPY(pose[3:6], unit='deg', order='xyz')

        ik_check = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, cast(SE3, self.target_pose), initial_q_rad
        )

        if not ik_check.success:
            error_str = "An intermediate point on the path is unreachable."
            if ik_check.violations:
                 error_str += f" Reason: Path violates joint limits: {ik_check.violations}"
            raise IKError(error_str)

        if self.velocity_percent is not None:
            # Calculate the total distance for translation and rotation
            tp = cast(SE3, self.target_pose)
            ip = cast(SE3, self.initial_pose)
            linear_distance = np.linalg.norm(tp.t - ip.t)
            angular_distance_rad = ip.angdist(tp)

            target_linear_speed = self.linmap_pct(self.velocity_percent, self.CART_LIN_JOG_MIN, self.CART_LIN_JOG_MAX)
            target_angular_speed_rad = np.deg2rad(self.linmap_pct(self.velocity_percent, self.CART_ANG_JOG_MIN, self.CART_ANG_JOG_MAX))

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
            self.log_debug("  -> Calculated MoveCart duration: %.2fs", self.duration)

        self.log_debug("  -> Command is valid and ready for execution.")
        if self.duration and float(self.duration) > 0.0:
            self.start_timer(float(self.duration))

    def execute_step(self, state) -> ExecutionStatus:
        dur = float(self.duration or 0.0)
        if dur <= 0.0:
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MOVECART complete")
        s = self.progress01(dur)
        s_scaled = quintic_scaling(float(s))

        assert self.initial_pose is not None and self.target_pose is not None
        _ctp = cast(SE3, self.initial_pose).interp(cast(SE3, self.target_pose), s_scaled)
        if not isinstance(_ctp, SE3):
            return ExecutionStatus.executing("Waiting for pose interpolation")
        current_target_pose = cast(SE3, _ctp)

        current_q_rad = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        # TODO: is it doing the expensive IK solving twice per command??? once in setup and once in execution??
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, current_target_pose, current_q_rad
        )

        if not ik_solution.success:
            error_str = "An intermediate point on the path is unreachable."
            if ik_solution.violations:
                 error_str += f" Reason: Path violates joint limits: {ik_solution.violations}"
            raise IKError(error_str)

        current_pos_rad = ik_solution.q

        # Send only the target position and let the firmware's P-controller handle speed.
        # Set feed-forward velocity to zero for smooth P-control.
        steps = PAROL6_ROBOT.ops.rad_to_steps(current_pos_rad)
        self.set_move_position(state, np.asarray(steps))

        if s >= 1.0:
            actual_elapsed = (time.perf_counter() - self._t0) if self._t0 is not None else dur
            self.log_info("MoveCart finished in ~%.2fs.", actual_elapsed)
            self.is_finished = True
            return ExecutionStatus.completed("MOVECART complete")

        return ExecutionStatus.executing("MoveCart")
