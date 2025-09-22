"""
Joint Movement Commands
Contains commands for direct joint angle movements
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.base import MotionCommand, ExecutionStatus, MotionProfile
from parol6.config import INTERVAL_S, TRACE, DEFAULT_ACCEL_PERCENT
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)


@register_command("MOVEJOINT")
class MoveJointCommand(MotionCommand):
    """
    A non-blocking command to move the robot's joints to a specific configuration.
    It pre-calculates the entire trajectory upon initialization.
    """
    def __init__(self):
        super().__init__()
        self.command_step = 0
        self.trajectory_steps = np.ndarray([])
        
        # Parameters (set in do_match())
        self.target_angles = None
        self.target_radians = None
        self.duration = None
        self.velocity_percent = None
        self.accel_percent = DEFAULT_ACCEL_PERCENT
        self.trajectory_type = 'trapezoid'
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse MOVEJOINT command parameters.
        
        Format: MOVEJOINT|j1|j2|j3|j4|j5|j6|duration|speed
        Example: MOVEJOINT|0|45|90|-45|30|0|None|50
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 9:
            return (False, "MOVEJOINT requires 8 parameters: 6 joint angles, duration, speed")
        
        try:
            # Parse joint angles
            self.target_angles = np.asarray([float(parts[i]) for i in range(1, 7)], dtype=float)
            
            # Parse duration and speed
            self.duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            self.velocity_percent = None if parts[8].upper() == 'NONE' else float(parts[8])
            
            # Validate joint limits
            self.target_radians = np.deg2rad(self.target_angles)
            for i in range(6):
                min_rad, max_rad = PAROL6_ROBOT.joint.limits.rad[i]
                if not (min_rad <= self.target_radians[i] <= max_rad):
                    return (False, f"Joint {i+1} target ({self.target_angles[i]} deg) is out of range")
            
            self.log_debug("Parsed MoveJoint: %s", self.target_angles)
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid MOVEJOINT parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing MOVEJOINT: {str(e)}")

    def do_setup(self, state):
        """Calculates the trajectory just before execution begins."""
        self.log_trace("Preparing trajectory for MoveJoint to %s...", self.target_angles)

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                self.log_trace("  -> INFO: Both duration and velocity were provided. Using duration.")
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(PAROL6_ROBOT.ops.rad_to_steps(self.target_radians), dtype=np.int32)
            dur = float(self.duration)
            self.trajectory_steps = MotionProfile.from_duration_steps(
                initial_pos_steps, target_pos_steps, dur, dt=INTERVAL_S
            )

        elif self.velocity_percent is not None:
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(PAROL6_ROBOT.ops.rad_to_steps(self.target_radians), dtype=np.int32)
            accel_percent = float(self.accel_percent) if self.accel_percent is not None else float(DEFAULT_ACCEL_PERCENT)
            self.trajectory_steps = MotionProfile.from_velocity_percent(
                initial_pos_steps,
                target_pos_steps,
                float(self.velocity_percent),
                accel_percent,
                dt=INTERVAL_S,
            )
            self.log_trace("  -> Command is valid (duration calculated from speed).")
        
        else:
            logger.log(TRACE, "  -> Using conservative values for MoveJoint.")
            command_len = 200
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(PAROL6_ROBOT.ops.rad_to_steps(self.target_radians), dtype=np.int32)
            total_dur = float(command_len) * INTERVAL_S
            self.trajectory_steps = MotionProfile.from_duration_steps(
                initial_pos_steps, target_pos_steps, total_dur, dt=INTERVAL_S
            )
        
        if len(self.trajectory_steps) == 0:
            self.log_warning(" -> Trajectory calculation resulted in no steps. Command is invalid.")
            self.is_valid = False
        self.log_trace(" -> Trajectory prepared with %s steps.", len(self.trajectory_steps))

    def execute_step(self, state) -> ExecutionStatus:
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        if self.command_step >= len(self.trajectory_steps):
            logger.log(TRACE, f"{type(self).__name__} finished.")
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MOVEJOINT")
        else:
            self.set_move_position(state, self.trajectory_steps[self.command_step])
            self.command_step += 1
            return ExecutionStatus.executing("MOVEJOINT")
