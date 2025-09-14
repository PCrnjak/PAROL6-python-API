"""
Joint Movement Commands
Contains commands for direct joint angle movements
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
import roboticstoolbox as rp
from .base import CommandBase, ExecutionStatus, ExecutionStatusCode
from parol6.config import INTERVAL_S
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)


@register_command("MOVEJOINT")
class MoveJointCommand(CommandBase):
    """
    A non-blocking command to move the robot's joints to a specific configuration.
    It pre-calculates the entire trajectory upon initialization.
    """
    def __init__(self):
        super().__init__()
        self.command_step = 0
        self.trajectory_steps = []
        
        # Parameters (set in match())
        self.target_angles = None
        self.duration = None
        self.velocity_percent = None
        self.accel_percent = 50
        self.trajectory_type = 'poly'
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
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
            self.target_angles = [float(parts[i]) for i in range(1, 7)]
            
            # Parse duration and speed
            self.duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            self.velocity_percent = None if parts[8].upper() == 'NONE' else float(parts[8])
            
            # Validate joint limits
            target_pos_rad = np.array([np.deg2rad(angle) for angle in self.target_angles])
            for i in range(6):
                min_rad, max_rad = PAROL6_ROBOT.Joint_limits_radian[i]
                if not (min_rad <= target_pos_rad[i] <= max_rad):
                    return (False, f"Joint {i+1} target ({self.target_angles[i]} deg) is out of range")
            
            logger.info(f"Parsed MoveJoint to {self.target_angles}")
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid MOVEJOINT parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing MOVEJOINT: {str(e)}")

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        """Calculates the trajectory just before execution begins."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

        logger.debug(f"  -> Preparing trajectory for MoveJoint to {self.target_angles}...")
        
        initial_pos_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
        target_pos_rad = np.array([np.deg2rad(angle) for angle in self.target_angles])

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
                logger.info("  -> Please check Joint_min/max_speed and Joint_min/max_acc values in PAROL6_ROBOT.py.")
                self.is_valid = False
                return
        
        else:
            logger.debug("  -> Using conservative values for MoveJoint.")
            command_len = 200
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len) # type: ignore
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
            return ExecutionStatus.completed("MOVEJOINT complete")
        else:
            pos_step, _ = self.trajectory_steps[self.command_step]
            state.Position_out[:] = pos_step
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.MOVE
            self.command_step += 1
            return ExecutionStatus.executing("MoveJoint")
