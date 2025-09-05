"""
Joint Movement Commands
Contains commands for direct joint angle movements
"""

import logging
import numpy as np
import PAROL6_ROBOT
import roboticstoolbox as rp

logger = logging.getLogger(__name__)

# Set interval - used for timing calculations
INTERVAL_S = 0.01

class MoveJointCommand:
    """
    A non-blocking command to move the robot's joints to a specific configuration.
    It pre-calculates the entire trajectory upon initialization.
    """
    def __init__(self, target_angles, duration=None, velocity_percent=None, accel_percent=50, trajectory_type='poly'):
        self.is_valid = False  # Will be set to True after basic validation
        self.is_finished = False
        self.command_step = 0
        self.trajectory_steps = []

        logger.info(f"Initializing MoveJoint to {target_angles}...")

        # --- MODIFICATION: Store parameters for deferred planning ---
        self.target_angles = target_angles
        self.duration = duration
        self.velocity_percent = velocity_percent
        self.accel_percent = accel_percent
        self.trajectory_type = trajectory_type

        # --- Perform only state-independent validation ---
        target_pos_rad = np.array([np.deg2rad(angle) for angle in self.target_angles])
        for i in range(6):
            min_rad, max_rad = PAROL6_ROBOT.Joint_limits_radian[i]
            if not (min_rad <= target_pos_rad[i] <= max_rad):
                logger.error(f"  -> VALIDATION FAILED: Target for Joint {i+1} ({self.target_angles[i]} deg) is out of range.")
                return
        
        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Calculates the trajectory just before execution begins."""
        logger.debug(f"  -> Preparing trajectory for MoveJoint to {self.target_angles}...")
        
        initial_pos_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(current_position_in)])
        target_pos_rad = np.array([np.deg2rad(angle) for angle in self.target_angles])

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                logger.info("  -> INFO: Both duration and velocity were provided. Using duration.")
            command_len = int(self.duration / INTERVAL_S)
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len)
            
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))

        elif self.velocity_percent is not None:
            try:
                accel_percent = self.accel_percent if self.accel_percent is not None else 50
                initial_pos_steps = np.array(current_position_in)
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
                logger.info(f"  -> Please check Joint_min/max_speed and Joint_min/max_acc values in PAROL6_ROBOT.py.")
                self.is_valid = False
                return
        
        else:
            logger.debug("  -> Using conservative values for MoveJoint.")
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

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, Position_out=None, **kwargs):
        # Get Position_out from kwargs if not provided
        if Position_out is None:
            Position_out = kwargs.get('Position_out', Position_in)
        
        # This method remains unchanged.
        if self.is_finished or not self.is_valid:
            return True

        if self.command_step >= len(self.trajectory_steps):
            logger.info(f"{type(self).__name__} finished.")
            self.is_finished = True
            Position_out[:] = Position_in[:]
            Speed_out[:] = [0] * 6
            Command_out.value = 156
            return True
        else:
            pos_step, _ = self.trajectory_steps[self.command_step]
            Position_out[:] = pos_step
            Speed_out[:] = [0] * 6
            Command_out.value = 156
            self.command_step += 1
            return False