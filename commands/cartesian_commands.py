"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, and MoveCart
"""

import logging
import numpy as np
import time
import PAROL6_ROBOT
from spatialmath import SE3
import roboticstoolbox as rp
from .ik_helpers import solve_ik_with_adaptive_tol_subdivision, quintic_scaling, AXIS_MAP

logger = logging.getLogger(__name__)

# Set interval - used for timing calculations
INTERVAL_S = 0.01

# Jogging uses a smaller IK iteration limit for more responsive performance
JOG_IK_ILIMIT = 20

class CartesianJogCommand:
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    This is the final, refactored version using clean, standard spatial math
    operations now that the core unit bug has been fixed.
    """
    def __init__(self, frame, axis, speed_percentage=50, duration=1.5, **kwargs):
        """
        Initializes and validates the Cartesian jog command.
        """
        self.is_valid = False
        self.is_finished = False
        logger.info(f"Initializing CartesianJog: Frame {frame}, Axis {axis}...")

        if axis not in AXIS_MAP:
            logger.warning(f"  -> VALIDATION FAILED: Invalid axis '{axis}'.")
            return
        
        # Store all necessary parameters for use in execute_step
        self.frame = frame
        self.axis_vectors = AXIS_MAP[axis]
        self.is_rotation = any(self.axis_vectors[1])
        self.speed_percentage = speed_percentage
        self.duration = duration
        self.end_time = time.time() + self.duration
        
        self.is_valid = True
        logger.debug("  -> Command is valid and ready.")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        if self.is_finished or not self.is_valid:
            return True

        # --- A. Check for completion ---
        if time.time() >= self.end_time:
            logger.info("Cartesian jog finished.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True

        # --- B. Calculate Target Pose using clean vector math ---
        Command_out.value = 123 # Set jog command
        
        q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
        T_current = PAROL6_ROBOT.robot.fkine(q_current)

        if not isinstance(T_current, SE3):
            return False # Wait for valid pose data

        # Calculate speed and displacement for this cycle
        linear_speed_ms = float(np.interp(self.speed_percentage, [0, 100], [PAROL6_ROBOT.Cartesian_linear_velocity_min_JOG, PAROL6_ROBOT.Cartesian_linear_velocity_max_JOG]))
        angular_speed_degs = float(np.interp(self.speed_percentage, [0, 100], [PAROL6_ROBOT.Cartesian_angular_velocity_min, PAROL6_ROBOT.Cartesian_angular_velocity_max]))

        delta_linear = linear_speed_ms * INTERVAL_S
        delta_angular_rad = np.deg2rad(angular_speed_degs * INTERVAL_S)

        # Create the small incremental transformation (delta_pose)
        trans_vec = np.array(self.axis_vectors[0]) * delta_linear
        rot_vec = np.array(self.axis_vectors[1]) * delta_angular_rad
        delta_pose = SE3.Rt(SE3.Eul(rot_vec).R, trans_vec)

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
                Speed_out[i] = int(PAROL6_ROBOT.SPEED_RAD2STEP(q_velocities[i], i))
        else:
            logger.warning("IK Warning: Could not find solution for jog step. Stopping.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True

        # --- D. Speed Scaling ---
        max_scale_factor = 1.0
        for i in range(6):
            if abs(Speed_out[i]) > PAROL6_ROBOT.Joint_max_speed[i]:
                scale = abs(Speed_out[i]) / PAROL6_ROBOT.Joint_max_speed[i]
                if scale > max_scale_factor:
                    max_scale_factor = scale
        
        if max_scale_factor > 1.0:
            for i in range(6):
                Speed_out[i] = int(Speed_out[i] / max_scale_factor)

        return False # Command is still running

class MovePoseCommand:
    """
    A non-blocking command to move the robot to a specific Cartesian pose.
    The movement itself is a joint-space interpolation.
    """
    def __init__(self, pose, duration=None, velocity_percent=None, accel_percent=50, trajectory_type='poly'):
        self.is_valid = True  # Assume valid; preparation step will confirm.
        self.is_finished = False
        self.command_step = 0
        self.trajectory_steps = []

        logger.info(f"Initializing MovePose to {pose}...")

        # --- MODIFICATION: Store parameters for deferred planning ---
        self.pose = pose
        self.duration = duration
        self.velocity_percent = velocity_percent
        self.accel_percent = accel_percent
        self.trajectory_type = trajectory_type

    """
        Initializes, validates, and pre-computes the trajectory for a move-to-pose command.

        Args:
            pose (list): A list of 6 values [x, y, z, r, p, y] for the target pose.
                         Positions are in mm, rotations are in degrees.
            duration (float, optional): The total time for the movement in seconds.
            velocity_percent (float, optional): The target velocity as a percentage (0-100).
            accel_percent (float, optional): The target acceleration as a percentage (0-100).
            trajectory_type (str, optional): The type of trajectory ('poly' or 'trap').
        """
    
    def prepare_for_execution(self, current_position_in):
        """Calculates the full trajectory just-in-time before execution."""
        logger.debug(f"  -> Preparing trajectory for MovePose to {self.pose}...")

        initial_pos_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(current_position_in)])
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

class MoveCartCommand:
    """
    A non-blocking command to move the robot's end-effector in a straight line
    in Cartesian space, completing the move in an exact duration.

    It works by:
    1. Pre-validating the final target pose.
    2. Interpolating the pose in Cartesian space in real-time.
    3. Solving Inverse Kinematics for each intermediate step to ensure path validity.
    """
    def __init__(self, pose, duration=None, velocity_percent=None):
        self.is_valid = False
        self.is_finished = False

        # --- MODIFICATION: Validate that at least one timing parameter is given ---
        if duration is None and velocity_percent is None:
            logger.error("  -> VALIDATION FAILED: MoveCartCommand requires either 'duration' or 'velocity_percent'.")
            return
        if duration is not None and velocity_percent is not None:
            logger.info("  -> INFO: Both duration and velocity_percent provided. Using duration.")
            self.velocity_percent = None # Prioritize duration
        else:
            self.velocity_percent = velocity_percent

        # --- Store parameters and set placeholders ---
        self.duration = duration
        self.pose = pose
        self.start_time = None
        self.initial_pose = None
        self.target_pose = None
        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Captures the initial state and validates the path just before execution."""
        logger.debug(f"  -> Preparing for MoveCart to {self.pose}...")
        
        # --- MOVED LOGIC: Capture initial state from live data ---
        initial_q_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(current_position_in)])
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

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, Position_out=None, **kwargs):
        # Get Position_out from kwargs if not provided
        if Position_out is None:
            Position_out = kwargs.get('Position_out', Position_in)
        
        if self.is_finished or not self.is_valid:
            return True

        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        s = min(elapsed_time / self.duration, 1.0)
        s_scaled = quintic_scaling(s)

        current_target_pose = self.initial_pose.interp(self.target_pose, s_scaled)

        current_q_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, current_target_pose, current_q_rad
        )

        if not ik_solution.success:
            logger.error("  -> ERROR: MoveCart failed. An intermediate point on the path is unreachable.")
            if ik_solution.violations:
                 logger.warning(f"     Reason: Path violates joint limits: {ik_solution.violations}")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True

        current_pos_rad = ik_solution.q

        # --- MODIFIED BLOCK ---
        # Send only the target position and let the firmware's P-controller handle speed.
        Position_out[:] = [int(PAROL6_ROBOT.RAD2STEPS(p, i)) for i, p in enumerate(current_pos_rad)]
        Speed_out[:] = [0] * 6 # Set feed-forward velocity to zero for smooth P-control.
        Command_out.value = 156
        # --- END MODIFIED BLOCK ---

        if s >= 1.0:
            logger.info(f"MoveCart finished in ~{elapsed_time:.2f}s.")
            self.is_finished = True
            # The main loop will handle holding the final position.

        return self.is_finished