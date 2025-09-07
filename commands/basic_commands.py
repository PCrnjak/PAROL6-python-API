"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and MultiJog
"""

import logging
import numpy as np
import PAROL6_ROBOT

logger = logging.getLogger(__name__)

# Set interval - used for timing calculations
INTERVAL_S = 0.01

class HomeCommand:
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """
    def __init__(self):
        self.is_valid = True
        self.is_finished = False
        # State machine: START -> WAIT_FOR_UNHOMED -> WAIT_FOR_HOMED -> FINISHED
        self.state = "START"
        # Counter to send the home command for multiple cycles
        self.start_cmd_counter = 10  # Send command 100 for 10 cycles (0.1s)
        # Safety timeout (20 seconds at 0.01s interval)
        self.timeout_counter = 2000
        logger.info("Initializing Home command...")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """
        Manages the homing command and monitors for completion using a state machine.
        """
        if self.is_finished:
            return True

        # --- State: START ---
        # On the first few executions, continuously send the 'home' (100) command.
        if self.state == "START":
            logger.debug(f"  -> Sending home signal (100)... Countdown: {self.start_cmd_counter}")
            Command_out.value = 100
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                # Once sent for enough cycles, move to the next state
                self.state = "WAITING_FOR_UNHOMED"
            return False

        # --- State: WAITING_FOR_UNHOMED ---
        # The robot's firmware should reset the homed status. We wait to see that happen.
        # During this time, we send 'idle' (255) to let the robot's controller take over.
        if self.state == "WAITING_FOR_UNHOMED":
            Command_out.value = 255
            # Homing sequence initiated detection
            if any(h == 0 for h in Homed_in[:6]):
                logger.info("  -> Homing sequence initiated by robot.")
                self.state = "WAITING_FOR_HOMED"
            # Homing timeout protection
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                logger.error("  -> ERROR: Timeout waiting for robot to start homing sequence.")
                self.is_finished = True
            return self.is_finished

        # --- State: WAITING_FOR_HOMED ---
        # Now we wait for all joints to report that they are homed (all flags are 1).
        if self.state == "WAITING_FOR_HOMED":
            Command_out.value = 255
            # Homing completion verification
            if all(h == 1 for h in Homed_in[:6]):
                logger.info("Homing sequence complete. All joints reported home.")
                self.is_finished = True
                Speed_out[:] = [0] * 6 # Ensure robot is stopped

        return self.is_finished

class JogCommand:
    """
    A non-blocking command to jog a joint for a specific duration or distance.
    It performs all safety and validity checks upon initialization.
    """
    def __init__(self, joint, speed_percentage=None, duration=None, distance_deg=None):
        """
        Initializes and validates the jog command. This is the SETUP phase.
        """
        self.is_valid = False
        self.is_finished = False
        self.mode = None
        self.command_step = 0

        # --- 1. Parameter Validation and Mode Selection ---
        if duration and distance_deg:
            self.mode = 'distance'
            logger.info(f"Initializing Jog: Joint {joint}, Distance {distance_deg} deg, Duration {duration}s.")
        elif duration:
            self.mode = 'time'
            logger.info(f"Initializing Jog: Joint {joint}, Speed {speed_percentage}%, Duration {duration}s.")
        elif distance_deg:
            self.mode = 'distance'
            logger.info(f"Initializing Jog: Joint {joint}, Speed {speed_percentage}%, Distance {distance_deg} deg.")
        else:
            logger.error("Error: JogCommand requires either 'duration', 'distance_deg', or both.")
            return

        # --- 2. Store parameters for deferred calculation ---
        self.joint = joint
        self.speed_percentage = speed_percentage
        self.duration = duration
        self.distance_deg = distance_deg

        # --- These will be calculated later ---
        self.direction = 1
        self.joint_index = 0
        self.speed_out = 0
        self.command_len = 0
        self.target_position = 0

        self.is_valid = True # Mark as valid for now; preparation step will confirm.


    def prepare_for_execution(self, current_position_in):
        """Pre-computes speeds and target positions using live data."""
        logger.debug("  -> Preparing for Jog command...")

        # Joint direction and index mapping
        self.direction = 1 if 0 <= self.joint <= 5 else -1
        self.joint_index = self.joint if self.direction == 1 else self.joint - 6
        
        distance_steps = 0
        if self.distance_deg:
            distance_steps = int(PAROL6_ROBOT.DEG2STEPS(abs(self.distance_deg), self.joint_index))
            self.target_position = current_position_in[self.joint_index] + (distance_steps * self.direction)
            
            min_limit, max_limit = PAROL6_ROBOT.Joint_limits_steps[self.joint_index]
            if not (min_limit <= self.target_position <= max_limit):
                # Convert to degrees for clearer error message
                target_deg = PAROL6_ROBOT.STEPS2DEG(self.target_position, self.joint_index)
                min_deg = PAROL6_ROBOT.STEPS2DEG(min_limit, self.joint_index)
                max_deg = PAROL6_ROBOT.STEPS2DEG(max_limit, self.joint_index)
                logger.warning(f"  -> VALIDATION FAILED: Target position {target_deg:.2f}° is out of joint limits ({min_deg:.2f}°, {max_deg:.2f}°).")
                self.is_valid = False
                return

        # Motion timing calculations
        speed_steps_per_sec = 0
        if self.mode == 'distance' and self.duration:
            speed_steps_per_sec = int(distance_steps / self.duration) if self.duration > 0 else 0
            max_joint_jog_speed = PAROL6_ROBOT.Joint_max_jog_speed[self.joint_index]
            if speed_steps_per_sec > max_joint_jog_speed:
                logger.warning(f"  -> VALIDATION FAILED: Required speed ({speed_steps_per_sec} steps/s) exceeds joint's max jog speed ({max_joint_jog_speed} steps/s).")
                self.is_valid = False
                return
            # Ensure speed is at least the minimum jog speed if not zero
            if speed_steps_per_sec > 0:
                speed_steps_per_sec = max(speed_steps_per_sec, PAROL6_ROBOT.Joint_min_jog_speed[self.joint_index])
        else:
            if self.speed_percentage is None:
                logger.error("Error: 'speed_percentage' must be provided if not calculating automatically.")
                self.is_valid = False
                return
            speed_steps_per_sec = int(np.interp(
                abs(self.speed_percentage),
                [0, 100],
                [PAROL6_ROBOT.Joint_min_jog_speed[self.joint_index],
                 PAROL6_ROBOT.Joint_max_jog_speed[self.joint_index]]
            ))

        self.speed_out = speed_steps_per_sec * self.direction
        self.command_len = int(self.duration / INTERVAL_S) if self.duration else float('inf')
        logger.debug("  -> Jog command is ready.")


    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.is_finished or not self.is_valid:
            return True

        stop_reason = None
        current_pos = Position_in[self.joint_index]

        if self.mode == 'time':
            if self.command_step >= self.command_len:
                stop_reason = "Timed jog finished."
        elif self.mode == 'distance':
            if (self.direction == 1 and current_pos >= self.target_position) or \
               (self.direction == -1 and current_pos <= self.target_position):
                stop_reason = "Distance jog finished."
        
        if not stop_reason:
            if (self.direction == 1 and current_pos >= PAROL6_ROBOT.Joint_limits_steps[self.joint_index][1]) or \
               (self.direction == -1 and current_pos <= PAROL6_ROBOT.Joint_limits_steps[self.joint_index][0]):
                stop_reason = f"Limit reached on joint {self.joint_index + 1}."

        if stop_reason:
            logger.info(stop_reason)
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        else:
            Speed_out[:] = [0] * 6
            Speed_out[self.joint_index] = self.speed_out
            Command_out.value = 123
            self.command_step += 1
            return False
        
class MultiJogCommand:
    """
    A non-blocking command to jog multiple joints simultaneously for a specific duration.
    It performs all safety and validity checks upon initialization.
    """
    def __init__(self, joints, speed_percentages, duration):
        """
        Initializes and validates the multi-jog command.
        """
        self.is_valid = False
        self.is_finished = False
        self.command_step = 0

        # --- 1. Parameter Validation ---
        if not isinstance(joints, list) or not isinstance(speed_percentages, list):
            logger.error("Error: MultiJogCommand requires 'joints' and 'speed_percentages' to be lists.")
            return

        if len(joints) != len(speed_percentages):
            logger.error("Error: The number of joints must match the number of speed percentages.")
            return

        if not duration or duration <= 0:
            logger.error("Error: MultiJogCommand requires a positive 'duration'.")
            return

        # ==========================================================
        # === NEW: Check for conflicting joint commands          ===
        # ==========================================================
        base_joints = set()
        for joint in joints:
            # Normalize the joint index to its base (0-5)
            base_joint = joint % 6
            # If the base joint is already in our set, it's a conflict.
            if base_joint in base_joints:
                logger.warning(f"  -> VALIDATION FAILED: Conflicting commands for Joint {base_joint + 1} (e.g., J1+ and J1-).")
                self.is_valid = False
                return
            base_joints.add(base_joint)
        # ==========================================================

        logger.info(f"Initializing MultiJog for joints {joints} with speeds {speed_percentages}% for {duration}s.")

        # --- 2. Store parameters ---
        self.joints = joints
        self.speed_percentages = speed_percentages
        self.duration = duration
        self.command_len = int(self.duration / INTERVAL_S)

        # --- This will be calculated in the prepare step ---
        self.speeds_out = [0] * 6

        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Pre-computes the speeds for each joint."""
        logger.debug("  -> Preparing for MultiJog command...")

        for i, joint in enumerate(self.joints):
            # Index mapping: 0-5 positive, 6-11 negative direction
            direction = 1 if 0 <= joint <= 5 else -1
            joint_index = joint if direction == 1 else joint - 6
            speed_percentage = self.speed_percentages[i]

            # Check for joint index validity
            if not (0 <= joint_index < 6):
                logger.warning(f"  -> VALIDATION FAILED: Invalid joint index {joint_index}.")
                self.is_valid = False
                return

            # Calculate speed in steps/sec
            speed_steps_per_sec = int(np.interp(
                speed_percentage,
                [0, 100],
                [PAROL6_ROBOT.Joint_min_jog_speed[joint_index],
                 PAROL6_ROBOT.Joint_max_jog_speed[joint_index]]
            ))
            self.speeds_out[joint_index] = speed_steps_per_sec * direction

        logger.debug("  -> MultiJog command is ready.")


    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.is_finished or not self.is_valid:
            return True

        # Stop if the duration has elapsed
        if self.command_step >= self.command_len:
            logger.info("Timed multi-jog finished.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        else:
            # Continuously check for joint limits during the jog
            for i in range(6):
                if self.speeds_out[i] != 0:
                    current_pos = Position_in[i]
                    # Hitting positive limit while moving positively
                    if self.speeds_out[i] > 0 and current_pos >= PAROL6_ROBOT.Joint_limits_steps[i][1]:
                         logger.warning(f"Limit reached on joint {i + 1}. Stopping jog.")
                         self.is_finished = True
                         Speed_out[:] = [0] * 6
                         Command_out.value = 255
                         return True
                    # Hitting negative limit while moving negatively
                    elif self.speeds_out[i] < 0 and current_pos <= PAROL6_ROBOT.Joint_limits_steps[i][0]:
                         logger.warning(f"Limit reached on joint {i + 1}. Stopping jog.")
                         self.is_finished = True
                         Speed_out[:] = [0] * 6
                         Command_out.value = 255
                         return True

            # If no limits are hit, apply the speeds
            Speed_out[:] = self.speeds_out
            Command_out.value = 123 # Jog command
            self.command_step += 1
            return False # Command is still running
