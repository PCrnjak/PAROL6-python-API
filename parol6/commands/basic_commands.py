"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and MultiJog
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from .base import CommandBase, ExecutionStatus, ExecutionStatusCode
from parol6.config import INTERVAL_S
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)


@register_command("HOME")
class HomeCommand(CommandBase):
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """
    def __init__(self):
        super().__init__(is_valid=True)
        # State machine: START -> WAIT_FOR_UNHOMED -> WAIT_FOR_HOMED -> FINISHED
        self.state = "START"
        # Counter to send the home command for multiple cycles
        self.start_cmd_counter = 10  # Send command 100 for 10 cycles (0.1s)
        # Safety timeout (20 seconds at 0.01s interval)
        self.timeout_counter = 2000
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse HOME command (no parameters).
        
        Format: HOME
        """
        if len(parts) != 1:
            return (False, "HOME command takes no parameters")
        
        logger.info("Parsed HOME command")
        return (True, None)

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """No pre-computation needed for HOME; bind dynamic context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

    def execute_step(self, state) -> ExecutionStatus:
        """
        Manages the homing command and monitors for completion using a state machine.
        """
        if self.is_finished:
            return ExecutionStatus.completed("Already finished")

        # --- State: START ---
        # On the first few executions, continuously send the 'home' (100) command.
        if self.state == "START":
            logger.debug(f"  -> Sending home signal (100)... Countdown: {self.start_cmd_counter}")
            state.Command_out = CommandCode.HOME
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                # Once sent for enough cycles, move to the next state
                self.state = "WAITING_FOR_UNHOMED"
            return ExecutionStatus.executing("Homing: start")

        # --- State: WAITING_FOR_UNHOMED ---
        # The robot's firmware should reset the homed status. We wait to see that happen.
        # During this time, we send 'idle' (255) to let the robot's controller take over.
        if self.state == "WAITING_FOR_UNHOMED":
            state.Command_out = CommandCode.IDLE
            # Homing sequence initiated detection
            if any(h == 0 for h in state.Homed_in[:6]):
                logger.info("  -> Homing sequence initiated by robot.")
                self.state = "WAITING_FOR_HOMED"
            # Homing timeout protection
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                logger.error("  -> ERROR: Timeout waiting for robot to start homing sequence.")
                self.is_finished = True
                return ExecutionStatus.failed("Homing timeout")
            return ExecutionStatus.executing("Homing: waiting for unhomed")

        # --- State: WAITING_FOR_HOMED ---
        # Now we wait for all joints to report that they are homed (all flags are 1).
        if self.state == "WAITING_FOR_HOMED":
            state.Command_out = CommandCode.IDLE
            # Homing completion verification
            if all(h == 1 for h in state.Homed_in[:6]):
                logger.info("Homing sequence complete. All joints reported home.")
                self.is_finished = True
                state.Speed_out[:] = [0] * 6  # Ensure robot is stopped
                return ExecutionStatus.completed("Homing complete")

            return ExecutionStatus.executing("Homing: waiting for homed")

        return ExecutionStatus.executing("Homing")


@register_command("JOG")
class JogCommand(CommandBase):
    """
    A non-blocking command to jog a joint for a specific duration or distance.
    It performs all safety and validity checks upon initialization.
    """
    def __init__(self):
        """
        Initializes the jog command.
        Parameters are parsed in match() method.
        """
        super().__init__()
        self.mode = None
        self.command_step = 0
        
        # Parameters (set in match())
        self.joint = None
        self.speed_percentage = None
        self.duration = None
        self.distance_deg = None
        
        # Calculated values
        self.direction = 1
        self.joint_index = 0
        self.speed_out = 0
        self.command_len = 0
        self.target_position = 0
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse JOG command parameters.
        
        Format: JOG|joint|speed_pct|duration|distance
        Example: JOG|0|50|2.0|None
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 5:
            return (False, "JOG requires 4 parameters: joint, speed, duration, distance")
        
        try:
            # Parse parameters
            self.joint = int(parts[1])
            self.speed_percentage = float(parts[2])
            self.duration = None if parts[3].upper() == 'NONE' else float(parts[3])
            self.distance_deg = None if parts[4].upper() == 'NONE' else float(parts[4])
            
            # Determine mode
            if self.duration and self.distance_deg:
                self.mode = 'distance'
                logger.info(f"Parsed Jog: Joint {self.joint}, Distance {self.distance_deg} deg, Duration {self.duration}s.")
            elif self.duration:
                self.mode = 'time'
                logger.info(f"Parsed Jog: Joint {self.joint}, Speed {self.speed_percentage}%, Duration {self.duration}s.")
            elif self.distance_deg:
                self.mode = 'distance'
                logger.info(f"Parsed Jog: Joint {self.joint}, Speed {self.speed_percentage}%, Distance {self.distance_deg} deg.")
            else:
                return (False, "JOG requires either duration or distance")
            
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid JOG parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing JOG: {str(e)}")

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        """Pre-computes speeds and target positions using live data."""
        # Bind dynamic context if provided
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

        logger.debug("  -> Preparing for Jog command...")

        # Joint direction and index mapping
        self.direction = 1 if 0 <= self.joint <= 5 else -1
        self.joint_index = self.joint if self.direction == 1 else self.joint - 6
        
        distance_steps = 0
        if self.distance_deg:
            distance_steps = int(PAROL6_ROBOT.DEG2STEPS(abs(self.distance_deg), self.joint_index))
            self.target_position = state.Position_in[self.joint_index] + (distance_steps * self.direction)
            
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

    def execute_step(self, state) -> ExecutionStatus:
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        stop_reason = None
        current_pos = state.Position_in[self.joint_index]

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
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.IDLE
            return ExecutionStatus.completed(stop_reason)
        else:
            state.Speed_out[:] = [0] * 6
            state.Speed_out[self.joint_index] = self.speed_out
            state.Command_out = CommandCode.JOG
            self.command_step += 1
            return ExecutionStatus.executing("Jogging")


@register_command("MULTIJOG")  
class MultiJogCommand(CommandBase):
    """
    A non-blocking command to jog multiple joints simultaneously for a specific duration.
    It performs all safety and validity checks upon initialization.
    """
    def __init__(self):
        """
        Initializes the multi-jog command.
        Parameters are parsed in match() method.
        """
        super().__init__()
        self.command_step = 0
        
        # Parameters (set in match())
        self.joints = None
        self.speed_percentages = None
        self.duration = None
        self.command_len = 0
        
        # Calculated values
        self.speeds_out = [0] * 6
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse MULTIJOG command parameters.
        
        Format: MULTIJOG|joints_csv|speeds_csv|duration
        Example: MULTIJOG|0,1,2|50,75,100|3.0
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 4:
            return (False, "MULTIJOG requires 3 parameters: joints, speeds, duration")
        
        try:
            # Parse parameters
            self.joints = [int(j) for j in parts[1].split(',')]
            self.speed_percentages = [float(s) for s in parts[2].split(',')]
            self.duration = float(parts[3])
            
            # Validate
            if len(self.joints) != len(self.speed_percentages):
                return (False, "Number of joints must match number of speeds")
            
            if self.duration <= 0:
                return (False, "Duration must be positive")
            
            # Check for conflicting joint commands
            base_joints = set()
            for joint in self.joints:
                # Normalize the joint index to its base (0-5)
                base_joint = joint % 6
                # If the base joint is already in our set, it's a conflict.
                if base_joint in base_joints:
                    return (False, f"Conflicting commands for Joint {base_joint + 1}")
                base_joints.add(base_joint)
            
            logger.info(f"Parsed MultiJog for joints {self.joints} with speeds {self.speed_percentages}% for {self.duration}s.")
            
            self.command_len = int(self.duration / INTERVAL_S)
            self.is_valid = True
            return (True, None)
            
        except ValueError as e:
            return (False, f"Invalid MULTIJOG parameters: {str(e)}")
        except Exception as e:
            return (False, f"Error parsing MULTIJOG: {str(e)}")

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        """Pre-computes the speeds for each joint."""
        # Bind dynamic context if provided
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

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

    def execute_step(self, state) -> ExecutionStatus:
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        # Stop if the duration has elapsed
        if self.command_step >= self.command_len:
            logger.info("Timed multi-jog finished.")
            self.is_finished = True
            state.Speed_out[:] = [0] * 6
            state.Command_out = CommandCode.IDLE
            return ExecutionStatus.completed("MultiJog complete")
        else:
            # Continuously check for joint limits during the jog
            for i in range(6):
                if self.speeds_out[i] != 0:
                    current_pos = state.Position_in[i]
                    # Hitting positive limit while moving positively
                    if self.speeds_out[i] > 0 and current_pos >= PAROL6_ROBOT.Joint_limits_steps[i][1]:
                         logger.warning(f"Limit reached on joint {i + 1}. Stopping jog.")
                         self.is_finished = True
                         state.Speed_out[:] = [0] * 6
                         state.Command_out = CommandCode.IDLE
                         return ExecutionStatus.completed(f"Limit reached on J{i+1}")
                    # Hitting negative limit while moving negatively
                    elif self.speeds_out[i] < 0 and current_pos <= PAROL6_ROBOT.Joint_limits_steps[i][0]:
                         logger.warning(f"Limit reached on joint {i + 1}. Stopping jog.")
                         self.is_finished = True
                         state.Speed_out[:] = [0] * 6
                         state.Command_out = CommandCode.IDLE
                         return ExecutionStatus.completed(f"Limit reached on J{i+1}")

            # If no limits are hit, apply the speeds
            state.Speed_out[:] = self.speeds_out
            state.Command_out = CommandCode.JOG
            self.command_step += 1
            return ExecutionStatus.executing("MultiJogging")
