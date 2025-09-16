"""
Gripper Control Commands
Contains commands for electric and pneumatic gripper control
"""

import logging
from typing import List, Tuple, Optional
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.base import MotionCommand, ExecutionStatus
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)


@register_command("PNEUMATICGRIPPER")
@register_command("ELECTRICGRIPPER")
class GripperCommand(MotionCommand):
    """
    A single, unified, non-blocking command to control all gripper functions.
    It internally selects the correct logic (position-based waiting, timed delay,
    or instantaneous) based on the specified action.
    """
    def __init__(self):
        """
        Initializes the Gripper command.
        Parameters are parsed in match() method.
        """
        super().__init__()
        self.state = "START"
        self.timeout_counter = 1000  # 10-second safety timeout for all waiting states
        self.detection_debounce_counter = 5  # 0.05s debounce for object detection
        self.wait_counter = 0
        
        # Parameters (set in match())
        self.gripper_type = None
        self.action = None
        self.target_position = None
        self.speed = None
        self.current = None
        self.state_to_set = None
        self.port_index = None
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse gripper command parameters.
        
        Formats:
        - PNEUMATICGRIPPER|action|port
        - ELECTRICGRIPPER|action|pos|spd|curr
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            Tuple of (can_handle, error_message)
        """
        command_name = parts[0].upper()
        
        if command_name == "PNEUMATICGRIPPER":
            if len(parts) != 3:
                return (False, "PNEUMATICGRIPPER requires 2 parameters: action, port")
            
            try:
                self.gripper_type = 'pneumatic'
                self.action = parts[1].lower()
                output_port = int(parts[2])
                
                # Validate action
                if self.action not in ['open', 'close']:
                    return (False, f"Invalid pneumatic gripper action: {self.action}")
                
                # Configure pneumatic settings
                self.state_to_set = 1 if self.action == 'open' else 0
                self.port_index = 2 if output_port == 1 else 3
                
                logger.info(f"Parsed PNEUMATICGRIPPER: action={self.action}, port={output_port}")
                self.is_valid = True
                return (True, None)
                
            except ValueError as e:
                return (False, f"Invalid PNEUMATICGRIPPER parameters: {str(e)}")
        
        elif command_name == "ELECTRICGRIPPER":
            if len(parts) != 5:
                return (False, "ELECTRICGRIPPER requires 4 parameters: action, position, speed, current")
            
            try:
                self.gripper_type = 'electric'
                
                # Parse action
                action_token = parts[1].upper()
                self.action = 'move' if action_token in ('NONE', 'MOVE') else parts[1].lower()
                
                # Parse numeric parameters
                position = int(parts[2])
                speed = int(parts[3])
                current = int(parts[4])
                
                # Configure based on action
                if self.action == 'move':
                    self.target_position = position
                    self.speed = speed
                    self.current = current
                    
                    # Validate ranges
                    if not (0 <= position <= 255):
                        return (False, f"Position must be 0-255, got {position}")
                    if not (0 <= speed <= 255):
                        return (False, f"Speed must be 0-255, got {speed}")
                    if not (100 <= current <= 1000):
                        return (False, f"Current must be 100-1000, got {current}")
                        
                elif self.action == 'calibrate':
                    self.wait_counter = 200  # 2-second fixed delay for calibration
                else:
                    return (False, f"Invalid electric gripper action: {self.action}")
                
                logger.info(f"Parsed ELECTRICGRIPPER: action={self.action}, pos={position}, speed={speed}, current={current}")
                self.is_valid = True
                return (True, None)
                
            except ValueError as e:
                return (False, f"Invalid ELECTRICGRIPPER parameters: {str(e)}")
        
        return (False, f"Unknown gripper command: {command_name}")

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind dynamic context if provided; no further precomputation required."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

    def execute_step(self, state) -> ExecutionStatus:
        """State-based execution for pneumatic and electric grippers."""
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            logger.error(f"  -> ERROR: Gripper command timed out in state {self.state}.")
            self.is_finished = True
            return ExecutionStatus.failed("Gripper timeout")

        # --- Pneumatic Logic (Instantaneous) ---
        if self.gripper_type == 'pneumatic':
            state.InOut_out[self.port_index] = self.state_to_set
            logger.info("  -> Pneumatic gripper command sent.")
            self.is_finished = True
            return ExecutionStatus.completed("Pneumatic gripper toggled")

        # --- Electric Gripper Logic ---
        if self.gripper_type == 'electric':
            # On the first run, transition to the correct state for the action
            if self.state == "START":
                if self.action == 'calibrate':
                    self.state = "SEND_CALIBRATE"
                else:  # 'move'
                    self.state = "WAIT_FOR_POSITION"
            
            # --- Calibrate Logic (Timed Delay) ---
            if self.state == "SEND_CALIBRATE":
                logger.debug("  -> Sending one-shot calibrate command...")
                state.Gripper_data_out[4] = 1  # Set mode to calibrate
                self.state = "WAITING_CALIBRATION"
                return ExecutionStatus.executing("Calibrating")

            if self.state == "WAITING_CALIBRATION":
                self.wait_counter -= 1
                if self.wait_counter <= 0:
                    logger.info("  -> Calibration delay finished.")
                    state.Gripper_data_out[4] = 0  # Reset to operation mode
                    self.is_finished = True
                    return ExecutionStatus.completed("Calibration complete")
                return ExecutionStatus.executing("Calibrating")

            # --- Move Logic (Position-Based) ---
            if self.state == "WAIT_FOR_POSITION":
                # Persistently send the move command
                state.Gripper_data_out[0] = self.target_position
                state.Gripper_data_out[1] = self.speed
                state.Gripper_data_out[2] = self.current
                state.Gripper_data_out[4] = 0  # Operation mode

                bitfield = [1, 1, not state.InOut_in[4], 1, 0, 0, 0, 0]
                fused = PAROL6_ROBOT.fuse_bitfield_2_bytearray(bitfield)
                state.Gripper_data_out[3] = int(fused.hex(), 16)

                object_detection = state.Gripper_data_in[5] if len(state.Gripper_data_in) > 5 else 0
                logger.debug(f" -> Gripper moving to {self.target_position} (current: {state.Gripper_data_in[1]}), object detected: {object_detection}")

                while self.detection_debounce_counter > 0 and object_detection != 0:
                    self.detection_debounce_counter -= 1

                # Check for completion
                current_position = state.Gripper_data_in[1]
                if abs(current_position - self.target_position) <= 5:
                    logger.info("  -> Gripper move complete.")
                    self.is_finished = True
                    # Set command back to idle
                    bitfield = [1, 0, not state.InOut_in[4], 1, 0, 0, 0, 0]
                    fused = PAROL6_ROBOT.fuse_bitfield_2_bytearray(bitfield)
                    state.Gripper_data_out[3] = int(fused.hex(), 16)
                    return ExecutionStatus.completed("Gripper move complete")
            
                if (object_detection == 1) and (self.target_position > current_position):
                    logger.info("  -> Gripper move holding position due to object detection when closing.")
                    self.is_finished = True
                    return ExecutionStatus.completed("Object detected while closing - hold")

                if (object_detection == 2) and (self.target_position < current_position):
                    logger.info("  -> Gripper move holding position due to object detection when opening.")
                    self.is_finished = True
                    return ExecutionStatus.completed("Object detected while opening - hold")
                
                return ExecutionStatus.executing("Moving gripper")

        # Should not reach here for known gripper types
        return ExecutionStatus.failed("Unknown gripper type")
