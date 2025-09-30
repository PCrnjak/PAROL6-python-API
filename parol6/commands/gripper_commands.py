"""
Gripper Control Commands
Contains commands for electric and pneumatic gripper control
"""

import logging
import parol6.PAROL6_ROBOT as PAROL6_ROBOT

logger = logging.getLogger(__name__)

class GripperCommand:
    """
    A single, unified, non-blocking command to control all gripper functions.
    It internally selects the correct logic (position-based waiting, timed delay,
    or instantaneous) based on the specified action.
    """
    def __init__(self, gripper_type, action=None, position=100, speed=100, current=500, output_port=1):
        """
        Initializes the Gripper command and configures its internal state machine
        based on the requested action.
        """
        self.is_valid = True
        self.is_finished = False
        self.gripper_type = gripper_type.lower()
        self.action = action.lower() if action else 'move'
        self.state = "START"
        self.timeout_counter = 1000 # 10-second safety timeout for all waiting states
        self.detection_debounce_counter = 5 # 0.05s debounce for object detection

        # --- Configure based on Gripper Type and Action ---
        if self.gripper_type == 'electric':
            if self.action == 'move':
                self.target_position = position
                self.speed = speed
                self.current = current
                if not (0 <= position <= 255 and 0 <= speed <= 255 and 100 <= current <= 1000):
                    self.is_valid = False
            elif self.action == 'calibrate':
                self.wait_counter = 200 # 2-second fixed delay for calibration
            else:
                self.is_valid = False # Invalid action

        elif self.gripper_type == 'pneumatic':
            if self.action not in ['open', 'close']:
                self.is_valid = False
            self.state_to_set = 1 if self.action == 'open' else 0
            self.port_index = 2 if output_port == 1 else 3
        else:
            self.is_valid = False

        if not self.is_valid:
            logger.error(f"  -> VALIDATION FAILED for GripperCommand with action: '{self.action}'")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, 
                     Gripper_data_out=None, InOut_out=None, 
                     Gripper_data_in=None, InOut_in=None, **kwargs):
        # Gripper commands require gripper data parameters
        if Gripper_data_out is None or InOut_out is None or Gripper_data_in is None or InOut_in is None:
            # Try to get from kwargs if not provided as positional arguments
            Gripper_data_out = kwargs.get('Gripper_data_out', Gripper_data_out)
            InOut_out = kwargs.get('InOut_out', InOut_out)
            Gripper_data_in = kwargs.get('Gripper_data_in', Gripper_data_in)
            InOut_in = kwargs.get('InOut_in', InOut_in)
            
            # If still None, we have a problem
            if Gripper_data_out is None or InOut_out is None:
                logger.error("GripperCommand requires Gripper_data_out and InOut_out parameters")
                self.is_finished = True
                return True
        
        if self.is_finished or not self.is_valid:
            return True

        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            logger.error(f"  -> ERROR: Gripper command timed out in state {self.state}.")
            self.is_finished = True
            return True

        # --- Pneumatic Logic (Instantaneous) ---
        if self.gripper_type == 'pneumatic':
            InOut_out[self.port_index] = self.state_to_set
            logger.info("  -> Pneumatic gripper command sent.")
            self.is_finished = True
            return True

        # --- Electric Gripper Logic ---
        if self.gripper_type == 'electric':
            # On the first run, transition to the correct state for the action
            if self.state == "START":
                if self.action == 'calibrate':
                    self.state = "SEND_CALIBRATE"
                else: # 'move'
                    self.state = "WAIT_FOR_POSITION"
            
            # --- Calibrate Logic (Timed Delay) ---
            if self.state == "SEND_CALIBRATE":
                logger.debug("  -> Sending one-shot calibrate command...")
                Gripper_data_out[4] = 1 # Set mode to calibrate
                self.state = "WAITING_CALIBRATION"
                return False

            if self.state == "WAITING_CALIBRATION":
                self.wait_counter -= 1
                if self.wait_counter <= 0:
                    logger.info("  -> Calibration delay finished.")
                    Gripper_data_out[4] = 0 # Reset to operation mode
                    self.is_finished = True
                    return True
                return False

            # --- Move Logic (Position-Based) ---
            if self.state == "WAIT_FOR_POSITION":
                # Persistently send the move command
                Gripper_data_out[0], Gripper_data_out[1], Gripper_data_out[2] = self.target_position, self.speed, self.current
                Gripper_data_out[4] = 0 # Operation mode
                bitfield = [1, 1, not InOut_in[4], 1, 0, 0, 0, 0]
                fused = PAROL6_ROBOT.fuse_bitfield_2_bytearray(bitfield)
                Gripper_data_out[3] = int(fused.hex(), 16)

                object_detection = Gripper_data_in[5] if len(Gripper_data_in) > 5 else 0
                logger.debug(f" -> Gripper moving to {self.target_position} (current: {Gripper_data_in[1]}), object detected: {object_detection}")

                while self.detection_debounce_counter > 0 and object_detection != 0:
                    self.detection_debounce_counter -= 1

                # Check for completion
                current_position = Gripper_data_in[1]
                if abs(current_position - self.target_position) <= 5:
                    logger.info("  -> Gripper move complete.")
                    self.is_finished = True
                    # Set command back to idle
                    bitfield = [1, 0, not InOut_in[4], 1, 0, 0, 0, 0]
                    fused = PAROL6_ROBOT.fuse_bitfield_2_bytearray(bitfield)
                    Gripper_data_out[3] = int(fused.hex(), 16)
                    return True
            
                if (object_detection == 1) and (self.target_position > current_position):
                    logger.info("  -> Gripper move holding position due to object detection when closing.")
                    self.is_finished = True
                    return True
                
                if (object_detection == 2) and (self.target_position < current_position):
                    logger.info("  -> Gripper move holding position due to object detection when opening.")
                    self.is_finished = True
                    return True
                
                return False
        
        return self.is_finished