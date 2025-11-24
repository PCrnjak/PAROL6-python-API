"""
Gripper Control Commands
Contains commands for electric and pneumatic gripper control
"""

import logging
from enum import Enum

from parol6.commands.base import Debouncer, ExecutionStatus, MotionCommand
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)

# Lifecycle TRACE is centralized in higher layers; keep semantic logs here only.


class GripperState(Enum):
    START = "START"
    SEND_CALIBRATE = "SEND_CALIBRATE"
    WAITING_CALIBRATION = "WAITING_CALIBRATION"
    WAIT_FOR_POSITION = "WAIT_FOR_POSITION"


@register_command("PNEUMATICGRIPPER")
@register_command("ELECTRICGRIPPER")
class GripperCommand(MotionCommand):
    """
    A single, unified, non-blocking command to control all gripper functions.
    It internally selects the correct logic (position-based waiting, timed delay,
    or instantaneous) based on the specified action.
    """

    __slots__ = (
        "state",
        "timeout_counter",
        "object_debouncer",
        "wait_counter",
        "gripper_type",
        "action",
        "target_position",
        "speed",
        "current",
        "state_to_set",
        "port_index",
    )

    def __init__(self):
        """
        Initializes the Gripper command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()
        self.state = GripperState.START
        self.timeout_counter = 1000  # 10-second safety timeout for all waiting states
        self.object_debouncer = Debouncer(5)  # 0.05s debounce for object detection
        self.wait_counter = 0

        # Parameters (set in do_match())
        self.gripper_type = None
        self.action = None
        self.target_position = None
        self.speed = None
        self.current = None
        self.state_to_set = None
        self.port_index = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
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

            self.gripper_type = "pneumatic"
            self.action = parts[1].lower()
            output_port = int(parts[2])

            # Validate action
            if self.action not in ["open", "close"]:
                return (False, f"Invalid pneumatic gripper action: {self.action}")

            # Configure pneumatic settings
            self.state_to_set = 1 if self.action == "open" else 0
            self.port_index = 2 if output_port == 1 else 3

            self.log_debug(
                "Parsed PNEUMATICGRIPPER: action=%s, port=%s", self.action, output_port
            )
            self.is_valid = True
            return (True, None)

        elif command_name == "ELECTRICGRIPPER":
            if len(parts) != 5:
                return (
                    False,
                    "ELECTRICGRIPPER requires 4 parameters: action, position, speed, current",
                )

            self.gripper_type = "electric"

            # Parse action
            action_token = parts[1].upper()
            self.action = (
                "move" if action_token in ("NONE", "MOVE") else parts[1].lower()
            )

            # Parse numeric parameters
            position = int(parts[2])
            speed = int(parts[3])
            current = int(parts[4])

            # Configure based on action
            if self.action == "move":
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

            elif self.action == "calibrate":
                self.wait_counter = 200  # 2-second fixed delay for calibration
            else:
                return (False, f"Invalid electric gripper action: {self.action}")

            self.log_debug(
                "Parsed ELECTRICGRIPPER: action=%s, pos=%s, speed=%s, current=%s",
                self.action,
                position,
                speed,
                current,
            )
            self.is_valid = True
            return (True, None)

        return (False, f"Unknown gripper command: {command_name}")

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """State-based execution for pneumatic and electric grippers."""
        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            raise TimeoutError(f"Gripper command timed out in state {self.state}.")

        # --- Pneumatic Logic (Instantaneous) ---
        if self.gripper_type == "pneumatic":
            state.InOut_out[self.port_index] = self.state_to_set
            logger.info("  -> Pneumatic gripper command sent.")
            self.is_finished = True
            return ExecutionStatus.completed("Pneumatic gripper toggled")

        # --- Electric Gripper Logic ---
        if self.gripper_type == "electric":
            # On the first run, transition to the correct state for the action
            if self.state == GripperState.START:
                if self.action == "calibrate":
                    self.state = GripperState.SEND_CALIBRATE
                else:  # 'move'
                    self.state = GripperState.WAIT_FOR_POSITION
            # --- Calibrate Logic (Timed Delay) ---
            if self.state == GripperState.SEND_CALIBRATE:
                logger.debug("  -> Sending one-shot calibrate command...")
                state.Gripper_data_out[4] = 1  # Set mode to calibrate
                self.state = GripperState.WAITING_CALIBRATION
                return ExecutionStatus.executing("Calibrating")

            if self.state == GripperState.WAITING_CALIBRATION:
                self.wait_counter -= 1
                if self.wait_counter <= 0:
                    logger.info("  -> Calibration delay finished.")
                    state.Gripper_data_out[4] = 0  # Reset to operation mode
                    self.is_finished = True
                    return ExecutionStatus.completed("Calibration complete")
                return ExecutionStatus.executing("Calibrating")

            # --- Move Logic (Position-Based) ---
            if self.state == GripperState.WAIT_FOR_POSITION:
                # Persistently send the move command
                state.Gripper_data_out[0] = self.target_position
                state.Gripper_data_out[1] = self.speed
                state.Gripper_data_out[2] = self.current
                state.Gripper_data_out[4] = 0  # Operation mode

                # Pack bitfield with direct bitwise operations (avoid bytearray/hex conversions)
                bits = [1, 1, int(not state.InOut_in[4]), 1, 0, 0, 0, 0]
                val = 0
                for b in bits:
                    val = (val << 1) | int(b)
                state.Gripper_data_out[3] = val

                object_detection = (
                    state.Gripper_data_in[5] if len(state.Gripper_data_in) > 5 else 0
                )
                logger.debug(
                    f" -> Gripper moving to {self.target_position} (current: {state.Gripper_data_in[1]}), object detected: {object_detection}"
                )

                # Use Debouncer from base class for object detection
                object_detected = self.object_debouncer.tick(object_detection != 0)

                # Check for completion
                current_position = state.Gripper_data_in[1]
                if abs(current_position - self.target_position) <= 5:
                    logger.info("  -> Gripper move complete.")
                    self.is_finished = True
                    # Set command back to idle
                    bits = [1, 0, int(not state.InOut_in[4]), 1, 0, 0, 0, 0]
                    val = 0
                    for b in bits:
                        val = (val << 1) | int(b)
                    state.Gripper_data_out[3] = val
                    return ExecutionStatus.completed("Gripper move complete")

                # Check for object detection after debouncing
                if object_detected:
                    if (object_detection == 1) and (
                        self.target_position > current_position
                    ):
                        logger.info(
                            "  -> Gripper move holding position due to object detection when closing."
                        )
                        self.is_finished = True
                        return ExecutionStatus.completed(
                            "Object detected while closing - hold"
                        )

                    if (object_detection == 2) and (
                        self.target_position < current_position
                    ):
                        logger.info(
                            "  -> Gripper move holding position due to object detection when opening."
                        )
                        self.is_finished = True
                        return ExecutionStatus.completed(
                            "Object detected while opening - hold"
                        )

                return ExecutionStatus.executing("Moving gripper")

        # Should not reach here for known gripper types
        return ExecutionStatus.failed("Unknown gripper type")
