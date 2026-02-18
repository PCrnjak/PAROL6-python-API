"""
Gripper Control Commands
Contains commands for electric and pneumatic gripper control
"""

import logging
from enum import Enum

from parol6.commands.base import Debouncer, ExecutionStatusCode, MotionCommand
from parol6.protocol.wire import CmdType, ElectricGripperCmd, PneumaticGripperCmd
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode

logger = logging.getLogger(__name__)


def _pack_gripper_bits(bits: list[int]) -> int:
    """Pack a list of 8 bit values into a single byte (MSB-first)."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


class GripperState(Enum):
    START = "START"
    SEND_CALIBRATE = "SEND_CALIBRATE"
    WAITING_CALIBRATION = "WAITING_CALIBRATION"
    WAIT_FOR_POSITION = "WAIT_FOR_POSITION"


@register_command(CmdType.PNEUMATICGRIPPER)
class PneumaticGripperCommand(MotionCommand[PneumaticGripperCmd]):
    """Control pneumatic gripper (open/close)."""

    PARAMS_TYPE = PneumaticGripperCmd

    __slots__ = (
        "state",
        "timeout_counter",
        "_state_to_set",
        "_port_index",
    )

    def __init__(self, p: PneumaticGripperCmd):
        super().__init__(p)
        self.state = GripperState.START
        self.timeout_counter = 1000
        self._state_to_set: int = 0
        self._port_index: int = 0

    def do_setup(self, state: "ControllerState") -> None:
        """Compute port index and state to set from params."""
        self._state_to_set = 1 if self.p.action == "open" else 0
        # port 1 -> index 2, port 2 -> index 3
        self._port_index = 2 if self.p.port == 1 else 3

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute pneumatic gripper command."""
        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            self.fail(make_error(ErrorCode.MOTN_GRIPPER_TIMEOUT, state=str(self.state)))
            return ExecutionStatusCode.FAILED

        state.InOut_out[self._port_index] = self._state_to_set
        logger.info("  -> Pneumatic gripper command sent.")
        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.ELECTRICGRIPPER)
class ElectricGripperCommand(MotionCommand[ElectricGripperCmd]):
    """Control electric gripper (move/calibrate)."""

    PARAMS_TYPE = ElectricGripperCmd

    __slots__ = (
        "state",
        "timeout_counter",
        "object_debouncer",
        "wait_counter",
        "_hw_position",
        "_hw_speed",
    )

    def __init__(self, p: ElectricGripperCmd):
        super().__init__(p)
        self.state = GripperState.START
        self.timeout_counter = 1000
        self.object_debouncer = Debouncer(5)
        self.wait_counter = 0
        self._hw_position = 0
        self._hw_speed = 1

    def do_setup(self, state: "ControllerState") -> None:
        """Scale normalized 0-1 values to hardware 0-255 range."""
        self._hw_position = int(round(self.p.position * 255))
        self._hw_speed = max(1, int(round(self.p.speed * 255)))
        if self.p.action == "calibrate":
            self.wait_counter = 200

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """State-based execution for electric gripper."""
        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            self.fail(make_error(ErrorCode.MOTN_GRIPPER_TIMEOUT, state=str(self.state)))
            return ExecutionStatusCode.FAILED

        if self.state == GripperState.START:
            if self.p.action == "calibrate":
                self.state = GripperState.SEND_CALIBRATE
            else:
                self.state = GripperState.WAIT_FOR_POSITION

        if self.state == GripperState.SEND_CALIBRATE:
            logger.debug("  -> Sending one-shot calibrate command...")
            state.Gripper_data_out[4] = 1
            self.state = GripperState.WAITING_CALIBRATION
            return ExecutionStatusCode.EXECUTING

        if self.state == GripperState.WAITING_CALIBRATION:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                logger.info("  -> Calibration delay finished.")
                state.Gripper_data_out[4] = 0
                self.finish()
                return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        if self.state == GripperState.WAIT_FOR_POSITION:
            state.Gripper_data_out[0] = self._hw_position
            state.Gripper_data_out[1] = self._hw_speed
            state.Gripper_data_out[2] = self.p.current
            state.Gripper_data_out[4] = 0

            state.Gripper_data_out[3] = _pack_gripper_bits(
                [1, 1, int(not state.InOut_in[4]), 1, 0, 0, 0, 0]
            )

            object_detection = (
                state.Gripper_data_in[5] if len(state.Gripper_data_in) > 5 else 0
            )
            logger.debug(
                f" -> Gripper moving to {self._hw_position} (current: {state.Gripper_data_in[1]}), object detected: {object_detection}"
            )

            object_detected = self.object_debouncer.tick(object_detection != 0)

            current_position = state.Gripper_data_in[1]
            if abs(current_position - self._hw_position) <= 5:
                logger.info("  -> Gripper move complete.")
                self.finish()
                state.Gripper_data_out[3] = _pack_gripper_bits(
                    [1, 0, int(not state.InOut_in[4]), 1, 0, 0, 0, 0]
                )
                return ExecutionStatusCode.COMPLETED

            if object_detected:
                if (object_detection == 1) and (self._hw_position > current_position):
                    logger.info(
                        "  -> Gripper move holding position due to object detection when closing."
                    )
                    self.finish()
                    return ExecutionStatusCode.COMPLETED

                if (object_detection == 2) and (self._hw_position < current_position):
                    logger.info(
                        "  -> Gripper move holding position due to object detection when opening."
                    )
                    self.finish()
                    return ExecutionStatusCode.COMPLETED

            return ExecutionStatusCode.EXECUTING

        self.fail(make_error(ErrorCode.MOTN_GRIPPER_UNKNOWN))
        return ExecutionStatusCode.FAILED
