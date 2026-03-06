"""
Gripper Control Commands — 100 Hz execution engines for tool actions.

These are instantiated by ToolActionCommand, not directly from wire structs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from parol6.commands.base import Debouncer, ExecutionStatusCode, MotionCommand
from parol6.server.state import ControllerState
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode

logger = logging.getLogger(__name__)


class ElectricGripperState(Enum):
    """State machine states for the electric gripper command engine."""

    START = "START"
    SEND_CALIBRATE = "SEND_CALIBRATE"
    WAITING_CALIBRATION = "WAITING_CALIBRATION"
    WAIT_FOR_POSITION = "WAIT_FOR_POSITION"


@dataclass(frozen=True)
class PneumaticGripperParams:
    """Parameters for pneumatic gripper action."""

    action: str  # "open" or "close"
    port: int  # 1 or 2


@dataclass(frozen=True)
class ElectricGripperParams:
    """Parameters for electric gripper action."""

    action: str  # "move" or "calibrate"
    position: float
    speed: float
    current: int


class PneumaticGripperCommand(MotionCommand[PneumaticGripperParams]):
    """Control pneumatic gripper (open/close)."""

    PARAMS_TYPE = None  # Not wire-registered — instantiated by ToolActionCommand

    __slots__ = (
        "timeout_counter",
        "_state_to_set",
        "_port_index",
    )

    def __init__(self, p: PneumaticGripperParams):
        super().__init__(p)
        self.timeout_counter = 1000
        self._state_to_set: int = 0
        self._port_index: int = 0

    @classmethod
    def from_tool_action(cls, *, action: str, port: int) -> PneumaticGripperCommand:
        return cls(PneumaticGripperParams(action=action, port=port))

    def do_setup(self, state: ControllerState) -> None:
        self._state_to_set = 1 if self.p.action == "open" else 0
        # port 1 -> index 2, port 2 -> index 3
        self._port_index = 2 if self.p.port == 1 else 3

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            self.fail(make_error(ErrorCode.MOTN_GRIPPER_TIMEOUT))
            return ExecutionStatusCode.FAILED

        state.InOut_out[self._port_index] = self._state_to_set
        logger.info("  -> Pneumatic gripper command sent.")
        self.finish()
        return ExecutionStatusCode.COMPLETED


class ElectricGripperCommand(MotionCommand[ElectricGripperParams]):
    """Control electric gripper (move/calibrate)."""

    PARAMS_TYPE = None  # Not wire-registered — instantiated by ToolActionCommand

    __slots__ = (
        "state",
        "timeout_counter",
        "object_debouncer",
        "wait_counter",
        "_hw_position",
        "_hw_speed",
    )

    def __init__(self, p: ElectricGripperParams):
        super().__init__(p)
        self.state = ElectricGripperState.START
        self.timeout_counter = 1000
        self.object_debouncer = Debouncer(5)
        self.wait_counter = 0
        self._hw_position = 0
        self._hw_speed = 1

    @classmethod
    def from_tool_action(
        cls, *, action: str, position: float = 0.0, speed: float = 0.5, current: int = 500
    ) -> ElectricGripperCommand:
        return cls(ElectricGripperParams(action=action, position=position, speed=speed, current=current))

    def do_setup(self, state: ControllerState) -> None:
        self._hw_position = int(round(self.p.position * 255))
        self._hw_speed = max(1, int(round(self.p.speed * 255)))
        if self.p.action == "calibrate":
            self.wait_counter = 200

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            self.fail(make_error(ErrorCode.MOTN_GRIPPER_TIMEOUT, state=str(self.state)))
            return ExecutionStatusCode.FAILED

        hw = state.gripper_hw

        if self.state == ElectricGripperState.START:
            if self.p.action == "calibrate":
                self.state = ElectricGripperState.SEND_CALIBRATE
            else:
                self.state = ElectricGripperState.WAIT_FOR_POSITION

        if self.state == ElectricGripperState.SEND_CALIBRATE:
            logger.debug("  -> Sending one-shot calibrate command...")
            hw.mode = 1
            self.state = ElectricGripperState.WAITING_CALIBRATION
            return ExecutionStatusCode.EXECUTING

        if self.state == ElectricGripperState.WAITING_CALIBRATION:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                logger.info("  -> Calibration delay finished.")
                hw.mode = 0
                self.finish()
                return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        if self.state == ElectricGripperState.WAIT_FOR_POSITION:
            hw.target_position = self._hw_position
            hw.target_speed = self._hw_speed
            hw.target_current = self.p.current
            hw.mode = 0

            estop = not state.InOut_in[4]
            hw.set_command_bits(move_active=True, estop=estop)

            object_detection = hw.object_detection

            object_detected = self.object_debouncer.tick(object_detection != 0)

            current_position = hw.feedback_position
            if abs(current_position - self._hw_position) <= 5:
                logger.info("  -> Gripper move complete.")
                self.finish()
                hw.set_command_bits(move_active=False, estop=estop)
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
