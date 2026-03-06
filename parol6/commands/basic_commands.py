"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and SetTool.
"""

import logging
from enum import Enum, auto
import numpy as np

from parol6.config import (
    JOG_MIN_STEPS,
    LIMITS,
    rad_to_steps,
    speed_steps_to_rad_scalar,
    steps_to_rad,
)
from parol6.protocol.wire import (
    CmdType,
    HomeCmd,
    JogJCmd,
    SetToolCmd,
)
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode

from .base import (
    ExecutionStatusCode,
    MotionCommand,
)

logger = logging.getLogger(__name__)


def _limit_hit_mask(pos_steps: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    return ((speeds > 0) & (pos_steps >= LIMITS.joint.position.steps[:, 1])) | (
        (speeds < 0) & (pos_steps <= LIMITS.joint.position.steps[:, 0])
    )


class HomeState(Enum):
    """State machine states for the homing sequence."""

    START = auto()
    WAITING_FOR_UNHOMED = auto()
    WAITING_FOR_HOMED = auto()


@register_command(CmdType.HOME)
class HomeCommand(MotionCommand[HomeCmd]):
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """

    PARAMS_TYPE = HomeCmd

    __slots__ = (
        "state",
        "start_cmd_counter",
        "timeout_counter",
    )

    def __init__(self, p: HomeCmd):
        super().__init__(p)
        self.state = HomeState.START
        self.start_cmd_counter = 10
        self.timeout_counter = 2000

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Manages the homing command and monitors for completion using a state machine."""
        if self.state == HomeState.START:
            logger.debug(
                "  -> Sending home signal (100)... Countdown: %d",
                self.start_cmd_counter,
            )
            state.Command_out = CommandCode.HOME
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                self.state = HomeState.WAITING_FOR_UNHOMED
            return ExecutionStatusCode.EXECUTING

        if self.state == HomeState.WAITING_FOR_UNHOMED:
            state.Command_out = CommandCode.IDLE
            if np.any(state.Homed_in[:6] == 0):
                logger.info("  -> Homing sequence initiated by robot.")
                self.state = HomeState.WAITING_FOR_HOMED
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                self.fail(make_error(ErrorCode.MOTN_HOME_TIMEOUT))
                self.stop_and_idle(state)
                return ExecutionStatusCode.FAILED
            return ExecutionStatusCode.EXECUTING

        if self.state == HomeState.WAITING_FOR_HOMED:
            state.Command_out = CommandCode.IDLE
            if np.all(state.Homed_in[:6] == 1):
                self.log_info("Homing sequence complete. All joints reported home.")
                self.finish()
                self.stop_and_idle(state)
                return ExecutionStatusCode.COMPLETED
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                self.fail(make_error(ErrorCode.MOTN_HOME_TIMEOUT))
                self.stop_and_idle(state)
                return ExecutionStatusCode.FAILED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.JOGJ)
class JogJCommand(MotionCommand[JogJCmd]):
    """
    A non-blocking command to jog joints for a specific duration.
    Uses static 6-element speed array on the wire (all joints, zeros for inactive).
    """

    PARAMS_TYPE = JogJCmd
    streamable = True

    __slots__ = (
        "speeds_out",
        "_jog_initialized",
        "_jog_vel_rad",
    )

    def __init__(self, p: JogJCmd):
        super().__init__(p)
        self.speeds_out = np.zeros(6, dtype=np.int32)
        self._jog_initialized = False
        self._jog_vel_rad = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute step speeds and rad/s velocities for all 6 joints."""
        for i in range(6):
            s = self.p.speeds[i]
            if s == 0.0:
                self.speeds_out[i] = 0
                self._jog_vel_rad[i] = 0.0
            else:
                frac = min(abs(s), 1.0)
                step_speed = int(
                    JOG_MIN_STEPS
                    + (LIMITS.joint.jog.velocity_steps[i] - JOG_MIN_STEPS) * frac
                )
                self.speeds_out[i] = step_speed if s > 0 else -step_speed
                self._jog_vel_rad[i] = speed_steps_to_rad_scalar(step_speed, i) * (
                    1 if s > 0 else -1
                )
        self.start_timer(self.p.duration)
        self._jog_initialized = False

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of joint jogging via StreamingExecutor."""
        se = state.streaming_executor

        # Sync position on first tick
        if not self._jog_initialized:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            se.sync_position(self._q_rad_buf)
            self._jog_initialized = True

        stop_reason = self._check_stop_conditions(state)

        if stop_reason:
            self._jog_vel_rad.fill(0.0)
            se.set_jog_velocity(self._jog_vel_rad)
            pos_rad, vel, finished = se.tick()
            self._q_rad_buf[:] = pos_rad
            rad_to_steps(self._q_rad_buf, self._steps_buf)
            self.set_move_position(state, self._steps_buf)

            if finished or all(abs(v) < 0.001 for v in vel):
                if stop_reason.startswith("Limit"):
                    logger.warning(stop_reason)
                else:
                    self.log_trace(stop_reason)
                self.finish()
                return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        se.set_jog_velocity(self._jog_vel_rad)
        pos_rad, _vel, _finished = se.tick()
        self._q_rad_buf[:] = pos_rad
        rad_to_steps(self._q_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        return ExecutionStatusCode.EXECUTING

    def _check_stop_conditions(self, state: "ControllerState") -> str | None:
        """Check if jog should stop. Returns stop reason or None."""
        if self.timer_expired():
            return "Timed jog finished."

        limit_mask = _limit_hit_mask(state.Position_in, self.speeds_out)
        if np.any(limit_mask):
            return f"Limit reached on joint {int(np.argmax(limit_mask)) + 1}."

        return None


@register_command(CmdType.SET_TOOL)
class SetToolCommand(MotionCommand[SetToolCmd]):
    """
    Set the current end-effector tool configuration.
    """

    PARAMS_TYPE = SetToolCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Set the tool in state and update robot kinematics."""
        tool_name = self.p.tool_name.strip().upper()

        # Update server state - property setter handles tool application and cache invalidation
        state.current_tool = tool_name

        self.log_info(f"Tool set to: {tool_name}")
        self.finish()
        return ExecutionStatusCode.COMPLETED
