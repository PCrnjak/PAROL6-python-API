"""
Utility Commands
Contains utility commands like Delay and Reset
"""

import logging

from parol6.commands.base import (
    CommandBase,
    ExecutionStatusCode,
    MotionCommand,
    SystemCommand,
)
from parol6.config import CONTROL_RATE_HZ
from parol6.protocol.wire import (
    CheckpointCmd,
    CmdType,
    DelayCmd,
    ResetLoopStatsCmd,
    ResetStateCmd,
    SetStatusRateCmd,
)
from parol6.server.status_cache import get_cache
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import ConfigurationError
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command(CmdType.DELAY)
class DelayCommand(CommandBase[DelayCmd]):
    """
    A non-blocking command that pauses execution for a specified duration.
    """

    PARAMS_TYPE = DelayCmd

    __slots__ = ()

    def do_setup(self, state: "ControllerState") -> None:
        self.start_timer(self.p.seconds)
        logger.info(f"  -> Delay starting for {self.p.seconds} seconds...")

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Keep the robot idle during the delay."""
        state.Command_out = CommandCode.IDLE
        state.Speed_out.fill(0)

        if self.timer_expired():
            logger.info(f"Delay finished after {self.p.seconds} seconds.")
            self.finish()
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.RESET_STATE)
class ResetStateCommand(SystemCommand[ResetStateCmd]):
    """
    Instantly reset controller state to initial values.
    """

    PARAMS_TYPE = ResetStateCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        state.reset()
        self._sync_mock = True
        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.RESET_LOOP_STATS)
class ResetLoopStatsCommand(SystemCommand[ResetLoopStatsCmd]):
    """
    Reset control loop timing statistics without affecting controller state.

    Resets: min/max period, overrun count, rolling statistics.
    Preserves: loop_count (uptime), robot state, command queues.
    """

    PARAMS_TYPE = ResetLoopStatsCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        state.loop_stats_reset_pending = True
        logger.debug("RESET_LOOP_STATS command executed")
        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.SET_STATUS_RATE)
class SetStatusRateCommand(SystemCommand[SetStatusRateCmd]):
    """Change the status broadcast rate for this session.

    Status is emitted every Nth control tick, so a rate that does not divide
    the control rate evenly cannot be served. It is refused rather than
    rounded to a neighbour: a capture taken at a rate nobody asked for is
    wrong in a way nothing reports.
    """

    PARAMS_TYPE = SetStatusRateCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        hz = float(self.p.hz)
        control = int(CONTROL_RATE_HZ)
        if hz <= 0.0 or hz > control or control % int(hz) != 0 or hz != int(hz):
            allowed = ", ".join(
                str(control // n) for n in range(1, control + 1) if control % n == 0
            )
            raise ConfigurationError(
                make_error(
                    ErrorCode.SYS_STATUS_RATE_INVALID,
                    requested=hz,
                    control=control,
                    allowed=allowed,
                )
            )
        state.status_rate_hz = hz
        get_cache().set_status_rate(hz)
        logger.info("Status broadcast rate set to %g Hz", hz)
        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.CHECKPOINT)
class CheckpointCommand(MotionCommand[CheckpointCmd]):
    """Queue marker that sets state.last_checkpoint on execution.

    Completes immediately on first tick. Used for progress tracking
    without affecting motion.
    """

    PARAMS_TYPE = CheckpointCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        state.last_checkpoint = self.p.label
        self.finish()
        self.log_info("Checkpoint reached: %s", self.p.label)
        return ExecutionStatusCode.COMPLETED
