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
from parol6.protocol.wire import (
    CheckpointCmd,
    CmdType,
    DelayCmd,
    ResetCmd,
    ResetLoopStatsCmd,
)
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
        """Start the delay timer."""
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


@register_command(CmdType.RESET)
class ResetCommand(SystemCommand[ResetCmd]):
    """
    Instantly reset controller state to initial values.
    """

    PARAMS_TYPE = ResetCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Reset state immediately."""
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
        """Signal controller to reset loop stats."""
        state.loop_stats_reset_pending = True
        logger.debug("RESET_LOOP_STATS command executed")
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
