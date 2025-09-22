"""
Utility Commands
Contains utility commands like Delay
"""

import logging
from typing import List, Tuple, Optional
from parol6.commands.base import CommandBase, ExecutionStatus, parse_float
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command

logger = logging.getLogger(__name__)


@register_command("DELAY")
class DelayCommand(CommandBase):
    """
    A non-blocking command that pauses execution for a specified duration.
    During the delay, it ensures the robot remains idle by sending the
    appropriate commands.
    """
    def __init__(self):
        """
        Initializes the Delay command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()
        self.duration = None

    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse DELAY command parameters.

        Format: DELAY|duration
        Example: DELAY|2.5
        """
        if len(parts) != 2:
            return (False, "DELAY requires 1 parameter: duration")

        try:
            self.duration = parse_float(parts[1])
            if self.duration is None or self.duration <= 0:
                return (False, f"Delay duration must be positive, got {parts[1]}")
            logger.info(f"Parsed Delay command for {self.duration} seconds")
            self.is_valid = True
            return (True, None)
        except Exception as e:
            return (False, f"Error parsing DELAY: {str(e)}")

    def setup(self, state):
        """Start the delay timer."""
        if self.duration:
            self.start_timer(self.duration)
            logger.info(f"  -> Delay starting for {self.duration} seconds...")

    def execute_step(self, state) -> ExecutionStatus:
        """
        Keep the robot idle during the delay and report status via ExecutionStatus.
        """
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        # Keep the robot idle during the delay
        state.Command_out = CommandCode.IDLE
        state.Speed_out.fill(0)

        # Check for completion
        if self.timer_expired():
            logger.info(f"Delay finished after {self.duration} seconds.")
            self.is_finished = True
            return ExecutionStatus.completed("Delay complete")

        return ExecutionStatus.executing("Delaying")
