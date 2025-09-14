"""
Utility Commands
Contains utility commands like Delay
"""

import logging
import time
from typing import List, Tuple, Optional
from parol6.commands.base import CommandBase, ExecutionStatus, ExecutionStatusCode
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
        Parameters are parsed in match() method.
        """
        super().__init__()
        self.duration = None
        self.end_time = None

    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse DELAY command parameters.

        Format: DELAY|duration
        Example: DELAY|2.5
        """
        if len(parts) != 2:
            return (False, "DELAY requires 1 parameter: duration")

        try:
            self.duration = float(parts[1])
            if self.duration <= 0:
                return (False, f"Delay duration must be positive, got {self.duration}")
            logger.info(f"Parsed Delay command for {self.duration} seconds")
            self.is_valid = True
            return (True, None)
        except ValueError:
            return (False, f"Invalid duration: {parts[1]}")
        except Exception as e:
            return (False, f"Error parsing DELAY: {str(e)}")

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Set the end time when the command actually starts."""
        # Bind dynamic context if provided (per policy); no-op otherwise
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter

        if self.duration:
            self.end_time = time.time() + self.duration
            logger.info(f"  -> Delay starting for {self.duration} seconds...")

    def execute_step(self, state) -> ExecutionStatus:
        """
        Keep the robot idle during the delay and report status via ExecutionStatus.
        """
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")

        # Keep the robot idle during the delay
        state.Command_out = CommandCode.IDLE
        state.Speed_out[:] = [0] * 6

        # Check for completion
        if self.end_time and time.time() >= self.end_time:
            logger.info(f"Delay finished after {self.duration} seconds.")
            self.is_finished = True
            return ExecutionStatus.completed("Delay complete")

        return ExecutionStatus.executing("Delaying")
