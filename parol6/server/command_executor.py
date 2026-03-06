"""Command queue management and execution."""

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from parol6.commands.base import (
    CommandBase,
    ExecutionStatusCode,
    MotionCommand,
)
from parol6.config import MAX_COMMAND_QUEUE_SIZE, TRACE
from parol6.protocol.wire import Command, decode_command
from waldoctl import ActionState

if TYPE_CHECKING:
    from parol6.server.state import ControllerState, StateManager

logger = logging.getLogger("parol6.server.command_executor")

_LONG_FLOAT_RE = re.compile(r"-?\d+\.\d{3,}")
_MAX_ACTION_PARAMS_LEN = 100


def _format_cmd_params(p: object) -> str:
    """Format wire command params for action log display.

    Strips the class name prefix and rounds long floats to 2 decimal places.
    """
    s = repr(p)
    i = s.find("(")
    if i >= 0:
        s = s[i:]
    return _LONG_FLOAT_RE.sub(lambda m: f"{float(m.group()):.2f}", s)[:_MAX_ACTION_PARAMS_LEN]


class QueueFullError(Exception):
    """Raised when the command queue is at capacity."""


@dataclass
class QueuedCommand:
    """Represents a command in the queue with metadata."""

    command: CommandBase
    command_id: str | None = None
    command_index: int = -1
    address: tuple[str, int] | None = None
    queued_time: float = field(default_factory=time.time)
    activated: bool = False
    first_tick_logged: bool = False


class CommandExecutor:
    """Manages the motion command queue and 100Hz execution loop.

    Responsibilities:
    - Queue management (queue_command, clear_queue, clear_streamable_commands)
    - 100Hz tick loop (execute_active_command)
    - Streaming fast-path (try_stream_fast_path)
    - Cancel operations (cancel_active_command, cancel_active_streamable)

    Immediate commands (system, query) are handled directly by the controller.
    """

    def __init__(self, state_manager: "StateManager"):
        self._state_manager = state_manager

        self.command_queue: deque[QueuedCommand] = deque(maxlen=MAX_COMMAND_QUEUE_SIZE)
        self.active_command: QueuedCommand | None = None

    def _update_queue_state(self, state: "ControllerState") -> None:
        """Update queue snapshot and next action in state."""
        # Reuse list to avoid allocation (clear + extend pattern)
        state.queue_nonstreamable.clear()
        for qc in self.command_queue:
            if not (
                isinstance(qc.command, MotionCommand)
                and qc.command.streamable
            ):
                state.queue_nonstreamable.append(type(qc.command).__name__)
        state.action_next = (
            state.queue_nonstreamable[0] if state.queue_nonstreamable else ""
        )

    # ---- Streaming fast-path ----

    def try_stream_fast_path(
        self,
        data: bytes,
        state: "ControllerState",
    ) -> bool | Command:
        """Attempt stream fast-path for active streamable command.

        When in stream mode with an active streamable command, this allows
        updating the command's parameters without full command creation/queueing.

        Returns:
            True if command was handled via fast-path.
            False if no decode was attempted (no active streamable, or decode failed).
            A Command struct if decoded successfully but type didn't match active command.
        """
        if not self.active_command:
            return False

        active_inst = self.active_command.command
        if not (isinstance(active_inst, MotionCommand) and active_inst.streamable):
            return False

        # Decode incoming command
        try:
            cmd_struct = decode_command(data)
        except Exception as e:
            logger.debug("Stream fast-path decode failed: %s", e)
            return False

        # Check if struct type matches active command's expected type
        active_params_type = getattr(active_inst, "PARAMS_TYPE", None)
        if active_params_type is None or type(cmd_struct) is not active_params_type:
            return cmd_struct  # Return decoded struct for caller to reuse

        logger.log(
            TRACE,
            "stream_fast_path active=%s incoming=%s",
            type(active_inst).__name__,
            type(cmd_struct).__name__,
        )

        # Assign new params (validation already done by decode)
        active_inst.assign_params(cmd_struct)

        # Re-setup with new params
        try:
            active_inst.setup(state)
            logger.log(TRACE, "stream_fast_path applied")
            return True
        except Exception as e:
            logger.error("Stream fast-path setup failed: %s", e)
            return False

    # ---- Command queueing ----

    def queue_command(
        self,
        address: tuple[str, int] | None,
        command: CommandBase,
        command_id: str | None = None,
    ) -> int:
        """Add a command to the execution queue.

        Args:
            address: Optional (ip, port) for acknowledgments.
            command: The command to queue.
            command_id: Optional ID for tracking.

        Returns:
            The assigned command index.

        Raises:
            QueueFullError: If the queue is at capacity.
        """
        if len(self.command_queue) >= MAX_COMMAND_QUEUE_SIZE:
            logger.warning("Command queue full (max %d)", MAX_COMMAND_QUEUE_SIZE)
            raise QueueFullError("Queue full")

        # Assign monotonic command index
        state = self._state_manager.get_state()
        cmd_index = state.next_command_index
        state.next_command_index += 1

        # Create queued command
        queued_cmd = QueuedCommand(
            command=command,
            command_id=command_id,
            command_index=cmd_index,
            address=address,
        )

        self.command_queue.append(queued_cmd)

        # Update queue snapshot
        self._update_queue_state(state)

        logger.log(
            TRACE,
            "Queued command: %s (ID: %s, index: %d)",
            type(command).__name__,
            command_id,
            cmd_index,
        )

        return cmd_index

    # ---- Active command execution (called every control loop tick) ----

    def execute_active_command(self) -> None:
        """Execute one step of the active command from the queue."""
        if not self._activate_next():
            return

        ac = self.active_command
        assert ac is not None  # _activate_next guarantees this

        try:
            state = self._state_manager.get_state()

            if not state.enabled:
                self.cancel_active_command("Controller disabled")
                return

            # One-time setup on first activation
            if not ac.activated:
                self._setup_active(ac, state)
                state.action_current = type(ac.command).__name__
                state.action_params = _format_cmd_params(ac.command.p)
                state.action_state = ActionState.EXECUTING
                state.executing_command_index = ac.command_index
                ac.activated = True
                logger.log(
                    TRACE,
                    "Activated command: %s (id=%s, index=%d)",
                    type(ac.command).__name__,
                    ac.command_id,
                    ac.command_index,
                )

            # Execute one tick
            if not ac.first_tick_logged:
                logger.log(TRACE, "tick_start name=%s", type(ac.command).__name__)
                ac.first_tick_logged = True

            code = ac.command.tick(state)
            self._process_tick_result(ac, code, state)

        except Exception as e:
            logger.error("Command execution error: %s", e)
            state.action_current = ""
            state.action_params = ""
            state.action_state = ActionState.IDLE
            self._update_queue_state(state)
            self.active_command = None

    def _activate_next(self) -> bool:
        """Promote next queued command to active.

        Returns True if there is an active command (existing or newly promoted).
        """
        if self.active_command is not None:
            return True
        if not self.command_queue:
            return False
        self.active_command = self.command_queue.popleft()
        self.active_command.activated = False
        return True

    def _setup_active(self, ac: QueuedCommand, state: "ControllerState") -> None:
        """Run one-time setup for a command."""
        ac.command.setup(state)

    def _process_tick_result(
        self,
        ac: QueuedCommand,
        code: ExecutionStatusCode,
        state: "ControllerState",
    ) -> None:
        """Handle post-tick bookkeeping: completion, failure."""
        if code == ExecutionStatusCode.COMPLETED:
            logger.log(
                TRACE,
                "Command completed: %s (id=%s, index=%d) at t=%f",
                type(ac.command).__name__,
                ac.command_id,
                ac.command_index,
                time.time(),
            )

            state.action_current = ""
            state.action_params = ""
            state.action_state = ActionState.IDLE
            state.completed_command_index = ac.command_index
            self._update_queue_state(state)
            self.active_command = None

        elif code == ExecutionStatusCode.FAILED:
            logger.debug(
                "Command failed: %s (id=%s) - %s at t=%.6f",
                type(ac.command).__name__,
                ac.command_id,
                ac.command.robot_error,
                time.time(),
            )

            state.action_current = ""
            state.action_params = ""
            state.action_state = ActionState.IDLE

            # Clear queued streamable commands on failure to prevent pileup
            if isinstance(ac.command, MotionCommand) and ac.command.streamable:
                removed = self.clear_streamable_commands(
                    f"Active streamable command failed: {ac.command.robot_error}"
                )
                if removed > 0:
                    logger.info(
                        "Cleared %d queued streamable commands due to active command failure",
                        removed,
                    )

            self._update_queue_state(state)
            self.active_command = None

    # ---- Cancellation and queue management ----

    def cancel_active_command(self, reason: str = "Cancelled by user") -> None:
        """Cancel the currently active command."""
        if not self.active_command:
            return

        logger.info(
            "Cancelling active command: %s - %s",
            type(self.active_command.command).__name__,
            reason,
        )

        state = self._state_manager.get_state()
        state.action_current = ""
        state.action_params = ""
        state.action_state = ActionState.IDLE

        self.active_command = None

    def cancel_active_streamable(self) -> bool:
        """Cancel active command if it's a streamable motion command.

        Returns:
            True if a command was cancelled.
        """
        ac = self.active_command
        if (
            ac
            and isinstance(ac.command, MotionCommand)
            and ac.command.streamable
        ):
            state = self._state_manager.get_state()
            state.action_current = ""
            state.action_params = ""
            state.action_state = ActionState.IDLE
            self.active_command = None
            return True
        return False

    def clear_queue(self, reason: str = "Queue cleared") -> None:
        """Clear all queued commands."""
        count = len(self.command_queue)
        self.command_queue.clear()

        logger.info("Cleared %d commands from queue: %s", count, reason)

        state = self._state_manager.get_state()
        state.queue_nonstreamable.clear()
        state.action_next = ""

    def clear_streamable_commands(
        self, reason: str = "Streamable commands cleared"
    ) -> int:
        """Clear all queued streamable motion commands."""
        removed_count = 0
        to_remove: list[QueuedCommand] = []

        for queued_cmd in self.command_queue:
            if isinstance(queued_cmd.command, MotionCommand) and queued_cmd.command.streamable:
                to_remove.append(queued_cmd)

        for queued_cmd in to_remove:
            self.command_queue.remove(queued_cmd)
            removed_count += 1

        if removed_count > 0:
            logger.debug(
                "Cleared %d streamable commands from queue: %s", removed_count, reason
            )

        return removed_count
