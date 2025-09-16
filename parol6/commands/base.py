"""
Base abstractions and helpers for command implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING, List
from abc import ABC, abstractmethod
from enum import Enum

import parol6.PAROL6_ROBOT as PAROL6_ROBOT

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


class ExecutionStatusCode(Enum):
    """Enumeration for command execution status codes."""
    QUEUED = "QUEUED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ExecutionStatus:
    """
    Status returned from command execution steps.
    """
    code: ExecutionStatusCode
    message: str
    error: Optional[Exception] = None
    details: Optional[Dict[str, Any]] = None

    @classmethod
    def executing(cls, message: str = "Executing", details: Optional[Dict[str, Any]] = None) -> "ExecutionStatus":
        """Create an EXECUTING status."""
        return cls(ExecutionStatusCode.EXECUTING, message, details=details)

    @classmethod
    def completed(cls, message: str = "Completed", details: Optional[Dict[str, Any]] = None) -> "ExecutionStatus":
        """Create a COMPLETED status."""
        return cls(ExecutionStatusCode.COMPLETED, message, details=details)

    @classmethod
    def failed(cls, message: str, error: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None) -> "ExecutionStatus":
        """Create a FAILED status."""
        return cls(ExecutionStatusCode.FAILED, message, error=error, details=details)


class CommandBase(ABC):
    """
    Reusable base for commands with shared lifecycle and safety helpers.
    """
    def __init__(self) -> None:
        self.is_valid: bool = True
        self.is_finished: bool = False
        self.error_state: bool = False
        self.error_message: str = ""
        # Optional context set by controller (commands "already have access" to these)
        self.udp_transport: Any = None
        self.addr: Any = None
        self.gcode_interpreter: Any = None

    # Ensure command objects are usable as dict keys (e.g., in server command_id_map)
    def __hash__(self) -> int:
        # Identity-based hash is appropriate for ephemeral command instances
        return id(self)

    @abstractmethod
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if this command can handle the given message parts.

        Args:
            parts: Pre-split message parts (e.g., ['JOG', '0', '50', '2.0', 'None'])

        Returns:
            Tuple of (can_handle, error_message)
            - can_handle: True if this command can process the message
            - error_message: Optional error message if the message is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def setup(self, state: "ControllerState", *, udp_transport: Any = None, addr: Any = None, gcode_interpreter: Any = None) -> None:
        """
        Prepare the command for execution using current robot state.

        Pass context that may change between creation and execution as keyword args.
        Commands should also read self.udp_transport/self.addr/self.gcode_interpreter set by controller.
        """
        # Default: bind any provided context to the instance
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter
        raise NotImplementedError

    @abstractmethod
    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """
        Execute one control-loop step and return an ExecutionStatus.

        Commands MUST interact with state.* arrays/buffers directly (Position_in/out, Speed_out, Command_out, etc.).
        """
        raise NotImplementedError

    def teardown(self, state: "ControllerState") -> None:
        """Optional cleanup hook after completion or failure."""
        return

    # ----- lifecycle helpers -----

    def finish(self) -> None:
        """Mark command as finished."""
        self.is_finished = True

    def fail(self, message: str) -> None:
        """Mark command as invalid/failed with an error message."""
        self.is_valid = False
        self.error_state = True
        self.error_message = message
        self.is_finished = True

    # ----- safety / limits helpers -----

    @staticmethod
    def within_limits(joint_index: int, position_steps: int) -> bool:
        """
        Check if a joint position (in steps) is within configured limits.
        """
        min_limit, max_limit = PAROL6_ROBOT.Joint_limits_steps[joint_index]
        return min_limit <= position_steps <= max_limit

    @staticmethod
    def clamp_to_limits(joint_index: int, position_steps: int) -> int:
        """
        Clamp a joint position (in steps) to configured limits.
        """
        min_limit, max_limit = PAROL6_ROBOT.Joint_limits_steps[joint_index]
        return max(min(position_steps, max_limit), min_limit)

    @staticmethod
    def joint_dir_and_index(joint_selector: int) -> Tuple[int, int]:
        """
        Convert "jog selector" (0..5 positive, 6..11 negative for reverse) into
        (direction, joint_index) where direction is +1 or -1 and joint_index is 0..5.
        """
        direction = 1 if 0 <= joint_selector <= 5 else -1
        joint_index = joint_selector if direction == 1 else joint_selector - 6
        return direction, joint_index


class QueryCommand(CommandBase):
    """
    Base class for query commands that execute immediately and bypass the queue.
    
    Query commands are read-only operations that return information about the robot state.
    They execute immediately without waiting in the command queue.
    """
    


class MotionCommand(CommandBase):
    """
    Base class for motion commands that require the controller to be enabled.
    
    Motion commands involve robot movement and require the controller to be in an enabled state.
    Some motion commands (like jog commands) can be replaced in stream mode.
    """
    streamable: bool = False  # Can be replaced in stream mode (only for jog commands)
    


class SystemCommand(CommandBase):
    """
    Base class for system control commands that can execute regardless of enable state.
    
    System commands control the overall state of the robot controller (enable/disable, stop, etc.)
    and can execute even when the controller is disabled.
    """
