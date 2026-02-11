"""
Base abstractions and helpers for command implementations.
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from parol6.config import TRACE
from parol6.protocol.wire import CmdType, Command, CommandCode, QueryType
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class ExecutionStatusCode(Enum):
    """Enumeration for command execution status codes."""

    QUEUED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


# ----- Small utilities -----


class Countdown:
    """Simple count-down timer."""

    def __init__(self, count: int):
        self.count = max(0, int(count))

    def tick(self) -> bool:
        """Decrement and return True when reaches zero."""
        if self.count > 0:
            self.count -= 1
        return self.count == 0


class Debouncer:
    """Simple count-based debouncer."""

    def __init__(self, count: int = 5) -> None:
        self.count_init = max(0, int(count))
        self.count = self.count_init

    def reset(self) -> None:
        self.count = self.count_init

    def tick(self, active: bool) -> bool:
        """
        Returns True exactly once when 'active' stays non-zero for 'count_init' ticks.
        Resets when 'active' becomes False.
        """
        if active:
            if self.count > 0:
                self.count -= 1
            return self.count == 0
        else:
            self.reset()
            return False


P = TypeVar("P")


class CommandBase(ABC, Generic[P]):
    """
    Reusable base for commands with shared lifecycle and safety helpers.

    Commands use typed msgspec structs for parameters. The PARAMS_TYPE class
    variable indicates which struct type this command expects.
    """

    # Set by @register_command decorator; used by controller stream fast-path
    _cmd_type: ClassVar[CmdType | None] = None

    # The params struct type this command expects (override in subclass)
    PARAMS_TYPE: ClassVar[type[Command] | None] = None

    __slots__ = (
        "p",
        "is_finished",
        "error_state",
        "error_message",
        "_t0",
        "_t_end",
        "_q_rad_buf",
        "_steps_buf",
    )

    def __init__(self, p: P) -> None:
        self.p = p
        self.is_finished: bool = False
        self.error_state: bool = False
        self.error_message: str = ""
        self._t0: float | None = None
        self._t_end: float | None = None
        # Pre-allocated buffers for zero-allocation unit conversions
        self._q_rad_buf: np.ndarray = np.zeros(6, dtype=np.float64)
        self._steps_buf: np.ndarray = np.zeros(6, dtype=np.int32)

    # Ensure command objects are usable as dict keys (e.g., in server command_id_map)
    def __hash__(self) -> int:
        # Identity-based hash is appropriate for ephemeral command instances
        return id(self)

    @property
    def name(self) -> str:
        return self._cmd_type.name if self._cmd_type else type(self).__name__

    # Logging helpers (uniform, include command identity)
    def log_trace(self, msg: str, *args: Any) -> None:
        logger.log(TRACE, "[%s] " + msg, self.name, *args)

    def log_debug(self, msg: str, *args: Any) -> None:
        logger.debug("[%s] " + msg, self.name, *args)

    def log_info(self, msg: str, *args: Any) -> None:
        logger.info("[%s] " + msg, self.name, *args)

    def log_warning(self, msg: str, *args: Any) -> None:
        logger.warning("[%s] " + msg, self.name, *args)

    def log_error(self, msg: str, *args: Any) -> None:
        logger.error("[%s] " + msg, self.name, *args)

    @staticmethod
    def stop_and_idle(state: ControllerState) -> None:
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.IDLE

    def assign_params(self, params: Command) -> None:
        """
        Assign pre-validated params struct.

        Called AFTER msgspec has decoded and validated the struct
        (via constraints and __post_init__). No validation here.
        """
        self.p = params

    def do_setup(self, state: ControllerState) -> None:
        """Subclass hook for preparation; override in subclasses."""
        return

    def setup(self, state: ControllerState) -> None:
        """Public setup wrapper providing centralized logging and error handling."""
        self.log_trace("setup start")
        try:
            self.do_setup(state)
            self.log_trace("setup ok")
        except Exception as e:
            # Mark invalid and propagate for higher-level lifecycle logging
            self.fail(f"Setup error: {e}")
            self.log_error("Setup error: %s", e)
            raise

    def tick(self, state: ControllerState) -> ExecutionStatusCode:
        """Template method: guards + execute_step + error handling."""
        if self.is_finished:
            return (
                ExecutionStatusCode.FAILED
                if self.error_state
                else ExecutionStatusCode.COMPLETED
            )
        try:
            return self.execute_step(state)
        except Exception as e:
            self._on_tick_error(state, e)
            return ExecutionStatusCode.FAILED

    def _on_tick_error(self, state: ControllerState, error: Exception) -> None:
        """Error-path cleanup. Override in subclasses for specialized behavior."""
        self.fail(str(error))
        self.log_error(str(error))

    @abstractmethod
    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """
        Execute one control-loop step.

        Returns ExecutionStatusCode.EXECUTING while in progress,
        COMPLETED when done, or FAILED on error.

        Commands MUST interact with state.* arrays/buffers directly
        (Position_in/out, Speed_out, Command_out, etc.).
        """
        raise NotImplementedError

    # ----- lifecycle helpers -----

    def finish(self) -> None:
        """Mark command as finished."""
        self.is_finished = True

    def fail(self, message: str) -> None:
        """Mark command as failed with an error message."""
        self.error_state = True
        self.error_message = message
        self.is_finished = True

    # ---- timing helpers ----
    def start_timer(self, duration_s: float) -> None:
        """Start a timer for the given duration in seconds."""
        self._t_end = time.perf_counter() + max(0.0, duration_s)

    def timer_expired(self) -> bool:
        """Check if the timer has expired."""
        return self._t_end is not None and time.perf_counter() >= self._t_end

    def progress01(self, duration_s: float) -> float:
        """Get progress as a value between 0 and 1."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        if duration_s <= 0.0:
            return 1.0
        p = (time.perf_counter() - self._t0) / duration_s
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)


class QueryCommand(CommandBase[P]):
    """
    Base class for query commands that execute immediately and bypass the queue.

    Query commands compute a result, pack it as a wire response, and return
    the bytes. The controller calls compute() and sends the result directly.
    Subclasses set QUERY_TYPE and implement compute().
    """

    QUERY_TYPE: ClassVar[QueryType]

    @abstractmethod
    def compute(self, state: ControllerState) -> bytes:
        """Compute the query result, pack it, and return response bytes."""
        ...

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        # Queries are dispatched via compute() by the controller.
        # This exists only to satisfy the abstract method.
        raise NotImplementedError("Queries use compute(), not execute_step()")


class MotionCommand(CommandBase[P]):
    """
    Base class for motion commands that require the controller to be enabled.

    Motion commands involve robot movement and require the controller to be in an enabled state.
    Some motion commands (like jog commands) can be replaced in stream mode.
    """

    streamable: bool = False

    def fail_and_idle(self, state: ControllerState, message: str) -> None:
        self.fail(message)
        self.stop_and_idle(state)

    def set_move_position(self, state: ControllerState, steps: np.ndarray) -> None:
        """Set position for MOVE command (zero speeds, Command=MOVE)."""
        state.Position_out[:] = steps
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.MOVE

    def _on_tick_error(self, state: ControllerState, error: Exception) -> None:
        """Zero speeds and set IDLE on error."""
        self.fail_and_idle(state, str(error))
        self.log_error(str(error))


class TrajectoryMoveCommandBase(MotionCommand[P]):
    """
    Base class for commands that execute pre-computed trajectories.

    Subclasses pre-compute trajectory_steps in do_setup(). Velocity/acceleration
    limits are enforced during trajectory building via local segment slowdown,
    so execute_step() simply outputs waypoints tick-by-tick.
    """

    __slots__ = ("trajectory_steps", "command_step", "_duration")

    def __init__(self, p: P):
        super().__init__(p)
        self.trajectory_steps: np.ndarray = np.empty((0, 6), dtype=np.int32)
        self.command_step = 0
        self._duration: float = 0.0

    @property
    def blend_radius(self) -> float:
        """Blend radius in mm. Default 0 (stop at target). Read from params.r if present."""
        return float(getattr(self.p, "r", 0.0))

    def do_setup_with_blend(
        self,
        state: ControllerState,
        next_cmds: "list[TrajectoryMoveCommandBase]",
    ) -> int:
        """Set up trajectory with blend through N next commands.

        Subclasses that support blending (MoveLCommand, JointMoveCommandBase)
        override this method. The default falls back to single-command setup.

        Returns:
            Number of *next_cmds* consumed (0 = no blending).
        """
        self.do_setup(state)
        return 0

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Execute trajectory by outputting pre-computed waypoints."""
        if self.command_step >= len(self.trajectory_steps):
            self.log_info("%s finished.", self.__class__.__name__)
            self.finish()
            self.stop_and_idle(state)
            return ExecutionStatusCode.COMPLETED

        target = self.trajectory_steps[self.command_step]
        state.Position_out[:] = target
        state.Command_out = CommandCode.MOVE
        self.command_step += 1

        return ExecutionStatusCode.EXECUTING


class SystemCommand(CommandBase[P]):
    """
    Base class for system control commands that can execute regardless of enable state.

    System commands control the overall state of the robot controller (enable/disable, stop, etc.)
    and can execute even when the controller is disabled.

    Side-effect signaling: commands that need infrastructure changes (simulator toggle,
    port switch, mock sync) set the corresponding attribute. The controller reads these
    after tick() and orchestrates the actual change.
    """

    __slots__ = ("_switch_simulator", "_switch_port", "_sync_mock")

    def __init__(self, p: P) -> None:
        super().__init__(p)
        self._switch_simulator: bool | None = None
        self._switch_port: str | None = None
        self._sync_mock: bool = False
