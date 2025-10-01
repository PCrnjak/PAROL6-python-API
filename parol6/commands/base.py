"""
Base abstractions and helpers for command implementations.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List, ClassVar, cast
from abc import ABC, abstractmethod
from enum import Enum
import logging
import json
import time

import roboticstoolbox as rp
import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.protocol.wire import CommandCode
from parol6.config import INTERVAL_S, TRACE
from parol6.utils.ik import AXIS_MAP
from parol6.server.state import ControllerState
    

logger = logging.getLogger(__name__)


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
    error_type: Optional[str] = None

    @classmethod
    def executing(cls, message: str = "Executing", details: Optional[Dict[str, Any]] = None) -> "ExecutionStatus":
        return cls(ExecutionStatusCode.EXECUTING, message, error=None, details=details)

    @classmethod
    def completed(cls, message: str = "Completed", details: Optional[Dict[str, Any]] = None) -> "ExecutionStatus":
        return cls(ExecutionStatusCode.COMPLETED, message, error=None, details=details)

    @classmethod
    def failed(cls, message: str, error: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None) -> "ExecutionStatus":
        et = type(error).__name__ if error is not None else None
        return cls(ExecutionStatusCode.FAILED, message, error=error, details=details, error_type=et)


# ----- Shared context and small utilities -----

@dataclass
class CommandContext:
    """Shared dynamic execution context for commands."""
    udp_transport: Any = None
    addr: Optional[tuple] = None
    gcode_interpreter: Any = None
    dt: float = INTERVAL_S

# Parsing utilities (lightweight, shared)
def _noneify(token: Any) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    return None if t == "" or t.upper() in ("NONE", "NULL") else t


def parse_int(token: Any) -> Optional[int]:
    t = _noneify(token)
    return None if t is None else int(t)


def parse_float(token: Any) -> Optional[float]:
    t = _noneify(token)
    return None if t is None else float(t)


def csv_ints(token: Any) -> List[int]:
    t = _noneify(token)
    return [] if t is None else [int(x) for x in t.split(",") if x != ""]


def csv_floats(token: Any) -> List[float]:
    t = _noneify(token)
    return [] if t is None else [float(x) for x in t.split(",") if x != ""]


def parse_bool(token: Any) -> bool:
    t = (str(token or "")).strip().lower()
    return t in ("1", "true", "yes", "on")


def typed(token: Any, type_=float):
    """Parse token with type, supporting None/Null/empty as None."""
    t = _noneify(token)
    if t is None:
        return None
    if type_ is bool:
        return parse_bool(t)
    return type_(t)


def expect_len(parts: List[str], n: int, cmd: str) -> None:
    """Ensure parts list has exactly n elements."""
    if len(parts) != n:
        raise ValueError(f"{cmd} requires {n-1} parameters, got {len(parts)-1}")


def at_least_len(parts: List[str], n: int, cmd: str) -> None:
    """Ensure parts list has at least n elements."""
    if len(parts) < n:
        raise ValueError(f"{cmd} requires at least {n-1} parameters, got {len(parts)-1}")


def parse_frame(token: Any) -> str:
    """Parse and validate frame token (WRF/TRF)."""
    t = (str(token or "")).strip().upper()
    if t not in ("WRF", "TRF"):
        raise ValueError(f"Invalid frame: {token}")
    return t


def parse_axis(token: Any) -> str:
    """Parse and validate axis token against AXIS_MAP."""
    t = (str(token or "")).strip().upper()
    # Convert to match AXIS_MAP format (e.g., +X -> X+, -Y -> Y-)
    if len(t) == 2 and t[0] in "+-" and t[1] in "XYZ":
        t = t[1] + t[0]  # Swap sign and axis
    elif len(t) == 3 and t[0] == "R" and t[2] in "+-":
        t = "R" + t[1] + t[2]  # Keep RX+ format
    if t not in AXIS_MAP:
        raise ValueError(f"Invalid axis: {token}")
    return t


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


class CommandBase(ABC):
    """
    Reusable base for commands with shared lifecycle and safety helpers.
    """
    # Set by @register_command decorator; used by controller stream fast-path
    _registered_name: ClassVar[str] = ""

    __slots__ = ("is_valid", "is_finished", "error_state", "error_message",
                 "udp_transport", "addr", "gcode_interpreter", "_t0", "_t_end")

    def __init__(self) -> None:
        self.is_valid: bool = True
        self.is_finished: bool = False
        self.error_state: bool = False
        self.error_message: str = ""
        self.udp_transport: Any = None
        self.addr: Any = None
        self.gcode_interpreter: Any = None
        self._t0: Optional[float] = None
        self._t_end: Optional[float] = None

    # Ensure command objects are usable as dict keys (e.g., in server command_id_map)
    def __hash__(self) -> int:
        # Identity-based hash is appropriate for ephemeral command instances
        return id(self)

    @property
    def name(self) -> str:
        return self._registered_name or type(self).__name__

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

    def bind(self, context: CommandContext):
        """
        Bind dynamic execution context. Controller should call this prior to setup().
        """
        self.udp_transport = context.udp_transport
        self.addr = context.addr
        self.gcode_interpreter = context.gcode_interpreter

    @abstractmethod
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
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

    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Wrapper that guards subclass do_match() to avoid propagating exceptions.
        Centralizes try/except so subclasses don't repeat it.
        """
        try:
            return self.do_match(parts)
        except Exception as e:
            # Do not log here to avoid duplicate noise; registry/controller provide lifecycle TRACE.
            return False, str(e)

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

    @abstractmethod
    def tick(self, state: ControllerState) -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling and calls do_execute().
        Controllers should prefer tick() over calling execute_step() directly.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """
        Execute one control-loop step and return an ExecutionStatus.

        Commands MUST interact with state.* arrays/buffers directly (Position_in/out, Speed_out, Command_out, etc.).
        """
        raise NotImplementedError

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

class QueryCommand(CommandBase):
    """
    Base class for query commands that execute immediately and bypass the queue.

    Query commands are read-only operations that return information about the robot state.
    They execute immediately without waiting in the command queue.
    """

    def reply_text(self, message: str) -> None:
        """Send an opaque ASCII message via UDP."""
        if self.udp_transport and self.addr:
            try:
                self.udp_transport.send_response(message, self.addr)
            except Exception as e:
                self.log_warning("Failed to send UDP reply: %s", e)

    def reply_ascii(self, prefix_or_message: str, payload: Optional[str] = None) -> None:
        """
        Reply as 'PREFIX|payload' if payload provided; otherwise send prefix_or_message verbatim.
        """
        if payload is None:
            self.reply_text(prefix_or_message)
        else:
            self.reply_text(f"{prefix_or_message}|{payload}")

    def reply_json(self, prefix: str, obj: Any) -> None:
        """Reply with JSON payload."""
        try:
            s = json.dumps(obj)
        except Exception:
            s = "{}"
        self.reply_ascii(prefix, s)

    def tick(self, state: ControllerState) -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling and calls do_execute().
        Controllers should prefer tick() over calling execute_step() directly.
        """
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        try:
            status = self.execute_step(state)
        except Exception as e:
            # Hard failure safeguards
            self.fail(str(e))
            return ExecutionStatus.failed("Execution error", error=e)
        return status


class MotionCommand(CommandBase):
    """
    Base class for motion commands that require the controller to be enabled.

    Motion commands involve robot movement and require the controller to be in an enabled state.
    Some motion commands (like jog commands) can be replaced in stream mode.
    """
    streamable: bool = False  # Can be replaced in stream mode (only for jog commands)

    # Limits and kinematic constants
    LIMS_STEPS: ClassVar[np.ndarray] = PAROL6_ROBOT.joint.limits.steps
    J_MIN: ClassVar[np.ndarray] = PAROL6_ROBOT.joint.speed.min
    J_MAX: ClassVar[np.ndarray] = PAROL6_ROBOT.joint.speed.max
    JOG_MIN: ClassVar[np.ndarray] = PAROL6_ROBOT.joint.speed.jog.min
    JOG_MAX: ClassVar[np.ndarray] = PAROL6_ROBOT.joint.speed.jog.max
    ACC_MIN_RAD: ClassVar[float] = PAROL6_ROBOT.joint.acc.min_rad
    ACC_MAX_RAD: ClassVar[float] = PAROL6_ROBOT.joint.acc.max_rad
    CART_LIN_JOG_MIN: ClassVar[float] = PAROL6_ROBOT.cart.vel.jog.min
    CART_LIN_JOG_MAX: ClassVar[float] = PAROL6_ROBOT.cart.vel.jog.max
    CART_ANG_JOG_MIN: ClassVar[float] = PAROL6_ROBOT.cart.vel.angular.min  # deg/s
    CART_ANG_JOG_MAX: ClassVar[float] = PAROL6_ROBOT.cart.vel.angular.max  # deg/s

    def __init__(self) -> None:
        super().__init__()

    # ---- mapping ----
    @staticmethod
    def linmap_pct(pct: float, lo: float, hi: float) -> float:
        if pct < 0.0:
            pct = 0.0
        elif pct > 100.0:
            pct = 100.0
        return lo + (hi - lo) * (pct / 100.0)

    # ---- per-joint max speed/acc ----
    def joint_vmax(self, velocity_percent: float) -> np.ndarray:
        return self.J_MIN + (self.J_MAX - self.J_MIN) * (max(0.0, min(100.0, velocity_percent)) / 100.0)

    def joint_amax_steps(self, accel_percent: float) -> np.ndarray:
        a_rad = self.linmap_pct(accel_percent, self.ACC_MIN_RAD, self.ACC_MAX_RAD)
        return np.asarray(PAROL6_ROBOT.ops.speed_rad_to_steps(np.full(6, a_rad)), dtype=float)

    # ---- speed scaling & limits ----
    def scale_speeds_to_joint_max(self, speeds: np.ndarray) -> np.ndarray:
        denom = np.where(self.J_MAX != 0.0, self.J_MAX, 1.0)
        scale = float(np.max(np.abs(speeds) / denom))
        if scale > 1.0:
            return np.rint(speeds / scale).astype(np.int32)
        else:
            return np.asarray(speeds, dtype=np.int32)

    def limit_hit_mask(self, pos_steps: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        return ((speeds > 0) & (pos_steps >= self.LIMS_STEPS[:, 1])) | ((speeds < 0) & (pos_steps <= self.LIMS_STEPS[:, 0]))

    # ---- trapezoid batch planner for step-space ----
    @staticmethod
    def plan_trapezoids(start_steps: np.ndarray, target_steps: np.ndarray, tgrid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(tgrid.size)
        q = np.empty((n, 6), dtype=float)
        qd = np.empty((n, 6), dtype=float)
        stationary = (target_steps == start_steps)
        if np.any(stationary):
            q[:, stationary] = start_steps[stationary]
            qd[:, stationary] = 0.0
        for i in np.flatnonzero(~stationary):
            jt = rp.trapezoidal(float(start_steps[i]), float(target_steps[i]), tgrid)
            q[:, i] = jt.q
            qd[:, i] = jt.qd
        return q, qd

    def fail_and_idle(self, state: ControllerState, message: str) -> None:
        self.fail(message)
        self.stop_and_idle(state)

    # ---- Higher-level IO helpers for common patterns ----
    def set_move_position(self, state: ControllerState, steps: np.ndarray) -> None:
        """Set position for MOVE command (zero speeds, Command=MOVE)."""
        np.copyto(state.Position_out, steps, casting='no')
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.MOVE

    def tick(self, state: ControllerState) -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling and calls do_execute().
        Controllers should prefer tick() over calling execute_step() directly.
        """
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")
        try:
            status = self.execute_step(state)
        except Exception as e:
            # Hard failure safeguards
            self.fail_and_idle(state, str(e))
            self.log_error(str(e))
            return ExecutionStatus.failed("Execution error", error=e)
        return status

# TODO: need to get and support the other motion profiles from the original program
class MotionProfile:
    """
    Utilities to build motion profiles in step-space using consistent trapezoids.
    """

    @staticmethod
    def from_duration_steps(
        start_steps: np.ndarray,
        target_steps: np.ndarray,
        duration_s: float,
        dt: float = INTERVAL_S,
    ) -> np.ndarray:
        """
        Build per-joint trapezoids to reach target in given duration.
        Returns array of shape (N, 6) steps (int32).
        """
        dur = float(max(0.0, duration_s))
        if dur == 0.0:
            # Degenerate: single step
            return np.asarray(target_steps, dtype=np.int32).reshape(1, -1)
        n = max(2, int(np.ceil(dur / max(1e-9, dt))))
        tgrid = np.linspace(0.0, dur, n, dtype=float)
        q, _qd = MotionCommand.plan_trapezoids(
            np.asarray(start_steps, dtype=float), np.asarray(target_steps, dtype=float), tgrid
        )
        return cast(np.ndarray, q.astype(np.int32, copy=False))

    @staticmethod
    def from_velocity_percent(
        start_steps: np.ndarray,
        target_steps: np.ndarray,
        velocity_percent: float,
        accel_percent: float,
        dt: float = INTERVAL_S,
    ) -> np.ndarray:
        """
        Build per-joint trapezoids sized by per-joint vmax and accel derived from percent settings.
        """
        start_steps = np.asarray(start_steps, dtype=float)
        target_steps = np.asarray(target_steps, dtype=float)

        # Per-joint vmax and amax (steps/s and steps/s^2)
        jmin = MotionCommand.J_MIN
        jmax = MotionCommand.J_MAX
        v_max_joint = jmin + (jmax - jmin) * (max(0.0, min(100.0, velocity_percent)) / 100.0)

        # Compute accel steps without instantiating MotionCommand
        a_rad = MotionCommand.linmap_pct(accel_percent, MotionCommand.ACC_MIN_RAD, MotionCommand.ACC_MAX_RAD)
        a_steps_vec = np.asarray(PAROL6_ROBOT.ops.speed_rad_to_steps(np.full(6, a_rad)), dtype=float)

        if np.any(v_max_joint <= 0) or np.any(a_steps_vec <= 0):
            raise ValueError("Invalid speed/acceleration (must be positive).")

        path = np.abs(target_steps - start_steps)
        t_accel = v_max_joint / a_steps_vec  # time to reach vmax per joint
        short_path = path < (v_max_joint * t_accel)

        # Safe accel time for short paths
        t_accel_adj = t_accel.copy()
        mask = short_path
        # Guard divide-by-zero
        safe = a_steps_vec[mask] > 0
        t_accel_adj[mask] = 0.0
        if np.any(mask):
            t_accel_adj[mask][safe] = np.sqrt(path[mask][safe] / a_steps_vec[mask][safe])  # type: ignore[index]

        # Per-joint total time, then horizon
        joint_time = np.where(short_path, 2.0 * t_accel_adj, path / v_max_joint + t_accel)
        total_time = float(np.max(joint_time))
        if total_time <= 0.0:
            return cast(np.ndarray, np.asarray(start_steps, dtype=np.int32).reshape(1, -1))
        if total_time < (2 * dt):
            total_time = 2 * dt

        n = max(2, int(np.ceil(total_time / max(dt, 1e-9))))
        tgrid = np.linspace(0.0, total_time, n, dtype=float)
        q, _qd = MotionCommand.plan_trapezoids(start_steps, target_steps, tgrid)
        return cast(np.ndarray, q.astype(np.int32, copy=False))


class SystemCommand(CommandBase):
    """
    Base class for system control commands that can execute regardless of enable state.
    
    System commands control the overall state of the robot controller (enable/disable, stop, etc.)
    and can execute even when the controller is disabled.
    """

    def tick(self, state: "ControllerState") -> ExecutionStatus:
        """
        Centralized lifecycle/error handling for system commands.
        """
        if self.is_finished or not self.is_valid:
            return ExecutionStatus.completed("Already finished") if self.is_finished else ExecutionStatus.failed("Invalid command")
        try:
            status = self.execute_step(state)
        except Exception as e:
            self.fail(str(e))
            self.log_error(str(e))
            return ExecutionStatus.failed("Execution error", error=e)
        return status
