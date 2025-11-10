from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any, Union, Sequence
from collections import deque
import threading
import time
import logging
import numpy as np
from parol6.protocol.wire import CommandCode
import parol6.PAROL6_ROBOT as PAROL6_ROBOT


@dataclass
class GripperModeResetTracker:
    """Tracks gripper mode for auto-reset functionality."""
    calibration_sent: bool = False  # Flag for calibration mode
    error_clear_sent: bool = False  # Flag for error clear mode


@dataclass
class ControllerState:
    """
    Centralized mutable state for the headless controller.

    Buffers use preallocated NumPy ndarrays for zero-copy, in-place operations.
    """
    # Serial/transport
    ser: Any = None
    last_reconnect_attempt: float = 0.0

    # Safety and control flags
    enabled: bool = True
    soft_error: bool = False
    disabled_reason: str = ""
    e_stop_active: bool = False
    stream_mode: bool = False
    
    # Tool configuration (affects kinematics and visualization)
    _current_tool: str = "NONE"

    # I/O buffers and protocol tracking (serial frame parsing state)
    input_byte: int = 0
    start_cond1: int = 0
    start_cond2: int = 0
    start_cond3: int = 0
    good_start: int = 0
    data_len: int = 0
    data_buffer: List[bytes] = field(default_factory=lambda: [b""] * 255)
    data_counter: int = 0

    # Robot telemetry and command buffers - using ndarray for efficiency
    Command_out: CommandCode = CommandCode.IDLE  # The command code to send to firmware

    # int32 joint buffers
    Position_out: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    Speed_out: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    Gripper_data_out: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))

    Position_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    Speed_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    Timing_data_in: np.ndarray = field(default_factory=lambda: np.zeros((1,), dtype=np.int32))
    Gripper_data_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))

    # uint8 flag/bitfield buffers
    Affected_joint_out: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    InOut_out: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    InOut_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Homed_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Temperature_error_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Position_error_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))

    Timeout_out: int = 0
    XTR_data: int = 0

    # Command queueing and tracking
    command_queue: Deque[Any] = field(default_factory=deque)
    incoming_command_buffer: Deque[Tuple[str, Tuple[str, int]]] = field(default_factory=deque)
    command_id_map: Dict[Any, Tuple[str, Tuple[str, int]]] = field(default_factory=dict)
    active_command: Any = None
    active_command_id: Optional[str] = None
    last_command_time: float = 0.0
    
    # Action tracking for status broadcast and queries
    action_current: str = ""
    action_state: str = "IDLE"  # IDLE, EXECUTING, COMPLETED, FAILED
    action_next: str = ""
    queue_nonstreamable: List[str] = field(default_factory=list)

    # Network setup and uptime
    ip: str = "127.0.0.1"
    port: int = 5001
    start_time: float = 0.0

    gripper_mode_tracker: GripperModeResetTracker = field(default_factory=GripperModeResetTracker)

    # Control loop runtime metrics (used by benchmarks/monitoring)
    loop_count: int = 0
    overrun_count: int = 0
    last_period_s: float = 0.0
    ema_period_s: float = 0.0
    
    # Command frequency metrics
    command_count: int = 0
    last_command_period_s: float = 0.0
    ema_command_period_s: float = 0.0
    command_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    
    # Forward kinematics cache (invalidated when Position_in or current_tool changes)
    _fkine_last_pos_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    _fkine_last_tool: str = ""
    _fkine_se3: Any = None  # SE3 instance from spatialmath
    _fkine_mat: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    _fkine_flat_mm: np.ndarray = field(default_factory=lambda: np.zeros((16,), dtype=np.float64))
    
    @property
    def current_tool(self) -> str:
        """Get the current tool name."""
        return self._current_tool
    
    @current_tool.setter
    def current_tool(self, tool_name: str) -> None:
        """Set the current tool and apply it to the robot model."""
        if tool_name != self._current_tool:
            self._current_tool = tool_name
            # Apply tool to robot model (rebuilds with tool as final link)
            PAROL6_ROBOT.apply_tool(tool_name)
            # Invalidate cache
            self._fkine_se3 = None
            logger.info(f"Tool changed to {tool_name}, fkine cache invalidated")


logger = logging.getLogger(__name__)


class StateManager:
    """
    Singleton manager for ControllerState with thread-safe operations.

    This class ensures that all state access is synchronized and provides
    convenience methods for common state operations.
    """

    _instance: Optional[StateManager] = None
    _lock: threading.Lock = threading.Lock()
    _state: Optional[ControllerState] = None

    def __new__(cls) -> StateManager:
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the state manager (only runs once due to singleton)."""
        if not hasattr(self, '_initialized'):
            self._state = ControllerState()
            self._state_lock = threading.RLock()  # Use RLock for re-entrant locking
            self._initialized = True
            logger.info("StateManager initialized with NumPy buffers")

    def get_state(self) -> ControllerState:
        """
        Get the current controller state.

        Note: This returns the actual state object. Modifications to it
        should be done through StateManager methods to ensure thread safety.

        Returns:
            The current ControllerState instance
        """
        with self._state_lock:
            if self._state is None:
                self._state = ControllerState()
            return self._state

    def reset_state(self) -> None:
        """
        Reset the controller state to a fresh instance.

        This is useful at controller startup to ensure buffers are initialized
        to known defaults. Callers must ensure they hold appropriate locks in
        higher layers if concurrent access is possible.
        """
        with self._state_lock:
            self._state = ControllerState()
            logger.info("Controller state reset")

# Global singleton instance accessor
_state_manager: Optional[StateManager] = None


def get_instance() -> StateManager:
    """
    Get the global StateManager instance.
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def get_state() -> ControllerState:
    """
    Convenience function to get the current controller state.
    """
    return get_instance().get_state()


# -----------------------------
# Forward kinematics cache management
# -----------------------------

def invalidate_fkine_cache() -> None:
    """
    Invalidate the fkine cache, forcing recomputation on next access.
    Call this when the robot model changes (e.g., tool change).
    """
    state = get_state()
    state._fkine_se3 = None
    state._fkine_last_tool = ""
    logger.debug("fkine cache invalidated")


def ensure_fkine_updated(state: ControllerState) -> None:
    """
    Ensure the fkine cache is up to date with current Position_in and tool.
    If Position_in or current_tool has changed, recalculate fkine and update cache.
    
    This function is thread-safe when called with state from get_state().
    
    Parameters
    ----------
    state : ControllerState
        The controller state to update
    """
    # Check if cache is valid
    pos_changed = not np.array_equal(state.Position_in, state._fkine_last_pos_in)
    tool_changed = state.current_tool != state._fkine_last_tool
    
    if pos_changed or tool_changed or state._fkine_se3 is None:
        # Recompute fkine
        q = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        T = PAROL6_ROBOT.robot.fkine(q)  # type: ignore[attr-defined]
        
        # Cache SE3 object
        state._fkine_se3 = T
        
        # Cache as 4x4 matrix
        mat = T.A.copy()
        np.copyto(state._fkine_mat, mat)
        
        # Cache as flattened 16-vector with mm translation
        flat = mat.reshape(-1).copy()
        flat[3] *= 1000.0   # X translation to mm
        flat[7] *= 1000.0   # Y translation to mm
        flat[11] *= 1000.0  # Z translation to mm
        np.copyto(state._fkine_flat_mm, flat)
        
        # Update cache tracking
        np.copyto(state._fkine_last_pos_in, state.Position_in)
        state._fkine_last_tool = state.current_tool


def get_fkine_se3(state: ControllerState | None = None):
    """
    Get the current end-effector pose as an SE3 object.
    Automatically updates cache if needed.
    
    Returns
    -------
    SE3
        Current end-effector pose
    """
    if state is None:
        state = get_state()
    ensure_fkine_updated(state)
    return state._fkine_se3


def get_fkine_matrix(state: ControllerState | None = None) -> np.ndarray:
    """
    Get the current end-effector pose as a 4x4 homogeneous transformation matrix.
    Automatically updates cache if needed.
    Translation is in meters.
    
    Returns
    -------
    np.ndarray
        4x4 transformation matrix (translation in meters)
    """
    if state is None:
        state = get_state()
    ensure_fkine_updated(state)
    return state._fkine_mat


def get_fkine_flat_mm(state: ControllerState | None = None) -> np.ndarray:
    """
    Get the current end-effector pose as a flattened 16-element array.
    Automatically updates cache if needed.
    Translation components (indices 3, 7, 11) are in millimeters for compatibility.
    
    Returns
    -------
    np.ndarray
        Flattened 16-element pose array (translation in mm)
    """
    if state is None:
        state = get_state()
    ensure_fkine_updated(state)
    return state._fkine_flat_mm
