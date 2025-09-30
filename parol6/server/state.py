from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any, Union, Sequence
from collections import deque
import threading
import time
import logging
import numpy as np
from parol6.protocol.wire import CommandCode


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
    last_command_time: float = 0.0
    last_command_period_s: float = 0.0
    ema_command_period_s: float = 0.0
    command_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=10))


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
