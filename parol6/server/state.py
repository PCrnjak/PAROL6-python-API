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
        Reset the controller state to defaults.

        This creates a new ControllerState instance with default values.
        """
        with self._state_lock:
            self._state = ControllerState()
            logger.info("Controller state reset to defaults")

    def update_telemetry(self,
                        Position_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        Speed_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        Homed_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        InOut_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        Temperature_error_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        Position_error_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        Timing_data_in: Optional[Union[int, Sequence[int], np.ndarray]] = None,
                        Gripper_data_in: Optional[Union[Sequence[int], np.ndarray]] = None,
                        XTR_data: Optional[int] = None) -> None:
        """
        Update telemetry data from serial frame using zero-copy ndarray operations when possible.
        """
        with self._state_lock:
            assert self._state
            if Position_in is not None:
                np.copyto(self._state.Position_in, np.asarray(Position_in, dtype=self._state.Position_in.dtype))
            if Speed_in is not None:
                np.copyto(self._state.Speed_in, np.asarray(Speed_in, dtype=self._state.Speed_in.dtype))
            if Homed_in is not None:
                np.copyto(self._state.Homed_in, np.asarray(Homed_in, dtype=self._state.Homed_in.dtype))
            if InOut_in is not None:
                np.copyto(self._state.InOut_in, np.asarray(InOut_in, dtype=self._state.InOut_in.dtype))
            if Temperature_error_in is not None:
                np.copyto(self._state.Temperature_error_in, np.asarray(Temperature_error_in, dtype=self._state.Temperature_error_in.dtype))
            if Position_error_in is not None:
                np.copyto(self._state.Position_error_in, np.asarray(Position_error_in, dtype=self._state.Position_error_in.dtype))
            if Timing_data_in is not None:
                if isinstance(Timing_data_in, int):
                    self._state.Timing_data_in[0] = Timing_data_in
                else:
                    np.copyto(self._state.Timing_data_in, np.asarray(Timing_data_in, dtype=self._state.Timing_data_in.dtype))
            if Gripper_data_in is not None:
                np.copyto(self._state.Gripper_data_in, np.asarray(Gripper_data_in, dtype=self._state.Gripper_data_in.dtype))
            if XTR_data is not None:
                self._state.XTR_data = XTR_data

    def update_telemetry_direct(self, frame_data: dict) -> None:
        """
        Update telemetry directly from unpacked frame data (dict).
        """
        with self._state_lock:
            assert self._state
            if "Position_in" in frame_data:
                np.copyto(self._state.Position_in, np.asarray(frame_data["Position_in"], dtype=self._state.Position_in.dtype))
            if "Speed_in" in frame_data:
                np.copyto(self._state.Speed_in, np.asarray(frame_data["Speed_in"], dtype=self._state.Speed_in.dtype))
            if "Homed_in" in frame_data:
                np.copyto(self._state.Homed_in, np.asarray(frame_data["Homed_in"], dtype=self._state.Homed_in.dtype))
            if "InOut_in" in frame_data:
                np.copyto(self._state.InOut_in, np.asarray(frame_data["InOut_in"], dtype=self._state.InOut_in.dtype))
            if "Temperature_error_in" in frame_data:
                np.copyto(self._state.Temperature_error_in, np.asarray(frame_data["Temperature_error_in"], dtype=self._state.Temperature_error_in.dtype))
            if "Position_error_in" in frame_data:
                np.copyto(self._state.Position_error_in, np.asarray(frame_data["Position_error_in"], dtype=self._state.Position_error_in.dtype))
            if "Timing_data_in" in frame_data:
                timing = frame_data["Timing_data_in"]
                if isinstance(timing, (list, tuple, np.ndarray)) and len(timing) > 0:
                    self._state.Timing_data_in[0] = int(timing[0])
            if "Gripper_data_in" in frame_data:
                np.copyto(self._state.Gripper_data_in, np.asarray(frame_data["Gripper_data_in"], dtype=self._state.Gripper_data_in.dtype))

    def update_command_outputs(self,
                              Position_out: Optional[Union[Sequence[int], np.ndarray]] = None,
                              Speed_out: Optional[Union[Sequence[int], np.ndarray]] = None,
                              Affected_joint_out: Optional[Union[Sequence[int], np.ndarray]] = None,
                              InOut_out: Optional[Union[Sequence[int], np.ndarray]] = None,
                              Timeout_out: Optional[int] = None,
                              Gripper_data_out: Optional[Union[Sequence[int], np.ndarray]] = None) -> None:
        """
        Update command output buffers using ndarray operations.
        """
        with self._state_lock:
            assert self._state
            if Position_out is not None:
                np.copyto(self._state.Position_out, np.asarray(Position_out, dtype=self._state.Position_out.dtype))
            if Speed_out is not None:
                np.copyto(self._state.Speed_out, np.asarray(Speed_out, dtype=self._state.Speed_out.dtype))
            if Affected_joint_out is not None:
                np.copyto(self._state.Affected_joint_out, np.asarray(Affected_joint_out, dtype=self._state.Affected_joint_out.dtype))
            if InOut_out is not None:
                np.copyto(self._state.InOut_out, np.asarray(InOut_out, dtype=self._state.InOut_out.dtype))
            if Timeout_out is not None:
                self._state.Timeout_out = int(Timeout_out)
            if Gripper_data_out is not None:
                np.copyto(self._state.Gripper_data_out, np.asarray(Gripper_data_out, dtype=self._state.Gripper_data_out.dtype))

    def set_serial_connection(self, ser: Any, port: str) -> None:
        """
        Set the serial connection object.
        """
        with self._state_lock:
            assert self._state
            self._state.ser = ser
            logger.info(f"Serial connection set: {port}")

    def clear_serial_connection(self) -> None:
        """Clear the serial connection."""
        with self._state_lock:
            assert self._state
            self._state.ser = None
            logger.info("Serial connection cleared")

    def is_connected(self) -> bool:
        """
        Check if serial connection is active.
        """
        with self._state_lock:
            assert self._state
            return self._state.ser is not None and self._state.ser.is_open if hasattr(self._state.ser, 'is_open') else False

    def set_enabled(self, enabled: bool, reason: str = "") -> None:
        """
        Set the enabled state of the controller.
        """
        with self._state_lock:
            assert self._state
            self._state.enabled = enabled
            if not enabled:
                self._state.disabled_reason = reason
                logger.info(f"Controller disabled: {reason}")
            else:
                self._state.disabled_reason = ""
                logger.info("Controller enabled")

    def is_enabled(self) -> bool:
        """
        Check if the controller is enabled.
        """
        with self._state_lock:
            assert self._state
            return self._state.enabled

    def set_estop(self, active: bool) -> None:
        """
        Set the E-stop state.
        """
        with self._state_lock:
            assert self._state
            self._state.e_stop_active = active
            if active:
                logger.warning("E-stop activated")
            else:
                logger.info("E-stop cleared")

    def is_estop_active(self) -> bool:
        """
        Check if E-stop is active.
        """
        with self._state_lock:
            assert self._state
            return self._state.e_stop_active

    def reset_estop(self) -> None:
        """
        Reset E-stop condition and clear any error states.
        """
        with self._state_lock:
            assert self._state
            if self._state.e_stop_active:
                # Clear E-stop flag
                self._state.e_stop_active = False

                # Clear any soft errors
                self._state.soft_error = False

                # Re-enable the controller
                self._state.enabled = True
                self._state.disabled_reason = ""

                # Clear command outputs to safe state
                self._state.Speed_out.fill(0)
                # Mirror current position
                np.copyto(self._state.Position_out, self._state.Position_in)

                logger.info("E-stop reset completed - controller re-enabled")

    def is_ready_for_motion(self) -> bool:
        """
        Check if the system is ready for motion commands.
        """
        with self._state_lock:
            assert self._state
            return (
                self._state.enabled
                and not self._state.e_stop_active
                and not self._state.soft_error
                and self._state.ser is not None
            )

    def get_active_command(self) -> Optional[Any]:
        """
        Get the currently active command.
        """
        with self._state_lock:
            assert self._state
            return self._state.active_command

    def set_active_command(self, command: Any, command_id: Optional[str] = None) -> None:
        """
        Set the active command.
        """
        with self._state_lock:
            assert self._state
            self._state.active_command = command
            self._state.active_command_id = command_id
            self._state.last_command_time = time.time()

    def clear_active_command(self) -> None:
        """Clear the active command."""
        with self._state_lock:
            assert self._state
            self._state.active_command = None
            self._state.active_command_id = None

    def get_command_queue_size(self) -> int:
        """
        Get the size of the command queue.
        """
        with self._state_lock:
            assert self._state
            return len(self._state.command_queue)

    def is_command_queue_empty(self) -> bool:
        """
        Check if the command queue is empty.
        """
        with self._state_lock:
            assert self._state
            return len(self._state.command_queue) == 0

    def set_network_config(self, ip: str, port: int) -> None:
        """
        Set network configuration.
        """
        with self._state_lock:
            assert self._state
            self._state.ip = ip
            self._state.port = port
            logger.info(f"Network config set: {ip}:{port}")

    def record_start_time(self) -> None:
        """Record the system start time."""
        with self._state_lock:
            assert self._state
            self._state.start_time = time.time()

    def get_uptime(self) -> float:
        """
        Get system uptime in seconds.
        """
        with self._state_lock:
            assert self._state
            if self._state.start_time > 0:
                return time.time() - self._state.start_time
            return 0.0


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


def is_ready_for_motion() -> bool:
    """
    Convenience function to check if system is ready for motion.
    """
    return get_instance().is_ready_for_motion()


def reset_estop() -> None:
    """
    Convenience function to reset E-stop condition.
    """
    get_instance().reset_estop()
