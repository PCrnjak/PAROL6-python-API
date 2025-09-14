from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import time
import logging
from parol6.protocol.wire import CommandCode


@dataclass
class CommandCooldownConfig:
    """Configuration for command processing cooldown."""
    cooldown_ms: int = 10  # Minimum milliseconds between command processing
    last_processed_time: float = 0.0  # Timestamp of last processed command
    enabled: bool = True  # Whether cooldown is active


@dataclass 
class GripperModeResetTracker:
    """Tracks gripper mode for auto-reset functionality."""
    calibration_sent: bool = False  # Flag for calibration mode
    error_clear_sent: bool = False  # Flag for error clear mode

@dataclass
class ControllerState:
    """
    Centralized mutable state for the headless controller.

    This dataclass is introduced as part of Phase 2 of the implementation plan
    to eliminate global variables and make the controller more testable. It is
    not yet wired into headless_commander.py; integration will be done incrementally.
    """
    # Serial/transport
    ser: Any = None
    com_port_str: Optional[str] = None
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

    # Robot telemetry and command buffers
    Command_out: CommandCode = CommandCode.IDLE  # The command code to send to firmware
    Position_out: List[int] = field(default_factory=lambda: [0] * 6)
    Speed_out: List[int] = field(default_factory=lambda: [0] * 6)
    Affected_joint_out: List[int] = field(default_factory=lambda: [0] * 8)
    InOut_out: List[int] = field(default_factory=lambda: [0] * 8)
    Timeout_out: int = 0
    Gripper_data_out: List[int] = field(default_factory=lambda: [0] * 6)

    Position_in: List[int] = field(default_factory=lambda: [0] * 6)
    Speed_in: List[int] = field(default_factory=lambda: [0] * 6)
    Homed_in: List[int] = field(default_factory=lambda: [0] * 8)
    InOut_in: List[int] = field(default_factory=lambda: [0] * 8)
    Temperature_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    Position_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    Timing_data_in: List[int] = field(default_factory=lambda: [0])
    XTR_data: int = 0
    Gripper_data_in: List[int] = field(default_factory=lambda: [0] * 6)

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
    
    # New fields for refactoring
    cooldown_config: CommandCooldownConfig = field(default_factory=CommandCooldownConfig)
    gripper_mode_tracker: GripperModeResetTracker = field(default_factory=GripperModeResetTracker)
    com_port_cache: Optional[str] = None


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
            logger.info("StateManager initialized")
    
    def get_state(self) -> ControllerState:
        """
        Get the current controller state.
        
        Note: This returns the actual state object. Modifications to it
        should be done through StateManager methods to ensure thread safety.
        
        Returns:
            The current ControllerState instance
        """
        with self._state_lock:
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
                        Position_in: Optional[List[int]] = None,
                        Speed_in: Optional[List[int]] = None,
                        Homed_in: Optional[List[int]] = None,
                        InOut_in: Optional[List[int]] = None,
                        Temperature_error_in: Optional[List[int]] = None,
                        Position_error_in: Optional[List[int]] = None,
                        Timing_data_in: Optional[int] = None,
                        Gripper_data_in: Optional[List[int]] = None,
                        XTR_data: Optional[int] = None) -> None:
        """
        Update telemetry data from serial frame.
        
        Args:
            Position_in: Joint position data
            Speed_in: Joint speed data
            Homed_in: Homing status data
            InOut_in: I/O status data
            Temperature_error_in: Temperature error flags
            Position_error_in: Position error flags
            Timing_data_in: Timing information
            Gripper_data_in: Gripper data
            XTR_data: Extra data field
        """
        with self._state_lock:
            if Position_in is not None:
                self._state.Position_in = Position_in.copy()
            if Speed_in is not None:
                self._state.Speed_in = Speed_in.copy()
            if Homed_in is not None:
                self._state.Homed_in = Homed_in.copy()
            if InOut_in is not None:
                self._state.InOut_in = InOut_in.copy()
            if Temperature_error_in is not None:
                self._state.Temperature_error_in = Temperature_error_in.copy()
            if Position_error_in is not None:
                self._state.Position_error_in = Position_error_in.copy()
            if Timing_data_in is not None:
                self._state.Timing_data_in = Timing_data_in
            if Gripper_data_in is not None:
                self._state.Gripper_data_in = Gripper_data_in.copy()
            if XTR_data is not None:
                self._state.XTR_data = XTR_data
    
    def update_command_outputs(self,
                              Position_out: Optional[List[int]] = None,
                              Speed_out: Optional[List[int]] = None,
                              Affected_joint_out: Optional[List[int]] = None,
                              InOut_out: Optional[List[int]] = None,
                              Timeout_out: Optional[int] = None,
                              Gripper_data_out: Optional[List[int]] = None) -> None:
        """
        Update command output buffers.
        
        Args:
            Position_out: Target position commands
            Speed_out: Speed commands
            Affected_joint_out: Affected joint flags
            InOut_out: I/O commands
            Timeout_out: Timeout value
            Gripper_data_out: Gripper commands
        """
        with self._state_lock:
            if Position_out is not None:
                self._state.Position_out = Position_out.copy()
            if Speed_out is not None:
                self._state.Speed_out = Speed_out.copy()
            if Affected_joint_out is not None:
                self._state.Affected_joint_out = Affected_joint_out.copy()
            if InOut_out is not None:
                self._state.InOut_out = InOut_out.copy()
            if Timeout_out is not None:
                self._state.Timeout_out = Timeout_out
            if Gripper_data_out is not None:
                self._state.Gripper_data_out = Gripper_data_out.copy()
    
    def set_serial_connection(self, ser: Any, port: str) -> None:
        """
        Set the serial connection object.
        
        Args:
            ser: Serial connection object
            port: COM port string
        """
        with self._state_lock:
            self._state.ser = ser
            self._state.com_port_str = port
            logger.info(f"Serial connection set: {port}")
    
    def clear_serial_connection(self) -> None:
        """Clear the serial connection."""
        with self._state_lock:
            self._state.ser = None
            self._state.com_port_str = None
            logger.info("Serial connection cleared")
    
    def is_connected(self) -> bool:
        """
        Check if serial connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        with self._state_lock:
            return self._state.ser is not None and self._state.ser.is_open if hasattr(self._state.ser, 'is_open') else False
    
    def set_enabled(self, enabled: bool, reason: str = "") -> None:
        """
        Set the enabled state of the controller.
        
        Args:
            enabled: True to enable, False to disable
            reason: Reason for disabling (if enabled=False)
        """
        with self._state_lock:
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
        
        Returns:
            True if enabled, False otherwise
        """
        with self._state_lock:
            return self._state.enabled
    
    def set_estop(self, active: bool) -> None:
        """
        Set the E-stop state.
        
        Args:
            active: True if E-stop is active, False otherwise
        """
        with self._state_lock:
            self._state.e_stop_active = active
            if active:
                logger.warning("E-stop activated")
            else:
                logger.info("E-stop cleared")
    
    def is_estop_active(self) -> bool:
        """
        Check if E-stop is active.
        
        Returns:
            True if E-stop is active, False otherwise
        """
        with self._state_lock:
            return self._state.e_stop_active
    
    def reset_estop(self) -> None:
        """
        Reset E-stop condition and clear any error states.
        
        This implements automatic E-stop recovery without requiring
        keyboard interaction.
        """
        with self._state_lock:
            if self._state.e_stop_active:
                # Clear E-stop flag
                self._state.e_stop_active = False
                
                # Clear any soft errors
                self._state.soft_error = False
                
                # Re-enable the controller
                self._state.enabled = True
                self._state.disabled_reason = ""
                
                # Clear command outputs to safe state
                self._state.Speed_out = [0] * 6
                self._state.Position_out = self._state.Position_in.copy()
                
                logger.info("E-stop reset completed - controller re-enabled")
    
    def is_ready_for_motion(self) -> bool:
        """
        Check if the system is ready for motion commands.
        
        Returns:
            True if ready for motion, False otherwise
        """
        with self._state_lock:
            return (
                self._state.enabled 
                and not self._state.e_stop_active 
                and not self._state.soft_error
                and self._state.ser is not None
            )
    
    def get_active_command(self) -> Optional[Any]:
        """
        Get the currently active command.
        
        Returns:
            The active command object or None
        """
        with self._state_lock:
            return self._state.active_command
    
    def set_active_command(self, command: Any, command_id: Optional[str] = None) -> None:
        """
        Set the active command.
        
        Args:
            command: The command object to set as active
            command_id: Optional command ID for tracking
        """
        with self._state_lock:
            self._state.active_command = command
            self._state.active_command_id = command_id
            self._state.last_command_time = time.time()
    
    def clear_active_command(self) -> None:
        """Clear the active command."""
        with self._state_lock:
            self._state.active_command = None
            self._state.active_command_id = None
    
    def get_command_queue_size(self) -> int:
        """
        Get the size of the command queue.
        
        Returns:
            Number of commands in the queue
        """
        with self._state_lock:
            return len(self._state.command_queue)
    
    def is_command_queue_empty(self) -> bool:
        """
        Check if the command queue is empty.
        
        Returns:
            True if empty, False otherwise
        """
        with self._state_lock:
            return len(self._state.command_queue) == 0
    
    def set_network_config(self, ip: str, port: int) -> None:
        """
        Set network configuration.
        
        Args:
            ip: IP address to bind to
            port: Port number to listen on
        """
        with self._state_lock:
            self._state.ip = ip
            self._state.port = port
            logger.info(f"Network config set: {ip}:{port}")
    
    def record_start_time(self) -> None:
        """Record the system start time."""
        with self._state_lock:
            self._state.start_time = time.time()
    
    def get_uptime(self) -> float:
        """
        Get system uptime in seconds.
        
        Returns:
            Uptime in seconds since start
        """
        with self._state_lock:
            if self._state.start_time > 0:
                return time.time() - self._state.start_time
            return 0.0


# Global singleton instance accessor
_state_manager: Optional[StateManager] = None


def get_instance() -> StateManager:
    """
    Get the global StateManager instance.
    
    Returns:
        The StateManager singleton instance
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def get_state() -> ControllerState:
    """
    Convenience function to get the current controller state.
    
    Returns:
        The current ControllerState instance
    """
    return get_instance().get_state()


def is_ready_for_motion() -> bool:
    """
    Convenience function to check if system is ready for motion.
    
    Returns:
        True if ready for motion, False otherwise
    """
    return get_instance().is_ready_for_motion()


def reset_estop() -> None:
    """
    Convenience function to reset E-stop condition.
    """
    get_instance().reset_estop()
