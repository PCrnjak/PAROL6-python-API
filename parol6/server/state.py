from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from pinokin import arrays_equal_6
from parol6.config import CONTROL_RATE_HZ, steps_to_rad
from parol6.motion import CartesianStreamingExecutor, StreamingExecutor
from parol6.protocol.wire import CommandCode
from parol6.utils.error_catalog import RobotError
from waldoctl import ActionState


class GripperHWState:
    """Named wrapper over the raw gripper numpy arrays.

    Provides human-readable property access while keeping the underlying
    int32 arrays intact for serial frame packing (zero-copy).

    ``data_out`` layout: [target_position, target_speed, target_current,
                          command_bits, mode, device_id]
    ``data_in``  layout: [device_id, feedback_position, feedback_speed,
                          feedback_current, status_byte, object_detection]
    """

    __slots__ = ("_out", "_in")

    def __init__(self, data_out: np.ndarray, data_in: np.ndarray) -> None:
        self._out = data_out
        self._in = data_in

    # -- Output (commands to gripper) --

    @property
    def target_position(self) -> int:
        return int(self._out[0])

    @target_position.setter
    def target_position(self, v: int) -> None:
        self._out[0] = v

    @property
    def target_speed(self) -> int:
        return int(self._out[1])

    @target_speed.setter
    def target_speed(self, v: int) -> None:
        self._out[1] = v

    @property
    def target_current(self) -> int:
        return int(self._out[2])

    @target_current.setter
    def target_current(self, v: int) -> None:
        self._out[2] = v

    @property
    def command_bits(self) -> int:
        return int(self._out[3])

    @command_bits.setter
    def command_bits(self, v: int) -> None:
        self._out[3] = v

    # -- Command bit helpers (MSB-first 8-bit field) --
    #
    # Bit layout: [enable, move_active, estop, grip_enable, 0, 0, 0, 0]
    #   bit 7 (0x80): enable        — always 1
    #   bit 6 (0x40): move_active   — 1 while moving, 0 when idle
    #   bit 5 (0x20): estop         — inverted e-stop sense
    #   bit 4 (0x10): grip_enable   — always 1
    #   bits 3-0:     reserved (0)

    _CMD_ENABLE: int = 0x80
    _CMD_MOVE_ACTIVE: int = 0x40
    _CMD_ESTOP: int = 0x20
    _CMD_GRIP_ENABLE: int = 0x10

    def set_command_bits(self, *, move_active: bool, estop: bool) -> None:
        """Pack and write the command bits byte.

        ``enable`` and ``grip_enable`` are always set.
        """
        v = self._CMD_ENABLE | self._CMD_GRIP_ENABLE
        if move_active:
            v |= self._CMD_MOVE_ACTIVE
        if estop:
            v |= self._CMD_ESTOP
        self._out[3] = v

    @property
    def mode(self) -> int:
        return int(self._out[4])

    @mode.setter
    def mode(self, v: int) -> None:
        self._out[4] = v

    @property
    def device_id_out(self) -> int:
        return int(self._out[5])

    @device_id_out.setter
    def device_id_out(self, v: int) -> None:
        self._out[5] = v

    # -- Input (feedback from gripper) --

    @property
    def device_id(self) -> int:
        return int(self._in[0])

    @property
    def feedback_position(self) -> int:
        return int(self._in[1])

    @property
    def feedback_speed(self) -> int:
        return int(self._in[2])

    @property
    def feedback_current(self) -> int:
        return int(self._in[3])

    @property
    def status_byte(self) -> int:
        return int(self._in[4])

    @property
    def object_detection(self) -> int:
        return int(self._in[5])


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
    hardware_connected: bool = False
    last_reconnect_attempt: float = 0.0

    # Safety and control flags
    enabled: bool = True
    soft_error: bool = False
    disabled_reason: str = ""
    e_stop_active: bool = False

    # Motion profile for all moves (TOPPRA, RUCKIG, QUINTIC, TRAPEZOID, LINEAR)
    # Note: RUCKIG is point-to-point only; Cartesian moves fall back to TOPPRA
    motion_profile: str = "TOPPRA"

    # Streaming executors for online motion (jogging/streaming)
    streaming_executor: StreamingExecutor = field(
        default_factory=lambda: StreamingExecutor(num_dofs=6, dt=1.0 / CONTROL_RATE_HZ)
    )
    cartesian_streaming_executor: CartesianStreamingExecutor = field(
        default_factory=lambda: CartesianStreamingExecutor(dt=1.0 / CONTROL_RATE_HZ)
    )

    # Tool configuration (affects kinematics and visualization)
    _current_tool: str = "NONE"
    _current_tool_variant: str = ""
    _tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Robot telemetry and command buffers - using ndarray for efficiency
    Command_out: CommandCode = CommandCode.IDLE  # The command code to send to firmware

    # int32 joint buffers
    Position_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    Speed_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    Gripper_data_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )

    Position_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    Speed_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    Timing_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((1,), dtype=np.int32)
    )
    Gripper_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )

    # Tool teleport: when >= 0, snap gripper to this position (0-255) on next tick
    tool_teleport_pos: float = -1.0

    # uint8 flag/bitfield buffers
    Affected_joint_out: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    InOut_out: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    InOut_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Homed_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Temperature_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    Position_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )

    Timeout_out: int = 0
    XTR_data: int = 0

    # Action tracking for status broadcast and queries
    action_current: str = ""
    action_params: str = ""
    action_state: ActionState = ActionState.IDLE  # IDLE, EXECUTING, ERROR
    action_next: str = ""
    queue_nonstreamable: list[str] = field(default_factory=list)

    # Queue progress tracking (monotonically increasing command indices)
    next_command_index: int = 0
    executing_command_index: int = -1
    completed_command_index: int = -1
    last_checkpoint: str = ""

    # Planning behavior (stop on first IK failure vs solve all for diagnostic)
    stop_on_failure: bool = True

    # Error state (set by segment player on planning failure)
    error: RobotError | None = None

    # Pipeline depth (maintained by segment player)
    queued_segments: int = 0
    queued_duration: float = 0.0

    # Network setup and uptime
    ip: str = "127.0.0.1"
    port: int = 5001
    start_time: float = 0.0

    gripper_mode_tracker: GripperModeResetTracker = field(
        default_factory=GripperModeResetTracker
    )

    # Control loop runtime metrics (used by benchmarks/monitoring)
    loop_count: int = 0
    overrun_count: int = 0

    # Rolling statistics from loop timer
    mean_period_s: float = 0.0
    std_period_s: float = 0.0
    min_period_s: float = 0.0
    max_period_s: float = 0.0
    p95_period_s: float = 0.0
    p99_period_s: float = 0.0

    # Flag to signal loop stats reset (picked up by controller)
    loop_stats_reset_pending: bool = False

    # Forward kinematics cache (invalidated when Position_in or current_tool changes)
    _fkine_last_pos_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    _fkine_last_tool_name: str = ""
    _fkine_last_tool_variant: str = ""
    _fkine_last_tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    _fkine_mat: np.ndarray = field(
        default_factory=lambda: np.asfortranarray(np.eye(4, dtype=np.float64))
    )
    _fkine_flat_mm: np.ndarray = field(
        default_factory=lambda: np.zeros((16,), dtype=np.float64)
    )
    _fkine_q_rad: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.float64)
    )

    # Named wrapper over raw gripper arrays (initialized in __post_init__)
    gripper_hw: GripperHWState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize E-stop to released state and named gripper wrapper."""
        self.InOut_in[4] = 1  # E-STOP released (0=pressed, 1=released)
        self.gripper_hw = GripperHWState(self.Gripper_data_out, self.Gripper_data_in)

    def reset(self) -> None:
        """
        Reset robot state to initial values without losing connection state.

        Preserves: ser, ip, port, start_time
        Resets: positions, speeds, I/O, queues, tool, errors, etc.
        """
        # Safety and control flags
        self.enabled = True
        self.soft_error = False
        self.disabled_reason = ""
        self.e_stop_active = False
        self.motion_profile = "TOPPRA"

        # Tool back to none
        self._current_tool = "NONE"
        self._current_tool_variant = ""
        self._tcp_offset_m = (0.0, 0.0, 0.0)
        PAROL6_ROBOT.apply_tool("NONE")

        # Command and telemetry buffers - zero out
        self.Command_out = CommandCode.IDLE
        self.Position_out.fill(0)
        self.Speed_out.fill(0)
        self.Gripper_data_out.fill(0)
        self.Position_in.fill(0)
        self.Speed_in.fill(0)
        self.Timing_data_in.fill(0)
        self.Gripper_data_in.fill(0)
        self.Affected_joint_out.fill(0)
        self.InOut_out.fill(0)
        self.InOut_in.fill(0)
        self.InOut_in[4] = 1  # E-STOP released (0=pressed, 1=released)
        self.Homed_in.fill(0)
        self.Temperature_error_in.fill(0)
        self.Position_error_in.fill(0)
        self.Timeout_out = 0
        self.XTR_data = 0

        # Action tracking
        self.action_current = ""
        self.action_params = ""
        self.action_state = ActionState.IDLE
        self.action_next = ""
        self.queue_nonstreamable.clear()

        # Queue progress tracking
        self.next_command_index = 0
        self.executing_command_index = -1
        self.completed_command_index = -1
        self.last_checkpoint = ""

        # Error and pipeline depth
        self.error = None
        self.queued_segments = 0
        self.queued_duration = 0.0

        # Gripper mode tracker
        self.gripper_mode_tracker = GripperModeResetTracker()

        # Invalidate fkine cache (SE3 is pre-allocated, just reset tracking)
        self._fkine_last_pos_in.fill(0)
        self._fkine_last_tool_name = ""
        self._fkine_last_tool_variant = ""

        # Reset streaming executors (clears reference_pose and Ruckig state)
        self.streaming_executor.reset()
        self.cartesian_streaming_executor.reset()

        logger.debug("Controller state reset (preserving connection)")

    @property
    def current_tool(self) -> str:
        """Get the current tool name."""
        return self._current_tool

    @property
    def current_tool_variant(self) -> str:
        """Get the current tool variant key."""
        return self._current_tool_variant

    def set_tool(self, tool_name: str, variant_key: str = "") -> None:
        """Set the current tool and apply it to the robot model.

        Resets TCP offset to zero (changing tools invalidates any prior offset).
        """
        if tool_name != self._current_tool or variant_key != self._current_tool_variant:
            self._current_tool = tool_name
            self._current_tool_variant = variant_key
            self._tcp_offset_m = (0.0, 0.0, 0.0)
            PAROL6_ROBOT.apply_tool(tool_name, variant_key=variant_key)
            label = f"{tool_name}:{variant_key}" if variant_key else tool_name
            logger.info(f"Tool changed to {label}")

    @property
    def tcp_offset_m(self) -> tuple[float, float, float]:
        """Current TCP offset in meters (tool-local frame)."""
        return self._tcp_offset_m

    def set_tcp_offset(self, offset_m: tuple[float, float, float]) -> None:
        """Set TCP offset and reapply tool transform with the composed offset."""
        self._tcp_offset_m = offset_m
        PAROL6_ROBOT.apply_tool(
            self._current_tool,
            variant_key=self._current_tool_variant,
            tcp_offset_m=offset_m,
        )
        logger.debug(
            "TCP offset set to (%.1f, %.1f, %.1f) mm",
            offset_m[0] * 1000,
            offset_m[1] * 1000,
            offset_m[2] * 1000,
        )


logger = logging.getLogger(__name__)


class StateManager:
    """Manager for ControllerState."""

    _state: ControllerState | None = None

    def __init__(self):
        """Initialize the state manager."""
        self._state = ControllerState()

    def get_state(self) -> ControllerState:
        """
        Get the current controller state.

        Returns:
            The current ControllerState instance
        """
        if self._state is None:
            self._state = ControllerState()
        return self._state

    def reset_state(self) -> None:
        """
        Reset the controller state to a fresh instance.

        This is useful at controller startup to ensure buffers are initialized
        to known defaults.
        """
        self._state = ControllerState()
        logger.debug("Controller state reset")


# Global singleton instance accessor
_state_manager: StateManager | None = None


@atexit.register
def _cleanup_state_manager() -> None:
    global _state_manager
    if _state_manager is not None:
        _state_manager._state = None
    _state_manager = None


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


def invalidate_fkine_cache(state: ControllerState | None = None) -> None:
    """
    Invalidate the fkine cache, forcing recomputation on next access.
    Called when the robot model changes (e.g., tool change).

    Parameters
    ----------
    state : ControllerState, optional
        The controller state to invalidate. If not provided, uses the global state.
    """
    if state is None:
        state = get_state()
    state._fkine_last_tool_name = ""
    state._fkine_last_tool_variant = ""
    logger.debug("fkine cache invalidated")


def ensure_fkine_updated(state: ControllerState) -> None:
    """
    Ensure the fkine cache is up to date with current Position_in and tool.
    If Position_in or current_tool has changed, recalculate fkine and update cache.

    Parameters
    ----------
    state : ControllerState
        The controller state to update
    """
    pos_changed = not arrays_equal_6(state.Position_in, state._fkine_last_pos_in)
    tool_changed = (
        state.current_tool != state._fkine_last_tool_name
        or state.current_tool_variant != state._fkine_last_tool_variant
        or state.tcp_offset_m != state._fkine_last_tcp_offset
    )

    if pos_changed or tool_changed:
        steps_to_rad(state.Position_in, state._fkine_q_rad)
        PAROL6_ROBOT.robot.fkine_into(state._fkine_q_rad, state._fkine_mat)

        # Cache as flattened 16-vector with mm translation (zero-allocation)
        state._fkine_flat_mm.reshape(4, 4)[:] = state._fkine_mat
        state._fkine_flat_mm[3] *= 1000.0  # X translation to mm
        state._fkine_flat_mm[7] *= 1000.0  # Y translation to mm
        state._fkine_flat_mm[11] *= 1000.0  # Z translation to mm

        # Update cache tracking
        state._fkine_last_pos_in[:] = state.Position_in
        state._fkine_last_tool_name = state.current_tool
        state._fkine_last_tool_variant = state.current_tool_variant
        state._fkine_last_tcp_offset = state.tcp_offset_m


def get_fkine_se3(state: ControllerState | None = None) -> np.ndarray:
    """
    Get the current end-effector pose as a 4x4 SE3 transformation matrix.
    Automatically updates cache if needed.

    Returns
    -------
    np.ndarray
        4x4 SE3 transformation matrix (translation in meters)
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
