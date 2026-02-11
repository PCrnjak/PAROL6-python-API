"""
Mock serial transport for simulation and testing.

This module provides a complete serial port simulation that generates
realistic robot responses without requiring hardware. The simulation
operates at the wire protocol level, making it transparent to the
controller code.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from parol6 import config as cfg
from parol6.config import LIMITS
from numba import njit  # type: ignore[import-untyped]

from parol6.protocol.wire import (
    CommandCode,
    _pack_bitfield,
    _pack_positions,
)
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@njit(cache=True)
def _simulate_motion_jit(
    position_f: np.ndarray,
    position_in: np.ndarray,
    speed_in: np.ndarray,
    speed_out: np.ndarray,
    position_out: np.ndarray,
    homed_in: np.ndarray,
    io_in: np.ndarray,
    prev_pos_f: np.ndarray,
    vmax_f: np.ndarray,
    jmin_f: np.ndarray,
    jmax_f: np.ndarray,
    home_angles_deg: np.ndarray,
    command_out: int,
    dt: float,
    homing_countdown: int,
) -> tuple[int, int]:
    """JIT-compiled motion simulation. Returns (new_homing_countdown, new_command_out).

    Note: CommandCode enums are used directly inside the function (resolved at compile time).
    Passing enums as arguments would add ~90µs overhead per call.
    """
    # Handle homing countdown
    if homing_countdown > 0:
        homing_countdown -= 1
        if homing_countdown == 0:
            # Homing complete
            homed_in.fill(1)
            for i in range(6):
                steps = cfg.deg_to_steps_scalar(home_angles_deg[i], i)
                position_in[i] = steps
                position_f[i] = float(steps)
                speed_in[i] = 0
            command_out = CommandCode.IDLE

    # Ensure E-stop stays released
    io_in[4] = 1

    # Simulate motion based on command type
    if command_out == CommandCode.HOME:
        if homing_countdown == 0:
            for i in range(6):
                homed_in[i] = 0
            homing_countdown = max(1, int(0.2 / cfg.INTERVAL_S + 0.5))
        for i in range(6):
            speed_in[i] = 0

    elif command_out == CommandCode.JOG or command_out == 123:
        prev_pos_f[:] = position_f

        # Scalar loop avoids allocations from np.clip().astype() and array arithmetic
        for i in range(6):
            v = float(speed_out[i])
            vmax = vmax_f[i]
            if v > vmax:
                v = vmax
            elif v < -vmax:
                v = -vmax

            new_pos = position_f[i] + v * dt
            if new_pos < jmin_f[i]:
                new_pos = jmin_f[i]
            elif new_pos > jmax_f[i]:
                new_pos = jmax_f[i]
            position_f[i] = new_pos

        if dt > 0:
            inv_dt = 1.0 / dt
            for i in range(6):
                v = round((position_f[i] - prev_pos_f[i]) * inv_dt)
                vmax = vmax_f[i]
                if v > vmax:
                    v = vmax
                elif v < -vmax:
                    v = -vmax
                speed_in[i] = int(v)
        else:
            speed_in.fill(0)

    elif command_out == CommandCode.MOVE or command_out == 156:
        prev_pos_f[:] = position_f

        for i in range(6):
            target = float(position_out[i])
            current_f = position_f[i]
            err_f = target - current_f

            max_step_f = vmax_f[i] * dt
            if max_step_f < 1.0:
                max_step_f = 1.0

            move = err_f
            if move > max_step_f:
                move = max_step_f
            elif move < -max_step_f:
                move = -max_step_f

            pos_f = current_f + move
            if pos_f < jmin_f[i]:
                pos_f = jmin_f[i]
            elif pos_f > jmax_f[i]:
                pos_f = jmax_f[i]
            position_f[i] = pos_f

        if dt > 0:
            inv_dt = 1.0 / dt
            for i in range(6):
                v = round((position_f[i] - prev_pos_f[i]) * inv_dt)
                vmax = vmax_f[i]
                if v > vmax:
                    v = vmax
                elif v < -vmax:
                    v = -vmax
                speed_in[i] = int(v)
        else:
            speed_in.fill(0)

    else:
        for i in range(6):
            speed_in[i] = 0

    # Sync integer position from float accumulator
    for i in range(6):
        position_in[i] = int(round(position_f[i]))

    return homing_countdown, command_out


@njit(cache=True)
def _write_frame_jit(
    state_position_out: np.ndarray,
    state_speed_out: np.ndarray,
    state_gripper_data_in: np.ndarray,
    position_out: np.ndarray,
    speed_out: np.ndarray,
    gripper_data_out: np.ndarray,
) -> None:
    """JIT-compiled frame write processing."""
    state_position_out[:] = position_out
    state_speed_out[:] = speed_out

    # Simulate gripper state updates
    if gripper_data_out[4] == 1:  # Calibration mode
        state_gripper_data_in[0] = gripper_data_out[5]
        state_gripper_data_in[4] = 0x40
    elif gripper_data_out[4] == 2:  # Error clear mode
        state_gripper_data_in[4] &= ~0x20

    if gripper_data_out[3] != 0:
        state_gripper_data_in[1] = gripper_data_out[0]
        state_gripper_data_in[2] = gripper_data_out[1]
        state_gripper_data_in[3] = gripper_data_out[2]


@njit(cache=True)
def _encode_payload_jit(
    out: memoryview,
    position_in: np.ndarray,
    speed_in: np.ndarray,
    homed_in: np.ndarray,
    io_in: np.ndarray,
    temp_err_in: np.ndarray,
    pos_err_in: np.ndarray,
    timing_in: np.ndarray,
    gripper_in: np.ndarray,
) -> None:
    """JIT-compiled payload encoding."""
    _pack_positions(out, position_in, 0)
    _pack_positions(out, speed_in, 18)

    out[36] = _pack_bitfield(homed_in)
    out[37] = _pack_bitfield(io_in)
    out[38] = _pack_bitfield(temp_err_in)
    out[39] = _pack_bitfield(pos_err_in)

    t = int(timing_in[0])
    out[40] = (t >> 8) & 0xFF
    out[41] = t & 0xFF
    out[42] = 0
    out[43] = 0

    out[44] = int(gripper_in[0]) & 0xFF
    pos = int(gripper_in[1]) & 0xFFFF
    spd = int(gripper_in[2]) & 0xFFFF
    cur = int(gripper_in[3]) & 0xFFFF
    out[45] = (pos >> 8) & 0xFF
    out[46] = pos & 0xFF
    out[47] = (spd >> 8) & 0xFF
    out[48] = spd & 0xFF
    out[49] = (cur >> 8) & 0xFF
    out[50] = cur & 0xFF
    out[51] = int(gripper_in[4]) & 0xFF


@dataclass
class MockRobotState:
    """Internal state of the simulated robot."""

    # Joint positions (in steps)
    position_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    # Floating accumulator for high-fidelity integration (steps, float)
    position_f: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.float64)
    )
    # Joint speeds (in steps/sec)
    speed_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    # Homed status per joint
    homed_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    # I/O states
    io_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )  # E-stop released
    # Error states
    temperature_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    position_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    # Gripper state
    gripper_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    # Timing data
    timing_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((1,), dtype=np.int32)
    )

    # Simulation parameters
    update_rate: float = cfg.INTERVAL_S  # match control loop cadence
    last_update: float = field(default_factory=time.time)
    homing_countdown: int = 0

    # Command state from controller
    command_out: int = CommandCode.IDLE
    position_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    speed_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )

    def __post_init__(self):
        """Initialize robot to standby position."""
        # Set initial positions to standby position for better IK
        for i in range(6):
            deg = float(cfg.STANDBY_ANGLES_DEG[i])
            steps = int(cfg.deg_to_steps_scalar(deg, i))
            self.position_in[i] = steps
        # Initialize float accumulator from integer steps
        self.position_f = self.position_in.astype(np.float64)

        # Ensure E-stop is not pressed (bit 4 = 1 means released)
        self.io_in[4] = 1


class MockSerialTransport:
    """
    Mock serial transport that simulates robot hardware responses.

    This class implements the exact same interface as SerialTransport,
    but generates simulated responses instead of communicating with
    real hardware. The simulation operates at the frame level, making
    it completely transparent to the controller.
    """

    def __init__(
        self, port: str | None = None, baudrate: int = 2000000, timeout: float = 0
    ):
        """
        Initialize the mock serial transport.

        Args:
            port: Ignored (for interface compatibility)
            baudrate: Ignored (for interface compatibility)
            timeout: Ignored (for interface compatibility)
        """
        self.port = port or "MOCK_SERIAL"
        self.baudrate = baudrate
        self.timeout = timeout

        # Internal robot state
        self._state = MockRobotState()

        # Frame generation tracking
        self._last_frame_time = time.time()
        self._frame_interval = cfg.INTERVAL_S  # match control loop cadence

        # Connection state
        self._connected = False

        # Statistics
        self._frames_received = 0

        # Latest-frame infrastructure (simulation publishes into this buffer)
        self._frame_buf = bytearray(64)  # payload 52B + headroom
        self._frame_mv = memoryview(self._frame_buf)[:52]
        self._frame_version = 0
        self._frame_ts = 0.0

        # Precompute motion simulation constants from LIMITS
        self._vmax_f = LIMITS.joint.hard.velocity_steps.astype(np.float64, copy=False)
        lims = np.asarray(LIMITS.joint.position.steps, dtype=np.int64)
        self._jmin_f = lims[:, 0].astype(np.float64, copy=False)
        self._jmax_f = lims[:, 1].astype(np.float64, copy=False)
        self._home_angles_deg = np.array(cfg.HOME_ANGLES_DEG, dtype=np.float64)

        # Scratch buffer for motion simulation (stores previous position)
        self._prev_pos_f = np.zeros((6,), dtype=np.float64)

        self._state.last_update = time.perf_counter()

        # Write initial frame so first read returns valid data
        self._encode_payload_into(self._frame_mv)
        self._frame_version = 1
        self._frame_ts = time.time()

        logger.info("MockSerialTransport initialized - simulation mode active")

    def connect(self, port: str | None = None) -> bool:
        """
        Simulate serial port connection.

        Args:
            port: Optional port name (ignored)

        Returns:
            Always returns True for mock
        """
        if port:
            self.port = port

        self._connected = True
        self._state = MockRobotState()  # Reset state on connect
        # Initialize time base to perf_counter for consistent scheduling
        self._state.last_update = time.perf_counter()
        logger.info(f"MockSerialTransport connected to simulated port: {self.port}")
        return True

    # Allow controller to sync the simulator pose/homing from live controller state
    def sync_from_controller_state(self, state: ControllerState) -> None:
        """
        Synchronize the mock robot internal state from a controller state snapshot.
        Expects arrays compatible with ControllerState (Position_in, Homed_in).
        """
        try:
            self._state.position_in = state.Position_in.copy()
            # keep high-fidelity accumulator in sync
            self._state.position_f = self._state.position_in.astype(np.float64)
            self._state.homed_in = state.Homed_in.copy()
            self._state.position_out = self._state.position_in.copy()
            self._state.last_update = time.perf_counter()
            self._state.homing_countdown = 0

            # Clear speeds and hold position
            self._state.speed_in = state.Speed_in.copy()
            self._state.command_out = CommandCode.IDLE
            logger.info("MockSerialTransport: state synchronized from controller")
        except Exception as e:
            logger.warning(
                "MockSerialTransport: failed to sync from controller state: %s", e
            )

    def disconnect(self) -> None:
        """Simulate serial port disconnection."""
        self._connected = False
        logger.info(f"MockSerialTransport disconnected from: {self.port}")

    def is_connected(self) -> bool:
        """
        Check if mock connection is active.

        Returns:
            Connection state
        """
        return self._connected

    def auto_reconnect(self) -> bool:
        """
        Mock auto-reconnect (always succeeds).

        Returns:
            True if not connected, False if already connected
        """
        if not self._connected:
            return self.connect(self.port)
        return False

    def write_frame(
        self,
        position_out: np.ndarray,
        speed_out: np.ndarray,
        command_out: int,
        affected_joint_out: np.ndarray,
        inout_out: np.ndarray,
        timeout_out: int,
        gripper_data_out: np.ndarray,
    ) -> bool:
        """Process a command frame from the controller."""
        if not self._connected:
            return False

        self._state.command_out = command_out
        self._frames_received += 1
        _write_frame_jit(
            self._state.position_out,
            self._state.speed_out,
            self._state.gripper_data_in,
            position_out,
            speed_out,
            gripper_data_out,
        )
        return True

    def tick_simulation(self) -> None:
        """
        Run one physics simulation step. Called by controller each tick.

        This advances the simulation by the elapsed time since last update,
        encodes the new state into the frame buffer, and increments the
        frame version for change detection.
        """
        if not self._connected:
            return

        now = time.perf_counter()
        dt = now - self._state.last_update
        self._state.last_update = now

        if dt > 0:
            state = self._state
            # CommandCode enums are resolved at compile time inside the JIT function.
            # Passing enums as arguments would add ~90µs overhead per call.
            state.homing_countdown, state.command_out = _simulate_motion_jit(
                state.position_f,
                state.position_in,
                state.speed_in,
                state.speed_out,
                state.position_out,
                state.homed_in,
                state.io_in,
                self._prev_pos_f,
                self._vmax_f,
                self._jmin_f,
                self._jmax_f,
                self._home_angles_deg,
                int(state.command_out),
                dt,
                state.homing_countdown,
            )

        self._encode_payload_into(self._frame_mv)
        self._frame_version += 1
        self._frame_ts = time.time()

    # ================================
    # Latest-frame API (reduced-copy)
    # ================================
    def poll_read(self) -> bool:
        """
        No-op for mock transport. Simulation is driven by tick_simulation().
        Returns True if connected (frame data is always available).
        """
        return self._connected

    def _encode_payload_into(self, out_mv: memoryview) -> None:
        """Build a 52-byte payload per firmware layout from simulated state."""
        st = self._state
        _encode_payload_jit(
            out_mv,
            st.position_in,
            st.speed_in,
            st.homed_in,
            st.io_in,
            st.temperature_error_in,
            st.position_error_in,
            st.timing_data_in,
            st.gripper_data_in,
        )

    def get_latest_frame_view(self) -> tuple[memoryview | None, int, float]:
        """
        Return latest 52-byte payload memoryview, version, timestamp.
        """
        mv = self._frame_mv if self._frame_version > 0 else None
        return (mv, self._frame_version, self._frame_ts)

    def get_info(self) -> dict:
        """
        Get information about the mock transport.

        Returns:
            Dictionary with transport information
        """
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "connected": self._connected,
            "timeout": self.timeout,
            "mode": "MOCK_SERIAL",
            "frames_received": self._frames_received,
            "simulation_rate_hz": int(1.0 / self._frame_interval),
            "robot_state": {
                "homed": all(self._state.homed_in[i] == 1 for i in range(6)),
                "estop": self._state.io_in[4] == 0,
                "command": self._state.command_out,
            },
        }
