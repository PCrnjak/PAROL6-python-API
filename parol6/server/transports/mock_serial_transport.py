"""
Mock serial transport for simulation and testing.

This module provides a complete serial port simulation that generates
realistic robot responses without requiring hardware. The simulation
operates at the wire protocol level, making it transparent to the
controller code.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6 import config as cfg
from parol6.protocol.wire import CommandCode, split_to_3_bytes
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@dataclass
class MockRobotState:
    """Internal state of the simulated robot."""

    # Joint positions (in steps)
    position_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    # Floating accumulator for high-fidelity integration (steps, float)
    position_f: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.float64))
    # Joint speeds (in steps/sec)
    speed_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    # Homed status per joint
    homed_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    # I/O states
    io_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )  # E-stop released
    # Error states
    temperature_error_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    position_error_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    # Gripper state
    gripper_data_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    # Timing data
    timing_data_in: np.ndarray = field(default_factory=lambda: np.zeros((1,), dtype=np.int32))

    # Simulation parameters
    update_rate: float = cfg.INTERVAL_S  # match control loop cadence
    last_update: float = field(default_factory=time.time)
    homing_countdown: int = 0

    # Command state from controller
    command_out: int = CommandCode.IDLE
    position_out: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    speed_out: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))

    def __post_init__(self):
        """Initialize robot to standby position."""
        # Set initial positions to standby position for better IK
        for i in range(6):
            deg = float(PAROL6_ROBOT.joint.standby.deg[i])
            steps = int(PAROL6_ROBOT.ops.deg_to_steps(deg, i))
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

    def __init__(self, port: str | None = None, baudrate: int = 2000000, timeout: float = 0):
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
        self._frames_to_send: list[bytes] = []
        self._last_frame_time = time.time()
        self._frame_interval = cfg.INTERVAL_S  # match control loop cadence

        # Connection state
        self._connected = False

        # Statistics
        self._frames_sent = 0
        self._frames_received = 0

        # Latest-frame infrastructure (simulation publishes into this buffer)
        self._frame_buf = bytearray(64)  # payload 52B + headroom
        self._frame_mv = memoryview(self._frame_buf)[:52]
        self._frame_version = 0
        self._frame_ts = 0.0
        self._reader_thread: threading.Thread | None = None
        self._reader_running = False

        # Precompute motion simulation constants
        self._vmax_f = PAROL6_ROBOT.joint.speed.max.astype(np.float64, copy=False)
        self._vmax_i32 = PAROL6_ROBOT.joint.speed.max.astype(np.int32, copy=False)
        lims = np.asarray(PAROL6_ROBOT.joint.limits.steps, dtype=np.int64)
        self._jmin_f = lims[:, 0].astype(np.float64, copy=False)
        self._jmax_f = lims[:, 1].astype(np.float64, copy=False)

        # Scratch buffers for motion simulation
        self._prev_pos_f = np.zeros((6,), dtype=np.float64)

        self._state.last_update = time.perf_counter()

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
            logger.warning("MockSerialTransport: failed to sync from controller state: %s", e)

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
        """
        Process a command frame from the controller.

        Instead of writing to serial, this updates the internal
        simulation state.

        Args:
            position_out: Target positions
            speed_out: Speed commands
            command_out: Command code
            affected_joint_out: Affected joint flags
            inout_out: I/O commands
            timeout_out: Timeout value
            gripper_data_out: Gripper commands

        Returns:
            True if processed successfully
        """
        if not self._connected:
            return False

        # Update simulation state with command
        self._state.command_out = command_out
        self._state.position_out = np.array(position_out, dtype=np.int32, copy=False)
        self._state.speed_out = np.array(speed_out, dtype=np.float64, copy=False)

        # Track frame reception
        self._frames_received += 1

        # Simulate gripper state updates
        if gripper_data_out[4] == 1:  # Calibration mode
            # Simulate gripper calibration
            self._state.gripper_data_in[0] = gripper_data_out[5]  # Set device ID
            self._state.gripper_data_in[4] = 0x40  # Set calibrated bit
        elif gripper_data_out[4] == 2:  # Error clear mode
            # Clear gripper errors
            self._state.gripper_data_in[4] &= ~0x20  # Clear error bit

        # Update gripper position/speed/current if commanded
        if gripper_data_out[3] != 0:  # Gripper command active
            self._state.gripper_data_in[1] = gripper_data_out[0]  # Position
            self._state.gripper_data_in[2] = gripper_data_out[1]  # Speed
            self._state.gripper_data_in[3] = gripper_data_out[2]  # Current

        return True

    def _simulate_motion(self, dt: float) -> None:
        """
        Simulate one step of robot motion.

        Args:
            dt: Time delta since last update
        """
        state = self._state

        # Handle homing countdown
        if state.homing_countdown > 0:
            state.homing_countdown -= 1
            if state.homing_countdown == 0:
                # Homing complete - mark all joints as homed and move to configured home posture
                target_deg = cfg.HOME_ANGLES_DEG
                # Mark all 8 homed bits as 1 to satisfy status bitfield expectations
                state.homed_in.fill(1)
                for i in range(6):
                    steps = int(PAROL6_ROBOT.ops.deg_to_steps(float(target_deg[i]), i))
                    state.position_in[i] = steps
                    state.position_f[i] = float(steps)
                    state.speed_in[i] = 0
                # Clear HOME command to avoid immediately restarting homing
                state.command_out = CommandCode.IDLE

        # Ensure E-stop stays released in simulation
        state.io_in[4] = 1

        # Simulate motion based on command type
        if state.command_out == CommandCode.HOME:
            # Start homing sequence
            if state.homing_countdown == 0:
                for i in range(6):
                    state.homed_in[i] = 0  # Mark as not homed
                # Schedule homing completion after ~0.2s (use fixed frame interval for determinism)
                state.homing_countdown = max(1, int(0.2 / self._frame_interval + 0.5))
            # Zero speeds during homing
            for i in range(6):
                state.speed_in[i] = 0

        elif state.command_out == CommandCode.JOG or state.command_out == 123:
            # Speed control mode (vectorized float-accumulated integration)
            np.copyto(self._prev_pos_f, state.position_f)

            # Clip commanded speeds to joint limits
            v_cmd = np.clip(
                state.speed_out.astype(np.float64, copy=False), -self._vmax_f, self._vmax_f
            )

            # Integrate position
            new_pos_f = state.position_f + v_cmd * dt

            # Apply joint limits
            np.clip(new_pos_f, self._jmin_f, self._jmax_f, out=state.position_f)

            # Report actual velocity based on realized motion
            if dt > 0:
                realized_v = np.rint((state.position_f - self._prev_pos_f) / dt).astype(np.int32)
                np.clip(realized_v, -self._vmax_i32, self._vmax_i32, out=state.speed_in)
            else:
                state.speed_in.fill(0)

        elif state.command_out == CommandCode.MOVE or state.command_out == 156:
            # Position control mode (float-accumulated and per-tick speed clamp)
            prev_pos_f = state.position_f.copy()
            for i in range(6):
                target = float(state.position_out[i])
                current_f = float(state.position_f[i])
                err_f = target - current_f

                # Calculate max move this tick from per-joint max speed
                max_step_f = float(PAROL6_ROBOT.joint.speed.max[i]) * float(dt)
                if max_step_f < 1.0:
                    # ensure some progress at very small dt
                    max_step_f = 1.0

                move = float(err_f)
                if move > max_step_f:
                    move = max_step_f
                elif move < -max_step_f:
                    move = -max_step_f

                new_pos_f = current_f + move

                # Apply joint limits
                jmin, jmax = PAROL6_ROBOT.joint.limits.steps[i]
                if new_pos_f < float(jmin):
                    new_pos_f = float(jmin)
                elif new_pos_f > float(jmax):
                    new_pos_f = float(jmax)

                state.position_f[i] = new_pos_f

            # Report actual velocity based on realized motion
            if dt > 0:
                realized_v = np.rint((state.position_f - prev_pos_f) / dt).astype(np.int32)
            else:
                realized_v = np.zeros(6, dtype=np.int32)
            vmax = PAROL6_ROBOT.joint.speed.max.astype(np.int32)
            state.speed_in[:] = np.clip(realized_v, -vmax, vmax)

        else:
            # Idle or unknown command - hold position
            for i in range(6):
                state.speed_in[i] = 0

        # Sync integer telemetry from high-fidelity accumulator
        state.position_in[:] = np.rint(state.position_f).astype(np.int32)

    # ================================
    # Latest-frame API (reduced-copy)
    # ================================
    def start_reader(self, shutdown_event: threading.Event) -> threading.Thread:
        """
        Start simulated latest-frame publisher thread.
        """
        if self._reader_thread and self._reader_thread.is_alive():
            return self._reader_thread

        def _run():
            self._reader_running = True
            period = self._frame_interval
            next_deadline = time.perf_counter()

            try:
                while not shutdown_event.is_set():
                    if not self._connected:
                        time.sleep(0.05)
                        continue

                    now = time.perf_counter()
                    if now >= next_deadline:
                        # Advance simulation before publishing a new frame
                        dt = now - self._state.last_update
                        if dt > 0:
                            self._simulate_motion(dt)
                            self._state.last_update = now

                        self._encode_payload_into(self._frame_mv)
                        self._frame_version += 1
                        self._frame_ts = time.time()

                        # Advance deadline
                        next_deadline += period
                        # If we fell far behind, resync to avoid tight catch-up loop
                        if next_deadline < now - period:
                            next_deadline = now + period
                    else:
                        # Sleep until next deadline (or at most 2ms to stay responsive)
                        sleep_time = min(next_deadline - now, 0.002)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
            finally:
                self._reader_running = False

        t = threading.Thread(target=_run, name="MockSerialReader", daemon=True)
        self._reader_thread = t
        t.start()
        return t

    def _encode_payload_into(self, out_mv: memoryview) -> None:
        """
        Build a 52-byte payload per firmware layout from the simulated state directly into memoryview.
        Zero-allocation version for use in the reader loop.
        """
        st = self._state
        out = out_mv
        # Positions (6 * 3 bytes)
        off = 0
        for i in range(6):
            b0, b1, b2 = split_to_3_bytes(int(st.position_in[i]))
            out[off] = b0
            out[off + 1] = b1
            out[off + 2] = b2
            off += 3
        # Speeds (6 * 3 bytes)
        off = 18
        for i in range(6):
            b0, b1, b2 = split_to_3_bytes(int(st.speed_in[i]))
            out[off] = b0
            out[off + 1] = b1
            out[off + 2] = b2
            off += 3

        def bits_to_byte(bits: np.ndarray) -> int:
            val = 0
            for b in bits[:8]:
                val = (val << 1) | (1 if b else 0)
            return val & 0xFF

        # Bitfields
        out[36] = bits_to_byte(st.homed_in)
        out[37] = bits_to_byte(st.io_in)
        out[38] = bits_to_byte(st.temperature_error_in)
        out[39] = bits_to_byte(st.position_error_in)

        # Timing (two bytes)
        t = int(st.timing_data_in[0]) if st.timing_data_in else 0
        out[40] = (t >> 8) & 0xFF
        out[41] = t & 0xFF

        # Reserved
        out[42] = 0
        out[43] = 0

        # Gripper
        gd = st.gripper_data_in
        dev_id = int(gd[0]) if gd.all() else 0
        pos = int(gd[1]) & 0xFFFF
        spd = int(gd[2]) & 0xFFFF
        cur = int(gd[3]) & 0xFFFF
        status = int(gd[4]) & 0xFF if gd.all() else 0

        out[44] = dev_id & 0xFF
        out[45] = (pos >> 8) & 0xFF
        out[46] = pos & 0xFF
        out[47] = (spd >> 8) & 0xFF
        out[48] = spd & 0xFF
        out[49] = (cur >> 8) & 0xFF
        out[50] = cur & 0xFF
        out[51] = status & 0xFF

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
            "frames_sent": self._frames_sent,
            "frames_received": self._frames_received,
            "simulation_rate_hz": int(1.0 / self._frame_interval),
            "robot_state": {
                "homed": all(self._state.homed_in[i] == 1 for i in range(6)),
                "estop": self._state.io_in[4] == 0,
                "command": self._state.command_out,
            },
        }


def create_mock_serial_transport() -> MockSerialTransport:
    """
    Factory function to create a mock serial transport.

    Returns:
        Configured MockSerialTransport instance
    """
    transport = MockSerialTransport()
    transport.connect()
    logger.info("Mock serial transport created and connected")
    return transport
