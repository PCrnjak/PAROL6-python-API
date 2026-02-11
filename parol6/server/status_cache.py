"""
Cache of the aggregate STATUS payload for binary msgpack broadcasting.

The heavy IK enablement computations are delegated to a separate subprocess
for true CPU parallelism, communicating via shared memory.
"""

import logging
import multiprocessing
import sys
import time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from parol6.config import steps_to_deg, steps_to_rad
from parol6.protocol.wire import pack_status
from parol6.server.ik_layout import (
    IK_INPUT_Q_OFFSET,
    IK_INPUT_SIZE,
    IK_INPUT_T_OFFSET,
    IK_OUTPUT_SIZE,
)
from parol6.server.ik_worker import ik_enablement_worker_main
from parol6.server.state import ControllerState, get_fkine_flat_mm, get_fkine_se3

logger = logging.getLogger(__name__)

# track parameter added in Python 3.13
_SHM_EXTRA_KWARGS = {"track": False} if sys.version_info >= (3, 13) else {}


def _cleanup_shm(shm: SharedMemory | None) -> None:
    """Safely close and unlink a shared memory segment."""
    if shm is None:
        return
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass


@njit(cache=True)
def _unpack_ik_response_into(
    buf_arr: np.ndarray,
    last_version: int,
    joint_en: np.ndarray,
    cart_en_wrf: np.ndarray,
    cart_en_trf: np.ndarray,
) -> int:
    """
    Check version and copy IK response if changed (zero-alloc hot path).

    Returns new version if data was copied, 0 if unchanged.
    """
    # Version is at offset 36, little-endian uint64
    version = np.uint64(0)
    for i in range(8):
        version |= np.uint64(buf_arr[36 + i]) << np.uint64(i * 8)

    if version == last_version or version == 0:
        return 0

    for i in range(12):
        joint_en[i] = buf_arr[i]
        cart_en_wrf[i] = buf_arr[12 + i]
        cart_en_trf[i] = buf_arr[24 + i]

    return int(version)


# Export for warmup.py
unpack_ik_response_into = _unpack_ik_response_into


@njit(cache=True)
def _update_arrays(
    pos_in: np.ndarray,
    io_in: np.ndarray,
    spd_in: np.ndarray,
    grip_in: np.ndarray,
    pos_last: np.ndarray,
    angles_deg: np.ndarray,
    q_rad_buf: np.ndarray,
    io_cached: np.ndarray,
    spd_cached: np.ndarray,
    grip_cached: np.ndarray,
) -> tuple[bool, bool, bool, bool]:
    """
    Check for changes and update cached arrays.
    Returns (pos_changed, io_changed, spd_changed, grip_changed).
    """
    pos_changed = not np.array_equal(pos_in, pos_last)
    io_changed = not np.array_equal(io_in, io_cached)
    spd_changed = not np.array_equal(spd_in, spd_cached)
    grip_changed = not np.array_equal(grip_in, grip_cached)

    if pos_changed:
        pos_last[:] = pos_in
        steps_to_deg(pos_in, angles_deg)
        steps_to_rad(pos_in, q_rad_buf)
    if io_changed:
        io_cached[:] = io_in
    if spd_changed:
        spd_cached[:] = spd_in
    if grip_changed:
        grip_cached[:] = grip_in

    return pos_changed, io_changed, spd_changed, grip_changed


class StatusCache:
    """
    Cache of the aggregate STATUS payload components and formatted ASCII.

    Fields:
      - angles_deg: 6 floats
      - speeds: 6 ints (steps/sec)
      - io: 5 ints [in1,in2,out1,out2,estop]
      - gripper: >=6 ints [id,pos,spd,cur,status,obj]
      - pose: 16 floats (flattened transform)
      - last_update_s: wall clock time of last cache update
    """

    def __init__(self) -> None:
        # Public snapshots (materialized only when they change)
        self.angles_deg: np.ndarray = np.zeros((6,), dtype=np.float64)
        self.speeds: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.io: np.ndarray = np.zeros((5,), dtype=np.uint8)
        self.gripper: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.pose: np.ndarray = np.zeros((16,), dtype=np.float64)

        self.last_serial_s: float = 0.0  # last time a fresh serial frame was observed
        self._last_tool_name: str = "NONE"  # Track tool changes

        # Action tracking fields
        self._action_current: str = ""
        self._action_state: str = "IDLE"

        # Queue tracking fields
        self._executing_index: int = -1
        self._completed_index: int = -1
        self._last_checkpoint: str = ""

        # Binary cache
        self._binary_cache: bytes = b""
        self._binary_dirty: bool = True

        # Change-detection caches to avoid expensive recomputation when inputs unchanged
        self._last_pos_in: np.ndarray = np.zeros((6,), dtype=np.int32)
        self._last_io_buf: np.ndarray = np.zeros((5,), dtype=np.uint8)

        # Pre-allocated buffer for IK request (avoids allocation per position change)
        self._q_rad_buf: np.ndarray = np.zeros(6, dtype=np.float64)

        # IK enablement results (pre-allocated for zero-alloc reads)
        self._joint_en = np.ones(12, dtype=np.uint8)
        self._cart_en_wrf = np.ones(12, dtype=np.uint8)
        self._cart_en_trf = np.ones(12, dtype=np.uint8)

        # IK worker subprocess state
        self._ik_stopped = False
        self._ik_last_version = 0
        shm_suffix = f"_{id(self)}"
        input_name = f"parol6_ik_in{shm_suffix}"
        output_name = f"parol6_ik_out{shm_suffix}"

        # Create shared memory segments
        self._ik_input_shm = SharedMemory(
            name=input_name, create=True, size=IK_INPUT_SIZE, **_SHM_EXTRA_KWARGS
        )
        self._ik_output_shm = SharedMemory(
            name=output_name, create=True, size=IK_OUTPUT_SIZE, **_SHM_EXTRA_KWARGS
        )

        # SharedMemory.buf is always non-None after successful __init__
        input_buf = self._ik_input_shm.buf
        output_buf = self._ik_output_shm.buf
        assert input_buf is not None
        assert output_buf is not None

        # Initialize with zeros
        np.frombuffer(input_buf, dtype=np.uint8)[:] = 0
        np.frombuffer(output_buf, dtype=np.uint8)[:] = 0

        # Memoryviews for cleanup
        self._ik_input_mv = memoryview(input_buf)
        self._ik_output_mv = memoryview(output_buf)

        # Zero-alloc input views: write directly into shared memory
        self._ik_input_q_view = np.frombuffer(
            input_buf,
            dtype=np.float64,
            count=6,
            offset=IK_INPUT_Q_OFFSET,
        )
        self._ik_input_T_view = np.frombuffer(
            input_buf,
            dtype=np.float64,
            count=16,
            offset=IK_INPUT_T_OFFSET,
        )

        # Zero-alloc output view for numba reader
        self._ik_output_arr = np.frombuffer(output_buf, dtype=np.uint8)

        # Spawn subprocess
        self._ik_shutdown_event: Event = multiprocessing.Event()
        self._ik_request_event: Event = multiprocessing.Event()
        self._ik_process: Process = Process(
            target=ik_enablement_worker_main,
            args=(
                input_name,
                output_name,
                self._ik_shutdown_event,
                self._ik_request_event,
            ),
            daemon=True,
            name="IKWorkerProcess",
        )
        self._ik_process.start()
        logger.info(f"IK worker started, PID: {self._ik_process.pid}")

    def _stop_ik_worker(self) -> None:
        """Shut down the IK worker subprocess and release resources."""
        if self._ik_stopped:
            return
        self._ik_stopped = True

        # Signal shutdown
        self._ik_shutdown_event.set()

        # Wait for process to exit
        if self._ik_process.is_alive():
            self._ik_process.join(timeout=2.0)
            if self._ik_process.is_alive():
                logger.warning("IK worker did not exit cleanly, terminating")
                self._ik_process.terminate()
                self._ik_process.join(timeout=1.0)

        # Wait for exitcode to be set (subprocess's finally block completed)
        deadline = time.time() + 1.0
        while self._ik_process.exitcode is None and time.time() < deadline:
            time.sleep(0.01)

        # Release numpy views before closing shared memory
        del self._ik_input_q_view
        del self._ik_input_T_view
        del self._ik_output_arr

        # Release memoryviews
        try:
            self._ik_input_mv.release()
        except BufferError:
            pass
        try:
            self._ik_output_mv.release()
        except BufferError:
            pass

        # Clean up shared memory
        _cleanup_shm(self._ik_input_shm)
        _cleanup_shm(self._ik_output_shm)
        logger.info("IK worker stopped")

    def _submit_ik_request(self, q_rad: np.ndarray, T_matrix: np.ndarray) -> None:
        """Submit an IK enablement request (non-blocking, zero-alloc)."""
        if self._ik_stopped:
            return
        self._ik_input_q_view[:] = q_rad[:6]
        self._ik_input_T_view[:] = T_matrix.flat[:16]
        self._ik_request_event.set()

    def _poll_ik_results(self) -> bool:
        """Check for new IK results (non-blocking, zero-alloc). Returns True if updated."""
        if self._ik_stopped:
            return False
        new_version = _unpack_ik_response_into(
            self._ik_output_arr,
            self._ik_last_version,
            self._joint_en,
            self._cart_en_wrf,
            self._cart_en_trf,
        )
        if new_version > 0:
            self._ik_last_version = new_version
            return True
        return False

    def close(self) -> None:
        """Shut down the IK worker subprocess and release resources."""
        self._stop_ik_worker()

    def __del__(self) -> None:
        """Safety net: ensure IK worker is stopped if close() was not called."""
        self.close()

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state with change gating:
          - Only recompute angles/pose when Position_in changes
          - Only refresh IO/speeds/gripper when their inputs actually change
          - IK enablement is computed asynchronously in a subprocess
        """
        # Do change detection
        self._last_io_buf[:] = state.InOut_in[:5]
        pos_changed, io_changed, spd_changed, grip_changed = _update_arrays(
            state.Position_in,
            self._last_io_buf,
            state.Speed_in,
            state.Gripper_data_in,
            self._last_pos_in,
            self.angles_deg,
            self._q_rad_buf,
            self.io,
            self.speeds,
            self.gripper,
        )
        tool_changed = state.current_tool != self._last_tool_name

        if pos_changed or tool_changed:
            if tool_changed:
                self._last_tool_name = state.current_tool
            self.pose[:] = get_fkine_flat_mm(state)
            # Submit IK request asynchronously
            try:
                T_matrix = get_fkine_se3(state)
                self._submit_ik_request(self._q_rad_buf, T_matrix)
            except (ValueError, OSError):
                pass

        # Poll for async IK results (non-blocking, zero-alloc)
        ik_changed = self._poll_ik_results()

        action_changed = (
            self._action_current != state.action_current
            or self._action_state != state.action_state
        )
        if action_changed:
            self._action_current = state.action_current
            self._action_state = state.action_state

        queue_changed = (
            self._executing_index != state.executing_command_index
            or self._completed_index != state.completed_command_index
            or self._last_checkpoint != state.last_checkpoint
        )
        if queue_changed:
            self._executing_index = state.executing_command_index
            self._completed_index = state.completed_command_index
            self._last_checkpoint = state.last_checkpoint

        # Mark binary cache dirty if anything changed
        if (
            pos_changed
            or tool_changed
            or io_changed
            or spd_changed
            or grip_changed
            or ik_changed
            or action_changed
            or queue_changed
        ):
            self._binary_dirty = True

    def to_binary(self) -> bytes:
        """Return the msgpack-encoded STATUS payload."""
        if self._binary_dirty:
            self._binary_cache = pack_status(
                self.pose,
                self.angles_deg,
                self.speeds,
                self.io,
                self.gripper,
                self._action_current,
                self._action_state,
                self._joint_en,
                self._cart_en_wrf,
                self._cart_en_trf,
                self._executing_index,
                self._completed_index,
                self._last_checkpoint,
            )
            self._binary_dirty = False
        return self._binary_cache

    def mark_serial_observed(self) -> None:
        """Mark that a fresh serial frame was observed just now."""
        self.last_serial_s = time.monotonic()

    def age_s(self) -> float:
        """Seconds since last fresh serial observation (used to gate broadcasting)."""
        if self.last_serial_s <= 0:
            return 1e9
        return time.monotonic() - self.last_serial_s

    @property
    def joint_en(self) -> np.ndarray:
        """Joint enablement flags (12 elements)."""
        return self._joint_en

    @property
    def cart_en_wrf(self) -> np.ndarray:
        """Cartesian enablement flags in world reference frame (12 elements)."""
        return self._cart_en_wrf

    @property
    def cart_en_trf(self) -> np.ndarray:
        """Cartesian enablement flags in tool reference frame (12 elements)."""
        return self._cart_en_trf


# Module-level singleton
_status_cache: StatusCache | None = None


def get_cache() -> StatusCache:
    global _status_cache
    if _status_cache is None:
        _status_cache = StatusCache()
    return _status_cache
