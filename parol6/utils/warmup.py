"""
JIT warmup utilities.

Call warmup_jit() on startup to pre-compile all numba functions before the control loop.
With cache=True, this is fast (~100ms) if the on-disk cache exists, slower (~10-30s on a
slow machine) on a cold first run.
"""

import logging
import time

import numpy as np

from parol6.commands.servo_commands import _max_vel_ratio_jit
from parol6.config import (
    deg_to_steps,
    deg_to_steps_scalar,
    rad_to_steps,
    rad_to_steps_scalar,
    speed_deg_to_steps,
    speed_deg_to_steps_scalar,
    speed_rad_to_steps,
    speed_rad_to_steps_scalar,
    speed_steps_to_deg,
    speed_steps_to_deg_scalar,
    speed_steps_to_rad,
    speed_steps_to_rad_scalar,
    steps_to_deg,
    steps_to_deg_scalar,
    steps_to_rad,
    steps_to_rad_scalar,
)
from parol6.motion.streaming_executors import (
    _pose_to_tangent_jit,
    _tangent_to_pose_jit,
)
from parol6.motion.trajectory import _smooth_singularity_outliers
from parol6.protocol.wire import (
    _pack_bitfield,
    _pack_positions,
    _unpack_bitfield,
    _unpack_positions,
    fuse_2_bytes,
    fuse_3_bytes,
    pack_tx_frame_into,
    split_to_3_bytes,
    unpack_rx_frame_into,
)
from parol6.server.ik_worker import (
    _AXIS_DIRS,
    _compute_joint_enable,
    _compute_target_poses,
)
from parol6.server.status_cache import unpack_ik_response_into
from parol6.server.loop_timer import (
    _compute_event_rate,
    _compute_loop_stats,
    _compute_phase_stats,
    _quickselect,
    _quickselect_partition,
)
from parol6.server.status_cache import _update_arrays
from parol6.server.transports.mock_serial_transport import (
    _encode_payload_jit,
    _simulate_gripper_ramp_jit,
    _simulate_motion_jit,
    _write_frame_jit,
)
from parol6.server.transports.serial_transport import (
    _append_to_ring_numba,
    _parse_frames_njit,
)
from parol6.utils.ik import _check_limits_core, _ik_safety_check
from pinokin import warmup_numba_se3

logger = logging.getLogger(__name__)


def warmup_jit() -> float:
    """
    Pre-compile all numba JIT functions by calling them with dummy data.

    Returns the time taken in seconds.
    """
    logger.info(
        "Warming up numba JIT compiler (first run is a one-time cold compile, "
        "~10-30s on a slow machine; cached afterwards so later starts are "
        "instant). This is normal startup, not a hang..."
    )
    start = time.perf_counter()

    def _progress(label: str) -> None:
        # Only chatter on a genuinely slow cold compile: warm-cache starts stay
        # quiet while a cold start visibly shows it isn't frozen.
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:
            logger.info("  ...JIT warmup: %s ready (%.1fs)", label, elapsed)

    # Warm up pinokin's zero-allocation SE3/SO3 functions
    warmup_numba_se3()

    dummy_6f = np.zeros(6, dtype=np.float64)
    dummy_6i = np.zeros(6, dtype=np.int32)
    out_6f = np.zeros(6, dtype=np.float64)
    out_6i = np.zeros(6, dtype=np.int32)

    # parol6/config.py
    deg_to_steps(dummy_6f, out_6i)
    deg_to_steps_scalar(0.0, 0)
    steps_to_deg(dummy_6i, out_6f)
    steps_to_deg_scalar(0, 0)
    rad_to_steps(dummy_6f, out_6i)
    rad_to_steps_scalar(0.0, 0)
    steps_to_rad(dummy_6i, out_6f)
    steps_to_rad_scalar(0, 0)
    speed_steps_to_deg(dummy_6i, out_6f)
    speed_steps_to_deg_scalar(0.0, 0)
    speed_deg_to_steps(dummy_6f, out_6i)
    speed_deg_to_steps_scalar(0.0, 0)
    speed_steps_to_rad(dummy_6i, out_6f)
    speed_steps_to_rad_scalar(0.0, 0)
    speed_rad_to_steps(dummy_6f, out_6i)
    speed_rad_to_steps_scalar(0.0, 0)
    _progress("joint conversions")

    # parol6/utils/ik.py
    _ik_safety_check(dummy_6f, dummy_6f, dummy_6f, dummy_6f, dummy_6f, dummy_6f)
    viol = np.zeros(6, dtype=np.bool_)
    _check_limits_core(
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        True,
        False,
        viol,
        viol,
        viol,
        viol,
        viol,
        viol,
    )

    # parol6/protocol/wire.py
    dummy_pos = np.zeros(6, dtype=np.int32)
    dummy_bits = np.zeros(8, dtype=np.uint8)
    dummy_out = np.zeros(20, dtype=np.uint8)
    _pack_positions(dummy_out, dummy_pos, 0)
    _unpack_positions(dummy_out, dummy_pos)
    _pack_bitfield(dummy_bits)
    _unpack_bitfield(0, dummy_bits)
    split_to_3_bytes(0)
    fuse_3_bytes(0, 0, 0)
    fuse_2_bytes(0, 0)

    # parol6/server/status_cache.py
    dummy_5u8 = np.zeros(5, dtype=np.uint8)
    _update_arrays(
        dummy_6i,
        dummy_5u8,
        dummy_6i,
        dummy_6i,
        dummy_6f,
        dummy_6f,
        dummy_5u8,
        dummy_6i,
    )

    # Dummy SE3 matrices for jit warmups below
    dummy_4x4 = np.zeros((4, 4), dtype=np.float64)
    dummy_4x4_b = np.zeros((4, 4), dtype=np.float64)
    dummy_4x4_out = np.zeros((4, 4), dtype=np.float64)

    # parol6/server/ik_worker.py
    dummy_qlim = np.zeros((2, 6), dtype=np.float64)
    dummy_12u8 = np.zeros(12, dtype=np.uint8)
    _compute_joint_enable(dummy_6f, dummy_qlim, dummy_12u8)
    dummy_targets = np.zeros((12, 4, 4), dtype=np.float64)
    _compute_target_poses(dummy_4x4, 0.001, 0.01, True, _AXIS_DIRS, dummy_targets)

    # parol6/server/ipc - IK response unpacking
    dummy_ik_buf = np.zeros(44, dtype=np.uint8)
    dummy_joint_en = np.zeros(12, dtype=np.uint8)
    dummy_cart_wrf = np.zeros(12, dtype=np.uint8)
    dummy_cart_trf = np.zeros(12, dtype=np.uint8)
    unpack_ik_response_into(
        dummy_ik_buf, 0, dummy_joint_en, dummy_cart_wrf, dummy_cart_trf
    )

    # parol6/protocol/wire.py - frame packing/unpacking
    dummy_tx_frame = memoryview(bytearray(64))
    dummy_gripper_data = np.zeros(6, dtype=np.int32)
    dummy_8u8_bitfield = np.zeros(8, dtype=np.uint8)
    # bitfield args need 8 elements for _pack_bitfield
    pack_tx_frame_into(
        dummy_tx_frame,
        dummy_6i,
        dummy_6i,
        0,
        dummy_8u8_bitfield,
        dummy_8u8_bitfield,
        0,
        dummy_gripper_data,
    )
    dummy_rx_frame = memoryview(bytearray(64))
    dummy_8u8_homed = np.zeros(8, dtype=np.uint8)
    dummy_8u8_io = np.zeros(8, dtype=np.uint8)
    dummy_8u8_temp = np.zeros(8, dtype=np.uint8)
    dummy_8u8_poserr = np.zeros(8, dtype=np.uint8)
    dummy_timing_out = np.zeros(1, dtype=np.int32)
    dummy_grip_out = np.zeros(6, dtype=np.int32)
    unpack_rx_frame_into(
        dummy_rx_frame,
        dummy_6i,
        dummy_6i,
        dummy_8u8_homed,
        dummy_8u8_io,
        dummy_8u8_temp,
        dummy_8u8_poserr,
        dummy_timing_out,
        dummy_grip_out,
    )

    # parol6/server/transports/serial_transport.py - real-hardware frame I/O.
    # Not exercised by the simulator, but warmed so a hardware controller does
    # not cold-compile these on its first serial frame.
    dummy_ring = np.zeros(256, dtype=np.uint8)
    dummy_src = np.zeros(8, dtype=np.uint8)
    _append_to_ring_numba(dummy_ring, dummy_src, 8, 256, 0, 0)
    _parse_frames_njit(dummy_ring, 0, 0, 256, np.zeros(64, dtype=np.uint8))
    _progress("protocol & kinematics")

    # parol6/server/loop_timer.py - stats computation
    dummy_1000f = np.zeros(1000, dtype=np.float64)
    dummy_1000f_scratch = np.zeros(1000, dtype=np.float64)
    # Fill with realistic timing data so the stats kernels warm a real code path
    dummy_1000f[:100] = np.linspace(0.004, 0.006, 100)
    _quickselect_partition(dummy_1000f_scratch[:10].copy(), 0, 9)
    _quickselect(dummy_1000f_scratch[:100].copy(), 50)
    _compute_phase_stats(dummy_1000f, dummy_1000f_scratch, 100)
    _compute_loop_stats(dummy_1000f, dummy_1000f_scratch, 100)
    _compute_event_rate(dummy_1000f, 100, 1.0, 1.0)

    # parol6/server/transports/mock_serial_transport.py
    dummy_pos_f = np.zeros(6, dtype=np.float64)
    dummy_8u8 = np.zeros(8, dtype=np.uint8)
    dummy_gripper_6i = np.zeros(6, dtype=np.int32)
    _simulate_motion_jit(
        dummy_pos_f,
        dummy_6i,
        dummy_6i,
        dummy_6i,
        dummy_6i,
        dummy_8u8,
        dummy_8u8,
        dummy_6f.copy(),
        dummy_6f.copy(),
        dummy_6f.copy(),
        dummy_6f.copy(),
        dummy_6f.copy(),
        0,
        0.004,
        0,
    )
    dummy_gripper_ramp = np.zeros(3, dtype=np.float64)
    _write_frame_jit(
        dummy_6i,
        dummy_6i,
        dummy_gripper_6i,
        dummy_6i,
        dummy_6i,
        dummy_gripper_6i,
        dummy_gripper_ramp,
    )
    _simulate_gripper_ramp_jit(
        dummy_gripper_ramp,
        dummy_gripper_6i,
        0.0,
        0.004,
        10432.0,
        40.0,
        80000.0,
    )
    dummy_payload = memoryview(bytearray(64))
    dummy_timing = np.zeros(1, dtype=np.int32)
    dummy_gripper_in = np.zeros(6, dtype=np.int32)
    # io_in needs 8 elements for _pack_bitfield
    _encode_payload_jit(
        dummy_payload,
        dummy_6i,
        dummy_6i,
        dummy_8u8,
        dummy_8u8,
        dummy_8u8,
        dummy_8u8,
        dummy_timing,
        dummy_gripper_in,
    )
    _progress("simulator & I/O")

    # Workspace arrays for jit functions below (SE3 funcs already warmed by pinokin)
    dummy_twist = np.zeros(6, dtype=np.float64)
    omega_ws = np.zeros(3, dtype=np.float64)
    R_ws = np.zeros((3, 3), dtype=np.float64)
    V_ws = np.zeros((3, 3), dtype=np.float64)
    V_inv_ws = np.zeros((3, 3), dtype=np.float64)

    # parol6/motion/streaming_executors.py
    ref_inv = np.zeros((4, 4), dtype=np.float64)
    delta_4x4 = np.zeros((4, 4), dtype=np.float64)
    _pose_to_tangent_jit(
        dummy_4x4,
        dummy_4x4_b,
        ref_inv,
        delta_4x4,
        dummy_twist,
        omega_ws,
        R_ws,
        V_inv_ws,
    )
    _tangent_to_pose_jit(
        dummy_4x4, dummy_twist, delta_4x4, dummy_4x4_out, omega_ws, R_ws, V_ws
    )

    # parol6/commands/servo_commands.py
    _max_vel_ratio_jit(dummy_6f, dummy_6f)

    # parol6/motion/trajectory.py — non-trivial array exercises every branch
    # (diff loop, median, bad-detection, interp loop).
    dummy_chain = np.zeros((10, 6), dtype=np.float64)
    for i in range(10):
        dummy_chain[i] = i * 0.01
    dummy_chain[5, 3] += 1.0  # synthetic outlier so the interp loop compiles
    _smooth_singularity_outliers(dummy_chain)

    elapsed = time.perf_counter() - start
    logger.info("JIT warmup complete (%.1fs).", elapsed)
    return elapsed
