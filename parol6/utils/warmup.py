"""
JIT warmup utilities.

Call warmup_jit() on startup to pre-compile all numba functions before the control loop.
With cache=True, this is fast (~100ms) if cache exists, slower (~3-10s) on first run.
"""

import logging
import time

import numpy as np

from parol6.commands.cartesian_commands import (
    _apply_velocity_delta_trf_jit,
    _apply_velocity_delta_wrf_jit,
)
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
from parol6.utils.ik import _check_limits_core, _ik_safety_check
from pinokin import warmup_numba_se3

logger = logging.getLogger(__name__)


def warmup_jit() -> float:
    """
    Pre-compile all numba JIT functions by calling them with dummy data.

    Returns the time taken in seconds.
    """
    logging.info("Warming JIT...")
    start = time.perf_counter()

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

    dummy_3x3 = np.eye(3, dtype=np.float64)

    # parol6/server/status_cache.py
    dummy_5u8 = np.zeros(5, dtype=np.uint8)
    _update_arrays(
        dummy_6i,  # pos_in
        dummy_5u8,  # io_in
        dummy_6i,  # spd_in
        dummy_6i,  # pos_last
        dummy_6f,  # angles_deg
        dummy_6f,  # q_rad_buf
        dummy_5u8,  # io_cached
        dummy_6i,  # spd_cached
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
    pack_tx_frame_into(
        dummy_tx_frame,  # out
        dummy_6i,  # position_out
        dummy_6i,  # speed_out
        0,  # command_code
        dummy_8u8_bitfield,  # affected_joint_out (8 elements for _pack_bitfield)
        dummy_8u8_bitfield,  # inout_out (8 elements for _pack_bitfield)
        0,  # timeout_out
        dummy_gripper_data,  # gripper_data_out
    )
    dummy_rx_frame = memoryview(bytearray(64))
    dummy_8u8_homed = np.zeros(8, dtype=np.uint8)
    dummy_8u8_io = np.zeros(8, dtype=np.uint8)
    dummy_8u8_temp = np.zeros(8, dtype=np.uint8)
    dummy_8u8_poserr = np.zeros(8, dtype=np.uint8)
    dummy_timing_out = np.zeros(1, dtype=np.int32)
    dummy_grip_out = np.zeros(6, dtype=np.int32)
    unpack_rx_frame_into(
        dummy_rx_frame,  # data
        dummy_6i,  # pos_out
        dummy_6i,  # spd_out
        dummy_8u8_homed,  # homed_out
        dummy_8u8_io,  # io_out
        dummy_8u8_temp,  # temp_out
        dummy_8u8_poserr,  # poserr_out
        dummy_timing_out,  # timing_out
        dummy_grip_out,  # grip_out
    )

    # parol6/server/loop_timer.py - stats computation
    dummy_1000f = np.zeros(1000, dtype=np.float64)
    dummy_1000f_scratch = np.zeros(1000, dtype=np.float64)
    # Fill with some data for realistic warmup
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
        dummy_pos_f,  # position_f
        dummy_6i,  # position_in
        dummy_6i,  # speed_in
        dummy_6i,  # speed_out
        dummy_6i,  # position_out
        dummy_8u8,  # homed_in
        dummy_8u8,  # io_in
        dummy_6f.copy(),  # prev_pos_f
        dummy_6f.copy(),  # vmax_f
        dummy_6f.copy(),  # jmin_f
        dummy_6f.copy(),  # jmax_f
        dummy_6f.copy(),  # home_angles_deg
        0,  # command_out
        0.004,  # dt
        0,  # homing_countdown
    )
    dummy_gripper_ramp = np.zeros(3, dtype=np.float64)
    _write_frame_jit(
        dummy_6i,  # state_position_out
        dummy_6i,  # state_speed_out
        dummy_gripper_6i,  # state_gripper_data_in
        dummy_6i,  # position_out
        dummy_6i,  # speed_out
        dummy_gripper_6i,  # gripper_data_out
        dummy_gripper_ramp,  # gripper_ramp
    )
    _simulate_gripper_ramp_jit(
        dummy_gripper_ramp,  # gripper_ramp
        dummy_gripper_6i,  # gripper_data_in
        0.0,  # gripper_pos_f
        0.004,  # dt
        10432.0,  # tick_range
        40.0,  # min_speed
        80000.0,  # max_speed
    )
    dummy_payload = memoryview(bytearray(64))
    dummy_timing = np.zeros(1, dtype=np.int32)
    dummy_gripper_in = np.zeros(6, dtype=np.int32)
    _encode_payload_jit(
        dummy_payload,  # out
        dummy_6i,  # position_in
        dummy_6i,  # speed_in
        dummy_8u8,  # homed_in
        dummy_8u8,  # io_in (8 elements for _pack_bitfield)
        dummy_8u8,  # temp_err_in
        dummy_8u8,  # pos_err_in
        dummy_timing,  # timing_in
        dummy_gripper_in,  # gripper_in
    )

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

    # parol6/commands/cartesian_commands.py
    vel_lin = np.zeros(3, dtype=np.float64)
    vel_ang = np.zeros(3, dtype=np.float64)
    world_twist = np.zeros(6, dtype=np.float64)
    body_twist = np.zeros(6, dtype=np.float64)
    _apply_velocity_delta_wrf_jit(
        dummy_3x3,  # R
        dummy_6f,  # smoothed_vel
        0.004,  # dt
        dummy_4x4,  # current_pose
        vel_lin,  # vel_lin
        vel_ang,  # vel_ang
        world_twist,  # world_twist
        delta_4x4,  # delta
        dummy_4x4_out,  # out
        omega_ws,  # omega_ws
        R_ws,  # R_ws
        V_ws,  # V_ws
    )
    _apply_velocity_delta_trf_jit(
        dummy_6f,  # smoothed_vel
        0.004,  # dt
        dummy_4x4,  # current_pose
        body_twist,  # body_twist
        delta_4x4,  # delta
        dummy_4x4_out,  # out
        omega_ws,  # omega_ws
        R_ws,  # R_ws
        V_ws,  # V_ws
    )
    _max_vel_ratio_jit(dummy_6f, dummy_6f)

    elapsed = time.perf_counter() - start
    logger.info(f"\tJIT warmup completed in {elapsed * 1000:.1f}ms")
    return elapsed
