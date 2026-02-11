"""
IK Worker subprocess.

This module runs the computationally expensive IK enablement calculations
in a separate process, communicating with the main process via shared memory.
"""

import logging
import signal
import sys
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from pinokin import se3_from_trans, se3_mul, se3_rx, se3_ry, se3_rz

from parol6.server.ik_layout import (
    IK_INPUT_Q_OFFSET,
    IK_INPUT_T_OFFSET,
    IK_OUTPUT_CART_TRF_OFFSET,
    IK_OUTPUT_CART_WRF_OFFSET,
    IK_OUTPUT_JOINT_OFFSET,
    IK_OUTPUT_VERSION_OFFSET,
)

logger = logging.getLogger(__name__)

# track parameter added in Python 3.13
_SHM_EXTRA_KWARGS = {"track": False} if sys.version_info >= (3, 13) else {}


def ik_enablement_worker_main(
    input_shm_name: str,
    output_shm_name: str,
    shutdown_event: Event,
    request_event: Event,
) -> None:
    """
    Main entry point for IK enablement worker subprocess.

    This worker waits for request signals, computes joint and cartesian
    enablement, and writes results to the output shared memory.

    Args:
        input_shm_name: Name of input shared memory segment
        output_shm_name: Name of output shared memory segment
        shutdown_event: Event to signal shutdown
        request_event: Event signaled when new request is available
    """
    # Ignore SIGINT in worker - main process handles shutdown via shutdown_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Attach to shared memory
    input_shm = SharedMemory(name=input_shm_name, create=False, **_SHM_EXTRA_KWARGS)
    output_shm = SharedMemory(name=output_shm_name, create=False, **_SHM_EXTRA_KWARGS)
    assert input_shm.buf is not None
    assert output_shm.buf is not None
    input_mv = memoryview(input_shm.buf)
    output_mv = memoryview(output_shm.buf)

    # Zero-alloc input views: read directly from shared memory
    q_rad = np.frombuffer(
        input_shm.buf, dtype=np.float64, count=6, offset=IK_INPUT_Q_OFFSET
    )
    T_flat = np.frombuffer(
        input_shm.buf, dtype=np.float64, count=16, offset=IK_INPUT_T_OFFSET
    )
    T_matrix = T_flat.reshape((4, 4))  # View, no copy

    # Zero-alloc output views: write directly to shared memory
    joint_en = np.frombuffer(
        output_shm.buf, dtype=np.uint8, count=12, offset=IK_OUTPUT_JOINT_OFFSET
    )
    cart_en_wrf = np.frombuffer(
        output_shm.buf, dtype=np.uint8, count=12, offset=IK_OUTPUT_CART_WRF_OFFSET
    )
    cart_en_trf = np.frombuffer(
        output_shm.buf, dtype=np.uint8, count=12, offset=IK_OUTPUT_CART_TRF_OFFSET
    )
    version_view = np.frombuffer(
        output_shm.buf, dtype=np.uint64, count=1, offset=IK_OUTPUT_VERSION_OFFSET
    )

    # Initialize outputs
    joint_en[:] = 1
    cart_en_wrf[:] = 1
    cart_en_trf[:] = 1

    # Initialize robot model in this process
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT
    from parol6.utils.ik import solve_ik

    robot = PAROL6_ROBOT.robot
    qlim = robot.qlim

    response_version = 0

    # Pre-allocate work array for cartesian targets
    cart_targets = np.zeros((12, 4, 4), dtype=np.float64)

    logger.info("IK worker subprocess started")

    try:
        while not shutdown_event.is_set():
            # Wait for request signal or timeout for shutdown check
            if not request_event.wait(timeout=0.1):
                continue  # Timeout - loop back to check shutdown

            request_event.clear()

            # Input data is already available via views (q_rad, T_matrix)

            # Compute joint enablement
            if qlim is not None:
                _compute_joint_enable(q_rad, qlim, joint_en)
            # else: joint_en stays all ones (pre-allocated default)

            # Compute cartesian enablement for both frames
            _compute_cart_enable(
                T_matrix,
                True,
                q_rad,
                robot,
                solve_ik,
                _AXIS_DIRS,
                cart_targets,
                cart_en_wrf,
            )
            _compute_cart_enable(
                T_matrix,
                False,
                q_rad,
                robot,
                solve_ik,
                _AXIS_DIRS,
                cart_targets,
                cart_en_trf,
            )

            # Output data written directly via views, just update version
            response_version += 1
            version_view[0] = response_version

    except Exception as e:
        logger.exception("IK worker subprocess error: %s", e)
    finally:
        # Release numpy views before closing shared memory
        del q_rad, T_flat, T_matrix
        del joint_en, cart_en_wrf, cart_en_trf, version_view

        # Release memoryviews
        try:
            input_mv.release()
        except BufferError:
            pass
        try:
            output_mv.release()
        except BufferError:
            pass

        input_shm.close()
        output_shm.close()
        logger.info("IK worker subprocess exiting")


@njit(cache=True)
def _compute_joint_enable(
    q_rad: np.ndarray,
    qlim: np.ndarray,
    out: np.ndarray,
    delta_rad: float = np.radians(0.2),
) -> None:
    """
    Compute per-joint +/- enable bits based on joint limits and a small delta.

    Writes to out array (12 elements): [J1+, J1-, J2+, J2-, ..., J6+, J6-]
    """
    for i in range(6):
        out[i * 2] = 1 if (q_rad[i] + delta_rad) <= qlim[1, i] else 0
        out[i * 2 + 1] = 1 if (q_rad[i] - delta_rad) >= qlim[0, i] else 0


# Axis directions: [dx, dy, dz, rx, ry, rz] for each of 12 axes
# Order: X+, X-, Y+, Y-, Z+, Z-, RX+, RX-, RY+, RY-, RZ+, RZ-
_AXIS_DIRS = np.array(
    [
        [1, 0, 0, 0, 0, 0],  # X+
        [-1, 0, 0, 0, 0, 0],  # X-
        [0, 1, 0, 0, 0, 0],  # Y+
        [0, -1, 0, 0, 0, 0],  # Y-
        [0, 0, 1, 0, 0, 0],  # Z+
        [0, 0, -1, 0, 0, 0],  # Z-
        [0, 0, 0, 1, 0, 0],  # RX+
        [0, 0, 0, -1, 0, 0],  # RX-
        [0, 0, 0, 0, 1, 0],  # RY+
        [0, 0, 0, 0, -1, 0],  # RY-
        [0, 0, 0, 0, 0, 1],  # RZ+
        [0, 0, 0, 0, 0, -1],  # RZ-
    ],
    dtype=np.float64,
)


@njit(cache=True)
def _compute_target_poses(
    T: np.ndarray,
    t_step: float,
    r_step: float,
    is_wrf: bool,
    axis_dirs: np.ndarray,
    targets: np.ndarray,
) -> None:
    """
    Compute 12 target poses for cartesian enablement checking.

    Args:
        T: Current pose (4x4 matrix)
        t_step: Translation step in meters
        r_step: Rotation step in radians
        is_wrf: True for world reference frame, False for tool reference frame
        axis_dirs: (12, 6) array of axis directions [dx, dy, dz, rx, ry, rz]
        targets: Output array (12, 4, 4) for target poses
    """
    dT = np.zeros((4, 4), dtype=np.float64)

    for i in range(12):
        d = axis_dirs[i]
        dx, dy, dz = d[0] * t_step, d[1] * t_step, d[2] * t_step
        rx, ry, rz = d[3] * r_step, d[4] * r_step, d[5] * r_step

        # Build delta transform (only one of trans/rot is non-zero per axis)
        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            se3_from_trans(dx, dy, dz, dT)
        elif rx != 0.0:
            se3_rx(rx, dT)
        elif ry != 0.0:
            se3_ry(ry, dT)
        elif rz != 0.0:
            se3_rz(rz, dT)

        # Apply in specified frame
        if is_wrf:
            se3_mul(dT, T, targets[i])
        else:
            se3_mul(T, dT, targets[i])


def _compute_cart_enable(
    T: np.ndarray,
    is_wrf: bool,
    q_rad: np.ndarray,
    robot,
    solve_ik,
    axis_dirs: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
    delta_mm: float = 0.5,
    delta_deg: float = 0.5,
) -> None:
    """
    Compute per-axis +/- enable bits for the given frame via small-step IK.

    Writes to out array (12 elements) in axis order:
    X+, X-, Y+, Y-, Z+, Z-, RX+, RX-, RY+, RY-, RZ+, RZ-
    """
    t_step = delta_mm / 1000.0
    r_step = np.radians(delta_deg)

    # Compute all 12 target poses in one numba call
    _compute_target_poses(T, t_step, r_step, is_wrf, axis_dirs, targets)

    # Check IK for each target
    for i in range(12):
        try:
            ik = solve_ik(robot, targets[i], q_rad, quiet_logging=True)
            out[i] = 1 if ik.success else 0
        except Exception:
            out[i] = 0
