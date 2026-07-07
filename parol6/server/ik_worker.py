"""
IK Worker subprocess.

This module runs the computationally expensive IK enablement calculations
in a separate process, communicating with the main process via shared memory.
"""

import logging
import signal
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from pinokin import se3_from_trans, se3_mul, se3_rx, se3_ry, se3_rz

from parol6.commands._collision_guard import _ESCAPE_TOL
from parol6.server.ik_layout import (
    IK_INPUT_Q_OFFSET,
    IK_INPUT_T_OFFSET,
    IK_INPUT_TOOL_OFFSET,
    IK_OUTPUT_CART_TRF_OFFSET,
    IK_OUTPUT_CART_WRF_OFFSET,
    IK_OUTPUT_JOINT_OFFSET,
    IK_OUTPUT_VERSION_OFFSET,
    SHM_EXTRA_KWARGS,
    unregister_shm,
)

if TYPE_CHECKING:
    import multiprocessing

logger = logging.getLogger(__name__)

# Directional collision enablement: when the arm is within _ENABLE_NEAR_M of
# collision, grey a direction whose _ENABLE_STEP_RAD step would collide.
_ENABLE_STEP_RAD = np.radians(2.0)
_ENABLE_NEAR_M = 0.05


@dataclass(frozen=True)
class SyncTool:
    """Sync the worker checker's tool geometry (mirrors the planner's SyncTool)."""

    tool_name: str
    variant_key: str = ""


@dataclass(frozen=True)
class SyncShapes:
    """Sync the worker checker's program-layer shapes (waldoctl Shape tuple)."""

    shapes: tuple


def _drain_sync(cmd_queue: "multiprocessing.Queue", robot_module) -> bool:
    """Apply queued geometry syncs to this process's checker; True if any applied.

    A failing sync (missing tool mesh, plugin import quirk in this spawned
    process) is logged and skipped — it must never kill the worker, which
    would freeze enablement for the rest of the session.
    """
    applied = False
    while True:
        try:
            msg = cmd_queue.get_nowait()
        except Empty:
            return applied
        try:
            if isinstance(msg, SyncTool):
                robot_module.apply_tool(msg.tool_name, variant_key=msg.variant_key)
                applied = True
            elif isinstance(msg, SyncShapes):
                robot_module.apply_shapes(msg.shapes)
                applied = True
        except Exception:
            logger.exception("IK worker geometry sync failed for %r", msg)


def ik_enablement_worker_main(
    input_shm_name: str,
    output_shm_name: str,
    shutdown_event: Event,
    request_event: Event,
    command_queue: "multiprocessing.Queue",
) -> None:
    """
    Main entry point for IK enablement worker subprocess.

    This worker waits for request signals, computes joint and cartesian
    enablement, and writes results to the output shared memory. Geometry
    syncs (tool / keep-out shapes) arrive via ``command_queue`` and trigger
    a recompute at the last requested pose so enablement never goes stale
    while the arm is stationary.

    Args:
        input_shm_name: Name of input shared memory segment
        output_shm_name: Name of output shared memory segment
        shutdown_event: Event to signal shutdown
        request_event: Event signaled when new request is available
        command_queue: SyncTool / SyncShapes geometry sync messages
    """
    # Ignore SIGINT in worker - main process handles shutdown via shutdown_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from parol6.server import set_pdeathsig

    set_pdeathsig()

    # Attach to shared memory
    input_shm = SharedMemory(name=input_shm_name, create=False, **SHM_EXTRA_KWARGS)
    output_shm = SharedMemory(name=output_shm_name, create=False, **SHM_EXTRA_KWARGS)
    unregister_shm(input_shm)
    unregister_shm(output_shm)
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

    # Tool transform view for detecting tool changes
    tool_T_flat = np.frombuffer(
        input_shm.buf, dtype=np.float64, count=16, offset=IK_INPUT_TOOL_OFFSET
    )
    tool_T = tool_T_flat.reshape(4, 4)
    last_tool_T = np.eye(4)
    _eye4 = np.eye(4)

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
    from parol6.tools import register_plugin_tools
    from parol6.utils.ik import solve_ik

    # Spawn-mode subprocess: the registry is freshly imported with only native
    # tools; SyncTool of a plugin tool needs them registered here too.
    register_plugin_tools()

    robot = PAROL6_ROBOT.robot
    qlim = robot.qlim
    qlim_rows = (
        (
            np.ascontiguousarray(qlim[0], dtype=np.float64),
            np.ascontiguousarray(qlim[1], dtype=np.float64),
        )
        if qlim is not None
        else None
    )

    response_version = 0
    have_request = False

    # Pre-allocate work arrays for cartesian targets + the joint-gate step.
    cart_targets = np.zeros((12, 4, 4), dtype=np.float64)
    q_step = np.zeros(6, dtype=np.float64)

    logger.debug("IK worker subprocess started")

    try:
        while not shutdown_event.is_set():
            signaled = request_event.wait(timeout=0.1)
            # A geometry sync recomputes at the last requested pose so greying
            # reflects a shape/tool change even while the arm is stationary.
            synced = _drain_sync(command_queue, PAROL6_ROBOT)
            if not signaled and not (synced and have_request):
                continue
            if signaled:
                request_event.clear()
                have_request = True

            # Apply tool transform if it changed since last request
            if not np.array_equal(tool_T, last_tool_T):
                last_tool_T[:] = tool_T
                if np.allclose(tool_T, _eye4):
                    robot.clear_tool_transform()
                else:
                    robot.set_tool_transform(tool_T.copy())
                logger.info("IK worker: tool transform updated")

            # Input data is already available via views (q_rad, T_matrix)

            # Compute joint enablement, then gate directions whose small step
            # would collide (self + tool + shapes; geometry kept in sync above).
            if qlim is not None:
                _compute_joint_enable(q_rad, qlim, joint_en)
            # else: joint_en stays all ones (pre-allocated default)
            checker = PAROL6_ROBOT.collision
            gate_joint_enable_collision(
                checker, q_rad, joint_en, q_step, qlim=qlim_rows
            )

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
                checker=checker,
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
                checker=checker,
            )

            # Output data written directly via views, just update version
            response_version += 1
            version_view[0] = response_version

    except (EOFError, OSError, BrokenPipeError, KeyboardInterrupt):
        # Expected when the parent shuts down: shared memory or the request
        # event get torn down before our shutdown_event check fires.
        pass
    except Exception as e:
        logger.exception("IK worker subprocess error: %s", e)
    finally:
        # Release numpy views before closing shared memory
        del q_rad, T_flat, T_matrix, tool_T_flat, tool_T
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
        logger.debug("IK worker subprocess exiting")


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


def gate_joint_enable_collision(checker, q_rad, joint_en, q_step, qlim=None) -> None:
    """Clear a joint direction in ``joint_en`` whose ``_ENABLE_STEP_RAD`` step
    collides — self, tool, or keep-out shape. Proximity-gated (skip the
    per-direction checks when the arm is farther than ``_ENABLE_NEAR_M`` from
    collision) so it stays cheap. When the arm is ALREADY colliding (a keep-out
    placed over it), escaping directions stay enabled — grey those that go
    deeper or contact anything new — mirroring the jog/planner escape
    semantics; otherwise every button would grey with no way out. ``qlim`` is
    optional ``(low, high)`` rows: the probe is clamped so a pose past the
    mechanical stop (which the jog itself can never reach) can't grey the
    button. Not njit — it calls the C++ checker.
    """
    if checker is None:
        return
    md_now = checker.min_distance(q_rad)
    if md_now >= _ENABLE_NEAR_M:
        return
    inside = checker.in_collision(q_rad)
    pairs_now = set(checker.colliding_pairs(q_rad)) if inside else None
    for j in range(6):
        for slot, sign in ((2 * j, 1.0), (2 * j + 1, -1.0)):
            if joint_en[slot]:
                q_step[:] = q_rad
                q_step[j] += sign * _ENABLE_STEP_RAD
                if qlim is not None:
                    if q_step[j] < qlim[0][j]:
                        q_step[j] = qlim[0][j]
                    elif q_step[j] > qlim[1][j]:
                        q_step[j] = qlim[1][j]
                if pairs_now is not None:
                    if set(checker.colliding_pairs(q_step)) - pairs_now or (
                        checker.min_distance(q_step) < md_now - _ESCAPE_TOL
                    ):
                        joint_en[slot] = 0
                elif checker.in_collision(q_step):
                    joint_en[slot] = 0


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
    checker=None,
) -> None:
    """
    Compute per-axis +/- enable bits for the given frame via small-step IK.

    A direction is disabled when IK fails or (near collision, like the joint
    gate) its solved config would collide — self, tool, or keep-out shape.

    Writes to out array (12 elements) in axis order:
    X+, X-, Y+, Y-, Z+, Z-, RX+, RX-, RY+, RY-, RZ+, RZ-
    """
    t_step = delta_mm / 1000.0
    r_step = np.radians(delta_deg)

    # Compute all 12 target poses in one numba call
    _compute_target_poses(T, t_step, r_step, is_wrf, axis_dirs, targets)

    md_now = checker.min_distance(q_rad) if checker is not None else np.inf
    near = md_now < _ENABLE_NEAR_M
    # Already colliding: keep escaping directions enabled (see the joint gate).
    inside = near and checker is not None and checker.in_collision(q_rad)
    pairs_now = set(checker.colliding_pairs(q_rad)) if inside else None

    # Check IK (and, when near collision, the solved config) for each target
    for i in range(12):
        try:
            ik = solve_ik(robot, targets[i], q_rad, quiet_logging=True)
            ok = bool(ik.success)
            if ok and near and checker is not None:
                if pairs_now is not None:
                    ok = checker.min_distance(ik.q) >= md_now - _ESCAPE_TOL and not (
                        set(checker.colliding_pairs(ik.q)) - pairs_now
                    )
                elif checker.in_collision(ik.q):
                    ok = False
            out[i] = 1 if ok else 0
        except Exception:
            out[i] = 0
