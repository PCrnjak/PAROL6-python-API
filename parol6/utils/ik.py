"""
IK Helper Functions and Utilities
Shared functions used by multiple command classes for inverse kinematics calculations.
"""

import atexit
import logging
import time
from dataclasses import dataclass

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from pinokin import Damping as _Damping, IKSolver as _IKSolver, Robot

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import IK_SAFETY_MARGINS_RAD

logger = logging.getLogger(__name__)

# Rate limiting for IK warnings (avoid log spam at 250Hz)
_ik_last_warn_time: float = 0.0
_IK_WARN_INTERVAL: float = 1.0  # Log at most once per second


def _rate_limited_warning(msg: str) -> None:
    """Log a warning with rate limiting to avoid spam."""
    global _ik_last_warn_time
    now = time.monotonic()
    if now - _ik_last_warn_time > _IK_WARN_INTERVAL:
        logger.warning(msg)
        _ik_last_warn_time = now


@dataclass
class SolveIKResult:
    """IK result with safety violation tracking."""

    q: NDArray[np.float64]
    success: bool
    iterations: int
    residual: float
    violations: str | None = None


# Cached solver state — invalidated when robot instance changes (e.g., tool change)
_cached_robot_id: int = -1
_cached_qlim: NDArray[np.float64] | None = None
_cached_solver: _IKSolver | None = None
_cached_buffered_min: NDArray[np.float64] | None = None
_cached_buffered_max: NDArray[np.float64] | None = None
_cached_qlim_min: NDArray[np.float64] | None = None
_cached_qlim_max: NDArray[np.float64] | None = None
_cached_result: SolveIKResult | None = None


@atexit.register
def _cleanup_ik_cache() -> None:
    global _cached_solver
    _cached_solver = None


def _ensure_cache(robot: Robot) -> None:
    """Rebuild cache if robot instance changed."""
    global _cached_robot_id, _cached_qlim, _cached_solver
    global _cached_buffered_min, _cached_buffered_max, _cached_result
    global _cached_qlim_min, _cached_qlim_max

    robot_id = id(robot)
    if _cached_robot_id == robot_id:
        return

    qlim = robot.qlim
    _cached_robot_id = robot_id
    _cached_qlim = qlim
    _cached_qlim_min = np.ascontiguousarray(qlim[0, :])
    _cached_qlim_max = np.ascontiguousarray(qlim[1, :])
    _cached_solver = _IKSolver(
        robot,
        damping=_Damping.Sugihara,
        tol=1e-12,
        lm_lambda=0.0,
        max_iter=10,
        max_restarts=10,
    )
    _cached_buffered_min = qlim[0, :] + IK_SAFETY_MARGINS_RAD[:, 0]
    _cached_buffered_max = qlim[1, :] - IK_SAFETY_MARGINS_RAD[:, 1]
    _cached_result = SolveIKResult(
        q=np.zeros(qlim.shape[1], dtype=np.float64),
        success=False,
        iterations=0,
        residual=0.0,
    )


# Unsigned axis vectors — direction comes from signed speed.
# Format: ([x, y, z], [rx, ry, rz])
AXIS_MAP = {
    "X": ([1, 0, 0], [0, 0, 0]),
    "Y": ([0, 1, 0], [0, 0, 0]),
    "Z": ([0, 0, 1], [0, 0, 0]),
    "RX": ([0, 0, 0], [1, 0, 0]),
    "RY": ([0, 0, 0], [0, 1, 0]),
    "RZ": ([0, 0, 0], [0, 0, 1]),
}


@njit(cache=True)
def _ik_safety_check(
    q: NDArray[np.float64],
    cq: NDArray[np.float64],
    buffered_min: NDArray[np.float64],
    buffered_max: NDArray[np.float64],
    qlim_min: NDArray[np.float64],
    qlim_max: NDArray[np.float64],
) -> tuple[bool, bool, int]:
    """
    JIT-compiled IK safety validation (zero-allocation scalar loop).
    Returns (ok, is_recovery_violation, violation_idx).
    """
    recovery_idx = -1
    safety_idx = -1
    for i in range(q.shape[0]):
        in_danger = cq[i] < buffered_min[i] or cq[i] > buffered_max[i]
        if in_danger:
            d_old = min(abs(cq[i] - qlim_min[i]), abs(cq[i] - qlim_max[i]))
            d_new = min(abs(q[i] - qlim_min[i]), abs(q[i] - qlim_max[i]))
            if d_new < d_old and recovery_idx < 0:
                recovery_idx = i
        else:
            if (q[i] < buffered_min[i] or q[i] > buffered_max[i]) and safety_idx < 0:
                safety_idx = i
    if recovery_idx >= 0:
        return False, True, recovery_idx
    if safety_idx >= 0:
        return False, False, safety_idx
    return True, False, -1


def solve_ik(
    robot: Robot,
    target_pose: NDArray[np.float64],
    current_q: NDArray[np.float64],
    quiet_logging: bool = False,
) -> SolveIKResult:
    """
    IK solver with per-joint safety margins.

    Per-joint safety margins (IK_SAFETY_MARGINS_RAD) are always enforced.
    Joints like J3 (elbow) have larger margins because backwards bend creates
    trap configurations that are hard to recover from.

    Parameters
    ----------
    robot : Robot
        pinokin Robot model
    target_pose : NDArray[np.float64]
        Target pose as 4x4 SE3 transformation matrix
    current_q : NDArray[np.float64]
        Current joint configuration in radians
    quiet_logging : bool, optional
        If True, suppress warning logs (default: False)

    Returns
    -------
    SolveIKResult
        success - True if solution found
        q - Joint configuration in radians
        iterations - Number of iterations used
        residual - Final error value
        violations - Error message if failed, None if successful
    """
    _ensure_cache(robot)
    assert _cached_solver is not None
    assert _cached_result is not None
    solver = _cached_solver
    result = _cached_result

    solver.solve(target_pose, q0=current_q)

    # Write into pre-allocated result buffer (zero-allocation)
    result.q[:] = solver.q
    result.success = solver.success
    result.iterations = solver.iterations
    result.residual = solver.residual
    result.violations = None

    if result.success:
        # JIT-compiled safety validation
        ok, is_recovery, idx = _ik_safety_check(
            result.q,
            current_q,
            _cached_buffered_min,
            _cached_buffered_max,  # type: ignore[arg-type]
            _cached_qlim_min,
            _cached_qlim_max,  # type: ignore[arg-type]
        )
        if not ok:
            result.success = False
            if is_recovery:
                result.violations = (
                    f"J{idx + 1} moving further into danger zone (recovery blocked)"
                )
            else:
                result.violations = (
                    f"J{idx + 1} would leave safe zone (buffer violated)"
                )
            if not quiet_logging:
                _rate_limited_warning(result.violations)

    if result.success:
        if not check_limits(
            current_q, result.q, allow_recovery=True, log=not quiet_logging
        ):
            result.success = False
    else:
        if result.violations is None:
            result.violations = "IK failed to solve."

    return result


# -----------------------------
# Fast, vectorized limit checking with edge-triggered logging
# -----------------------------
# Pre-allocated buffers for check_limits (avoid per-call allocation)
_cl_viol = np.zeros(6, dtype=np.bool_)
_cl_below = np.zeros(6, dtype=np.bool_)
_cl_above = np.zeros(6, dtype=np.bool_)
_cl_cur_viol = np.zeros(6, dtype=np.bool_)
_cl_t_below = np.zeros(6, dtype=np.bool_)
_cl_t_above = np.zeros(6, dtype=np.bool_)
_cl_dummy_target = np.zeros(6, dtype=np.float64)
_cl_mn = np.ascontiguousarray(PAROL6_ROBOT._joint_limits_radian[:, 0])
_cl_mx = np.ascontiguousarray(PAROL6_ROBOT._joint_limits_radian[:, 1])
_last_violation_mask = np.zeros(6, dtype=np.bool_)
_last_any_violation = False


@njit(cache=True)
def _check_limits_core(
    q_arr: NDArray[np.float64],
    t_arr: NDArray[np.float64],
    mn: NDArray[np.float64],
    mx: NDArray[np.float64],
    allow_recovery: bool,
    has_target: bool,
    viol_out: NDArray[np.bool_],
    below_out: NDArray[np.bool_],
    above_out: NDArray[np.bool_],
    cur_viol_out: NDArray[np.bool_],
    t_below_out: NDArray[np.bool_],
    t_above_out: NDArray[np.bool_],
) -> bool:
    """JIT-compiled core of check_limits. Writes masks to output buffers."""
    for i in range(6):
        below_out[i] = q_arr[i] < mn[i]
        above_out[i] = q_arr[i] > mx[i]
        cur_viol_out[i] = below_out[i] or above_out[i]

    if not has_target:
        all_ok = True
        for i in range(6):
            viol_out[i] = cur_viol_out[i]
            if viol_out[i]:
                all_ok = False
        return all_ok

    for i in range(6):
        t_below_out[i] = t_arr[i] < mn[i]
        t_above_out[i] = t_arr[i] > mx[i]

    all_ok = True
    for i in range(6):
        t_viol = t_below_out[i] or t_above_out[i]
        if allow_recovery:
            rec_ok = (above_out[i] and t_arr[i] <= q_arr[i]) or (
                below_out[i] and t_arr[i] >= q_arr[i]
            )
        else:
            rec_ok = False
        ok = (not cur_viol_out[i] and not t_viol) or (cur_viol_out[i] and rec_ok)
        viol_out[i] = not ok
        if not ok:
            all_ok = False

    return all_ok


def check_limits(
    q: ArrayLike,
    target_q: ArrayLike | None = None,
    allow_recovery: bool = True,
    *,
    log: bool = True,
) -> bool:
    """
    Vectorized limits check in radians.
    - q: current joint angles in radians (array-like)
    - target_q: optional target joint angles in radians (array-like)
    - allow_recovery: allow movement that heads back toward valid range if currently violating
    - log: emit edge-triggered warning/info logs on violation state changes

    Returns True if move is allowed (within limits or valid recovery), False otherwise.
    """
    global _last_any_violation

    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    has_target = target_q is not None
    t_arr = (
        np.asarray(target_q, dtype=np.float64).reshape(-1)
        if has_target
        else _cl_dummy_target
    )

    all_ok = _check_limits_core(
        q_arr,
        t_arr,
        _cl_mn,
        _cl_mx,
        allow_recovery,
        has_target,
        _cl_viol,
        _cl_below,
        _cl_above,
        _cl_cur_viol,
        _cl_t_below,
        _cl_t_above,
    )

    if log:
        any_viol = not all_ok

        # Edge-triggered violation logs
        if any_viol and (
            np.any(_cl_viol != _last_violation_mask) or not _last_any_violation
        ):
            idxs = np.where(_cl_viol)[0]
            tokens = []
            for i in idxs:
                if _cl_cur_viol[i]:
                    tokens.append(
                        f"J{i + 1}:" + ("cur<min" if _cl_below[i] else "cur>max")
                    )
                elif has_target and _cl_t_below[i]:
                    tokens.append(f"J{i + 1}:target<min")
                elif has_target and _cl_t_above[i]:
                    tokens.append(f"J{i + 1}:target>max")
                else:
                    tokens.append(f"J{i + 1}:violation")
            logger.warning("LIMIT VIOLATION: %s", " ".join(tokens))
        elif (not any_viol) and _last_any_violation:
            logger.info("Limits back in range")

        _last_violation_mask[:] = _cl_viol
        _last_any_violation = any_viol

    return all_ok
