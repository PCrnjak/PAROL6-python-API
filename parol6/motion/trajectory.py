"""
Unified trajectory generation pipeline using TOPP-RA for time-optimal path parameterization.

This module provides the shared trajectory infrastructure for all motion commands.
Path generation (geometry) stays in command files; this handles time parameterization.

Pipeline:
  1. Command generates Cartesian poses (for cart commands) or joint targets
  2. JointPath holds uniformly-sampled joint positions
  3. TrajectoryBuilder applies TOPP-RA + motion profile to produce Trajectory
  4. Trajectory contains motor steps ready for tick-by-tick execution
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.interpolate import PPoly
from ruckig import InputParameter, OutputParameter, Result, Ruckig  # type: ignore[unresolved-import, ty:unresolved-import]

import toppra.algorithm as algo
import toppra.constraint as constraint
from toppra.parametrizer import ParametrizeConstAccel

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import INTERVAL_S, LIMITS, rad_to_steps
from parol6.motion.geometry import _rotation_angle, cartesian_path_knots
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError
from parol6.utils.joint_limits import HALF_STEP_RAD, TRAVEL_MAX_RAD, TRAVEL_MIN_RAD


from pinokin import Damping, IKSolver, se3_interp


logger = logging.getLogger(__name__)


def _rad_to_steps_alloc(rad: NDArray) -> NDArray[np.int32]:
    """Convert radians to steps, allocating output. For planning phase only."""
    out = np.zeros(rad.shape, dtype=np.int32)
    if rad.ndim == 1:
        rad_to_steps(rad, out)
    else:
        for i in range(rad.shape[0]):
            rad_to_steps(rad[i], out[i])
    return out


def _trapezoid_duration(distance: float, v_max: float, a_max: float) -> float:
    """Duration of a trapezoidal profile over ``distance``, starting and ending at rest."""
    distance = abs(distance)
    if distance < 1e-12:
        return 0.0
    if distance * a_max >= v_max * v_max:
        return v_max / a_max + distance / v_max
    return 2.0 * float(np.sqrt(distance / a_max))


def _trapezoid_samples(
    times: NDArray[np.float64], q0: float, q1: float, v_max: float, a_max: float
) -> NDArray[np.float64]:
    """Sample a trapezoidal profile from ``q0`` to ``q1``, starting and ending at rest."""
    distance = abs(q1 - q0)
    duration = _trapezoid_duration(distance, v_max, a_max)
    if duration <= 0.0:
        return np.full(times.shape, q0, dtype=np.float64)

    if distance * a_max >= v_max * v_max:
        t_accel = v_max / a_max
        v_peak = v_max
    else:
        t_accel = duration / 2.0
        v_peak = a_max * t_accel

    t = np.clip(times, 0.0, duration)
    t_decel = duration - t_accel
    travelled = np.where(
        t < t_accel,
        0.5 * a_max * t * t,
        np.where(
            t < t_decel,
            0.5 * v_peak * t_accel + v_peak * (t - t_accel),
            distance - 0.5 * a_max * (duration - t) ** 2,
        ),
    )
    return q0 + np.sign(q1 - q0) * travelled


def _quintic_samples(
    times: NDArray[np.float64], q0: float, q1: float, duration: float
) -> NDArray[np.float64]:
    """Sample a quintic profile from ``q0`` to ``q1``, at rest and unaccelerated at both ends."""
    s = np.clip(times / duration, 0.0, 1.0)
    return q0 + (q1 - q0) * s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


# Peak ds/dt and d²s/dt² of a quintic from s=0 to s=1 over one second.
_QUINTIC_PEAK_VEL = 1.875
_QUINTIC_PEAK_ACC = 10.0 / math.sqrt(3.0)

# A timing solver's duration can land a hair past a whole number of ticks.
_TICK_ROUNDING = 1e-6


def _tick_intervals(duration: float, dt: float) -> int:
    """Control ticks a move of ``duration`` spans, rounded up to a whole
    number. The player plays one row per tick, so a trajectory's rows sit
    exactly a tick apart; stretched to whole ticks, it lasts as long as
    it plans and accelerates no harder."""
    return max(1, math.ceil(duration / dt - _TICK_ROUNDING))


# Relative slack on a segment rate: below it a rate is taken as settled.
_RATE_TOLERANCE = 1e-9


def _slowed(rows: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """Uniformly stretch tick-spaced rows without increasing their played limits."""
    return JointPath(positions=rows).sample_many(np.linspace(0.0, 1.0, count))


def _distinct(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    # Seidel treats intervals below 1e-10 as singular.
    return np.concatenate(([True], np.diff(values) > 1e-9))


def _resample(
    rows: NDArray[np.float64], knots: NDArray[np.float64], at: NDArray[np.float64]
) -> NDArray[np.float64]:
    indices = np.clip(np.searchsorted(knots, at, side="right") - 1, 0, len(knots) - 2)
    fraction = ((at - knots[indices]) / (knots[indices + 1] - knots[indices]))[:, None]
    return rows[indices] + fraction * (rows[indices + 1] - rows[indices])


@njit(cache=True)
def _segment_rates(
    deltas: NDArray[np.float64],
    v_max: NDArray[np.float64],
    dv_max: NDArray[np.float64],
    rate_max: float,
) -> NDArray[np.float64]:
    """The fastest rate (one over its time) each segment between rows can
    be run at: no faster than ``rate_max``, every joint under ``v_max``,
    and every joint's velocity changing by at most ``dv_max`` from one
    segment to the next, from rest into the first and into rest out of the
    last.

    A joint that its two neighbouring segments move in opposite directions
    turns at the row between them, and each side takes half the change.
    Every other bound caps a segment's rate by its own limits or by a
    rising function of a neighbour's rate, so lowering the rates until no
    bound is broken reaches the fastest timing that breaks none: a forward
    pass for a joint speeding up into a segment, a backward pass for one
    slowing into the next, repeated while either still lowers a rate.
    """
    padded = np.zeros((len(deltas) + 2, deltas.shape[1]))
    padded[1:-1] = deltas
    deltas = padded
    n, dof = deltas.shape
    rates = np.full(n, rate_max)
    for i in range(n):
        for j in range(dof):
            d = abs(deltas[i, j])
            if d > 0.0:
                rates[i] = min(rates[i], v_max[j] / d)
    for i in range(n - 1):
        for j in range(dof):
            a = deltas[i, j]
            b = deltas[i + 1, j]
            if a * b < 0.0:
                rates[i] = min(rates[i], 0.5 * dv_max[j] / abs(a))
                rates[i + 1] = min(rates[i + 1], 0.5 * dv_max[j] / abs(b))
            elif a == 0.0 and b != 0.0:
                rates[i + 1] = min(rates[i + 1], dv_max[j] / abs(b))
            elif b == 0.0 and a != 0.0:
                rates[i] = min(rates[i], dv_max[j] / abs(a))
    lowered = True
    while lowered:
        lowered = False
        for i in range(n - 1):
            for j in range(dof):
                a = deltas[i, j]
                b = deltas[i + 1, j]
                if a * b > 0.0:
                    cap = (abs(a) * rates[i] + dv_max[j]) / abs(b)
                    if cap < rates[i + 1] * (1.0 - _RATE_TOLERANCE):
                        rates[i + 1] = cap
                        lowered = True
        for i in range(n - 2, -1, -1):
            for j in range(dof):
                a = deltas[i, j]
                b = deltas[i + 1, j]
                if a * b > 0.0:
                    cap = (abs(b) * rates[i + 1] + dv_max[j]) / abs(a)
                    if cap < rates[i] * (1.0 - _RATE_TOLERANCE):
                        rates[i] = cap
                        lowered = True
    return rates[1:-1]


class _BlendedPath:
    """C1 parabolic blends bounded by motor resolution and tick-sized corner cutting."""

    def __init__(
        self,
        knots: NDArray[np.float64],
        positions: NDArray[np.float64],
        a_max: NDArray[np.float64],
        dt: float,
    ) -> None:
        self.dof = positions.shape[1]
        self.path_interval = (float(knots[0]), float(knots[-1]))
        intervals = np.diff(knots)
        self.slopes = np.diff(positions, axis=0) / intervals[:, None]
        delta = np.diff(self.slopes, axis=0)
        magnitude = np.max(np.abs(delta), axis=1)
        corners = np.flatnonzero(magnitude > 1e-9)
        with np.errstate(divide="ignore"):
            # Playback itself cuts corners by up to a*dt²/4.
            tick_width = dt**2 * np.min(a_max / np.abs(delta[corners]), axis=1)
        half_width = np.minimum(
            0.5 * np.minimum(intervals[:-1], intervals[1:])[corners],
            np.maximum(
                4.0 * float(np.min(HALF_STEP_RAD)) / magnitude[corners], tick_width
            ),
        )
        starts = knots[corners + 1] - half_width
        ends = knots[corners + 1] + half_width
        unblended = np.delete(knots, corners + 1)
        breaks = np.unique(np.concatenate([unblended, starts, ends]))
        at = breaks[:-1]
        segments = np.clip(
            np.searchsorted(knots, at, side="right") - 1, 0, len(self.slopes) - 1
        )
        coefficients = np.zeros((3, len(at), self.dof))
        coefficients[1] = self.slopes[segments]
        coefficients[2] = (
            positions[segments]
            + self.slopes[segments] * (at - knots[segments])[:, None]
        )
        if len(corners):
            blends = np.maximum(np.searchsorted(starts, at, side="right") - 1, 0)
            inside = (at >= starts[blends]) & (at < ends[blends])
            selected = blends[inside]
            curvature = delta[corners[selected]] / (2.0 * half_width[selected, None])
            offset = (at[inside] - starts[selected])[:, None]
            slope = self.slopes[corners[selected]]
            position = (
                positions[corners[selected] + 1] - slope * half_width[selected, None]
            )
            coefficients[0, inside] = 0.5 * curvature
            coefficients[1, inside] = slope + offset * curvature
            coefficients[2, inside] = position + offset * (
                slope + 0.5 * offset * curvature
            )
        self._pp = PPoly(coefficients, breaks)

    @property
    def breaks(self) -> NDArray[np.float64]:
        return self._pp.x

    def __call__(self, s: float | NDArray, order: int = 0) -> NDArray[np.float64]:
        return self._pp(s, order)


class _PathSpeedConstraint(constraint.LinearConstraint):
    """A scalar path-speed ceiling expressed directly as a bound on squared speed."""

    def __init__(self, caps: float | NDArray[np.float64]) -> None:
        super().__init__()
        self.caps = caps

    def compute_constraint_params(self, path: _BlendedPath, gridpoints: NDArray):
        bounds = np.zeros((len(gridpoints), 2))
        bounds[:, 1] = np.square(self.caps)
        return None, None, None, None, None, None, bounds


class ProfileType(Enum):
    """Available trajectory profile types for motion planning."""

    TOPPRA = "toppra"  # Time-optimal path following (default)
    RUCKIG = "ruckig"  # Point-to-point jerk-limited (can't follow Cartesian paths)
    QUINTIC = "quintic"  # Quintic polynomial (C² smooth, predictable shape)
    TRAPEZOID = "trapezoid"  # Trapezoidal velocity profile
    LINEAR = "linear"  # Direct linear interpolation (no smoothing)

    @classmethod
    def from_string(cls, name: str) -> ProfileType:
        """Convert string to ProfileType, case-insensitive."""
        name_upper = name.upper()
        if name_upper == "NONE":
            return cls.LINEAR
        try:
            return cls[name_upper]
        except KeyError:
            logger.warning("Unknown profile type '%s', using TOPPRA", name)
            return cls.TOPPRA


# Largest joint change allowed between consecutive cartesian IK waypoints.
# A bigger step is solved again at poses in between; one that survives is
# the solver hopping to another IK branch, and the commanded path would
# whip the arm through the hop: the move is refused rather than the hop
# smoothed over (par6's `move_l_max_joint_step_rad`).
IK_MAX_JOINT_STEP_RAD: float = 0.35

# How far the tool may move while the wrist reconfigures at a singularity
# for the reconfiguration to count as one: with J5 at zero the J4/J6 pair
# is a null motion and the tool stands still; a hop that moves the tool
# more than this is a branch flip, and refused.
_WRIST_NULL_MOTION_POS_M: float = 5e-4
_WRIST_NULL_MOTION_ROT_RAD: float = np.radians(0.5)


def _ik_branch_hop(positions: NDArray[np.float64]) -> int | None:
    """Index of the first waypoint the chain reaches with a joint step past
    ``IK_MAX_JOINT_STEP_RAD``, or None when the chain stays on one branch."""
    if len(positions) < 2:
        return None
    steps = np.max(np.abs(np.diff(positions, axis=0)), axis=1)
    hops = np.nonzero(steps > IK_MAX_JOINT_STEP_RAD)[0]
    if len(hops) == 0:
        return None
    return int(hops[0]) + 1


# The turns of J4 tried, in order, when a chain has to leave a wrist
# singularity: the two quarter turns first, since a tilt out of the
# arm's plane is what a move from the standby pose usually asks for.
_WRIST_TURNS_RAD: tuple[float, ...] = (
    np.pi / 2,
    -np.pi / 2,
    np.pi / 4,
    -np.pi / 4,
    3 * np.pi / 4,
    -3 * np.pi / 4,
    np.pi,
)


def _wrist_turn(
    q_from: NDArray[np.float64], turn: float, pose_from: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """``q_from`` with its wrist turned by ``turn``: J4 forward, J6 back.
    None when the turn leaves the joint window or moves the tool, which it
    does not at the singularity and does a little near it."""
    bridge = q_from.copy()
    bridge[3] += turn
    bridge[5] -= turn
    if np.any(bridge < TRAVEL_MIN_RAD) or np.any(bridge > TRAVEL_MAX_RAD):
        return None
    robot = PAROL6_ROBOT.robot
    for frac in (0.5, 1.0):
        pose = robot.fkine(q_from + frac * (bridge - q_from))
        if np.linalg.norm(pose[:3, 3] - pose_from[:3, 3]) > _WRIST_NULL_MOTION_POS_M:
            return None
        if _rotation_angle(pose_from, pose) > _WRIST_NULL_MOTION_ROT_RAD:
            return None
    return bridge


# Largest joint step between the rows of a wrist turn: the collision guard
# checks rows, so the turn carries enough of them to be checked along its
# sweep and not only at its ends.
_WRIST_TURN_ROW_RAD = np.radians(5.0)


def _leave_wrist_singularity(
    solver: IKSolver,
    se3_poses: list[NDArray[np.float64]],
    q_from: NDArray[np.float64],
    q_hint: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], int] | None:
    """The joint chain for ``se3_poses`` from a wrist standing at its
    singularity, led by the turn of the wrist the chain needs, and the
    number of rows the turn adds ahead of the path; None when no turn
    gives one.

    With J5 at zero, J4 and J6 share an axis: turning J4 by an angle and J6
    back by the same angle leaves the tool where it is. A pose a hair off
    the singularity fixes the split between them, so the chain's first
    step can ask for a quarter turn of J4 that the tool never sees, or
    the solver can find no step at all from a seed whose jacobian has
    lost a rank. The turn is made first, as a joint move of its own, and
    the chain is solved again from the turned wrist. The turns are tried
    in a fixed order, the solver's own answer for the first pose
    (``q_hint``) last, and the first whose chain steps no joint further
    between two rows than the turn's own rows do is taken, so the same
    move always turns the wrist the same way. The split the turn settles
    on decides whether the solver snaps the wrist round in one row
    further along, where the tool has barely left the singularity; when
    every chain snaps, the one that snaps least is taken.
    """
    turns: list[float] = list(_WRIST_TURNS_RAD)
    if q_hint is not None:
        turns.append(float(q_hint[3] - q_from[3]))
    least: tuple[NDArray[np.float64], int] | None = None
    least_step = np.inf
    for turn in turns:
        bridge = _wrist_turn(q_from, turn, se3_poses[0])
        if bridge is None:
            continue
        result = solver.batch_ik(se3_poses[1:], bridge, stop_on_failure=True)
        if not result.all_valid:
            continue
        path = np.asarray(result.joint_positions, dtype=np.float64)
        chain = np.concatenate([bridge[np.newaxis], path])
        if _ik_branch_hop(chain) is not None:
            continue
        step = float(np.max(np.abs(np.diff(chain, axis=0)))) if len(chain) > 1 else 0.0
        if step > _WRIST_TURN_ROW_RAD and step >= least_step:
            continue
        rows = max(1, int(np.ceil(abs(turn) / _WRIST_TURN_ROW_RAD)))
        fractions = np.arange(1, rows + 1, dtype=np.float64)[:, np.newaxis] / rows
        sweep = q_from + fractions * (bridge - q_from)
        least = (np.concatenate([q_from[np.newaxis], sweep, path]), rows)
        least_step = step
        if step <= _WRIST_TURN_ROW_RAD:
            break
    return least


# Halvings a joint step past IK_MAX_JOINT_STEP_RAD gets before it is taken
# for a hop between IK branches. A fast but continuous swing, such as the
# wrist turning past a hair off its singularity, splits into steps within
# the limit a level or two down; a jump between branches stays a jump
# however finely the path is cut.
_HOP_BISECTIONS: int = 6
# How far the bridge's solution at a waypoint may differ from the batch
# solve's before the rest of the chain is solved again from the bridge.
_SAME_SOLUTION_RAD: float = 1e-6


def _bridge(
    solver: IKSolver,
    pose_a: NDArray[np.float64],
    q_a: NDArray[np.float64],
    pose_b: NDArray[np.float64],
    depth: int,
    poses: list[NDArray[np.float64]],
    qs: list[NDArray[np.float64]],
) -> bool:
    """Solve ``pose_b`` on from ``q_a`` at ``pose_a``, halving the stretch
    between them while a joint steps further than ``IK_MAX_JOINT_STEP_RAD``.
    The poses solved on the way, ``pose_b`` last, and their solutions are
    appended to ``poses`` and ``qs``. False when a step survives ``depth``
    halvings."""
    if solver.solve(pose_b, q0=q_a):
        q_b = np.array(solver.q, dtype=np.float64)
        if float(np.max(np.abs(q_b - q_a))) <= IK_MAX_JOINT_STEP_RAD:
            poses.append(pose_b)
            qs.append(q_b)
            return True
    if depth == 0:
        return False
    # Halfway as a straight segment runs: position lerp, rotation geodesic.
    mid = np.empty((4, 4), dtype=np.float64)
    se3_interp(pose_a, pose_b, 0.5, mid)
    mid[:3, 3] = 0.5 * (pose_a[:3, 3] + pose_b[:3, 3])
    if not _bridge(solver, pose_a, q_a, mid, depth - 1, poses, qs):
        return False
    return _bridge(solver, mid, qs[-1], pose_b, depth - 1, poses, qs)


def _bridged_chain(
    solver: IKSolver,
    se3_poses: list[NDArray[np.float64]],
    positions: NDArray[np.float64],
    hop: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """The poses and joint chain with every hop from ``hop`` on bridged by
    :func:`_bridge`, the poses it solved spliced in between the path's own;
    None when a hop survives the bridge, a jump between IK branches. Where
    the bridge lands on another solution than the batch solve did, the
    rest of the chain is solved again from where it landed."""
    out_poses = list(se3_poses[:hop])
    out_q = list(positions[:hop])
    i = hop
    while True:
        if not _bridge(
            solver,
            out_poses[-1],
            out_q[-1],
            se3_poses[i],
            _HOP_BISECTIONS,
            out_poses,
            out_q,
        ):
            return None
        if float(np.max(np.abs(out_q[-1] - positions[i]))) > _SAME_SOLUTION_RAD:
            result = solver.batch_ik(se3_poses[i:], out_q[-1], stop_on_failure=True)
            if not result.all_valid:
                return None
            positions = np.concatenate(
                [positions[:i], np.asarray(result.joint_positions, dtype=np.float64)]
            )
        nxt = _ik_branch_hop(positions[i:])
        end = len(se3_poses) if nxt is None else i + nxt
        out_poses.extend(se3_poses[i + 1 : end])
        out_q.extend(positions[i + 1 : end])
        if nxt is None:
            return np.asarray(out_poses), np.asarray(out_q)
        i = end


@dataclass
class JointPath:
    """
    Joint-space path uniformly sampled in path space.

    This is the common abstraction for all motion commands. Cartesian commands
    solve IK to produce this; joint commands interpolate directly.

    Attributes:
        positions: (N, 6) array of joint angles in radians
        valid: Per-row IK validity. None means all rows are valid.
        prefix: Leading rows that are a wrist reconfiguration at a
            singularity, run as a joint move before the path proper; row
            ``prefix`` is the path's first pose. Zero for a plain path.
        knots: For a path solved from cartesian poses, each pose's
            normalized distance along the path (``cartesian_path_knots``),
            one per row from row ``prefix`` on: the path parameter to time
            it by. None for a joint-space path.
    """

    positions: NDArray[np.float64]  # (N, 6) joint angles in radians
    valid: NDArray[np.bool_] | None = None  # (N,) per-row validity, None = all valid
    prefix: int = 0
    knots: NDArray[np.float64] | None = None

    @property
    def is_partial(self) -> bool:
        """True if some IK solutions failed."""
        return self.valid is not None

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> NDArray[np.float64]:
        return self.positions[idx]

    @classmethod
    def from_poses(
        cls,
        poses: NDArray[np.float64],
        seed_q: NDArray[np.float64],
        stop_on_failure: bool = True,
    ) -> JointPath:
        """
        Solve IK for poses with seeded chain.

        Each IK solve uses the previous solution as seed, maintaining continuity.

        Args:
            poses: (N, 4, 4) SE3 poses
            seed_q: Initial joint angles for IK seeding (radians)
            stop_on_failure: If True, stop solving after first IK failure
                (real controller). If False, solve all poses (diagnostic).

        A joint step past ``IK_MAX_JOINT_STEP_RAD`` between two poses is
        solved again at poses in between; a step that stays that large
        however finely the path is cut is a hop between IK branches.

        Returns:
            JointPath with solved joint positions, and the poses' knots. If
            some poses failed, ``valid`` is set to a per-row bool array
            (``is_partial`` is True).

        Raises:
            IKError: If fewer than 2 consecutive valid poses from the start,
                or the chain hops between IK branches.
        """
        se3_poses = [poses[i] for i in range(len(poses))]

        solver = IKSolver(
            PAROL6_ROBOT.robot,
            damping=Damping.Sugihara,
            tol=1e-12,
            lm_lambda=0.0,
            max_iter=10,
            max_restarts=10,
        )
        result = solver.batch_ik(
            se3_poses,
            np.asarray(seed_q, dtype=np.float64),
            stop_on_failure=stop_on_failure,
        )

        if result.all_valid:
            positions = np.asarray(result.joint_positions, dtype=np.float64)
            hop = _ik_branch_hop(positions)
            if hop is None:
                return cls(positions=positions, knots=cartesian_path_knots(poses))
            if hop == 1:
                # A path leaving a wrist singularity turns the wrist first.
                turned = _leave_wrist_singularity(
                    solver, se3_poses, positions[0], positions[1]
                )
                if turned is not None:
                    return cls(
                        positions=turned[0],
                        prefix=turned[1],
                        knots=cartesian_path_knots(poses),
                    )
            bridged = _bridged_chain(solver, se3_poses, positions, hop)
            if bridged is not None:
                return cls(positions=bridged[1], knots=cartesian_path_knots(bridged[0]))
            raise IKError(
                make_error(
                    ErrorCode.IK_PARTIAL_PATH,
                    valid=str(hop),
                    total=str(len(se3_poses)),
                )
            )

        valid = np.array(result.valid, dtype=np.bool_)
        first_fail = int(np.argmin(valid))  # first False index
        if first_fail == 1:
            # A seed at a wrist singularity can leave the solver no step
            # to take; the turned wrist is a seed it can solve from. The
            # preview takes the same way out, or it would refuse a move
            # the controller runs.
            turned = _leave_wrist_singularity(
                solver,
                se3_poses,
                np.asarray(result.joint_positions, dtype=np.float64)[0],
                None,
            )
            if turned is not None:
                return cls(
                    positions=turned[0],
                    prefix=turned[1],
                    knots=cartesian_path_knots(poses),
                )
        if first_fail < 2:
            if stop_on_failure:
                raise IKError(
                    make_error(
                        ErrorCode.IK_PARTIAL_PATH,
                        valid=str(first_fail),
                        total=str(len(se3_poses)),
                    )
                )
            # Diagnostic mode: return partial data for visualization
            return cls(positions=result.joint_positions, valid=valid)

        return cls(positions=result.joint_positions, valid=valid)

    @classmethod
    def interpolate(
        cls,
        start_rad: NDArray[np.float64],
        end_rad: NDArray[np.float64],
        n_samples: int,
    ) -> JointPath:
        """
        Direct joint-space linear interpolation (for MovePose/MoveJoint).

        Args:
            start_rad: Starting joint angles in radians
            end_rad: Ending joint angles in radians
            n_samples: Number of samples (minimum 2)

        Returns:
            JointPath with interpolated positions
        """
        n_samples = max(2, n_samples)
        start = np.asarray(start_rad, dtype=np.float64)
        end = np.asarray(end_rad, dtype=np.float64)

        t = np.linspace(0, 1, n_samples).reshape(-1, 1)
        positions = start + t * (end - start)

        return cls(positions=positions)

    def append(self, other: JointPath) -> JointPath:
        """
        Concatenate paths (for path blending).

        Args:
            other: Path to append

        Returns:
            New JointPath with concatenated positions
        """
        combined = np.concatenate([self.positions, other.positions], axis=0)
        return JointPath(positions=combined)

    def sample(self, s: float) -> NDArray[np.float64]:
        """
        Sample path at normalized position s in [0, 1].

        Uses linear interpolation between path points.

        Args:
            s: Path position from 0 (start) to 1 (end)

        Returns:
            Interpolated joint position
        """
        s = np.clip(s, 0.0, 1.0)
        n = len(self.positions)
        if n < 2:
            return self.positions[0].copy()

        idx_float = s * (n - 1)
        idx_lo = int(idx_float)
        idx_hi = min(idx_lo + 1, n - 1)
        frac = idx_float - idx_lo

        return self.positions[idx_lo] * (1 - frac) + self.positions[idx_hi] * frac

    def sample_many(self, s_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Vectorized sampling at multiple path positions.

        Args:
            s_values: Array of path positions from 0 (start) to 1 (end)

        Returns:
            (N, 6) array of interpolated joint positions
        """
        s_values = np.clip(s_values, 0.0, 1.0)
        n = len(self.positions)
        if n < 2:
            return np.tile(self.positions[0], (len(s_values), 1))

        idx_float = s_values * (n - 1)
        idx_lo = idx_float.astype(np.intp)
        idx_hi = np.minimum(idx_lo + 1, n - 1)
        frac = (idx_float - idx_lo).reshape(-1, 1)

        return self.positions[idx_lo] * (1 - frac) + self.positions[idx_hi] * frac


@dataclass
class Trajectory:
    """
    Ready-to-execute trajectory with motor steps at control rate.

    Precomputed trajectories are sent directly to the controller without smoothing.
    StreamingExecutor is only used for online targets (jogging/streaming).

    Attributes:
        steps: (M, 6) motor steps at each control tick
        duration: How long the player takes over it [s]: one control tick
            per row, the first holding the pose the move starts from. Zero
            for a move that goes nowhere.
    """

    steps: NDArray[np.int32]  # (M, 6) motor steps
    duration: float  # seconds
    positions_rad: NDArray[np.float64]  # Before motor-step quantization

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> NDArray[np.int32]:
        return self.steps[idx]


class TrajectoryBuilder:
    """
    Converts JointPath to executable Trajectory.

    Along-path profiles use TOPP-RA to time a blended path within the
    selected profile's speed envelope and the joint limits.

    All limits come from PAROL6_ROBOT config - no hardcoded fallbacks.
    """

    def __init__(
        self,
        joint_path: JointPath,
        profile: ProfileType | str,
        velocity_frac: float = 1.0,
        accel_frac: float = 1.0,
        jerk_frac: float = 1.0,
        duration: float | None = None,
        dt: float = INTERVAL_S,
        cart_vel_limit: float | None = None,
        cart_acc_limit: float | None = None,
        path_knots: NDArray[np.float64] | None = None,
        constant_tool_speed: bool = False,
    ):
        """
        Initialize trajectory builder.

        Args:
            joint_path: Path in joint space
            profile: Motion profile to apply
            velocity_frac: Scale joint velocity limits (0.0-1.0), default 1.0
            accel_frac: Scale joint acceleration limits (0.0-1.0), default 1.0
            jerk_frac: Scale jerk limits (0.0-1.0), default 1.0
            duration: Uniformly stretch the final rows when longer than their fastest timing
            dt: Control loop time step
            cart_vel_limit: Cartesian linear velocity limit in m/s (for Cartesian commands)
            cart_acc_limit: Cartesian linear acceleration limit in m/s² (for Cartesian commands)
            path_knots: Path-parameter value of each joint waypoint, strictly
                increasing from 0 to 1 — cumulative tool distance for a
                cartesian path, so that a constant ``ds/dt`` is a constant
                tool speed. ``None`` spaces the waypoints evenly.
            constant_tool_speed: Hold the whole path to one ``ds/dt``, the
                fastest the steepest stretch and the cartesian ceiling allow,
                rather than running each stretch as fast as it can (what a
                process move promises).
        """
        self.joint_path = joint_path
        self.path_knots = path_knots
        self.constant_tool_speed = constant_tool_speed
        self.profile = (
            ProfileType.from_string(profile) if isinstance(profile, str) else profile
        )

        # RUCKIG is point-to-point only - if Cartesian limits are set, we need path following
        if self.profile == ProfileType.RUCKIG and (
            cart_vel_limit is not None or cart_acc_limit is not None
        ):
            logger.warning("RUCKIG cannot follow Cartesian paths, using TOPPRA")
            self.profile = ProfileType.TOPPRA

        self.velocity_frac = velocity_frac
        self.accel_frac = accel_frac
        self.jerk_frac = jerk_frac
        self.duration = duration
        self.dt = dt
        self.cart_vel_limit = cart_vel_limit
        self.cart_acc_limit = cart_acc_limit

        # Joint limits scaled by user fractions.
        # Apply 1% safety margin to account for floating-point precision in
        # trajectory libraries and integer rounding in rad→steps conversion.
        limit_margin = 0.99
        self.v_max = LIMITS.joint.hard.velocity * self.velocity_frac * limit_margin
        self.a_max = LIMITS.joint.hard.acceleration * self.accel_frac * limit_margin
        self.j_max = LIMITS.joint.hard.jerk * self.jerk_frac * limit_margin

        # Pre-compute limit arrays for TOPP-RA (avoids allocation per build() call)
        self._vlim = np.column_stack([-self.v_max, self.v_max])
        self._alim = np.column_stack([-self.a_max, self.a_max])

    def build(self) -> Trajectory:
        """Time the fastest selected profile, enforce played joint limits across
        the complete move, then uniformly stretch to a longer requested duration.
        """
        trajectory = self._build()
        if trajectory.duration == 0.0:
            return trajectory
        rows = self._enforce_segment_limits(trajectory.positions_rad)
        fastest = (len(rows) - 1) * self.dt
        if self.duration:
            if self.duration < fastest:
                logger.warning(
                    "Extending duration from %.3fs to %.3fs to respect the joint limits",
                    self.duration,
                    fastest,
                )
            elif self.duration > fastest:
                rows = _slowed(rows, _tick_intervals(self.duration, self.dt) + 1)
        elif len(rows) > len(trajectory):
            duration = (len(trajectory) - 1) * self.dt
            level = logging.WARNING if fastest > 1.1 * duration else logging.DEBUG
            logger.log(
                level,
                "Extending duration from %.3fs to %.3fs (%.1f%% increase) to respect velocity/acceleration limits",
                duration,
                fastest,
                (fastest / duration - 1) * 100,
            )
        return self._trajectory(rows)

    def _build(self) -> Trajectory:
        """The trajectory as the selected profile times it."""
        positions = self.joint_path.positions
        # A path that goes nowhere is done where it stands: there is no
        # distance for a timing solver to spread over, and every profile
        # would either refuse it or divide by it. Nowhere is within a
        # motor step, the resolution the arm is driven at.
        if len(self.joint_path) < 2 or self._within_a_step(positions):
            steps = _rad_to_steps_alloc(positions[0:1])  # Keep 2D shape (1, 6)
            return Trajectory(
                steps=steps,
                duration=0.0,
                positions_rad=positions[0:1].copy(),
            )

        if self.joint_path.prefix > 0:
            return self._build_with_prefix()

        along_path = self._is_cartesian_path() or not self._is_point_to_point()
        if self.profile == ProfileType.RUCKIG and not along_path:
            return self._build_ruckig_trajectory()
        elif self.profile == ProfileType.LINEAR:
            return self._build_simple_trajectory()
        elif self.profile == ProfileType.QUINTIC:
            if along_path:
                return self._build_quintic_trajectory_along_path()
            return self._build_quintic_trajectory_joint()
        elif self.profile == ProfileType.TRAPEZOID:
            if along_path:
                # A trapezoid of the path coordinate is what LINEAR runs.
                return self._build_simple_trajectory()
            return self._build_trapezoid_trajectory_joint()
        else:
            return self._build_toppra_trajectory()

    def _trajectory(self, trajectory_rad: NDArray[np.float64]) -> Trajectory:
        """The rows as the player takes them, one per control tick: the
        trajectory lasts as many ticks as it has rows, the queue time the
        controller reports for it."""
        return Trajectory(
            steps=_rad_to_steps_alloc(trajectory_rad),
            duration=len(trajectory_rad) * self.dt,
            positions_rad=trajectory_rad,
        )

    def _blended_path(self) -> _BlendedPath:
        positions = self.joint_path.positions
        if self.path_knots is not None:
            knots = np.asarray(self.path_knots, dtype=np.float64)
        else:
            # Bezier sampling density must not create artificial speed changes.
            knots = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            )
            knots /= knots[-1]
        keep = _distinct(knots)
        if np.count_nonzero(keep) < 2:
            raise TrajectoryPlanningError(
                make_error(
                    ErrorCode.TRAJ_NO_STEPS,
                    detail="the path covers no distance to time",
                )
            )
        return _BlendedPath(knots[keep], positions[keep], self.a_max, self.dt)

    @staticmethod
    def _within_a_step(positions: NDArray[np.float64]) -> bool:
        """The path ends on the step it starts on and strays no more than a
        step between: a move of even one step still goes there."""
        ends = _rad_to_steps_alloc(positions[[0, -1]])
        if not np.array_equal(ends[0], ends[1]):
            return False
        steps = _rad_to_steps_alloc(positions)
        return bool(np.max(np.abs(steps - steps[0])) <= 1)

    def _is_point_to_point(self) -> bool:
        """The rows run straight from the first to the last and never turn
        back: the path per-joint profiles, which see only its two ends,
        reproduce."""
        positions = self.joint_path.positions
        chord = positions[-1] - positions[0]
        length_sq = float(np.dot(chord, chord))
        if length_sq < 1e-18:
            return False
        rel = positions - positions[0]
        along = rel @ chord / length_sq
        off = rel - np.outer(along, chord)
        return bool(np.max(np.abs(off)) < 1e-9 and np.all(np.diff(along) >= -1e-12))

    def _build_with_prefix(self) -> Trajectory:
        """A wrist reconfiguration ahead of the path is its own joint move,
        timed by the joint limits alone, and the path follows it from
        rest: the cartesian timing (knots, tool ceiling, constant tool
        speed) applies to the path, which starts at the reconfigured pose,
        and a requested duration covers the two together."""
        p = self.joint_path.prefix
        turn = TrajectoryBuilder(
            joint_path=JointPath(positions=self.joint_path.positions[: p + 1]),
            profile=self.profile,
            velocity_frac=self.velocity_frac,
            accel_frac=self.accel_frac,
            jerk_frac=self.jerk_frac,
            dt=self.dt,
        )._build()
        path = TrajectoryBuilder(
            joint_path=JointPath(positions=self.joint_path.positions[p:]),
            profile=self.profile,
            velocity_frac=self.velocity_frac,
            accel_frac=self.accel_frac,
            jerk_frac=self.jerk_frac,
            dt=self.dt,
            cart_vel_limit=self.cart_vel_limit,
            cart_acc_limit=self.cart_acc_limit,
            path_knots=self.path_knots,
            constant_tool_speed=self.constant_tool_speed,
        )._build()
        return self._trajectory(
            np.concatenate([turn.positions_rad, path.positions_rad[1:]])
        )

    def _build_toppra_trajectory(self) -> Trajectory:
        """
        Build trajectory using TOPP-RA's time-optimal path parameterization.

        Bounded corner blends expose curvature to the timing solver while
        preserving the waypoint parameter and Cartesian velocity limits.
        """
        return self._time_path(self._blended_path())

    def _time_path(
        self,
        path: _BlendedPath,
        profile_speed: Callable[[NDArray], NDArray] | None = None,
    ) -> Trajectory:
        """Time the blended path within the selected profile's speed envelope."""
        positions = self.joint_path.positions
        gridpoints = self._gridpoints(path)
        joint_vel_constraint = constraint.JointVelocityConstraint(self._vlim)
        joint_acc_constraint = constraint.JointAccelerationConstraint(self._alim)
        constraints: list[constraint.Constraint] = [
            joint_vel_constraint,
            joint_acc_constraint,
        ]

        if self._is_cartesian_path():
            constraints.append(self._build_cart_vel_constraint(path, gridpoints))
        if self.constant_tool_speed:
            constraints.append(self._build_path_speed_cap(path))
        if profile_speed is not None:
            constraints.append(_PathSpeedConstraint(profile_speed(gridpoints)))

        try:
            instance = algo.TOPPRA(
                constraints,
                path,
                gridpoints=gridpoints,
                parametrizer="ParametrizeConstAccel",
            )
            jnt_traj = instance.compute_trajectory()

            if jnt_traj is None:
                raise RuntimeError("TOPP-RA failed to compute trajectory")

            assert isinstance(jnt_traj, ParametrizeConstAccel)
            duration = float(jnt_traj.duration)
            if not np.isfinite(duration) or duration <= 0:
                raise RuntimeError("TOPP-RA returned an invalid duration")

            logger.debug(
                "TrajectoryBuilder: TOPP-RA duration=%.3f, path_len=%d",
                duration,
                len(positions),
            )

            intervals = _tick_intervals(duration, self.dt)
            times = np.arange(intervals) * (duration / intervals)
            trajectory_rad = np.empty((intervals + 1, 6), dtype=np.float64)
            trajectory_rad[:-1] = jnt_traj(times)
            trajectory_rad[-1] = positions[-1]
            if not np.all(np.isfinite(trajectory_rad)):
                raise RuntimeError("TOPP-RA returned non-finite joint positions")
            duration = len(trajectory_rad) * self.dt

            logger.debug(
                "TrajectoryBuilder: output_samples=%d, duration=%.3f",
                len(trajectory_rad),
                duration,
            )

            return self._trajectory(trajectory_rad)

        except Exception as e:
            # A move the solver cannot time is refused, never quietly run
            # under a different profile than the one selected.
            raise TrajectoryPlanningError(
                make_error(ErrorCode.TRAJ_NO_STEPS, detail=f"TOPPRA failed: {e}")
            ) from e

    def _build_simple_trajectory(self) -> Trajectory:
        """A trapezoidal path-speed envelope, slowed ahead of bends as needed."""
        path = self._blended_path()
        vmax_s, amax_s, _ = self._compute_s_profile_limits()
        vmax_s = min(vmax_s, self._cruise_cap(path))
        if not self._is_cartesian_path() and self._is_point_to_point():
            duration = _trapezoid_duration(1.0, vmax_s, amax_s)
            intervals = _tick_intervals(duration, self.dt)
            profile_s = _trapezoid_samples(
                np.linspace(0.0, duration, intervals + 1), 0.0, 1.0, vmax_s, amax_s
            )
            return self._trajectory(path(profile_s))

        def speed(s: NDArray) -> NDArray:
            return np.sqrt(np.minimum(vmax_s**2, 2.0 * amax_s * np.minimum(s, 1.0 - s)))

        return self._time_path(path, speed)

    def _is_cartesian_path(self) -> bool:
        """Check if this is a Cartesian path (has Cartesian velocity limits set)."""
        return self.cart_vel_limit is not None and self.cart_vel_limit > 0

    def _cruise_cap(self, path: _BlendedPath) -> float:
        """The ``ds/dt`` ceiling on a path timed through its path coordinate:
        on a process move, one tool speed the steepest stretch and the tool
        ceiling allow; on any other cartesian path, the tool ceiling alone,
        the joints slowing only where they must; none off one."""
        if self.constant_tool_speed:
            return self._path_speed_cap(path)
        return self._tool_speed_limit(path)

    def _compute_s_profile_limits(self) -> tuple[float, float, float]:
        """
        Compute path parameter (s) limits derived from joint limits.

        Along a path in joint space from s=0 to s=1, joint j travels
        L[j] = sum of |delta q[j]| over the rows, so on average:
            joint_velocity = L[j] * (ds/dt)
            joint_acceleration = L[j] * (d²s/dt²)
            joint_jerk = L[j] * (d³s/dt³)

        So the s-profile limits are:
            vmax_s = min(v_max[j] / L[j]) for all joints
            amax_s = min(a_max[j] / L[j]) for all joints
            jmax_s = min(j_max[j] / L[j]) for all joints

        The path's length rather than its endpoint delta: a path that comes
        back to where it started still has somewhere to go.

        Returns:
            (vmax_s, amax_s, jmax_s): Limits for the path parameter profile
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return (1.0, 1.0, 1.0)

        length = np.sum(np.abs(np.diff(positions, axis=0)), axis=0)

        # Avoid division by zero for joints that don't move
        with np.errstate(divide="ignore", invalid="ignore"):
            vmax_s_per_joint = np.where(length > 1e-9, self.v_max / length, np.inf)
            amax_s_per_joint = np.where(length > 1e-9, self.a_max / length, np.inf)
            jmax_s_per_joint = np.where(length > 1e-9, self.j_max / length, np.inf)

        # The limiting joint determines the s-profile limits
        vmax_s = float(np.min(vmax_s_per_joint))
        amax_s = float(np.min(amax_s_per_joint))
        jmax_s = float(np.min(jmax_s_per_joint))

        return (vmax_s, amax_s, jmax_s)

    def _enforce_segment_limits(
        self, trajectory_rad: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Trim residual velocity and acceleration excess in the final played
        rows, including the seam after a wrist-turn prefix. Continuous
        timing can miss extrema between gridpoints and across that seam.

        Each segment between rows is given the shortest time, never less
        than a tick, that keeps every joint under its velocity limit and
        its velocity change from one segment to the next within a tick of
        its acceleration limit (:func:`_segment_rates`). The rows are then
        resampled a tick apart along that timing; a stretch of segments
        all at least a tick long, sampled a tick apart, changes velocity
        no faster than its segments do.

        Args:
            trajectory_rad: Joint positions in radians, shape (N, 6), one
                control tick apart

        Returns:
            The trajectory resampled with locally stretched segments, its
            rows still one tick apart; the input itself when nothing needs
            stretching.
        """
        n_points = len(trajectory_rad)
        if n_points < 2:
            return trajectory_rad

        deltas = np.diff(trajectory_rad, axis=0)
        rates = _segment_rates(deltas, self.v_max, self.a_max * self.dt, 1.0 / self.dt)
        if rates.min() * self.dt >= 1.0 - _RATE_TOLERANCE:
            return trajectory_rad

        cumulative_times = np.zeros(n_points)
        cumulative_times[1:] = np.cumsum(1.0 / rates)
        new_duration = float(cumulative_times[-1])

        intervals = _tick_intervals(new_duration, self.dt)
        output_times = np.arange(intervals + 1) * (new_duration / intervals)

        return _resample(trajectory_rad, cumulative_times, output_times)

    def _compute_joint_duration_trapezoid(self) -> float:
        """
        Compute duration for joint paths using trapezoidal profile.

        For each joint, computes the minimum duration for its displacement
        given its velocity/acceleration limits.
        Returns the maximum (slowest joint determines overall duration).
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = positions[-1] - positions[0]
        max_duration = 0.0

        for j in range(6):
            delta = abs(total_delta[j])
            if delta < 1e-6:
                continue

            duration = _trapezoid_duration(delta, self.v_max[j], self.a_max[j])
            max_duration = max(max_duration, duration)

        return max(max_duration, self.dt * 2)

    def _compute_joint_duration_quintic(self) -> float:
        """
        Compute duration for joint paths using quintic polynomial profile.

        For quintic polynomials with zero-velocity endpoints:
        - Peak velocity at t=T/2: v_peak = 1.875 * delta / T
        - Peak acceleration at t=T*(3-sqrt(3))/6: a_peak = 5.77 * delta / T²

        For velocity limit: T = 1.875 * delta / v_max
        For acceleration limit: T = sqrt(5.77 * delta / a_max)

        Returns the maximum duration across all joints.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = np.abs(positions[-1] - positions[0])

        time_vel = _QUINTIC_PEAK_VEL * total_delta / self.v_max

        with np.errstate(divide="ignore", invalid="ignore"):
            time_acc = np.where(
                self.a_max > 0,
                np.sqrt(_QUINTIC_PEAK_ACC * total_delta / self.a_max),
                0.0,
            )

        time_per_joint = np.maximum(time_vel, time_acc)
        return max(float(np.max(time_per_joint)), self.dt * 2)

    def _build_quintic_trajectory_joint(self) -> Trajectory:
        """
        Build per-joint quintic trajectory.

        Each joint independently follows a quintic polynomial profile,
        synchronized to finish at the same time.
        """
        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        duration = self._compute_joint_duration_quintic()

        intervals = _tick_intervals(duration, self.dt)
        duration = intervals * self.dt
        times = np.arange(intervals + 1) * self.dt
        trajectory_rad = np.empty((intervals + 1, 6), dtype=np.float64)

        for j in range(6):
            delta = end_pos[j] - start_pos[j]
            if abs(delta) < 1e-9:
                trajectory_rad[:, j] = start_pos[j]
                continue

            trajectory_rad[:, j] = _quintic_samples(
                times, start_pos[j], end_pos[j], duration
            )

        return self._trajectory(trajectory_rad)

    def _build_quintic_trajectory_along_path(self) -> Trajectory:
        """A quintic path-speed envelope, or a steady cruise for process moves."""
        if self.constant_tool_speed:
            return self._build_simple_trajectory()
        path = self._blended_path()
        vmax_s, amax_s, _ = self._compute_s_profile_limits()
        duration = max(
            _QUINTIC_PEAK_VEL / min(vmax_s, self._cruise_cap(path)),
            np.sqrt(_QUINTIC_PEAK_ACC / amax_s),
            self.dt * 2,
        )
        times = np.linspace(0.0, 1.0, 1001)
        profile_s = _quintic_samples(times, 0.0, 1.0, 1.0)
        profile_v = 30.0 * times**2 * (1.0 - times) ** 2 / duration
        return self._time_path(path, lambda s: np.interp(s, profile_s, profile_v))

    def _build_trapezoid_trajectory_joint(self) -> Trajectory:
        """
        Build per-joint trapezoidal trajectory.

        Each joint independently follows a trapezoidal velocity profile,
        synchronized to finish at the same time.
        """
        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        duration = self._compute_joint_duration_trapezoid()

        intervals = _tick_intervals(duration, self.dt)
        duration = intervals * self.dt
        times = np.arange(intervals + 1) * self.dt
        trajectory_rad = np.empty((intervals + 1, 6), dtype=np.float64)

        for j in range(6):
            delta = end_pos[j] - start_pos[j]
            if abs(delta) < 1e-9:
                trajectory_rad[:, j] = start_pos[j]
                continue

            profile_duration = _trapezoid_duration(delta, self.v_max[j], self.a_max[j])

            # Scale this joint's own profile time onto the synchronized duration
            time_scale = profile_duration / duration

            trajectory_rad[:, j] = _trapezoid_samples(
                times * time_scale,
                start_pos[j],
                end_pos[j],
                self.v_max[j],
                self.a_max[j],
            )

        return self._trajectory(trajectory_rad)

    def _gridpoints(self, path: _BlendedPath) -> NDArray[np.float64]:
        breaks = path.breaks
        widths = np.diff(breaks)
        midpoints = 0.5 * (breaks[:-1] + breaks[1:])
        curvature = np.max(np.abs(path(midpoints, 2)), axis=1)
        divisions = np.maximum(1, np.ceil(widths / 0.05)).astype(int)
        divisions = np.maximum(
            divisions, np.ceil(np.sqrt(0.5 * curvature * widths**2 / 1e-4)).astype(int)
        )
        divisions = np.maximum(divisions, np.where(curvature > 1e-9, 2, 1))
        points = np.concatenate(
            [
                np.linspace(left, right, int(count) + 1)
                for left, right, count in zip(breaks[:-1], breaks[1:], divisions)
            ]
        )
        curved = curvature > 1e-9
        # Narrow bends must not constrain an entire adjacent straight interval.
        outside = np.concatenate(
            [
                breaks[:-1][curved] - widths[curved] / 2,
                breaks[1:][curved] + widths[curved] / 2,
            ]
        )
        points = np.unique(
            np.concatenate([points, np.clip(outside, *path.path_interval)])
        )
        midpoints = 0.5 * (points[:-1] + points[1:])
        straight = np.max(np.abs(path(midpoints, 2)), axis=1) < 1e-9
        points = np.sort(np.concatenate([points, midpoints[straight]]))
        points = points[_distinct(points)]
        points[[0, -1]] = path.path_interval
        return points

    def _build_cart_vel_constraint(
        self, path: _BlendedPath, gridpoints: NDArray[np.float64]
    ) -> _PathSpeedConstraint:
        assert self.cart_vel_limit is not None
        positions = path(gridpoints)
        slopes = path(gridpoints, 1)
        speeds = self._tool_speeds(positions, slopes)
        with np.errstate(divide="ignore"):
            caps = self.cart_vel_limit / speeds
        return _PathSpeedConstraint(caps)

    def _build_path_speed_cap(self, path: _BlendedPath) -> _PathSpeedConstraint:
        return _PathSpeedConstraint(self._path_speed_cap(path))

    def _tool_speeds(
        self, positions: NDArray[np.float64], slopes: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        robot = PAROL6_ROBOT.robot
        jac = np.zeros((6, 6), dtype=np.float64, order="F")
        speeds = np.empty(len(slopes))
        for i in range(len(slopes)):
            robot.jacob0_into(positions[i], jac)
            speeds[i] = np.linalg.norm(jac[:3, :] @ slopes[i])
        return speeds

    def _tool_speed_limit(self, path: _BlendedPath) -> float:
        if not self._is_cartesian_path():
            return np.inf
        assert self.cart_vel_limit is not None
        fastest = float(
            np.max(self._tool_speeds(path(path.breaks), path(path.breaks, 1)))
        )
        return self.cart_vel_limit / fastest if fastest > 1e-9 else np.inf

    def _path_speed_cap(self, path: _BlendedPath) -> float:
        """One feasible cruise speed under joint velocity, bend acceleration and tool limits."""
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.min(self.v_max / np.abs(path.slopes))
            bend = np.sqrt(np.min(self.a_max / np.abs(path(path.breaks, 2))))
        cap = min(float(velocity), float(bend), self._tool_speed_limit(path))
        if not np.isfinite(cap) or cap <= 0.0:
            raise TrajectoryPlanningError(
                make_error(
                    ErrorCode.TRAJ_NO_STEPS,
                    detail="the path covers no tool distance to hold a speed along",
                )
            )
        return cap

    def _build_ruckig_trajectory(self) -> Trajectory:
        """
        Build trajectory using Ruckig for jerk-limited point-to-point motion.

        Note: This does NOT follow the path waypoints - it goes directly from
        start to end. Use TOPP-RA profiles for path-following motion.
        """
        n_dofs = 6
        gen = Ruckig(n_dofs, self.dt)
        inp = InputParameter(n_dofs)
        out = OutputParameter(n_dofs)

        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        # Ruckig requires Python lists for input parameters
        inp.current_position = start_pos.tolist()
        inp.current_velocity = [0.0] * n_dofs
        inp.current_acceleration = [0.0] * n_dofs
        inp.target_position = end_pos.tolist()
        inp.target_velocity = [0.0] * n_dofs
        inp.target_acceleration = [0.0] * n_dofs
        inp.max_velocity = self.v_max.tolist()
        inp.max_acceleration = self.a_max.tolist()
        inp.max_jerk = self.j_max.tolist()
        if self.duration is not None:
            inp.minimum_duration = self.duration

        # Pre-size the buffer from an estimated duration; the loop below
        # grows it if Ruckig runs longer than expected.
        est_duration = self._estimate_simple_duration()
        max_iters = int(est_duration / self.dt) + 500  # generous margin
        trajectory_rad = np.empty((max_iters, n_dofs), dtype=np.float64)

        # Row 0 is the start, played on the first tick like every profile's:
        # Ruckig's first output is already a tick along.
        trajectory_rad[0] = start_pos
        count = 1
        result = Result.Working

        while result == Result.Working:
            result = gen.update(inp, out)
            if count >= len(trajectory_rad):
                new_buf = np.empty((len(trajectory_rad) * 2, n_dofs), dtype=np.float64)
                new_buf[:count] = trajectory_rad[:count]
                trajectory_rad = new_buf
            trajectory_rad[count] = out.new_position
            count += 1
            out.pass_to_input(inp)

        if result == Result.Error:
            raise RuntimeError("Ruckig failed to compute trajectory")

        return self._trajectory(trajectory_rad[:count])

    def _estimate_simple_duration(self) -> float:
        """Estimate minimum duration based on joint velocity limits.

        With adaptive time distribution, each segment gets time proportional
        to its joint movement, so total duration is sum of per-segment times.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        deltas = np.diff(positions, axis=0)  # (N-1, 6)
        segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)

        return max(float(np.sum(segment_times)), self.dt * 2)
