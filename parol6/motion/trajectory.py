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
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from ruckig import InputParameter, OutputParameter, Result, Ruckig  # type: ignore[unresolved-import, ty:unresolved-import]

from scipy.interpolate import PPoly

import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from toppra.interpolator import SplineInterpolator

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import INTERVAL_S, LIMITS, rad_to_steps
from parol6.motion.geometry import _rotation_angle, cartesian_path_knots
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError
from parol6.utils.joint_limits import TRAVEL_MAX_RAD, TRAVEL_MIN_RAD


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


class _LinearPath:
    """Piecewise linear path wrapper for TOPPRA compatibility.

    Wraps a scipy PPoly (degree-1) to satisfy the toppra path interface:
    __call__(s, order), .dof, .path_interval.  Linear segments prevent
    overshoot between waypoints — critical near wrist singularities where
    cubic spline bulge amplifies orientation error.
    """

    __slots__ = ("_pp", "dof", "path_interval")

    def __init__(self, ppoly: PPoly, dof: int) -> None:
        self._pp = ppoly
        self.dof = dof
        self.path_interval = [float(ppoly.x[0]), float(ppoly.x[-1])]

    def __call__(self, s_in: float | NDArray, order: int = 0) -> NDArray[np.float64]:
        scalar = np.isscalar(s_in)
        s = np.atleast_1d(np.asarray(s_in, dtype=float))
        if order <= 1:
            result = self._pp(s, order)
        else:
            # Second+ derivative of a piecewise-linear path is zero
            result = np.zeros((len(s), self.dof))
        return result[0] if scalar and result.ndim > 1 else result


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

    Uses TOPP-RA to compute maximum allowable path speed, then applies
    the selected motion profile (clamped to TOPP-RA limits).

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
            duration: Override duration (stretches profile if longer than TOPP-RA min)
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
        """
        Generate time-parameterized trajectory.

        Uses TOPP-RA to compute time-optimal trajectory, then samples it directly
        at the control rate. No interpolation of the original joint path is needed
        since TOPP-RA's trajectory already provides smooth, continuous positions.

        RUCKIG runs point to point, from the first row to the last; a path
        that is not a straight run between them is timed by TOPP-RA
        instead. QUINTIC and TRAPEZOID run each joint on its own profile
        from end to end on a straight run, and the path coordinate along
        any other path.

        Returns:
            Trajectory ready for execution, its rows one control tick apart
        """
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

        if self.path_knots is not None and self.profile in (
            ProfileType.LINEAR,
            ProfileType.QUINTIC,
            ProfileType.TRAPEZOID,
        ):
            # These profiles time a path coordinate spaced by row; spaced
            # by the tool's own distance instead, their cruise is one tool
            # speed, as TOPPRA holds it through the knots.
            self.joint_path = self._resampled_by_knots(self.path_knots)
            self.path_knots = None

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

    def _resampled_by_knots(self, knots: NDArray[np.float64]) -> JointPath:
        """The joint path resampled at even steps of the path parameter the
        knots give each row (the tool's normalized distance), as many rows
        as it had. Repeated knots keep their first row."""
        positions = self.joint_path.positions
        knots = np.asarray(knots, dtype=np.float64)
        keep = np.concatenate(([True], np.diff(knots) > 1e-12))
        at = np.linspace(0.0, 1.0, len(positions))
        resampled = np.empty_like(positions)
        for j in range(positions.shape[1]):
            resampled[:, j] = np.interp(at, knots[keep], positions[keep, j])
        return JointPath(positions=resampled)

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
        ).build()
        # A requested duration is the whole move's: the path gets what the
        # turn's motion leaves of it, and runs as fast as it can when the
        # turn left none, as any request shorter than the move can be does.
        path_duration = None
        if self.duration:
            remaining = self.duration - (len(turn) - 1) * self.dt
            path_duration = remaining if remaining > 0 else None
        path = TrajectoryBuilder(
            joint_path=JointPath(positions=self.joint_path.positions[p:]),
            profile=self.profile,
            velocity_frac=self.velocity_frac,
            accel_frac=self.accel_frac,
            jerk_frac=self.jerk_frac,
            duration=path_duration,
            dt=self.dt,
            cart_vel_limit=self.cart_vel_limit,
            cart_acc_limit=self.cart_acc_limit,
            path_knots=self.path_knots,
            constant_tool_speed=self.constant_tool_speed,
        ).build()
        return self._trajectory(
            np.concatenate([turn.positions_rad, path.positions_rad[1:]])
        )

    def _build_toppra_trajectory(self) -> Trajectory:
        """
        Build trajectory using TOPP-RA's time-optimal path parameterization.

        Uses piecewise linear interpolation through waypoints (no overshoot)
        and computes time-optimal velocity profile respecting joint limits
        and optional Cartesian velocity limits.
        """
        positions = self.joint_path.positions
        if self.path_knots is not None:
            # Waypoints that cover no distance would give a zero-width
            # segment; the path keeps the first of any such run.
            keep = np.concatenate(([True], np.diff(self.path_knots) > 1e-12))
            positions = positions[keep]
            ss_waypoints = np.asarray(self.path_knots, dtype=np.float64)[keep]
            if len(positions) < 2:
                raise TrajectoryPlanningError(
                    make_error(
                        ErrorCode.TRAJ_NO_STEPS,
                        detail="the path covers no tool distance to time",
                    )
                )
        else:
            ss_waypoints = np.linspace(0.0, 1.0, len(positions))
        n_points = len(positions)

        # Piecewise linear PPoly — prevents cubic spline overshoot that
        # amplifies orientation error near wrist singularities
        n_seg = n_points - 1
        dof = positions.shape[1]
        c = np.zeros((2, n_seg, dof))
        for i in range(n_seg):
            dx = ss_waypoints[i + 1] - ss_waypoints[i]
            c[0, i, :] = (positions[i + 1] - positions[i]) / dx
            c[1, i, :] = positions[i]
        path = _LinearPath(PPoly(c, ss_waypoints), dof)

        joint_vel_constraint = constraint.JointVelocityConstraint(self._vlim)
        joint_acc_constraint = constraint.JointAccelerationConstraint(self._alim)
        constraints: list[constraint.Constraint] = [
            joint_vel_constraint,
            joint_acc_constraint,
        ]

        if self.cart_vel_limit is not None and self.cart_vel_limit > 0:
            cart_constraint = self._build_cart_vel_constraint(path, ss_waypoints)
            if cart_constraint is not None:
                constraints.append(cart_constraint)
        if self.constant_tool_speed:
            constraints.append(self._build_path_speed_cap(path, positions, c[0]))

        try:
            # Use evenly-spaced gridpoints - TOPPRA docs recommend "at least a few times
            # the number of waypoints". Auto-selection can cluster points near
            # discontinuities, causing TOPPRAsd to produce incorrect durations.
            n_gridpoints = n_points * 3
            gridpoints = np.linspace(0.0, 1.0, n_gridpoints)

            if self.duration is not None and self.duration > 0:
                instance = algo.TOPPRAsd(constraints, path, gridpoints=gridpoints)
                instance.set_desired_duration(self.duration)
                jnt_traj = instance.compute_trajectory()
                if jnt_traj is not None:
                    duration = self.duration
                    logger.debug(
                        "TrajectoryBuilder: TOPPRAsd target_duration=%.3f, path_len=%d",
                        duration,
                        n_points,
                    )
                else:
                    # Fall back to time-optimal if TOPPRAsd fails
                    logger.warning("TOPPRAsd failed, trying time-optimal TOPPRA")
                    instance = algo.TOPPRA(constraints, path, gridpoints=gridpoints)
                    jnt_traj = instance.compute_trajectory()
            else:
                instance = algo.TOPPRA(constraints, path, gridpoints=gridpoints)
                jnt_traj = instance.compute_trajectory()

            if not isinstance(jnt_traj, SplineInterpolator):
                raise RuntimeError("TOPP-RA failed to compute trajectory")

            duration = float(jnt_traj.duration)

            logger.debug(
                "TrajectoryBuilder: TOPP-RA duration=%.3f, path_len=%d",
                duration,
                n_points,
            )

            # One row per tick, the solved trajectory slowed onto a whole
            # number of ticks and ending exactly at its endpoint.
            intervals = _tick_intervals(duration, self.dt)
            times = np.arange(intervals) * (duration / intervals)
            trajectory_rad = np.empty((intervals + 1, 6), dtype=np.float64)
            trajectory_rad[:-1] = jnt_traj(times)
            trajectory_rad[-1] = jnt_traj(duration)
            duration = intervals * self.dt

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
        """
        Build the LINEAR profile: constant velocity along the path with ramps
        at the acceleration limit at either end.

        The path coordinate runs a trapezoid whose cruise is the fastest the
        steepest joint allows and whose ramps are at that joint's
        acceleration limit, so the profile never steps its velocity. The
        limits come from the path's own length, segment by segment, so a
        wrist flip or a reconfiguration mid-path costs the time it takes
        rather than being averaged away by the endpoint delta. On a
        cartesian path the cruise also keeps the tool under its ceiling.
        """
        vmax_s, amax_s, _ = self._compute_s_profile_limits()
        vmax_s = min(vmax_s, self._cruise_cap())
        if not np.isfinite(vmax_s) or not np.isfinite(amax_s):
            vmax_s, amax_s = 1.0, 1.0

        profile_duration = _trapezoid_duration(1.0, vmax_s, amax_s)
        duration = max(profile_duration, self.duration or 0.0, self.dt * 2)
        intervals = _tick_intervals(duration, self.dt)
        duration = intervals * self.dt
        times = np.arange(intervals + 1) * self.dt
        profile_s = _trapezoid_samples(
            times * (profile_duration / duration), 0.0, 1.0, vmax_s, amax_s
        )
        trajectory_rad = self.joint_path.sample_many(profile_s)

        trajectory_rad = self._enforce_segment_limits(trajectory_rad)

        return self._trajectory(trajectory_rad)

    def _is_cartesian_path(self) -> bool:
        """Check if this is a Cartesian path (has Cartesian velocity limits set)."""
        return self.cart_vel_limit is not None and self.cart_vel_limit > 0

    def _cruise_cap(self) -> float:
        """The ``ds/dt`` ceiling on a path timed through its path coordinate:
        on a process move, one tool speed the steepest stretch and the tool
        ceiling allow; on any other cartesian path, the tool ceiling alone,
        the joints slowing only where they must; none off one."""
        if self.constant_tool_speed:
            return self._row_speed_cap()
        return self._tool_speed_cap()

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
        Enforce velocity limits by locally stretching segments that exceed limits.

        Walks through each segment and checks if joint velocities exceed limits.
        Where they do, stretches that segment's time. This handles singularities
        and wrist flips by slowing only where necessary, not globally.

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

        initial_dt = self.dt
        duration = (n_points - 1) * initial_dt

        deltas = np.diff(trajectory_rad, axis=0)  # (N-1, 6)

        # Minimum time per segment to respect velocity limits:
        # max(|delta[j]| / v_max[j]) over joints
        min_segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)
        segment_times = np.maximum(min_segment_times, initial_dt)

        # Approximate acceleration check: is the velocity change between
        # adjacent segments feasible at the times they take once slowed for
        # velocity? Measured at the original spacing instead, a segment
        # slowed for velocity counts its excess twice.
        if n_points > 2:
            velocities = deltas / segment_times[:, np.newaxis]
            spans = 0.5 * (segment_times[:-1] + segment_times[1:])
            accel = np.diff(velocities, axis=0) / spans[:, np.newaxis]  # (N-2, 6)
            accel_times = np.zeros(n_points - 1)
            for i in range(len(accel)):
                max_accel_ratio = np.max(np.abs(accel[i]) / self.a_max)
                if max_accel_ratio > 1.0:
                    # Spread the extra time across adjacent segments
                    stretch = np.sqrt(max_accel_ratio)
                    accel_times[i] = max(accel_times[i], min_segment_times[i] * stretch)
                    accel_times[i + 1] = max(
                        accel_times[i + 1], min_segment_times[i + 1] * stretch
                    )
            segment_times = np.maximum(segment_times, accel_times)

        new_duration = float(np.sum(segment_times))
        if new_duration <= duration * 1.001:  # No significant change
            return trajectory_rad

        logger.warning(
            "Extending duration from %.3fs to %.3fs (%.1f%% increase) to respect velocity/acceleration limits",
            duration,
            new_duration,
            (new_duration / duration - 1) * 100,
        )

        # Resample at control rate with the new per-segment timing
        cumulative_times = np.zeros(n_points)
        cumulative_times[1:] = np.cumsum(segment_times)

        intervals = _tick_intervals(new_duration, self.dt)
        output_times = np.arange(intervals + 1) * (new_duration / intervals)

        new_trajectory = np.empty((intervals + 1, 6), dtype=np.float64)
        for j in range(6):
            new_trajectory[:, j] = np.interp(
                output_times, cumulative_times, trajectory_rad[:, j]
            )

        return new_trajectory

    def _requested_or(self, fastest: float) -> float:
        """The requested duration, stretched to ``fastest`` when shorter:
        a duration no joint can keep is not kept by overrunning its
        limits."""
        if not self.duration:
            return fastest
        if self.duration < fastest:
            logger.warning(
                "Extending duration from %.3fs to %.3fs to respect velocity/acceleration limits",
                self.duration,
                fastest,
            )
            return fastest
        return self.duration

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

    def _compute_duration_along_rows(self) -> float:
        """
        Compute duration for a path followed row by row, from per-segment
        joint requirements.

        This properly handles singularities and wrist flips by analyzing
        the maximum joint movement required in each path segment, not just
        the total start-to-end displacement.

        Returns the sum of minimum segment times, ensuring the path can be
        traversed without violating joint velocity limits at any point.
        """
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        # Per-segment time from the max joint movement within each segment;
        # summing these handles singularities/wrist flips that total
        # start-to-end displacement would miss.
        deltas = np.diff(positions, axis=0)  # (N-1, 6)
        segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)

        segment_times = np.maximum(segment_times, self.dt)

        return max(float(np.sum(segment_times)), self.dt * 2)

    def _build_quintic_trajectory_joint(self) -> Trajectory:
        """
        Build per-joint quintic trajectory.

        Each joint independently follows a quintic polynomial profile,
        synchronized to finish at the same time.
        """
        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        duration = self._requested_or(self._compute_joint_duration_quintic())

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

        trajectory_rad = self._enforce_segment_limits(trajectory_rad)

        return self._trajectory(trajectory_rad)

    def _build_quintic_trajectory_along_path(self) -> Trajectory:
        """
        Build a quintic trajectory of the path coordinate.

        The path is followed row by row along a quintic profile of s, with
        local slowdown where velocity limits would be exceeded. On a
        cartesian path the profile's peak keeps the tool under its ceiling.
        """
        if self.duration:
            duration = self.duration
        else:
            # Use per-segment analysis to handle singularities and wrist flips
            duration = self._compute_duration_along_rows()
        duration = max(duration, _QUINTIC_PEAK_VEL / self._cruise_cap())

        intervals = _tick_intervals(duration, self.dt)
        duration = intervals * self.dt
        times = np.arange(intervals + 1) * self.dt

        profile_s = _quintic_samples(times, 0.0, 1.0, duration)

        trajectory_rad = self.joint_path.sample_many(profile_s)

        trajectory_rad = self._enforce_segment_limits(trajectory_rad)

        return self._trajectory(trajectory_rad)

    def _build_trapezoid_trajectory_joint(self) -> Trajectory:
        """
        Build per-joint trapezoidal trajectory.

        Each joint independently follows a trapezoidal velocity profile,
        synchronized to finish at the same time.
        """
        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        duration = self._requested_or(self._compute_joint_duration_trapezoid())

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

        trajectory_rad = self._enforce_segment_limits(trajectory_rad)

        return self._trajectory(trajectory_rad)

    def _build_cart_vel_constraint(
        self, path: ta.SplineInterpolator | _LinearPath, ss_waypoints: NDArray
    ) -> constraint.JointVelocityConstraintVarying | None:
        """
        Build Cartesian velocity constraint for TOPP-RA using path-tangent method.

        Uses the path tangent (dq/ds) to compute accurate Cartesian velocity limits.
        At each path point s:
        - cart_vel = J_lin @ q_dot = J_lin @ (dq/ds * s_dot)
        - ||cart_vel|| = ||J_lin @ dq/ds|| * |s_dot|
        - For ||cart_vel|| <= v_max: |s_dot| <= v_max / ||J_lin @ dq/ds||

        This is more accurate than the column-norm method as it considers the
        actual direction of motion along the path.

        Args:
            path: The spline path through joint space
            ss_waypoints: Path parameter values at each waypoint

        Returns:
            JointVelocityConstraintVarying with path-dependent limits, or None if error
        """
        if self.cart_vel_limit is None or self.cart_vel_limit <= 0:
            return None

        try:
            robot = PAROL6_ROBOT.robot

            # cart_vel_limit is already in m/s (SI units)
            v_max_m_s = self.cart_vel_limit
            # Scaled joint limits respect the user's velocity_frac
            v_max_joint = self.v_max

            # Pre-allocate; vlim_func is called once per gridpoint
            vlim_buffer = np.empty((6, 2), dtype=np.float64)
            _jac_buf = np.zeros((6, 6), dtype=np.float64, order="F")

            def vlim_func(s: float) -> NDArray:
                """Compute velocity limits at path position s using path tangent."""
                q = path(s)
                dq_ds = path(s, 1)  # Path tangent (first derivative)

                # Linear (translational) part of the Jacobian is the first 3 rows
                robot.jacob0_into(q, _jac_buf)
                J_lin = _jac_buf[:3, :]

                cart_vel_per_sdot = np.linalg.norm(J_lin @ dq_ds)

                if cart_vel_per_sdot < 1e-6:
                    # Near-zero path tangent (at waypoint or singular), use joint limits
                    vlim_buffer[:, 0] = -v_max_joint
                    vlim_buffer[:, 1] = v_max_joint
                    return vlim_buffer.copy()

                max_sdot = v_max_m_s / cart_vel_per_sdot

                # The Cartesian constraint limits s_dot, not individual joint velocities.
                # We scale ALL joint velocity limits uniformly by the ratio of
                # (Cartesian-limited s_dot) / (fastest achievable s_dot from joint limits).
                #
                # This ensures the path velocity respects the Cartesian limit while
                # keeping joints at their relative proportions.
                abs_dq_ds = np.abs(dq_ds)

                with np.errstate(divide="ignore", invalid="ignore"):
                    s_dot_per_joint = np.where(
                        abs_dq_ds > 1e-9,
                        v_max_joint / abs_dq_ds,
                        np.inf,
                    )

                # The binding joint limit determines max achievable s_dot
                s_dot_from_joints = float(np.min(s_dot_per_joint))

                if max_sdot < s_dot_from_joints and s_dot_from_joints > 0:
                    # Cartesian constraint is the tighter one; scale all limits down
                    scale = max_sdot / s_dot_from_joints
                    q_dot_max = v_max_joint * scale
                else:
                    q_dot_max = v_max_joint

                vlim_buffer[:, 0] = -q_dot_max
                vlim_buffer[:, 1] = q_dot_max

                return vlim_buffer.copy()

            return constraint.JointVelocityConstraintVarying(vlim_func)

        except Exception as e:
            logger.warning("Failed to build Cartesian velocity constraint: %s", e)
            return None

    def _build_path_speed_cap(
        self,
        path: _LinearPath,
        positions: NDArray[np.float64],
        slopes: NDArray[np.float64],
    ) -> constraint.Constraint:
        """:meth:`_path_speed_cap` as a TOPP-RA constraint. Holding every
        stretch to it is what a constant tool speed costs; on a process
        move that is the point rather than the price."""
        cap = self._path_speed_cap(positions, slopes)
        vlim_buffer = np.empty((6, 2), dtype=np.float64)

        def vlim_func(s: float) -> NDArray:
            dq_ds = np.abs(path(s, 1))
            q_dot_max = np.maximum(dq_ds * cap, 1e-6)
            vlim_buffer[:, 0] = -q_dot_max
            vlim_buffer[:, 1] = q_dot_max
            return vlim_buffer.copy()

        return constraint.JointVelocityConstraintVarying(vlim_func)

    def _row_slopes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The rows each segment starts at, and each segment's ``dq/ds``,
        for a path whose rows are evenly spaced in its parameter, as the
        profiles that time ``s`` directly take it."""
        positions = self.joint_path.positions
        return positions[:-1], np.diff(positions, axis=0) * (len(positions) - 1)

    def _row_speed_cap(self) -> float:
        """:meth:`_path_speed_cap` over the rows (:meth:`_row_slopes`)."""
        return self._path_speed_cap(*self._row_slopes())

    def _tool_speed_cap(self) -> float:
        """:meth:`_tool_speed_limit` over the rows (:meth:`_row_slopes`)."""
        return self._tool_speed_limit(*self._row_slopes())

    def _tool_speed_limit(
        self, positions: NDArray[np.float64], slopes: NDArray[np.float64]
    ) -> float:
        """The fastest ``ds/dt`` that keeps the tool under the cartesian
        ceiling wherever it moves fastest per unit of path; unbounded off
        a cartesian path, or on one where the tool only turns.
        ``positions`` are the rows the path runs through, one per segment
        start in ``slopes``."""
        if not self._is_cartesian_path():
            return np.inf
        assert self.cart_vel_limit is not None
        robot = PAROL6_ROBOT.robot
        jac = np.zeros((6, 6), dtype=np.float64, order="F")
        fastest = 0.0
        for i in range(len(slopes)):
            robot.jacob0_into(positions[i], jac)
            fastest = max(fastest, float(np.linalg.norm(jac[:3, :] @ slopes[i])))
        return self.cart_vel_limit / fastest if fastest > 1e-9 else np.inf

    def _path_speed_cap(
        self, positions: NDArray[np.float64], slopes: NDArray[np.float64]
    ) -> float:
        """One ``ds/dt`` ceiling for the whole path: the fastest constant the
        steepest stretch allows under the joint limits, and under the
        cartesian ceiling (:meth:`_tool_speed_limit`). ``positions`` are
        the rows the path runs through, one per segment start in
        ``slopes``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            per_joint = np.where(
                np.abs(slopes) > 1e-9, self.v_max / np.abs(slopes), np.inf
            )
        cap = min(float(np.min(per_joint)), self._tool_speed_limit(positions, slopes))
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
