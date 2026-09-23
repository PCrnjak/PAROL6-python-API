"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, MoveCart, MoveCartRelTrf
"""

import logging
from typing import cast

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands._collision_guard import collision_blocked, guard_joint_path
from parol6.config import (
    INTERVAL_S,
    LIMITS,
    PATH_SAMPLES,
    rad_to_steps,
    steps_to_rad,
)
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.motion.geometry import (
    ArcSegment,
    LineSegment,
    build_blended_path,
    cartesian_path_knots,
)
from parol6.protocol.wire import (
    CmdType,
    JogLCmd,
    MoveLCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import TrajectoryPlanningError
from parol6.utils.ik import RateLimitedWarning, solve_ik
from pinokin import se3_from_rpy, se3_interp, se3_rpy

from parol6.commands.servo_commands import _max_vel_ratio_jit

from .base import (
    ExecutionStatusCode,
    MotionCommand,
    TrajectoryMoveCommandBase,
    guard_homed,
)

logger = logging.getLogger(__name__)

# Full-scale jog_l rates: a velocity fraction of ±1 maps to these.
_CART_ANG_JOG_MAX_RAD: float = float(LIMITS.cart.jog.velocity.angular)
_CART_LIN_JOG_MAX_MS: float = float(LIMITS.cart.jog.velocity.linear)


def jog_twist(fractions: list[float], out: np.ndarray) -> None:
    """The TCP twist `[vx, vy, vz, wx, wy, wz]` (m/s, rad/s) a jog_l's six
    signed fractions ask for: each component is a pure linear map from
    zero onto its full-scale rate, so an all-zero vector asks for rest and
    a diagonal keeps the direction it was given."""
    for i in range(6):
        f = fractions[i]
        if f > 1.0:
            f = 1.0
        elif f < -1.0:
            f = -1.0
        out[i] = f * (_CART_LIN_JOG_MAX_MS if i < 3 else _CART_ANG_JOG_MAX_RAD)


_ik_warn = RateLimitedWarning()


@register_command(CmdType.JOGL)
class JogLCommand(MotionCommand[JogLCmd]):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.

    The CSE drives the commanded 6-DOF TCP twist (Ruckig-smoothed); IK
    converts each smoothed pose to joint space. Velocity clamping and
    commanded-position tracking match servo_l for smooth, deterministic
    joint trajectories. An unreachable pose brakes the tool along its
    twist; a predicted collision brakes it and ends the jog with the
    collision latched, as a planned move is refused.
    """

    PARAMS_TYPE = JogLCmd
    streamable = True

    __slots__ = (
        "_ik_stopping",
        "_collision_stopping",
        "_twist",
        "_dot_buf",
        "_q_commanded",
        "_q_ik_seed",
        "_dq_buf",
        "_pos_rad_buf",
        "_vel_ratio",
    )

    def __init__(self, p: JogLCmd):
        super().__init__(p)
        self._ik_stopping = False
        self._collision_stopping = False
        self._vel_ratio = 1.0

        self._twist = np.zeros(6, dtype=np.float64)
        self._dot_buf = np.zeros((), dtype=np.float64)
        self._q_commanded = np.zeros(6, dtype=np.float64)
        self._q_ik_seed = np.zeros(6, dtype=np.float64)
        self._dq_buf = np.zeros(6, dtype=np.float64)
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Resolve the twist and start the timer."""
        guard_homed(state)
        jog_twist(self.p.velocities, self._twist)
        self.start_timer(self.p.duration)
        self._ik_stopping = False
        self._collision_stopping = False

    def _track_and_send(self, state: "ControllerState", ik_q: np.ndarray) -> None:
        """Velocity-clamp IK result, update tracked position, send MOVE."""
        self._q_ik_seed[:] = ik_q
        dq = self._dq_buf
        for i in range(6):
            dq[i] = float(ik_q[i]) - self._q_commanded[i]
        ratio = _max_vel_ratio_jit(ik_q, self._q_commanded)
        if ratio > 1.0:
            for i in range(6):
                self._q_commanded[i] += dq[i] / ratio
            self._vel_ratio = ratio
        else:
            self._q_commanded[:] = ik_q
            self._vel_ratio = 1.0
        self._pos_rad_buf[:] = self._q_commanded
        rad_to_steps(self._pos_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

    def _command_twist(self, cse, scale: float) -> None:
        """Re-command the twist, held back by the factor the joints were:
        the tool keeps its direction and loses only speed."""
        if scale != 1.0:
            np.multiply(self._twist, scale, out=self._pos_rad_buf)
            cse.set_jog_twist(self._pos_rad_buf, self.p.frame == "WRF")
        else:
            cse.set_jog_twist(self._twist, self.p.frame == "WRF")

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of Cartesian jogging."""
        cse = state.cartesian_streaming_executor

        # Initialize only if not already active (preserve velocity across streaming)
        if not cse.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(1.0, self.p.accel)
            self._q_commanded[:] = self._q_rad_buf
            self._q_ik_seed[:] = self._q_rad_buf
            self._vel_ratio = 1.0

        # Handle timer expiry - stop smoothly
        if self.timer_expired():
            cse.stop()
            smoothed_pose, smoothed_vel, finished = cse.tick()

            np.dot(smoothed_vel, smoothed_vel, out=self._dot_buf)
            if not finished and self._dot_buf > 1e-8:
                ik_result = solve_ik(PAROL6_ROBOT.robot, smoothed_pose, self._q_ik_seed)
                if ik_result.success and ik_result.q is not None:
                    # Keep streaming while escaping from inside a keep-out,
                    # else the target freezes at release and the arm jerks.
                    checker = PAROL6_ROBOT.collision
                    if checker is None or not collision_blocked(
                        checker, self._q_commanded, ik_result.q
                    ):
                        self._track_and_send(state, ik_result.q)
                return ExecutionStatusCode.EXECUTING

            cse.active = False
            self.finish()
            self.stop_and_idle(state)
            return ExecutionStatusCode.COMPLETED

        # While stopping, leave the CSE target at zero — re-commanding the
        # twist every tick would defeat cse.stop()'s deceleration.
        if not self._ik_stopping and not self._collision_stopping:
            self._command_twist(cse, 1.0 / self._vel_ratio)

        smoothed_pose, smoothed_vel, _finished = cse.tick()

        if self._collision_stopping:
            # Braking to rest with the collision latched; the jog ends there
            # and does not resume on its own, like a refused planned move.
            np.dot(smoothed_vel, smoothed_vel, out=self._dot_buf)
            if self._dot_buf < 1e-8:
                cse.sync_pose(get_fkine_se3(state))
                cse.active = False
                self.fail_and_idle(
                    state,
                    make_error(
                        ErrorCode.SYS_SELF_COLLISION,
                        detail="jog_l stopped short of a predicted collision",
                    ),
                )
                return ExecutionStatusCode.FAILED
            # The brake's own configurations are gated too: the ones that
            # would reach the contact are withheld, and the arm holds the
            # last clear one while the smoother runs down.
            ik_result = solve_ik(PAROL6_ROBOT.robot, smoothed_pose, self._q_ik_seed)
            checker = PAROL6_ROBOT.collision
            if (
                ik_result.success
                and ik_result.q is not None
                and (
                    checker is None
                    or not collision_blocked(checker, self._q_commanded, ik_result.q)
                )
            ):
                self._track_and_send(state, ik_result.q)
            return ExecutionStatusCode.EXECUTING

        ik_result = solve_ik(
            PAROL6_ROBOT.robot,
            smoothed_pose,
            self._q_ik_seed,
        )
        if not ik_result.success or ik_result.q is None:
            if not self._ik_stopping:
                _ik_warn(
                    logger,
                    "[JOGL] IK failed - initiating graceful stop: pos=%s",
                    smoothed_pose[:3, 3],
                )
                cse.stop()
                self._ik_stopping = True
            else:
                # Still failing, check if we've stopped decelerating
                np.dot(smoothed_vel, smoothed_vel, out=self._dot_buf)
                if self._dot_buf < 1e-8:
                    cse.sync_pose(get_fkine_se3(state))
                    cse.active = False
                    self.finish()
                    return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        # A predicted collision brakes like an IK failure (no mid-jog
        # raise) but does not resume: the jog ends where it stopped with the
        # collision latched. Escaping from inside a keep-out stays allowed,
        # mirroring the planner guard.
        checker = PAROL6_ROBOT.collision
        if checker is not None and collision_blocked(
            checker, self._q_commanded, ik_result.q
        ):
            _ik_warn(
                logger,
                "[JOGL] collision predicted - stopping",
            )
            # Captured once on the stop transition (not every decel tick).
            state.collision_pairs = tuple(
                PAROL6_ROBOT.display_pairs(checker.colliding_pairs(ik_result.q))
            )
            state.collision_active = True
            cse.stop()
            self._collision_stopping = True
            return ExecutionStatusCode.EXECUTING

        # Reachable again — resume jogging.
        if self._ik_stopping:
            logger.info("[JOGL] pose reachable again - resuming jog")
            steps_to_rad(state.Position_in, self._q_rad_buf)
            cse.sync_pose(get_fkine_se3(state))
            self._q_commanded[:] = self._q_rad_buf
            self._q_ik_seed[:] = self._q_rad_buf
            self._vel_ratio = 1.0
            self._ik_stopping = False
            self._command_twist(cse, 1.0)

        self._track_and_send(state, ik_result.q)

        return ExecutionStatusCode.EXECUTING


def resolve_pose(
    start: np.ndarray, pose: "list[float]", frame: str, rel: bool
) -> np.ndarray:
    """The SE3 target a wire pose ``[x, y, z, rx, ry, rz]`` (mm, degrees)
    names, resolved against the pose the move starts from.

    A TRF pose is an offset in the tool frame at the start of the move,
    with or without ``rel``. A WRF pose is absolute unless ``rel``, in which
    case its rotation is applied about the TCP and its translation is added
    in world coordinates.
    """
    delta_se3 = np.zeros((4, 4), dtype=np.float64)
    se3_from_rpy(
        pose[0] / 1000.0,
        pose[1] / 1000.0,
        pose[2] / 1000.0,
        np.radians(pose[3]),
        np.radians(pose[4]),
        np.radians(pose[5]),
        delta_se3,
    )
    if frame == "TRF":
        return start @ delta_se3
    if rel:
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = delta_se3[:3, :3] @ start[:3, :3]
        target[:3, 3] = start[:3, 3] + delta_se3[:3, 3]
        return target
    return delta_se3


class CartesianChainLink:
    """A cartesian move that can join a blend chain: it contributes one
    segment, resolved against the pose the move before it ends at."""

    def chain_segment(
        self, previous: np.ndarray, state: "ControllerState"
    ) -> tuple[LineSegment | ArcSegment, np.ndarray]:
        """The segment this move traces from ``previous``, and its end pose."""
        raise NotImplementedError


def setup_cartesian_chain(
    head: "TrajectoryMoveCommandBase",
    state: "ControllerState",
    next_cmds: "list[TrajectoryMoveCommandBase]",
) -> int:
    """Plan ``head`` and the cartesian moves blended behind it as ONE path
    whose junctions are rounded. Returns how many of ``next_cmds`` the chain
    consumed; the head's trajectory covers them all. Falls back to the
    head's own setup when there is nothing to chain."""
    assert isinstance(head, CartesianChainLink)
    if head.blend_radius <= 0 or not next_cmds:
        head.do_setup(state)
        return 0

    chain: list[TrajectoryMoveCommandBase] = [head]
    for cmd in next_cmds:
        if isinstance(cmd, CartesianChainLink):
            chain.append(cmd)
            if cmd.blend_radius <= 0:
                break
        else:
            break
    if len(chain) < 2:
        head.do_setup(state)
        return 0

    initial_pose = get_fkine_se3(state).copy()
    segments: list[LineSegment | ArcSegment] = []
    blend_radii: list[float] = []
    previous = initial_pose
    for i, cmd in enumerate(chain):
        assert isinstance(cmd, CartesianChainLink)
        segment, end = cmd.chain_segment(previous, state)
        segments.append(segment)
        previous = end
        if i < len(chain) - 1:
            blend_radii.append(cmd.blend_radius)

    composite_poses = build_blended_path(
        segments, blend_radii, samples_per_segment=PATH_SAMPLES
    )
    if len(composite_poses) == 0:
        head.do_setup(state)
        return 0

    steps_to_rad(state.Position_in, head._q_rad_buf)
    joint_path = JointPath.from_poses(composite_poses, head._q_rad_buf)
    if joint_path.is_partial:
        assert joint_path.valid is not None
        raise TrajectoryPlanningError(
            make_error(
                ErrorCode.IK_PARTIAL_PATH,
                valid=str(int(joint_path.valid.sum())),
                total=str(len(joint_path)),
            )
        )
    guard_joint_path(joint_path.positions)

    # The chain runs under the slowest speed and acceleration fraction in
    # it; durations add up when every move carries one.
    min_speed = head.p.resolved_speed
    min_accel = head.p.accel
    total_duration = head.p.resolved_duration
    all_have_duration = total_duration is not None
    for cmd in chain[1:]:
        min_speed = min(min_speed, cmd.p.resolved_speed)
        min_accel = min(min_accel, cmd.p.accel)
        d = cmd.p.resolved_duration
        if all_have_duration and d is not None:
            assert total_duration is not None
            total_duration += d
        else:
            all_have_duration = False
            total_duration = None

    builder = TrajectoryBuilder(
        joint_path=joint_path,
        profile=state.motion_profile,
        velocity_frac=min_speed,
        accel_frac=min_accel,
        duration=total_duration,
        dt=INTERVAL_S,
        cart_vel_limit=LIMITS.cart.hard.velocity.linear * min_speed,
        cart_acc_limit=LIMITS.cart.hard.acceleration.linear * min_accel,
        path_knots=cartesian_path_knots(composite_poses),
    )
    trajectory = builder.build()
    head.trajectory_steps = trajectory.steps
    head.trajectory_rad = trajectory.positions_rad
    head._duration = trajectory.duration
    return len(chain) - 1


@register_command(CmdType.MOVEL)
class MoveLCommand(TrajectoryMoveCommandBase[MoveLCmd], CartesianChainLink):
    """Move the robot's end-effector in a straight line to a Cartesian pose.

    Supports absolute and relative modes via the `rel` field, and WRF/TRF frames.
    """

    PARAMS_TYPE = MoveLCmd

    __slots__ = (
        "initial_pose",
        "target_pose",
        "cartesian_diagnostic",
        "_cart_poses_buf",
    )

    def __init__(self, p: MoveLCmd):
        super().__init__(p)
        self.initial_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self.cartesian_diagnostic: dict | None = None
        self._cart_poses_buf = np.empty((PATH_SAMPLES, 4, 4), dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Set up the move - compute target pose and pre-compute trajectory."""
        guard_homed(state)
        self.initial_pose = get_fkine_se3(state)
        self._compute_target_pose(state)
        self._precompute_trajectory(state)

    def _precompute_trajectory(self, state: "ControllerState") -> None:
        """Pre-compute joint trajectory that follows straight-line Cartesian path."""
        from parol6.utils.errors import IKError

        assert self.initial_pose is not None and self.target_pose is not None

        steps_to_rad(state.Position_in, self._q_rad_buf)
        current_rad = self._q_rad_buf

        cart_poses = self._cart_poses_buf
        for i in range(PATH_SAMPLES):
            s = i / (PATH_SAMPLES - 1)
            se3_interp(self.initial_pose, self.target_pose, s, cart_poses[i])

        stop_on_failure = state.stop_on_failure
        joint_path = JointPath.from_poses(
            cart_poses,
            current_rad,
            stop_on_failure=stop_on_failure,
        )

        if not joint_path.is_partial:
            guard_joint_path(joint_path.positions)

        if joint_path.is_partial:
            ik_valid = joint_path.valid
            assert ik_valid is not None
            # Extract TCP poses (x,y,z,rx,ry,rz) in meters+radians from SE3
            n = len(cart_poses)
            tcp_poses = np.empty((n, 6), dtype=np.float64)
            _rpy_buf = np.empty(3, dtype=np.float64)
            for i in range(n):
                tcp_poses[i, :3] = cart_poses[i][:3, 3]
                se3_rpy(cart_poses[i], _rpy_buf)
                tcp_poses[i, 3:] = _rpy_buf
            self.cartesian_diagnostic = {
                "tcp_poses": tcp_poses,
                "ik_valid": ik_valid,
            }
            raise IKError(
                make_error(
                    ErrorCode.IK_PARTIAL_PATH,
                    valid=str(int(ik_valid.sum())),
                    total=str(len(ik_valid)),
                )
            )

        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=state.motion_profile,
            velocity_frac=self.p.resolved_speed,
            accel_frac=self.p.accel,
            duration=self.p.resolved_duration,
            dt=INTERVAL_S,
            cart_vel_limit=LIMITS.cart.hard.velocity.linear * self.p.resolved_speed,
            cart_acc_limit=LIMITS.cart.hard.acceleration.linear * self.p.accel,
            path_knots=cartesian_path_knots(cart_poses),
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self.trajectory_rad = trajectory.positions_rad
        self._duration = trajectory.duration

        self.log_debug(
            "  -> Pre-computed Cartesian path: profile=%s, steps=%d, duration=%.3fs",
            state.motion_profile,
            len(self.trajectory_steps),
            float(self._duration),
        )

    def _compute_target_pose(self, state: "ControllerState") -> None:
        self.target_pose = resolve_pose(
            cast(np.ndarray, self.initial_pose), self.p.pose, self.p.frame, self.p.rel
        )

    def chain_segment(
        self, previous: np.ndarray, state: "ControllerState"
    ) -> tuple[LineSegment | ArcSegment, np.ndarray]:
        end = resolve_pose(previous, self.p.pose, self.p.frame, self.p.rel)
        return LineSegment(previous, end), end

    def do_setup_with_blend(
        self,
        state: "ControllerState",
        next_cmds: "list[TrajectoryMoveCommandBase]",
    ) -> int:
        """Build one cartesian trajectory through the moves blended behind
        this one, straight or circular, with the junctions rounded."""
        guard_homed(state)
        return setup_cartesian_chain(self, state, next_cmds)
