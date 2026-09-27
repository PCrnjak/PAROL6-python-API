"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, MoveCart, MoveCartRelTrf
"""

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands._collision_guard import (
    collision_blocked,
    collision_stop,
    guard_cartesian_path,
)
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
)
from parol6.protocol.wire import (
    CmdType,
    JogLCmd,
    MoveLCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.error_catalog import RobotError, make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError
from parol6.utils.ik import RateLimitedWarning, solve_ik
from pinokin import se3_from_rpy, se3_rpy

from parol6.commands.servo_commands import _step_toward_jit
from parol6.motion.streaming_executors import below_speed

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
    joint trajectories. The twist is resolved against the tool as it now
    stands, every tick: a tool-frame jog moves along the tool's axes as
    they are after any turn before it, a world-frame turn is about the
    world axis. An unreachable pose brakes the tool along its twist; a
    predicted collision brakes it and ends the jog with the collision
    latched, as a planned move is refused. The duration is counted in
    control ticks.
    """

    PARAMS_TYPE = JogLCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_accel_applied",
        "_ik_stopping",
        "_released",
        "_collision_error",
        "_twist",
        "_scaled_twist",
        "_q_commanded",
        "_q_ik_seed",
        "_vel_ratio",
    )

    def __init__(self, p: JogLCmd):
        super().__init__(p)
        self._initialized = False
        self._accel_applied = -1.0
        self._ik_stopping = False
        # The duration ran out and the brake has been handed to the CSE.
        self._released = False
        # Set once a contact stops the jog; latched across datagrams.
        self._collision_error: RobotError | None = None
        self._vel_ratio = 1.0

        self._twist = np.zeros(6, dtype=np.float64)
        self._scaled_twist = np.zeros(6, dtype=np.float64)
        self._q_commanded = np.zeros(6, dtype=np.float64)
        self._q_ik_seed = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Resolve the twist and start the timer. A collision stop stays
        latched across the datagrams that keep the stream alive: the jog
        ends where it braked, like a refused planned move."""
        guard_homed(state)
        jog_twist(self.p.velocities, self._twist)
        self.start_tick_timer(self.p.duration)
        self._ik_stopping = False

    def _sync(self, state: "ControllerState") -> None:
        """Start the CSE and the joint tracking from the arm, at rest."""
        steps_to_rad(state.Position_in, self._q_rad_buf)
        state.cartesian_streaming_executor.sync_pose(get_fkine_se3(state))
        self._q_commanded[:] = self._q_rad_buf
        self._q_ik_seed[:] = self._q_rad_buf
        self._vel_ratio = 1.0

    def _track_and_send(self, state: "ControllerState", ik_q: np.ndarray) -> None:
        """Velocity-clamp IK result, update tracked position, send MOVE."""
        self._q_ik_seed[:] = ik_q
        ratio = _step_toward_jit(self._q_commanded, ik_q)
        self._vel_ratio = ratio if ratio > 1.0 else 1.0
        rad_to_steps(self._q_commanded, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

    def _send_if_clear(self, state: "ControllerState", pose: np.ndarray) -> None:
        """Solve *pose* and send it, unless the step there would reach a
        contact: a brake's configurations are gated like the jog's own, and
        the arm holds the last clear one instead."""
        ik_result = solve_ik(PAROL6_ROBOT.robot, pose, self._q_ik_seed)
        if not ik_result.success or ik_result.q is None:
            return
        checker = PAROL6_ROBOT.collision
        if checker is None or not collision_blocked(
            checker, self._q_commanded, ik_result.q
        ):
            self._track_and_send(state, ik_result.q)

    def _command_twist(self, cse) -> None:
        """Command the twist, held back by the factor the joints were: the
        tool keeps its direction and loses only speed."""
        self._released = False
        if self._vel_ratio != 1.0:
            np.multiply(self._twist, 1.0 / self._vel_ratio, out=self._scaled_twist)
            cse.set_jog_twist(self._scaled_twist, self.p.frame == "WRF")
        else:
            cse.set_jog_twist(self._twist, self.p.frame == "WRF")

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of Cartesian jogging."""
        cse = state.cartesian_streaming_executor

        # A new jog starts from the arm; one continued by the next datagram
        # keeps the velocity it is at.
        if not self._initialized or not cse.active:
            self._sync(state)
            self._accel_applied = -1.0
            self._initialized = True
        if self.p.accel != self._accel_applied:
            cse.set_limits(1.0, self.p.accel)
            self._accel_applied = self.p.accel

        if self._collision_error is not None:
            # Braking to rest with the collision latched, whatever the timer
            # says; the jog ends there in error and does not resume on its
            # own, like a refused planned move.
            smoothed_pose, smoothed_vel, _finished = cse.tick()
            if below_speed(smoothed_vel, 1e-8):
                cse.active = False
                self.fail_and_idle(state, self._collision_error)
                return ExecutionStatusCode.FAILED
            self._send_if_clear(state, smoothed_pose)
            return ExecutionStatusCode.EXECUTING

        # Handle timer expiry - stop smoothly
        if self.tick_timer_expired():
            if not self._released:
                cse.stop()
                self._released = True
            smoothed_pose, smoothed_vel, finished = cse.tick()

            if not finished and not below_speed(smoothed_vel, 1e-8):
                # Keep streaming while escaping from inside a keep-out,
                # else the target freezes at release and the arm jerks.
                self._send_if_clear(state, smoothed_pose)
                return ExecutionStatusCode.EXECUTING

            cse.active = False
            self.finish()
            self.stop_and_idle(state)
            return ExecutionStatusCode.COMPLETED

        # While stopping, leave the CSE target at zero — re-commanding the
        # twist every tick would defeat cse.stop()'s deceleration.
        if not self._ik_stopping:
            self._command_twist(cse)

        smoothed_pose, smoothed_vel, _finished = cse.tick()

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
            elif below_speed(smoothed_vel, 1e-8):
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
            self._collision_error = collision_stop(state, checker, ik_result.q)
            cse.stop()
            return ExecutionStatusCode.EXECUTING

        # Reachable again — resume jogging.
        if self._ik_stopping:
            logger.info("[JOGL] pose reachable again - resuming jog")
            self._sync(state)
            self._ik_stopping = False
            self._command_twist(cse)

        self._track_and_send(state, ik_result.q)

        return ExecutionStatusCode.EXECUTING


def pose6_to_se3(pose: Sequence[float], out: np.ndarray) -> np.ndarray:
    """Write the SE3 of a wire pose ``[x, y, z, rx, ry, rz]`` (mm, degrees)
    into ``out`` and return it."""
    se3_from_rpy(
        pose[0] / 1000.0,
        pose[1] / 1000.0,
        pose[2] / 1000.0,
        np.radians(pose[3]),
        np.radians(pose[4]),
        np.radians(pose[5]),
        out,
    )
    return out


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
    delta_se3 = pose6_to_se3(pose, np.zeros((4, 4), dtype=np.float64))
    if frame == "TRF":
        return start @ delta_se3
    if rel:
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = delta_se3[:3, :3] @ start[:3, :3]
        target[:3, 3] = start[:3, 3] + delta_se3[:3, 3]
        return target
    return delta_se3


def cartesian_diagnostic(poses: np.ndarray, ik_valid: np.ndarray) -> dict:
    """What a dry run draws of a cartesian path IK could not solve all of:
    the path's TCP poses (x, y, z in m, roll, pitch, yaw in rad) and which
    of them solved."""
    n = len(poses)
    tcp_poses = np.empty((n, 6), dtype=np.float64)
    rpy = np.empty(3, dtype=np.float64)
    for i in range(n):
        tcp_poses[i, :3] = poses[i][:3, 3]
        se3_rpy(poses[i], rpy)
        tcp_poses[i, 3:] = rpy
    return {"tcp_poses": tcp_poses, "ik_valid": ik_valid}


class CartesianChainLink:
    """A cartesian move that can join a blend chain: it contributes one
    segment, resolved against the pose the move before it ends at."""

    #: Where the move ends, once planning has resolved it: a dry run that
    #: fails the move carries on from there.
    target_pose: np.ndarray | None = None
    #: The path a dry run draws for a move that fails in IK.
    cartesian_diagnostic: dict | None = None

    def chain_segment(
        self, previous: np.ndarray, state: "ControllerState"
    ) -> tuple[LineSegment | ArcSegment, np.ndarray]:
        """The segment this move traces from ``previous``, and its end pose."""
        raise NotImplementedError

    def do_setup_with_blend(
        self,
        state: "ControllerState",
        next_cmds: "list[TrajectoryMoveCommandBase]",
    ) -> int:
        """Build one cartesian trajectory through the moves blended behind
        this one, straight or circular, with the junctions rounded."""
        assert isinstance(self, TrajectoryMoveCommandBase)
        guard_homed(state)
        return setup_cartesian_chain(self, state, next_cmds)


def setup_cartesian_chain(
    head: "TrajectoryMoveCommandBase",
    state: "ControllerState",
    next_cmds: "list[TrajectoryMoveCommandBase]",
) -> int:
    """Plan ``head`` and the cartesian moves blended behind it as ONE path
    whose junctions are rounded. Returns how many of ``next_cmds`` the chain
    consumed; the head's trajectory covers them all. Falls back to the
    head's own setup when there is nothing to chain.

    A move whose segment cannot be built (a move_c whose via names no
    circle) ends the chain ahead of it: the moves before it run, stopping
    where it would have started, and it fails on its own setup as it
    would alone."""
    assert isinstance(head, CartesianChainLink)
    chain: list[TrajectoryMoveCommandBase] = [head]
    if head.blend_radius > 0:
        for cmd in next_cmds:
            if not isinstance(cmd, CartesianChainLink):
                break
            chain.append(cmd)
            if cmd.blend_radius <= 0:
                break
    if len(chain) < 2:
        head.do_setup(state)
        return 0

    segments: list[LineSegment | ArcSegment] = []
    previous = get_fkine_se3(state).copy()
    for i, cmd in enumerate(chain):
        assert isinstance(cmd, CartesianChainLink)
        try:
            segment, end = cmd.chain_segment(previous, state)
        except TrajectoryPlanningError:
            if i == 0:
                raise
            del chain[i:]
            break
        segments.append(segment)
        previous = end
        if i == 0:
            head.target_pose = end
    if len(chain) < 2:
        head.do_setup(state)
        return 0
    blend_radii = [cmd.blend_radius for cmd in chain[:-1]]

    composite_poses = build_blended_path(
        segments, blend_radii, samples_per_segment=PATH_SAMPLES
    )

    steps_to_rad(state.Position_in, head._q_rad_buf)
    joint_path = JointPath.from_poses(
        composite_poses, head._q_rad_buf, stop_on_failure=state.stop_on_failure
    )
    if joint_path.is_partial:
        assert joint_path.valid is not None
        head.cartesian_diagnostic = cartesian_diagnostic(
            composite_poses, joint_path.valid
        )
        raise IKError(
            make_error(
                ErrorCode.IK_PARTIAL_PATH,
                valid=str(int(joint_path.valid.sum())),
                total=str(len(joint_path)),
            )
        )
    guard_cartesian_path(joint_path)

    # The chain runs under the slowest speed and acceleration fraction in
    # it; durations add up when every move carries one.
    min_speed = min(c.p.resolved_speed for c in chain)
    min_accel = min(c.p.accel for c in chain)
    durations = [c.p.resolved_duration for c in chain]
    total_duration = (
        sum(cast(list[float], durations)) if None not in durations else None
    )

    builder = TrajectoryBuilder(
        joint_path=joint_path,
        profile=state.motion_profile,
        velocity_frac=min_speed,
        accel_frac=min_accel,
        duration=total_duration,
        dt=INTERVAL_S,
        cart_vel_limit=LIMITS.cart.hard.velocity.linear * min_speed,
        cart_acc_limit=LIMITS.cart.hard.acceleration.linear * min_accel,
        path_knots=joint_path.knots,
    )
    trajectory = builder.build()
    head.trajectory_steps = trajectory.steps
    head.trajectory_rad = trajectory.positions_rad
    head._duration = trajectory.duration
    return len(chain) - 1


@register_command(CmdType.MOVEL)
class MoveLCommand(CartesianChainLink, TrajectoryMoveCommandBase[MoveLCmd]):
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
        assert self.initial_pose is not None and self.target_pose is not None

        steps_to_rad(state.Position_in, self._q_rad_buf)
        current_rad = self._q_rad_buf

        cart_poses = self._cart_poses_buf
        LineSegment(self.initial_pose, self.target_pose).sample_into(
            cart_poses, 0.0, 1.0, 0
        )

        stop_on_failure = state.stop_on_failure
        joint_path = JointPath.from_poses(
            cart_poses,
            current_rad,
            stop_on_failure=stop_on_failure,
        )

        if not joint_path.is_partial:
            guard_cartesian_path(joint_path)

        if joint_path.is_partial:
            ik_valid = joint_path.valid
            assert ik_valid is not None
            self.cartesian_diagnostic = cartesian_diagnostic(cart_poses, ik_valid)
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
            path_knots=joint_path.knots,
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
