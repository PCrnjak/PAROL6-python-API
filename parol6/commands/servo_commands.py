"""
Servo Commands — streaming position targets (not queued).

ServoJ: joint-space position target via StreamingExecutor
ServoJPose: joint-space target from Cartesian pose (IK + StreamingExecutor)
ServoL: Cartesian-space target via CartesianStreamingExecutor + IK

Every servo stream keeps out of collision as a jog does: a target that
would collide is refused on arrival, and each tick the configuration the
stream is heading for one lookahead horizon on is checked; a contact ahead
brakes the stream to rest short of it, and it ends there with the
collision as its error. No step of that brake is commanded into contact.
"""

import logging
import math
from typing import TypeVar

import numpy as np
from numba import njit

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands._collision_guard import (
    collision_blocked,
    collision_stop,
    stream_lookahead,
)
from parol6.config import (
    INTERVAL_S,
    LIMITS,
    rad_to_steps,
    steps_to_rad,
)
from parol6.motion.streaming_executors import below_speed
from parol6.protocol.wire import CmdType, ServoJCmd, ServoJPoseCmd, ServoLCmd
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.error_catalog import RobotError, make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError
from parol6.utils.ik import RateLimitedWarning, solve_ik
from pinokin import se3_from_rpy

from .base import ExecutionStatusCode, MotionCommand, guard_homed

logger = logging.getLogger(__name__)

# A servo stream that goes silent for this long is braked to rest and
# held, rather than driven on to a target the client stopped refreshing.
SERVO_GRACE_S: float = 0.25

# Velocity ratio uses hardware limits (jog limits only apply to jog_j/jog_l)
_JOINT_MAX_STEP_INV = 1.0 / (
    np.array(LIMITS.joint.hard.velocity, dtype=np.float64) * INTERVAL_S
)
# Differential wrist coupling: motor 6 = J6*ratio[5] + J4*ratio[3] in step space

_J4_STEP_FACTOR: float = (
    float(PAROL6_ROBOT.joint.ratio[3]) / PAROL6_ROBOT.radian_per_step_constant
)
_J6_STEP_FACTOR: float = (
    float(PAROL6_ROBOT.joint.ratio[5]) / PAROL6_ROBOT.radian_per_step_constant
)
_MOTOR6_MAX_STEP_INV: float = 1.0 / (
    float(PAROL6_ROBOT._joint_max_speed_hw[5]) * INTERVAL_S
)
_ik_warn = RateLimitedWarning()

# Squared speed under which a braking stream counts as at rest.
_REST_SQ = 1e-8


@njit(cache=True)
def _max_vel_ratio_jit(
    target_q: np.ndarray,
    current_q: np.ndarray,
) -> float:
    """Max per-tick velocity ratio across all joints. >1.0 means limit exceeded.

    Accounts for the differential wrist coupling: motor6 drives both J6 and
    compensates for J4 rotation. The effective motor 6 step velocity is
    ``dJ6 * ratio[5] + dJ4 * ratio[3]`` (in step space), which must stay
    within motor 6's hardware speed limit.
    """
    max_ratio = 0.0
    n = target_q.shape[0]
    for i in range(n):
        r = abs(target_q[i] - current_q[i]) * _JOINT_MAX_STEP_INV[i]
        if r > max_ratio:
            max_ratio = r
    # Differential wrist: motor 6 effective speed includes J4 coupling
    if n >= 6:
        dq4_steps = abs(target_q[3] - current_q[3]) * _J4_STEP_FACTOR
        dq6_steps = abs(target_q[5] - current_q[5]) * _J6_STEP_FACTOR
        motor6_ratio = (dq4_steps + dq6_steps) * _MOTOR6_MAX_STEP_INV
        if motor6_ratio > max_ratio:
            max_ratio = motor6_ratio
    return max_ratio


@njit(cache=True)
def _step_toward_jit(q_commanded: np.ndarray, q_target: np.ndarray) -> float:
    """Move ``q_commanded`` toward ``q_target`` in place, as far as one tick
    of every joint's hardware speed allows: every joint's step is divided by
    the worst joint's share of its budget, so the arm keeps its direction
    through joint space and only loses speed. Returns that share; at or
    under 1 the step landed on ``q_target``."""
    ratio = _max_vel_ratio_jit(q_target, q_commanded)
    if ratio > 1.0:
        inv = 1.0 / ratio
        for i in range(q_commanded.shape[0]):
            q_commanded[i] += (q_target[i] - q_commanded[i]) * inv
    else:
        for i in range(q_commanded.shape[0]):
            q_commanded[i] = q_target[i]
    return ratio


#: The target a braking joint stream ramps toward. Read, never written:
#: ``set_jog_velocity`` copies it into the executor's own buffer.
_ZERO_JOINT_VEL = np.zeros(6, dtype=np.float64)


_JP = TypeVar("_JP", ServoJCmd, ServoJPoseCmd)


class _JointServoCommand(MotionCommand[_JP]):
    """A joint-space servo stream: the StreamingExecutor interpolates to
    each target; a contact ahead, an unreachable target or a client gone
    silent brakes it in joint space to a hold."""

    streamable = True

    __slots__ = (
        "_initialized",
        "_retarget",
        "_speed_applied",
        "_accel_applied",
        "_braking",
        "_collision",
        "_brake_error",
        "_target_rad",
        "_target_q",
        "_q_sent",
        "_la_buf",
    )

    def __init__(self, p: _JP):
        super().__init__(p)
        self._initialized = False
        self._retarget = True
        self._speed_applied = -1.0
        self._accel_applied = -1.0
        self._braking = False
        # A collision brake stays latched across the datagrams that keep
        # the stream alive: the stream ends where it stopped, in error.
        self._collision = False
        self._brake_error: RobotError | None = None
        self._target_rad = [0.0] * 6
        self._target_q = np.zeros(6, dtype=np.float64)
        # The configuration last commanded.
        self._q_sent = np.zeros(6, dtype=np.float64)
        self._la_buf = np.zeros(6, dtype=np.float64)

    def _set_target(
        self, q: "list[float] | np.ndarray", state: ControllerState
    ) -> None:
        """Aim the stream at ``q`` [rad], refusing it on arrival if it would
        collide: a running stream brakes to rest with the collision as its
        error, one not yet running never starts."""
        for i in range(6):
            v = float(q[i])
            self._target_rad[i] = v
            self._target_q[i] = v
        checker = PAROL6_ROBOT.collision
        if checker is None:
            return
        running = self._initialized and state.streaming_executor.active
        if not running:
            steps_to_rad(state.Position_in, self._q_sent)
        if not collision_blocked(checker, self._q_sent, self._target_q):
            return
        error = collision_stop(state, checker, self._target_q)
        if not running:
            raise TrajectoryPlanningError(error)
        self._braking = True
        self._collision = True
        self._brake_error = error

    def _step_clear(
        self, state: ControllerState, pos: np.ndarray, vel: np.ndarray
    ) -> bool:
        """Whether the stream may command ``pos``, reached at ``vel``. While
        it tracks its target, a contact one lookahead horizon ahead — never
        past the target, where it stops — starts the collision brake. A
        brake comes to rest short of any horizon, so each of its steps is
        checked on its own instead; one that would reach a contact is held
        back, and the stream ends there in collision."""
        checker = PAROL6_ROBOT.collision
        if checker is None:
            return True
        if self._braking:
            if not collision_blocked(checker, self._q_sent, pos):
                return True
            if not self._collision:
                self._brake_error = collision_stop(state, checker, pos)
                self._collision = True
            return False
        stream_lookahead(pos, vel, self._la_buf, self._target_q)
        if not collision_blocked(checker, self._q_sent, self._la_buf):
            return True
        logger.warning("[%s] collision predicted - braking", self.name)
        self._brake_error = collision_stop(state, checker, self._la_buf)
        self._braking = True
        self._collision = True
        return not collision_blocked(checker, self._q_sent, pos)

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        se = state.streaming_executor

        if not self._initialized or not se.active:
            steps_to_rad(state.Position_in, self._q_sent)
            se.sync_position(self._q_sent)
            self._initialized = True
            self._retarget = True
            self._speed_applied = -1.0
            self._accel_applied = -1.0
        if self.p.speed != self._speed_applied or self.p.accel != self._accel_applied:
            # A stream re-targets through assign_params + do_setup, so a
            # change of speed or accel mid-stream reaches the limiter here.
            se.set_limits(self.p.speed, self.p.accel)
            self._speed_applied = self.p.speed
            self._accel_applied = self.p.accel

        if self._braking or self.timer_expired():
            self._braking = True
            se.set_jog_velocity(_ZERO_JOINT_VEL)
        elif self._retarget:
            se.set_position_target(self._target_rad)
            self._retarget = False
        pos_rad, vel, finished = se.tick()
        if self._step_clear(state, pos_rad, vel):
            self._q_sent[:] = pos_rad
        rad_to_steps(self._q_sent, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        if self._braking:
            if not (finished or below_speed(vel, _REST_SQ)):
                return ExecutionStatusCode.EXECUTING
            se.active = False
            if self._brake_error is not None:
                self.fail(self._brake_error)
                return ExecutionStatusCode.FAILED
            self.finish()
            return ExecutionStatusCode.COMPLETED

        if finished:
            se.active = False
            self.finish()
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.SERVOJ)
class ServoJCommand(_JointServoCommand[ServoJCmd]):
    """Streaming joint position target.

    Uses StreamingExecutor with set_position_target() for smooth Ruckig-
    interpolated motion to the target joint angles.
    """

    PARAMS_TYPE = ServoJCmd

    __slots__ = ("_angles_seen", "_angles_rad")

    def __init__(self, p: ServoJCmd):
        super().__init__(p)
        self._angles_seen: list[float] | None = None
        self._angles_rad = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
        guard_homed(state)
        self.start_timer(SERVO_GRACE_S)
        if self._collision:
            return
        if self._braking:
            # A client heard from again ends the brake its silence began.
            self._braking = False
            self._retarget = True
        angles = self.p.angles
        if angles == self._angles_seen:
            return
        self._angles_seen = angles
        for i in range(6):
            self._angles_rad[i] = math.radians(angles[i])
        self._retarget = True
        self._set_target(self._angles_rad, state)


@register_command(CmdType.SERVOJ_POSE)
class ServoJPoseCommand(_JointServoCommand[ServoJPoseCmd]):
    """Streaming joint position target via Cartesian pose.

    Solves IK for the target pose, then uses StreamingExecutor like ServoJ.
    """

    PARAMS_TYPE = ServoJPoseCmd

    __slots__ = ("_target_se3", "_pose_seen", "_unreachable")

    def __init__(self, p: ServoJPoseCmd):
        super().__init__(p)
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)
        self._pose_seen: list[float] | None = None
        # The refusal of the pose last seen, when the solver could not
        # reach it: a client resending it gets the same answer unsolved.
        self._unreachable: RobotError | None = None

    def do_setup(self, state: ControllerState) -> None:
        guard_homed(state)
        self.start_timer(SERVO_GRACE_S)
        if self._collision:
            return
        pose = self.p.pose
        if pose == self._pose_seen:
            if self._braking and self._unreachable is None:
                # A client heard from again ends the brake its silence began.
                self._braking = False
                self._retarget = True
            return
        self._pose_seen = pose
        self._unreachable = None
        self._braking = False
        self._brake_error = None
        self._retarget = True

        # Build target SE3 from [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            math.radians(pose[3]),
            math.radians(pose[4]),
            math.radians(pose[5]),
            self._target_se3,
        )

        # Seed IK with current joint angles for branch continuity
        steps_to_rad(state.Position_in, self._q_rad_buf)
        ik_result = solve_ik(PAROL6_ROBOT.robot, self._target_se3, self._q_rad_buf)
        if not ik_result.success or ik_result.q is None:
            # Unreachable: the stream brakes to rest where it is and ends in
            # error there, rather than stopping dead on a setup failure; the
            # next reachable target starts it again. A stream that was not
            # running has nothing to brake and is refused outright.
            error = make_error(
                ErrorCode.IK_TARGET_UNREACHABLE,
                detail=f"SERVOJ_POSE: IK failed for pose {[round(v, 1) for v in pose]}",
            )
            if not (self._initialized and state.streaming_executor.active):
                raise IKError(error)
            self._unreachable = error
            self._braking = True
            self._brake_error = error
            return
        self._set_target(ik_result.q, state)


@register_command(CmdType.SERVOL)
class ServoLCommand(MotionCommand[ServoLCmd]):
    """Streaming Cartesian position target.

    CSE drives the Cartesian path (with its own internal Ruckig for smooth
    TCP motion).  IK converts each smoothed pose to joint space.  If any
    joint's per-tick delta exceeds its hardware velocity limit, all deltas
    are scaled proportionally — on every path, the brakes included — and
    the stream ends only once the joints have caught up with the tool.

    A pose the solver cannot reach brakes the tool along its line and holds
    it there; the next target the client sends that differs from the one
    it braked away from resumes the stream from where the tool is.
    """

    PARAMS_TYPE = ServoLCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_ik_stopping",
        "_held",
        "_silent",
        "_collision_error",
        "_target_se3",
        "_pose_seen",
        "_brake_pose",
        "_q_commanded",
        "_q_prev",
        "_q_ik_seed",
        "_q_target",
        "_target_solved",
        "_dq_buf",
        "_la_buf",
    )

    def __init__(self, p: ServoLCmd):
        super().__init__(p)
        self._initialized = False
        self._ik_stopping = False
        # Braked to rest short of an unreachable target: nothing moves
        # until the client names somewhere new or goes silent.
        self._held = False
        self._silent = False
        # Set once a contact stops the stream; latched across datagrams.
        self._collision_error: RobotError | None = None
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)
        self._pose_seen: list[float] | None = None
        # The wire pose the running brake gave up on.
        self._brake_pose: list[float] | None = None
        self._q_commanded = np.zeros(6, dtype=np.float64)
        self._q_prev = np.zeros(6, dtype=np.float64)
        self._q_ik_seed = np.zeros(6, dtype=np.float64)
        # The joints the target solves to, when the solver reached it.
        self._q_target = np.zeros(6, dtype=np.float64)
        self._target_solved = False
        self._dq_buf = np.zeros(6, dtype=np.float64)
        self._la_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
        guard_homed(state)
        self.start_timer(SERVO_GRACE_S)
        self._silent = False
        running = self._initialized and state.cartesian_streaming_executor.active
        pose = self.p.pose
        if self._ik_stopping and pose != self._brake_pose:
            # Somewhere new ends the brake; the same pose keeps it. The
            # brake ran on past what the solver reaches while the arm held,
            # so the stream resumes from the arm, not from the brake.
            self._ik_stopping = False
            self._held = False
            self._initialized = False
        if pose == self._pose_seen:
            return
        self._pose_seen = pose
        self._target_solved = False

        # Build target SE3 from [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            math.radians(pose[3]),
            math.radians(pose[4]),
            math.radians(pose[5]),
            self._target_se3,
        )
        if self._collision_error is None:
            self._refuse_colliding_target(state, running)

    def _refuse_colliding_target(self, state: ControllerState, running: bool) -> None:
        """Refuse a target whose solution would collide, on arrival: a
        ``running`` stream brakes to rest with the collision as its error,
        one not yet running never starts. A target the solver cannot reach
        is left to the stream, which brakes short of it."""
        checker = PAROL6_ROBOT.collision
        if checker is None:
            return
        cse = state.cartesian_streaming_executor
        if not running:
            steps_to_rad(state.Position_in, self._q_commanded)
            self._q_ik_seed[:] = self._q_commanded
        ik_result = solve_ik(PAROL6_ROBOT.robot, self._target_se3, self._q_ik_seed)
        if not ik_result.success:
            return
        self._q_target[:] = ik_result.q
        self._target_solved = True
        if not collision_blocked(checker, self._q_commanded, self._q_target):
            return
        error = collision_stop(state, checker, self._q_target)
        if not running:
            raise TrajectoryPlanningError(error)
        self._collision_error = error
        cse.stop()

    def _step_clear(self, state: ControllerState, braking: bool) -> bool:
        """Whether the step just taken, from ``_q_prev`` to ``_q_commanded``,
        may be commanded; a blocked one is taken back. While the stream
        tracks its target, a contact one lookahead horizon ahead — never
        past the target's joints, where it stops — starts the collision
        brake. A brake comes to rest short of any horizon, so each of its
        steps is checked on its own instead; one that would reach a contact
        ends the stream there in collision."""
        checker = PAROL6_ROBOT.collision
        if checker is None:
            return True
        if not braking:
            np.subtract(self._q_commanded, self._q_prev, out=self._dq_buf)
            self._dq_buf *= 1.0 / INTERVAL_S
            stream_lookahead(
                self._q_commanded,
                self._dq_buf,
                self._la_buf,
                self._q_target if self._target_solved else None,
            )
            if not collision_blocked(checker, self._q_prev, self._la_buf):
                return True
            logger.warning("[SERVOL] collision predicted - braking")
            self._collision_error = collision_stop(state, checker, self._la_buf)
            state.cartesian_streaming_executor.stop()
        if not collision_blocked(checker, self._q_prev, self._q_commanded):
            return True
        if self._collision_error is None:
            self._collision_error = collision_stop(state, checker, self._q_commanded)
        self._q_commanded[:] = self._q_prev
        return False

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        cse = state.cartesian_streaming_executor

        if not self._initialized or not cse.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(self.p.speed, self.p.accel)
            self._q_commanded[:] = self._q_rad_buf
            self._q_ik_seed[:] = self._q_rad_buf
            self._held = False
            self._initialized = True

        # A client that has gone silent stops refreshing its target: the
        # tool brakes along its line and holds where it stops.
        if not self._silent and self.timer_expired():
            self._silent = True
            cse.stop()
        if self._held:
            self._send(state)
            if self._silent:
                cse.active = False
                self.finish()
                return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        # A brake owns the limiter: re-aiming it at the pose it is braking
        # away from would undo the brake on the next tick.
        braking = self._collision_error is not None or self._ik_stopping or self._silent
        if not braking:
            cse.set_pose_target(self._target_se3)
        smoothed_pose, vel, finished = cse.tick()

        # Solve IK seeded from previous IK result (branch continuity)
        ik_result = solve_ik(
            PAROL6_ROBOT.robot,
            smoothed_pose,
            self._q_ik_seed,
        )
        # Whether the joints have caught up with the tool.
        landed = True
        if ik_result.success and ik_result.q is not None:
            self._q_ik_seed[:] = ik_result.q
            self._q_prev[:] = self._q_commanded
            ratio = _step_toward_jit(self._q_commanded, ik_result.q)
            landed = ratio <= 1.0
            if not self._step_clear(state, braking):
                landed = True
            elif not braking:
                # Slow the tool by the factor the joints were held back.
                cse.set_limits(self.p.speed / max(ratio, 1.0), self.p.accel)
        elif not braking:
            _ik_warn(
                logger,
                "[SERVOL] IK failed — decelerating: pos=%s",
                smoothed_pose[:3, 3],
            )
            cse.stop()
            self._ik_stopping = True
            self._brake_pose = self.p.pose

        self._send(state)

        if self._collision_error is not None or self._silent or self._ik_stopping:
            if not landed or not (finished or below_speed(vel, _REST_SQ)):
                return ExecutionStatusCode.EXECUTING
            if self._collision_error is not None:
                cse.active = False
                self.fail_and_idle(state, self._collision_error)
                return ExecutionStatusCode.FAILED
            if self._silent:
                cse.active = False
                self.finish()
                return ExecutionStatusCode.COMPLETED
            self._held = True
            return ExecutionStatusCode.EXECUTING

        if finished and landed:
            self.finish()
            cse.active = False
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING

    def _send(self, state: ControllerState) -> None:
        rad_to_steps(self._q_commanded, self._steps_buf)
        self.set_move_position(state, self._steps_buf)
