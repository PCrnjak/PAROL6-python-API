"""
Servo Commands — streaming position targets (not queued).

ServoJ: joint-space position target via StreamingExecutor
ServoJPose: joint-space target from Cartesian pose (IK + StreamingExecutor)
ServoL: Cartesian-space target via CartesianStreamingExecutor + IK
"""

import logging
import math

import numpy as np
from numba import njit

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import (
    INTERVAL_S,
    LIMITS,
    rad_to_steps,
    steps_to_rad,
)
from parol6.protocol.wire import CmdType, ServoJCmd, ServoJPoseCmd, ServoLCmd
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.error_catalog import RobotError, make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError
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


#: The target a braking joint stream ramps toward. Read, never written:
#: ``set_jog_velocity`` copies it into the executor's own buffer.
_ZERO_JOINT_VEL = np.zeros(6, dtype=np.float64)


def _streaming_joint_step(
    cmd: "ServoJCommand | ServoJPoseCommand", state: ControllerState
) -> ExecutionStatusCode:
    """Shared execute_step for ServoJ and ServoJPose commands."""
    se = state.streaming_executor

    if not cmd._initialized or not se.active:
        steps_to_rad(state.Position_in, cmd._q_rad_buf)
        se.sync_position(cmd._q_rad_buf)
        cmd._initialized = True
        cmd._speed_applied = -1.0
        cmd._accel_applied = -1.0
    if cmd.p.speed != cmd._speed_applied or cmd.p.accel != cmd._accel_applied:
        # A stream re-targets through assign_params + do_setup, so a change
        # of speed or accel mid-stream reaches the limiter here.
        se.set_limits(cmd.p.speed, cmd.p.accel)
        cmd._speed_applied = cmd.p.speed
        cmd._accel_applied = cmd.p.accel

    # A target the arm cannot reach, or a client that has gone silent, ends
    # the stream by braking in joint space and holding where it stops.
    if cmd._braking or cmd.timer_expired():
        cmd._braking = True
        se.set_jog_velocity(_ZERO_JOINT_VEL)
    else:
        se.set_position_target(cmd._target_rad)
    pos_rad, vel, finished = se.tick()
    cmd._pos_rad_buf[:] = pos_rad
    rad_to_steps(cmd._pos_rad_buf, cmd._steps_buf)
    cmd.set_move_position(state, cmd._steps_buf)

    if cmd._braking:
        if not (finished or np.dot(vel, vel) < 1e-8):
            return ExecutionStatusCode.EXECUTING
        se.active = False
        if cmd._brake_error is not None:
            cmd.fail(cmd._brake_error)
            return ExecutionStatusCode.FAILED
        cmd.finish()
        return ExecutionStatusCode.COMPLETED

    if finished:
        se.active = False
        cmd.finish()
        return ExecutionStatusCode.COMPLETED

    return ExecutionStatusCode.EXECUTING


@register_command(CmdType.SERVOJ)
class ServoJCommand(MotionCommand[ServoJCmd]):
    """Streaming joint position target.

    Uses StreamingExecutor with set_position_target() for smooth Ruckig-
    interpolated motion to the target joint angles.
    """

    PARAMS_TYPE = ServoJCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_speed_applied",
        "_accel_applied",
        "_braking",
        "_brake_error",
        "_target_rad",
        "_pos_rad_buf",
    )

    def __init__(self, p: ServoJCmd):
        super().__init__(p)
        self._initialized = False
        self._speed_applied = -1.0
        self._accel_applied = -1.0
        self._braking = False
        self._brake_error: RobotError | None = None
        self._target_rad = [0.0] * 6
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
        guard_homed(state)
        # Target arrives in degrees; convert into pre-allocated radian buffer
        for i in range(6):
            self._target_rad[i] = math.radians(self.p.angles[i])
        self.start_timer(SERVO_GRACE_S)
        self._braking = False
        self._brake_error = None

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        return _streaming_joint_step(self, state)


@register_command(CmdType.SERVOJ_POSE)
class ServoJPoseCommand(MotionCommand[ServoJPoseCmd]):
    """Streaming joint position target via Cartesian pose.

    Solves IK for the target pose, then uses StreamingExecutor like ServoJ.
    """

    PARAMS_TYPE = ServoJPoseCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_speed_applied",
        "_accel_applied",
        "_braking",
        "_brake_error",
        "_target_rad",
        "_pos_rad_buf",
        "_target_se3",
    )

    def __init__(self, p: ServoJPoseCmd):
        super().__init__(p)
        self._initialized = False
        self._speed_applied = -1.0
        self._accel_applied = -1.0
        self._braking = False
        self._brake_error: RobotError | None = None
        self._target_rad = [0.0] * 6
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
        guard_homed(state)
        self.start_timer(SERVO_GRACE_S)
        self._braking = False
        self._brake_error = None
        pose = self.p.pose

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
            if not state.streaming_executor.active:
                raise IKError(error)
            self._braking = True
            self._brake_error = error
            return

        for i in range(6):
            self._target_rad[i] = float(ik_result.q[i])

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        return _streaming_joint_step(self, state)


@register_command(CmdType.SERVOL)
class ServoLCommand(MotionCommand[ServoLCmd]):
    """Streaming Cartesian position target.

    CSE drives the Cartesian path (with its own internal Ruckig for smooth
    TCP motion).  IK converts each smoothed pose to joint space.  If any
    joint's per-tick delta exceeds its hardware velocity limit, all deltas
    are scaled proportionally.

    A pose the solver cannot reach brakes the tool along its line and holds
    it there; the next target the client sends that differs from the one
    it braked away from resumes the stream from where the tool is.
    """

    PARAMS_TYPE = ServoLCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_ik_stopping",
        "_silent",
        "_target_se3",
        "_brake_pose",
        "_pos_rad_buf",
        "_q_commanded",
        "_q_ik_seed",
        "_dq_buf",
    )

    def __init__(self, p: ServoLCmd):
        super().__init__(p)
        self._initialized = False
        self._ik_stopping = False
        self._silent = False
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)
        # The wire pose the running brake gave up on.
        self._brake_pose = np.zeros(6, dtype=np.float64)
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)
        self._q_commanded = np.zeros(6, dtype=np.float64)
        self._q_ik_seed = np.zeros(6, dtype=np.float64)
        self._dq_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
        guard_homed(state)
        self.start_timer(SERVO_GRACE_S)
        self._silent = False
        pose = self.p.pose
        if self._ik_stopping:
            # Somewhere new ends the brake; the same pose keeps it. The
            # brake ran on past what the solver reaches while the arm held,
            # so the stream resumes from the arm, not from the brake.
            for i in range(6):
                if pose[i] != self._brake_pose[i]:
                    self._ik_stopping = False
                    self._initialized = False
                    break

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

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        cse = state.cartesian_streaming_executor

        if not self._initialized or not cse.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(self.p.speed, self.p.accel)
            self._q_commanded[:] = self._q_rad_buf
            self._q_ik_seed[:] = self._q_rad_buf
            self._initialized = True

        # A client that has gone silent stops refreshing its target: the
        # tool brakes along its line and holds where it stops.
        if not self._silent and self.timer_expired():
            self._silent = True
            cse.stop()
        # A brake owns the limiter: re-aiming it at the pose it is braking
        # away from would undo the brake on the next tick.
        if not (self._ik_stopping or self._silent):
            cse.set_pose_target(self._target_se3)
        smoothed_pose, vel, finished = cse.tick()

        # Solve IK seeded from previous IK result (branch continuity)
        ik_result = solve_ik(
            PAROL6_ROBOT.robot,
            smoothed_pose,
            self._q_ik_seed,
        )
        at_rest = finished or float(np.dot(vel, vel)) < 1e-8
        if self._silent:
            if ik_result.success and ik_result.q is not None:
                self._q_ik_seed[:] = ik_result.q
                self._q_commanded[:] = ik_result.q
            self._send(state)
            if at_rest:
                cse.active = False
                self.finish()
                return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING
        if ik_result.success and ik_result.q is not None:
            if self._ik_stopping:
                # Braking along the line: follow the brake's own poses.
                self._q_ik_seed[:] = ik_result.q
                self._q_commanded[:] = ik_result.q
            else:
                self._q_ik_seed[:] = ik_result.q

                dq = self._dq_buf
                for i in range(6):
                    dq[i] = float(ik_result.q[i]) - self._q_commanded[i]

                # Velocity ratio: worst-case joint vs its per-tick hard limit
                ratio = _max_vel_ratio_jit(ik_result.q, self._q_commanded)

                if ratio > 1.0:
                    for i in range(6):
                        self._q_commanded[i] += dq[i] / ratio
                    cse.set_limits(self.p.speed / ratio, self.p.accel)
                else:
                    self._q_commanded[:] = ik_result.q
                    cse.set_limits(self.p.speed, self.p.accel)
        elif not self._ik_stopping:
            # IK failed — graceful deceleration
            _ik_warn(
                logger,
                "[SERVOL] IK failed — decelerating: pos=%s",
                smoothed_pose[:3, 3],
            )
            cse.stop()
            self._ik_stopping = True
            self._brake_pose[:] = self.p.pose

        self._send(state)

        if finished and not self._ik_stopping:
            self.finish()
            cse.active = False
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING

    def _send(self, state: ControllerState) -> None:
        self._pos_rad_buf[:] = self._q_commanded
        rad_to_steps(self._pos_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)
