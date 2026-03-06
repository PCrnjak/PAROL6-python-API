"""
Servo Commands — streaming position targets (not queued).

ServoJ: joint-space position target via StreamingExecutor
ServoJPose: joint-space target from Cartesian pose (IK + StreamingExecutor)
ServoL: Cartesian-space target via CartesianStreamingExecutor + IK
"""

import logging
import math
import time

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
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError
from parol6.utils.ik import solve_ik
from pinokin import se3_from_rpy

from .base import ExecutionStatusCode, MotionCommand

logger = logging.getLogger(__name__)

# Velocity ratio uses hardware limits (jog limits only apply to jogJ/jogL)
_JOINT_MAX_STEP_INV = 1.0 / (
    np.array(LIMITS.joint.hard.velocity, dtype=np.float64) * INTERVAL_S
)
# Rate-limiting for IK failure warnings
_IK_WARN_INTERVAL: float = 1.0
_last_servo_ik_warn: float = 0.0


@njit(cache=True)
def _max_vel_ratio_jit(
    target_q: np.ndarray,
    current_q: np.ndarray,
) -> float:
    """Max per-tick velocity ratio across all joints. >1.0 means limit exceeded."""
    max_ratio = 0.0
    n = target_q.shape[0]
    for i in range(n):
        r = abs(target_q[i] - current_q[i]) * _JOINT_MAX_STEP_INV[i]
        if r > max_ratio:
            max_ratio = r
    return max_ratio


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
        "_target_rad",
        "_pos_rad_buf",
    )

    def __init__(self, p: ServoJCmd):
        super().__init__(p)
        self._initialized = False
        self._target_rad = [0.0] * 6
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
        # Convert target from degrees to radians into pre-allocated list
        for i in range(6):
            self._target_rad[i] = math.radians(self.p.target[i])

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        se = state.streaming_executor

        # Sync position on first tick or if executor not active
        if not self._initialized or not se.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            se.sync_position(self._q_rad_buf)
            se.set_limits(self.p.speed, self.p.accel)
            self._initialized = True

        se.set_position_target(self._target_rad)
        pos_rad, _vel, finished = se.tick()

        self._pos_rad_buf[:] = pos_rad
        rad_to_steps(self._pos_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        if finished:
            self.finish()
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.SERVOJ_POSE)
class ServoJPoseCommand(MotionCommand[ServoJPoseCmd]):
    """Streaming joint position target via Cartesian pose.

    Solves IK for the target pose, then uses StreamingExecutor like ServoJ.
    """

    PARAMS_TYPE = ServoJPoseCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_target_rad",
        "_pos_rad_buf",
        "_target_se3",
    )

    def __init__(self, p: ServoJPoseCmd):
        super().__init__(p)
        self._initialized = False
        self._target_rad = [0.0] * 6
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
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

        # Solve IK using current joint angles as seed
        steps_to_rad(state.Position_in, self._q_rad_buf)
        ik_result = solve_ik(PAROL6_ROBOT.robot, self._target_se3, self._q_rad_buf)
        if not ik_result.success or ik_result.q is None:
            raise IKError(
                make_error(
                    ErrorCode.IK_TARGET_UNREACHABLE,
                    detail=f"SERVOJ_POSE: IK failed for pose {[round(v, 1) for v in pose]}",
                )
            )

        for i in range(6):
            self._target_rad[i] = float(ik_result.q[i])

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        se = state.streaming_executor

        # Sync position on first tick or if executor not active
        if not self._initialized or not se.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            se.sync_position(self._q_rad_buf)
            se.set_limits(self.p.speed, self.p.accel)
            self._initialized = True

        se.set_position_target(self._target_rad)
        pos_rad, _vel, finished = se.tick()

        self._pos_rad_buf[:] = pos_rad
        rad_to_steps(self._pos_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        if finished:
            self.finish()
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.SERVOL)
class ServoLCommand(MotionCommand[ServoLCmd]):
    """Streaming Cartesian position target.

    CSE drives the Cartesian path (with its own internal Ruckig for smooth
    TCP motion).  IK converts each smoothed pose to joint space.  If any
    joint's per-tick delta exceeds its hardware velocity limit, all deltas
    are scaled proportionally and CSE speed is reduced by the same ratio.
    """

    PARAMS_TYPE = ServoLCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_ik_stopping",
        "_target_se3",
        "_pos_rad_buf",
        "_q_commanded",
        "_q_ik_seed",
        "_dq_buf",
    )

    def __init__(self, p: ServoLCmd):
        super().__init__(p)
        self._initialized = False
        self._ik_stopping = False
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)
        self._q_commanded = np.zeros(6, dtype=np.float64)
        self._q_ik_seed = np.zeros(6, dtype=np.float64)
        self._dq_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: ControllerState) -> None:
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

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        cse = state.cartesian_streaming_executor

        if not self._initialized or not cse.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(self.p.speed, self.p.accel)
            self._q_commanded[:] = self._q_rad_buf
            self._q_ik_seed[:] = self._q_rad_buf
            self._initialized = True

        # CSE drives Cartesian path
        cse.set_pose_target(self._target_se3)
        smoothed_pose, vel, finished = cse.tick()

        # Solve IK seeded from previous IK result (branch continuity)
        ik_result = solve_ik(
            PAROL6_ROBOT.robot,
            smoothed_pose,
            self._q_ik_seed,
        )
        if ik_result.success and ik_result.q is not None:
            # IK recovered after failure — re-sync from encoder
            if self._ik_stopping:
                logger.info("[SERVOL] IK recovered — resuming")
                steps_to_rad(state.Position_in, self._q_rad_buf)
                cse.sync_pose(get_fkine_se3(state))
                self._q_commanded[:] = self._q_rad_buf
                self._q_ik_seed[:] = self._q_rad_buf
                self._ik_stopping = False
                # Let next tick handle normal tracking
            else:
                self._q_ik_seed[:] = ik_result.q

                # Compute per-joint delta from commanded position
                dq = self._dq_buf
                for i in range(6):
                    dq[i] = float(ik_result.q[i]) - self._q_commanded[i]

                # Velocity ratio: worst-case joint vs its per-tick hard limit
                ratio = _max_vel_ratio_jit(ik_result.q, self._q_commanded)

                if ratio > 1.0:
                    # Scale all deltas proportionally
                    for i in range(6):
                        self._q_commanded[i] += dq[i] / ratio
                    cse.set_limits(max(0.01, self.p.speed / ratio), self.p.accel)
                else:
                    self._q_commanded[:] = ik_result.q
                    cse.set_limits(self.p.speed, self.p.accel)
        else:
            # IK failed — graceful deceleration
            if not self._ik_stopping:
                global _last_servo_ik_warn
                now = time.monotonic()
                if now - _last_servo_ik_warn > _IK_WARN_INTERVAL:
                    logger.warning(
                        "[SERVOL] IK failed — decelerating: pos=%s",
                        smoothed_pose[:3, 3],
                    )
                    _last_servo_ik_warn = now
                cse.stop()
                self._ik_stopping = True

        self._pos_rad_buf[:] = self._q_commanded
        rad_to_steps(self._pos_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        if finished and not self._ik_stopping:
            self.finish()
            cse.active = False
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING
