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
from numba import njit  # type: ignore[import-untyped]

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import (
    INTERVAL_S,
    LIMITS,
    _rad_to_steps_jit,
    rad_to_steps,
    steps_to_rad,
)
from parol6.protocol.wire import CmdType, ServoJCmd, ServoJPoseCmd, ServoLCmd
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.ik import solve_ik
from pinokin import se3_from_rpy

from .base import ExecutionStatusCode, MotionCommand

logger = logging.getLogger(__name__)

# Wrist-flip velocity limiting constants (module globals → numba compile-time constants)
_JOINT_MAX_STEP_INV = 1.0 / (
    np.array(LIMITS.joint.jog.velocity, dtype=np.float64) * INTERVAL_S
)
_RAD_STEP_INV = 1.0 / PAROL6_ROBOT.radian_per_step_constant
_JOINT_RATIO = np.ascontiguousarray(PAROL6_ROBOT.joint.ratio, dtype=np.float64)

# Rate-limiting for IK failure warnings
_IK_WARN_INTERVAL: float = 1.0
_last_servo_ik_warn: float = 0.0


@njit(cache=True)
def _vel_scale_and_convert_jit(
    target_q: np.ndarray,
    current_q: np.ndarray,
    scratch: np.ndarray,
    out_steps: np.ndarray,
    flip_target_q: np.ndarray,
) -> bool:
    """Velocity-limit joint step and convert to motor steps. Zero-allocation.

    If any joint exceeds its per-tick jog velocity limit, uniformly scales
    so the worst joint is exactly at its limit, and copies target_q into
    flip_target_q. Converts the result to motor steps via _rad_to_steps_jit.

    Always writes final motor steps into out_steps.
    Returns True if scaling was applied (flip detected), False otherwise.
    """
    n = target_q.shape[0]
    max_ratio = 0.0

    for i in range(n):
        d = target_q[i] - current_q[i]
        scratch[i] = d
        r = abs(d) * _JOINT_MAX_STEP_INV[i]
        if r > max_ratio:
            max_ratio = r

    if max_ratio > 1.0:
        inv = 1.0 / max_ratio
        for i in range(n):
            scratch[i] = current_q[i] + scratch[i] * inv
            flip_target_q[i] = target_q[i]
    else:
        for i in range(n):
            scratch[i] = target_q[i]

    _rad_to_steps_jit(scratch, out_steps, scratch, _RAD_STEP_INV, _JOINT_RATIO)
    return max_ratio > 1.0


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
            se.sync_position(list(self._q_rad_buf))
            se.set_limits(self.p.speed, self.p.accel)
            self._initialized = True

        se.set_position_target(self._target_rad)
        pos_rad, _vel, finished = se.tick()

        for i in range(6):
            self._pos_rad_buf[i] = pos_rad[i]
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
            raise ValueError(
                f"SERVOJ_POSE: IK failed for pose {[round(v, 1) for v in pose]}"
            )

        for i in range(6):
            self._target_rad[i] = float(ik_result.q[i])

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        se = state.streaming_executor

        # Sync position on first tick or if executor not active
        if not self._initialized or not se.active:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            se.sync_position(list(self._q_rad_buf))
            se.set_limits(self.p.speed, self.p.accel)
            self._initialized = True

        se.set_position_target(self._target_rad)
        pos_rad, _vel, finished = se.tick()

        for i in range(6):
            self._pos_rad_buf[i] = pos_rad[i]
        rad_to_steps(self._pos_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        if finished:
            self.finish()
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.SERVOL)
class ServoLCommand(MotionCommand[ServoLCmd]):
    """Streaming Cartesian position target.

    Uses CartesianStreamingExecutor for smooth Ruckig-interpolated motion
    along a straight-line Cartesian path. Each tick: CSE smoothing, IK solve,
    wrist-flip velocity limiting.
    """

    PARAMS_TYPE = ServoLCmd
    streamable = True

    __slots__ = (
        "_initialized",
        "_ik_stopping",
        "_target_se3",
        # Wrist-flip handling buffers
        "_flip_target_q",
        "_flipping",
        "_scratch_buf",
    )

    def __init__(self, p: ServoLCmd):
        super().__init__(p)
        self._initialized = False
        self._ik_stopping = False
        self._target_se3 = np.zeros((4, 4), dtype=np.float64)
        self._flip_target_q = np.zeros(6, dtype=np.float64)
        self._flipping = False
        self._scratch_buf = np.zeros(6, dtype=np.float64)

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

        steps_to_rad(state.Position_in, self._q_rad_buf)

        # Initialize on first tick or if executor not active
        if not self._initialized or not cse.active:
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(self.p.speed, self.p.accel)
            self._initialized = True
            self._ik_stopping = False
            self._flipping = False

        cse.set_pose_target(self._target_se3)
        smoothed_pose, vel, finished = cse.tick()

        # Solve IK for the smoothed Cartesian pose
        ik_result = solve_ik(PAROL6_ROBOT.robot, smoothed_pose, self._q_rad_buf)
        if not ik_result.success or ik_result.q is None:
            return self._handle_ik_failure(state, cse, vel, smoothed_pose)

        # IK succeeded — if we were stopping, recover
        if self._ik_stopping:
            logger.info("[SERVOL] IK recovered — resuming motion")
            cse.sync_pose(get_fkine_se3(state))
            cse.set_pose_target(self._target_se3)
            self._ik_stopping = False

        # Velocity-limit and convert to steps (wrist-flip safe)
        self._flipping = _vel_scale_and_convert_jit(
            ik_result.q,
            self._q_rad_buf,
            self._scratch_buf,
            self._steps_buf,
            self._flip_target_q,
        )
        self.set_move_position(state, self._steps_buf)

        if finished and not self._flipping:
            self.finish()
            cse.active = False
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING

    def _handle_ik_failure(
        self, state: ControllerState, cse, vel, smoothed_pose
    ) -> ExecutionStatusCode:
        """Handle IK failure with graceful stop and rate-limited warnings."""
        global _last_servo_ik_warn

        if not self._ik_stopping:
            now = time.monotonic()
            if now - _last_servo_ik_warn > _IK_WARN_INTERVAL:
                logger.warning(
                    "[SERVOL] IK failed — initiating graceful stop: pos=%s",
                    smoothed_pose[:3, 3],
                )
                _last_servo_ik_warn = now
            cse.stop()
            self._ik_stopping = True
        else:
            # Still failing, check if deceleration complete
            if np.dot(vel, vel) < 1e-8:
                cse.sync_pose(get_fkine_se3(state))
                self.finish()
                return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING
