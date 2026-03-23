"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, MoveCart, MoveCartRelTrf
"""

import logging
from typing import cast

import numpy as np
from numba import njit

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import (
    CART_ANG_JOG_MIN,
    CART_LIN_JOG_MIN,
    INTERVAL_S,
    LIMITS,
    PATH_SAMPLES,
    rad_to_steps,
    steps_to_rad,
)
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.protocol.wire import (
    CmdType,
    CommandCode,
    JogLCmd,
    MoveLCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.ik import RateLimitedWarning, solve_ik
from pinokin import se3_exp_ws, se3_from_rpy, se3_interp, se3_mul, se3_rpy

from parol6.commands.servo_commands import _max_vel_ratio_jit

from .base import (
    ExecutionStatusCode,
    MotionCommand,
    TrajectoryMoveCommandBase,
)

logger = logging.getLogger(__name__)

# Pre-computed Cartesian jog limit constants (avoid per-tick recomputation)
_CART_ANG_JOG_MIN_RAD: float = float(np.deg2rad(CART_ANG_JOG_MIN))
_CART_ANG_JOG_MAX_RAD: float = float(LIMITS.cart.jog.velocity.angular)
_CART_LIN_JOG_MIN_MS: float = CART_LIN_JOG_MIN / 1000.0
_CART_LIN_JOG_MAX_MS: float = float(LIMITS.cart.jog.velocity.linear)


def _linmap_frac(frac: float, lo: float, hi: float) -> float:
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return lo + (hi - lo) * frac


_ik_warn = RateLimitedWarning()


@njit(cache=True)
def _apply_velocity_delta_wrf_jit(
    R: np.ndarray,
    smoothed_vel: np.ndarray,
    dt: float,
    current_pose: np.ndarray,
    vel_lin: np.ndarray,
    vel_ang: np.ndarray,
    world_twist: np.ndarray,
    delta: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """Apply smoothed velocity delta in World Reference Frame.

    Transforms body-frame velocity to world-frame and left-multiplies.
    WRF: target = delta @ current (world-frame delta applied first)

    Args:
        R: 3x3 rotation matrix (reference pose rotation for WRF)
        smoothed_vel: 6D body-frame velocity [vx, vy, vz, wx, wy, wz]
        dt: Time step
        current_pose: Current pose as 4x4 SE3
        vel_lin: Workspace buffer for linear velocity (3,)
        vel_ang: Workspace buffer for angular velocity (3,)
        world_twist: Workspace buffer for world-frame twist (6,)
        delta: Workspace buffer for delta transform (4x4)
        out: Output pose (4x4 SE3)
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    # Transform velocity to world frame: R @ vel
    for i in range(3):
        vel_lin[i] = (
            R[i, 0] * smoothed_vel[0]
            + R[i, 1] * smoothed_vel[1]
            + R[i, 2] * smoothed_vel[2]
        )
        vel_ang[i] = (
            R[i, 0] * smoothed_vel[3]
            + R[i, 1] * smoothed_vel[4]
            + R[i, 2] * smoothed_vel[5]
        )

    # Build world-frame twist scaled by dt
    world_twist[0] = vel_lin[0] * dt
    world_twist[1] = vel_lin[1] * dt
    world_twist[2] = vel_lin[2] * dt
    world_twist[3] = vel_ang[0] * dt
    world_twist[4] = vel_ang[1] * dt
    world_twist[5] = vel_ang[2] * dt

    # Exponential map and apply (world frame = left multiply)
    se3_exp_ws(world_twist, delta, omega_ws, R_ws, V_ws)
    se3_mul(delta, current_pose, out)


@njit(cache=True)
def _apply_velocity_delta_trf_jit(
    smoothed_vel: np.ndarray,
    dt: float,
    current_pose: np.ndarray,
    body_twist: np.ndarray,
    delta: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """Apply smoothed velocity delta in Tool Reference Frame.

    Uses body-frame velocity directly and right-multiplies.
    TRF: target = current @ delta (body-frame delta applied in tool frame)

    Args:
        smoothed_vel: 6D body-frame velocity [vx, vy, vz, wx, wy, wz]
        dt: Time step
        current_pose: Current pose as 4x4 SE3
        body_twist: Workspace buffer for body-frame twist (6,)
        delta: Workspace buffer for delta transform (4x4)
        out: Output pose (4x4 SE3)
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    # Build body-frame twist scaled by dt (no transformation needed)
    body_twist[0] = smoothed_vel[0] * dt
    body_twist[1] = smoothed_vel[1] * dt
    body_twist[2] = smoothed_vel[2] * dt
    body_twist[3] = smoothed_vel[3] * dt
    body_twist[4] = smoothed_vel[4] * dt
    body_twist[5] = smoothed_vel[5] * dt

    # Exponential map and apply (tool frame = right multiply)
    se3_exp_ws(body_twist, delta, omega_ws, R_ws, V_ws)
    se3_mul(current_pose, delta, out)


@register_command(CmdType.JOGL)
class JogLCommand(MotionCommand[JogLCmd]):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    Uses static 6-element velocity vector [vx,vy,vz,wx,wy,wz] on the wire.
    """

    PARAMS_TYPE = JogLCmd
    streamable = True

    __slots__ = (
        "is_rotation",
        "_ik_stopping",
        "_axis_index",
        "_axis_sign",
        # Pre-allocated buffers (allocated once in __init__, reused across streaming)
        "_world_twist_buf",
        "_vel_lin_buf",
        "_vel_ang_buf",
        "_delta_se3_buf",
        "_target_pose_buf",
        "_omega_ws",
        "_R_ws",
        "_V_ws",
        "_dot_buf",
        "_q_clamped_buf",
    )

    def __init__(self, p: JogLCmd):
        super().__init__(p)
        self.is_rotation = False
        self._ik_stopping = False
        self._axis_index = 0
        self._axis_sign = 1.0

        self._world_twist_buf = np.zeros(6, dtype=np.float64)
        self._vel_lin_buf = np.zeros(3, dtype=np.float64)
        self._vel_ang_buf = np.zeros(3, dtype=np.float64)
        self._delta_se3_buf = np.zeros((4, 4), dtype=np.float64)
        self._target_pose_buf = np.zeros((4, 4), dtype=np.float64)
        self._omega_ws = np.zeros(3, dtype=np.float64)
        self._R_ws = np.zeros((3, 3), dtype=np.float64)
        self._V_ws = np.zeros((3, 3), dtype=np.float64)
        self._dot_buf = np.zeros((), dtype=np.float64)
        self._q_clamped_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Find dominant axis and start timer."""
        vels = self.p.velocities
        max_idx = 0
        max_abs = abs(vels[0])
        for i in range(1, 6):
            a = abs(vels[i])
            if a > max_abs:
                max_abs = a
                max_idx = i
        self.is_rotation = max_idx >= 3
        self._axis_index = max_idx - 3 if max_idx >= 3 else max_idx
        self._axis_sign = 1.0 if vels[max_idx] >= 0 else -1.0
        self.start_timer(self.p.duration)
        self._ik_stopping = False

    def _compute_target_pose_from_velocity(
        self, state: "ControllerState", smoothed_vel: np.ndarray
    ) -> None:
        """Compute target pose from smoothed velocity."""
        cse = state.cartesian_streaming_executor
        current_pose = get_fkine_se3(state)

        if self.p.frame == "WRF":
            # WRF: transform velocity to world frame and left-multiply
            assert cse.reference_pose is not None
            R = cse.reference_pose[:3, :3]
            _apply_velocity_delta_wrf_jit(
                R,
                smoothed_vel,
                cse.dt,
                current_pose,
                self._vel_lin_buf,
                self._vel_ang_buf,
                self._world_twist_buf,
                self._delta_se3_buf,
                self._target_pose_buf,
                self._omega_ws,
                self._R_ws,
                self._V_ws,
            )
        else:
            # TRF: use body-frame velocity directly and right-multiply
            _apply_velocity_delta_trf_jit(
                smoothed_vel,
                cse.dt,
                current_pose,
                self._world_twist_buf,
                self._delta_se3_buf,
                self._target_pose_buf,
                self._omega_ws,
                self._R_ws,
                self._V_ws,
            )

    def _clamp_and_send(self, state: "ControllerState", q: np.ndarray) -> None:
        """Velocity-clamp IK result and send to motors."""
        ratio = _max_vel_ratio_jit(q, self._q_rad_buf)
        if ratio > 1.0:
            for i in range(6):
                self._q_clamped_buf[i] = (
                    self._q_rad_buf[i] + (q[i] - self._q_rad_buf[i]) / ratio
                )
            rad_to_steps(self._q_clamped_buf, self._steps_buf)
        else:
            rad_to_steps(q, self._steps_buf)
        # Send as JOG command — firmware uses velocity directly (no averaging)
        for i in range(6):
            state.Speed_out[i] = int(
                (self._steps_buf[i] - state.Position_in[i]) / INTERVAL_S
            )
        state.Command_out = CommandCode.JOG

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of Cartesian jogging."""
        cse = state.cartesian_streaming_executor

        steps_to_rad(state.Position_in, self._q_rad_buf)

        # Initialize only if not already active (preserve velocity across streaming)
        if not cse.active:
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(1.0, self.p.accel)

        # Handle timer expiry - stop smoothly
        if self.timer_expired():
            cse.set_jog_velocity_1dof(self._axis_index, 0.0, self.is_rotation)
            _smoothed_pose, smoothed_vel, finished = cse.tick()

            np.dot(smoothed_vel, smoothed_vel, out=self._dot_buf)
            if not finished and self._dot_buf > 1e-8:
                self._compute_target_pose_from_velocity(state, smoothed_vel)
                ik_result = solve_ik(
                    PAROL6_ROBOT.robot, self._target_pose_buf, self._q_rad_buf
                )
                if ik_result.success and ik_result.q is not None:
                    self._clamp_and_send(state, ik_result.q)
                return ExecutionStatusCode.EXECUTING

            cse.active = False
            self.finish()
            self.stop_and_idle(state)
            return ExecutionStatusCode.COMPLETED

        # Compute target velocity based on speed fraction from velocity vector
        vels = self.p.velocities
        speed_mag = abs(vels[self._axis_index + (3 if self.is_rotation else 0)])
        if self.is_rotation:
            velocity = (
                _linmap_frac(speed_mag, _CART_ANG_JOG_MIN_RAD, _CART_ANG_JOG_MAX_RAD)
                * self._axis_sign
            )
        else:
            velocity = (
                _linmap_frac(speed_mag, _CART_LIN_JOG_MIN_MS, _CART_LIN_JOG_MAX_MS)
                * self._axis_sign
            )

        # Set target velocity (WRF transforms to body frame, TRF uses body directly)
        if self.p.frame == "WRF":
            cse.set_jog_velocity_1dof_wrf(self._axis_index, velocity, self.is_rotation)
        else:
            cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        _smoothed_pose, smoothed_vel, _finished = cse.tick()
        self._compute_target_pose_from_velocity(state, smoothed_vel)

        ik_result = solve_ik(
            PAROL6_ROBOT.robot,
            self._target_pose_buf,
            self._q_rad_buf,
        )
        if not ik_result.success or ik_result.q is None:
            if not self._ik_stopping:
                _ik_warn(
                    logger,
                    "[CARTJOG] IK failed - initiating graceful stop: pos=%s",
                    self._target_pose_buf[:3, 3],
                )
                cse.stop()
                self._ik_stopping = True
            else:
                # Still failing, check if we've stopped decelerating
                np.dot(smoothed_vel, smoothed_vel, out=self._dot_buf)
                if self._dot_buf < 1e-8:
                    # Sync CSE to actual robot pose now that we've stopped
                    # This allows recovery by jogging in a different direction
                    cse.sync_pose(get_fkine_se3(state))
                    cse.active = False
                    self.finish()
                    return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        # IK succeeded - if we were stopping, recover by resuming jogging
        if self._ik_stopping:
            logger.info("[CARTJOG] IK recovered - resuming jog")
            # Sync to actual robot pose before resuming (CSE drifted during stop)
            cse.sync_pose(get_fkine_se3(state))
            self._ik_stopping = False
            # Re-apply the jog velocity to resume motion
            if self.p.frame == "WRF":
                cse.set_jog_velocity_1dof_wrf(
                    self._axis_index, velocity, self.is_rotation
                )
            else:
                cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        self._clamp_and_send(state, ik_result.q)

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.MOVEL)
class MoveLCommand(TrajectoryMoveCommandBase[MoveLCmd]):
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
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self._duration = trajectory.duration

        self.log_debug(
            "  -> Pre-computed Cartesian path: profile=%s, steps=%d, duration=%.3fs",
            state.motion_profile,
            len(self.trajectory_steps),
            float(self._duration),
        )

    def _compute_target_pose(self, state: "ControllerState") -> None:
        """Compute target pose — absolute or relative based on rel flag."""
        pose = self.p.pose

        if self.p.rel:
            # Relative move: compute delta SE3, then apply in tool frame (TRF)
            # or world frame (WRF) depending on self.p.frame
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
            if self.p.frame == "TRF":
                # Post-multiply for tool-relative motion
                self.target_pose = cast(np.ndarray, self.initial_pose) @ delta_se3
            else:
                # Pre-multiply for world-relative motion
                self.target_pose = delta_se3 @ cast(np.ndarray, self.initial_pose)
        else:
            # Absolute target pose
            self.target_pose = np.zeros((4, 4), dtype=np.float64)
            se3_from_rpy(
                pose[0] / 1000.0,
                pose[1] / 1000.0,
                pose[2] / 1000.0,
                np.radians(pose[3]),
                np.radians(pose[4]),
                np.radians(pose[5]),
                self.target_pose,
            )

    def do_setup_with_blend(
        self,
        state: "ControllerState",
        next_cmds: "list[TrajectoryMoveCommandBase]",
    ) -> int:
        """Build composite Cartesian trajectory with blend zones."""
        if self.blend_radius <= 0 or not next_cmds:
            self.do_setup(state)
            return 0

        chain: list[MoveLCommand] = [self]
        for cmd in next_cmds:
            if isinstance(cmd, MoveLCommand):
                chain.append(cmd)
            else:
                break
        if len(chain) < 2:
            self.do_setup(state)
            return 0

        from parol6.motion.geometry import build_composite_cartesian_path

        initial_pose = get_fkine_se3(state)
        self.initial_pose = initial_pose

        waypoints = [initial_pose]
        blend_radii: list[float] = []
        prev_pose = initial_pose

        for i, movel in enumerate(chain):
            movel.initial_pose = prev_pose
            movel._compute_target_pose(state)
            if movel.target_pose is None:
                if i < 2:
                    self.do_setup(state)
                    return 0
                chain = chain[:i]
                break
            waypoints.append(movel.target_pose)
            prev_pose = movel.target_pose
            if i < len(chain) - 1:
                blend_radii.append(movel.blend_radius)

        if len(waypoints) < 3:
            self.do_setup(state)
            return 0

        composite_poses = build_composite_cartesian_path(
            waypoints,
            blend_radii,
            samples_per_segment=PATH_SAMPLES,
        )

        if len(composite_poses) == 0:
            self.do_setup(state)
            return 0

        steps_to_rad(state.Position_in, self._q_rad_buf)

        try:
            joint_path = JointPath.from_poses(composite_poses, self._q_rad_buf)
        except Exception:
            self.log_warning(
                "Blend IK failed for %d-segment Cartesian path, falling back",
                len(waypoints) - 1,
            )
            self.do_setup(state)
            return 0

        if joint_path.is_partial:
            self.log_warning(
                "Blend IK partial for %d-segment Cartesian path, falling back",
                len(waypoints) - 1,
            )
            self.do_setup(state)
            return 0

        # Use minimum speed/accel across chain, sum durations when all duration-based
        min_speed = self.p.resolved_speed
        min_accel = self.p.accel
        total_duration = self.p.resolved_duration
        all_have_duration = total_duration is not None

        for i in range(1, len(chain)):
            cmd = chain[i]
            s = cmd.p.resolved_speed
            a = cmd.p.accel
            if s < min_speed:
                min_speed = s
            if a < min_accel:
                min_accel = a
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
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self._duration = trajectory.duration

        consumed = len(chain) - 1
        self.log_info(
            "  -> Blended Cartesian trajectory: %d segments, steps=%d, duration=%.3fs",
            len(waypoints) - 1,
            len(self.trajectory_steps),
            trajectory.duration,
        )
        return consumed
