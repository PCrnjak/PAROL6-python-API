"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, MoveCart, MoveCartRelTrf
"""

import logging
from typing import cast

import numpy as np

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
    JogLCmd,
    MoveLCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.ik import RateLimitedWarning, solve_ik
from pinokin import se3_from_rpy, se3_interp, se3_rpy

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


@register_command(CmdType.JOGL)
class JogLCommand(MotionCommand[JogLCmd]):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.

    CSE drives Cartesian velocity (Ruckig-smoothed).  IK converts each
    smoothed pose to joint space.  Velocity clamping and commanded-position
    tracking match servoL for smooth, deterministic joint trajectories.
    """

    PARAMS_TYPE = JogLCmd
    streamable = True

    __slots__ = (
        "is_rotation",
        "_ik_stopping",
        "_axis_index",
        "_axis_sign",
        "_dot_buf",
        "_q_commanded",
        "_q_ik_seed",
        "_dq_buf",
        "_pos_rad_buf",
        "_vel_ratio",
    )

    def __init__(self, p: JogLCmd):
        super().__init__(p)
        self.is_rotation = False
        self._ik_stopping = False
        self._axis_index = 0
        self._axis_sign = 1.0
        self._vel_ratio = 1.0

        self._dot_buf = np.zeros((), dtype=np.float64)
        self._q_commanded = np.zeros(6, dtype=np.float64)
        self._q_ik_seed = np.zeros(6, dtype=np.float64)
        self._dq_buf = np.zeros(6, dtype=np.float64)
        self._pos_rad_buf = np.zeros(6, dtype=np.float64)

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
            cse.set_jog_velocity_1dof(self._axis_index, 0.0, self.is_rotation)
            smoothed_pose, smoothed_vel, finished = cse.tick()

            np.dot(smoothed_vel, smoothed_vel, out=self._dot_buf)
            if not finished and self._dot_buf > 1e-8:
                ik_result = solve_ik(PAROL6_ROBOT.robot, smoothed_pose, self._q_ik_seed)
                if ik_result.success and ik_result.q is not None:
                    self._track_and_send(state, ik_result.q)
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

        # Scale velocity by previous tick's clamping ratio to keep CSE
        # in sync with joint-velocity-limited motion
        if self._vel_ratio > 1.0:
            velocity /= self._vel_ratio

        # Set target velocity (WRF transforms to body frame, TRF uses body directly)
        if self.p.frame == "WRF":
            cse.set_jog_velocity_1dof_wrf(self._axis_index, velocity, self.is_rotation)
        else:
            cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

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
                    "[CARTJOG] IK failed - initiating graceful stop: pos=%s",
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

        # IK succeeded - if we were stopping, recover by resuming jogging
        if self._ik_stopping:
            logger.info("[CARTJOG] IK recovered - resuming jog")
            steps_to_rad(state.Position_in, self._q_rad_buf)
            cse.sync_pose(get_fkine_se3(state))
            self._q_commanded[:] = self._q_rad_buf
            self._q_ik_seed[:] = self._q_rad_buf
            self._vel_ratio = 1.0
            self._ik_stopping = False
            if self.p.frame == "WRF":
                cse.set_jog_velocity_1dof_wrf(
                    self._axis_index, velocity, self.is_rotation
                )
            else:
                cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        self._track_and_send(state, ik_result.q)

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
