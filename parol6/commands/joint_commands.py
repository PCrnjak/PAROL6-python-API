"""
Joint Movement Commands
Contains commands for joint-space movements with unified trajectory execution.

Uses unified motion pipeline with TOPP-RA for time-optimal path parameterization.
All commands inherit from JointMoveCommandBase which uses MotionExecutor for
jerk-limited smoothing during execution.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.base import TrajectoryMoveCommandBase
from parol6.config import (
    INTERVAL_S,
    MAX_BLEND_LOOKAHEAD,
    steps_to_rad,
)
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.protocol.wire import CmdType, MoveJCmd, MoveJPoseCmd, MotionParamsMixin
from parol6.server.command_registry import register_command
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError
from parol6.utils.ik import solve_ik
from pinokin import se3_from_rpy

_MP = TypeVar("_MP", bound=MotionParamsMixin)

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class JointMoveCommandBase(TrajectoryMoveCommandBase[_MP]):
    """Base class for joint-space trajectory commands.

    Subclasses must implement:
    - _get_target_rad(): Return target joint positions in radians

    This base class provides:
    - do_setup(): Builds trajectory via JointPath.interpolate + TrajectoryBuilder
    - execute_step(): Inherited from TrajectoryMoveCommandBase (uses MotionExecutor)
    """

    __slots__ = (
        "_T_buf",
        "_q_full_buf",
        "_diff_buf",
        "_current_rad_buf",
        "_tcp_mm_buf",
    )

    def __init__(self, p: _MP) -> None:
        super().__init__(p)
        nq = PAROL6_ROBOT.robot.nq
        self._T_buf = np.zeros((4, 4), dtype=np.float64, order="F")
        self._q_full_buf = np.zeros(nq, dtype=np.float64)
        self._diff_buf = np.empty(3, dtype=np.float64)
        self._current_rad_buf = np.zeros(6, dtype=np.float64)
        self._tcp_mm_buf = np.empty((MAX_BLEND_LOOKAHEAD + 2, 3), dtype=np.float64)

    @abstractmethod
    def _get_target_rad(
        self, state: ControllerState, current_rad: np.ndarray
    ) -> np.ndarray:
        """Return target joint positions in radians.

        Args:
            state: Controller state
            current_rad: Current joint positions in radians (for IK seed if needed)
        """
        ...

    def do_setup(self, state: ControllerState) -> None:
        """Build trajectory from current position to target using unified motion pipeline."""
        steps_to_rad(state.Position_in, self._q_rad_buf)
        target_rad = self._get_target_rad(state, self._q_rad_buf)
        current_rad = self._q_rad_buf

        joint_path = JointPath.interpolate(current_rad, target_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=state.motion_profile,
            velocity_frac=self.p.resolved_speed,
            accel_frac=self.p.accel,
            duration=self.p.resolved_duration,
            dt=INTERVAL_S,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self._duration = trajectory.duration

        if len(self.trajectory_steps) == 0:
            raise TrajectoryPlanningError(
                make_error(ErrorCode.TRAJ_NO_STEPS, detail="")
            )

        self.log_trace(
            "  -> Using profile: %s, duration: %.3fs, steps: %d",
            state.motion_profile,
            trajectory.duration,
            len(self.trajectory_steps),
        )

    def do_setup_with_blend(
        self,
        state: ControllerState,
        next_cmds: "list[TrajectoryMoveCommandBase]",
    ) -> int:
        """Build composite joint-space trajectory with blend zones."""
        if self.blend_radius <= 0 or not next_cmds:
            self.do_setup(state)
            return 0

        chain: list[JointMoveCommandBase] = [self]
        for cmd in next_cmds:
            if isinstance(cmd, JointMoveCommandBase):
                chain.append(cmd)
            else:
                break
        if len(chain) < 2:
            self.do_setup(state)
            return 0

        from parol6.motion.geometry import build_composite_joint_path

        steps_to_rad(state.Position_in, self._q_rad_buf)
        self._current_rad_buf[:] = self._q_rad_buf
        current_rad = self._current_rad_buf

        waypoints_rad: list[np.ndarray] = [current_rad]
        blend_radii_mm: list[float] = []

        for i, cmd in enumerate(chain):
            target_rad = cmd._get_target_rad(state, current_rad)
            waypoints_rad.append(target_rad)
            if i < len(chain) - 1:
                blend_radii_mm.append(cmd.blend_radius)
            current_rad = target_rad

        if len(waypoints_rad) < 3:
            self.do_setup(state)
            return 0

        # FK at each waypoint for TCP positions (zone sizing)
        nq = PAROL6_ROBOT.robot.nq
        T_buf = self._T_buf
        T_buf.fill(0)
        q_full = self._q_full_buf
        q_full.fill(0)
        n_wp = len(waypoints_rad)
        tcp_mm = self._tcp_mm_buf[:n_wp]
        for wi, q in enumerate(waypoints_rad):
            nj = min(len(q), nq)
            q_full[:nj] = q[:nj]
            q_full[nj:] = 0.0
            PAROL6_ROBOT.robot.fkine_into(q_full, T_buf)
            tcp_mm[wi, 0] = T_buf[0, 3] * 1000.0
            tcp_mm[wi, 1] = T_buf[1, 3] * 1000.0
            tcp_mm[wi, 2] = T_buf[2, 3] * 1000.0

        # Convert mm blend radii to segment fractions via TCP distances
        blend_fracs: list[tuple[float, float]] = []
        diff_buf = self._diff_buf
        for i in range(len(blend_radii_mm)):
            wp_idx = i + 1
            np.subtract(tcp_mm[wp_idx], tcp_mm[wp_idx - 1], diff_buf)
            seg_before = float(np.linalg.norm(diff_buf))
            np.subtract(tcp_mm[wp_idx + 1], tcp_mm[wp_idx], diff_buf)
            seg_after = float(np.linalg.norm(diff_buf))
            r = blend_radii_mm[i]
            frac_before = min(r / seg_before, 0.5) if seg_before > 1e-6 else 0.0
            frac_after = min(r / seg_after, 0.5) if seg_after > 1e-6 else 0.0
            blend_fracs.append((frac_before, frac_after))

        try:
            positions = build_composite_joint_path(
                waypoints_rad,
                blend_fracs,
                samples_per_segment=50,
            )
        except Exception as e:
            self.log_warning("Joint blend path failed: %s, falling back", e)
            self.do_setup(state)
            return 0

        joint_path = JointPath(positions=positions)

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
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self._duration = trajectory.duration

        consumed = len(chain) - 1
        self.log_info(
            "  -> Blended joint trajectory: %d segments, steps=%d, duration=%.3fs",
            len(waypoints_rad) - 1,
            len(self.trajectory_steps),
            trajectory.duration,
        )
        return consumed


@register_command(CmdType.MOVEJ)
class MoveJCommand(JointMoveCommandBase[MoveJCmd]):
    """Move the robot's joints to a specific configuration."""

    PARAMS_TYPE = MoveJCmd

    __slots__ = ()

    def _get_target_rad(
        self, state: ControllerState, current_rad: np.ndarray
    ) -> np.ndarray:
        """Return target joint positions in radians."""
        target = np.deg2rad(self.p.angles)
        if self.p.rel:
            target += current_rad
        return target


@register_command(CmdType.MOVEJ_POSE)
class MoveJPoseCommand(JointMoveCommandBase[MoveJPoseCmd]):
    """Move the robot to a specific Cartesian pose via joint-space interpolation.

    Uses IK to find the target joint configuration, then interpolates in joint space.
    This is different from MoveL which follows a straight-line Cartesian path.
    """

    PARAMS_TYPE = MoveJPoseCmd

    __slots__ = ()

    def _get_target_rad(
        self, state: ControllerState, current_rad: np.ndarray
    ) -> np.ndarray:
        """Solve IK for target pose and return joint positions in radians."""
        pose = self.p.pose

        target_pose = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            np.radians(pose[3]),
            np.radians(pose[4]),
            np.radians(pose[5]),
            target_pose,
        )

        ik_solution = solve_ik(PAROL6_ROBOT.robot, target_pose, current_rad)
        if not ik_solution.success:
            detail = ik_solution.violations or ""
            raise IKError(make_error(ErrorCode.IK_TARGET_UNREACHABLE, detail=detail))

        return np.asarray(ik_solution.q, dtype=np.float64)
