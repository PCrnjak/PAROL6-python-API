"""
Smooth Geometry Commands

Commands for generating smooth geometric paths: circles, arcs, and splines.
These use the unified motion pipeline with TOPP-RA for time-optimal path parameterization.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from parol6.commands.base import TrajectoryMoveCommandBase
from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import CircularMotion, JointPath, SplineMotion, TrajectoryBuilder
from parol6.protocol.wire import (
    CmdType,
    MoveCCmd,
    MotionParamsMixin,
    MovePCmd,
    MoveSCmd,
)
from parol6.motion.geometry import compute_circle_from_3_points
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_se3
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError
from pinokin import se3_from_rpy, se3_interp, se3_rpy

_MP = TypeVar("_MP", bound=MotionParamsMixin)

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


# =============================================================================
# TRF/WRF Transformation Utilities
# =============================================================================

# Pre-allocated workspace buffers for TRF/WRF transformations (command setup phase)
_pose_trf_buf: np.ndarray = np.zeros((4, 4), dtype=np.float64)
_pose_wrf_buf: np.ndarray = np.zeros((4, 4), dtype=np.float64)
_rpy_rad_buf: np.ndarray = np.zeros(3, dtype=np.float64)


def _pose6_trf_to_wrf(
    pose6_mm_deg: Sequence[float], tool_pose: np.ndarray, out: np.ndarray
) -> None:
    """Convert 6D pose [x,y,z,rx,ry,rz] from TRF to WRF (mm, degrees)."""
    se3_from_rpy(
        pose6_mm_deg[0] / 1000.0,
        pose6_mm_deg[1] / 1000.0,
        pose6_mm_deg[2] / 1000.0,
        np.radians(pose6_mm_deg[3]),
        np.radians(pose6_mm_deg[4]),
        np.radians(pose6_mm_deg[5]),
        _pose_trf_buf,
    )
    np.matmul(tool_pose, _pose_trf_buf, out=_pose_wrf_buf)
    se3_rpy(_pose_wrf_buf, _rpy_rad_buf)
    out[:3] = _pose_wrf_buf[:3, 3] * 1000.0
    np.degrees(_rpy_rad_buf, out=out[3:])


def _transform_waypoints_trf_to_wrf(
    waypoints: Sequence[Sequence[float]], frame: str, state: "ControllerState"
) -> np.ndarray:
    """Transform 6D waypoint poses from TRF to WRF. Returns (N, 6) array."""
    n = len(waypoints)
    result = np.empty((n, 6), dtype=np.float64)
    if frame == "WRF":
        for i in range(n):
            result[i] = waypoints[i]
        return result
    tool_pose = get_fkine_se3(state)
    for i in range(n):
        _pose6_trf_to_wrf(waypoints[i], tool_pose, out=result[i])
    return result


# =============================================================================
# Smooth Motion Command Base
# =============================================================================


class BaseSmoothMotionCommand(TrajectoryMoveCommandBase[_MP]):
    """Base class for smooth geometry commands (circle, arc, helix, spline).

    Subclasses implement generate_main_trajectory() to create Cartesian geometry.
    This base class handles IK conversion and trajectory building.
    """

    __slots__ = (
        "_rpy_rad_buf",
        "_pose6_buf",
    )

    def __init__(self, p: _MP) -> None:
        super().__init__(p)
        self._rpy_rad_buf = np.zeros(3, dtype=np.float64)
        self._pose6_buf = np.zeros(6, dtype=np.float64)

    def get_current_pose(self, state: "ControllerState") -> np.ndarray:
        """Get current TCP pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        current_se3 = get_fkine_se3(state)
        se3_rpy(current_se3, self._rpy_rad_buf)
        self._pose6_buf[:3] = current_se3[:3, 3] * 1000  # m -> mm
        np.degrees(self._rpy_rad_buf, out=self._pose6_buf[3:])
        return self._pose6_buf

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute trajectory from current position."""
        self.log_debug("  -> Preparing %s...", self.name)

        current_pose = self.get_current_pose(state)
        self.log_info(
            "  -> Generating %s from position: %s",
            self.name,
            [round(p, 1) for p in current_pose[:3]],
        )

        cartesian_trajectory = self.generate_main_trajectory(current_pose)
        if cartesian_trajectory is None or len(cartesian_trajectory) == 0:
            self.fail(make_error(ErrorCode.TRAJ_EMPTY_RESULT, detail=""))
            return

        steps_to_rad(state.Position_in, self._q_rad_buf)

        try:
            joint_path = JointPath.from_poses(cartesian_trajectory, self._q_rad_buf)
        except IKError as e:
            self.log_error("  -> ERROR: IK failed during trajectory generation: %s", e)
            self.fail(e.robot_error)
            return

        if joint_path.is_partial:
            self.log_error("  -> ERROR: Partial IK during trajectory generation")
            self.fail(make_error(ErrorCode.IK_PARTIAL_PATH, valid="?", total="?"))
            return

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

        self.log_info(
            "  -> Trajectory prepared: %d steps, %.2fs duration",
            len(self.trajectory_steps),
            trajectory.duration,
        )

    def generate_main_trajectory(self, effective_start_pose):
        """Override this in subclasses to generate the specific motion trajectory."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")


@register_command(CmdType.MOVEC)
class MoveCCommand(BaseSmoothMotionCommand[MoveCCmd]):
    """Execute circular arc motion through current → via → end (3-point arc).

    Computes circle center and normal from the 3 points, then delegates to
    CircularMotion.generate_arc().
    """

    PARAMS_TYPE = MoveCCmd

    __slots__ = ("_via", "_end")

    def __init__(self, p: MoveCCmd) -> None:
        super().__init__(p)
        self._via: np.ndarray = np.asarray(p.via, dtype=np.float64)
        self._end: np.ndarray = np.asarray(p.end, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Transform via/end from TRF if needed, then compute arc."""
        if self.p.frame == "TRF":
            tool_pose = get_fkine_se3(state)
            _pose6_trf_to_wrf(self.p.via, tool_pose, out=self._via)
            _pose6_trf_to_wrf(self.p.end, tool_pose, out=self._end)
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc geometry from current position through via to end."""
        start_xyz = effective_start_pose[:3]
        via_xyz = self._via[:3]
        end_xyz = self._end[:3]

        center, _radius, normal = compute_circle_from_3_points(
            start_xyz, via_xyz, end_xyz
        )

        return CircularMotion().generate_arc(
            start_pose=effective_start_pose,
            end_pose=self._end,
            center=center,
            normal=normal,
            clockwise=False,
        )


@register_command(CmdType.MOVES)
class MoveSCommand(BaseSmoothMotionCommand[MoveSCmd]):
    """Execute smooth spline motion through waypoints."""

    PARAMS_TYPE = MoveSCmd

    __slots__ = ("_waypoints",)

    def __init__(self, p: MoveSCmd) -> None:
        super().__init__(p)
        self._waypoints: np.ndarray | None = None

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        self._waypoints = _transform_waypoints_trf_to_wrf(
            self.p.waypoints, self.p.frame, state
        )
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate spline starting from actual position."""
        assert self._waypoints is not None

        wps = self._waypoints
        motion_gen = SplineMotion()

        first_wp_error = float(np.linalg.norm(wps[0, :3] - effective_start_pose[:3]))

        if first_wp_error > 5.0:
            modified_waypoints = np.vstack([effective_start_pose[np.newaxis], wps])
            logger.info(
                f"    Added start position as first waypoint (distance: {first_wp_error:.1f}mm)"
            )
        else:
            modified_waypoints = np.vstack([effective_start_pose[np.newaxis], wps[1:]])
            logger.info("    Replaced first waypoint with actual start position")

        duration = self.p.resolved_duration
        trajectory = motion_gen.generate_spline(
            waypoints=modified_waypoints,
            duration=duration,
        )

        logger.debug(f"    Generated spline with {len(trajectory)} points")

        return trajectory


# Number of SE3 samples per linear segment for moveP
_MOVEP_SAMPLES_PER_SEGMENT: int = 20


@register_command(CmdType.MOVEP)
class MovePCommand(BaseSmoothMotionCommand[MovePCmd]):
    """Process move — constant TCP speed through waypoints with piecewise linear segments.

    Phase 3 will add auto-blending at corners (Bézier blend zones).
    Currently uses sharp piecewise-linear interpolation.
    """

    PARAMS_TYPE = MovePCmd

    __slots__ = ("_waypoints", "_se3_buf_a", "_se3_buf_b", "_interp_buf")

    def __init__(self, p: MovePCmd) -> None:
        super().__init__(p)
        self._waypoints: np.ndarray | None = None
        self._se3_buf_a = np.zeros((4, 4), dtype=np.float64)
        self._se3_buf_b = np.zeros((4, 4), dtype=np.float64)
        self._interp_buf = np.zeros((4, 4), dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if TRF, build trajectory with constant TCP speed."""
        self._waypoints = _transform_waypoints_trf_to_wrf(
            self.p.waypoints, self.p.frame, state
        )

        self.log_debug("  -> Preparing %s...", self.name)
        current_pose = self.get_current_pose(state)
        self.log_info(
            "  -> Generating %s from position: %s",
            self.name,
            [round(p, 1) for p in current_pose[:3]],
        )

        cartesian_trajectory = self.generate_main_trajectory(current_pose)
        if cartesian_trajectory is None or len(cartesian_trajectory) == 0:
            self.fail(make_error(ErrorCode.TRAJ_EMPTY_RESULT, detail=""))
            return

        steps_to_rad(state.Position_in, self._q_rad_buf)

        try:
            joint_path = JointPath.from_poses(cartesian_trajectory, self._q_rad_buf)
        except IKError as e:
            self.log_error("  -> ERROR: IK failed during trajectory generation: %s", e)
            self.fail(e.robot_error)
            return

        if joint_path.is_partial:
            self.log_error("  -> ERROR: Partial IK during trajectory generation")
            self.fail(make_error(ErrorCode.IK_PARTIAL_PATH, valid="?", total="?"))
            return

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

        self.log_info(
            "  -> Trajectory prepared: %d steps, %.2fs duration",
            len(self.trajectory_steps),
            trajectory.duration,
        )

    def generate_main_trajectory(self, effective_start_pose):
        """Generate piecewise-linear Cartesian path through waypoints.

        Each segment is linearly interpolated in SE3 space.
        Phase 3 adds Bézier blend zones at corner points.
        """
        assert self._waypoints is not None

        wps = self._waypoints

        first_wp_error = float(np.linalg.norm(wps[0, :3] - effective_start_pose[:3]))
        if first_wp_error > 5.0:
            all_waypoints = np.vstack([effective_start_pose[np.newaxis], wps])
        else:
            all_waypoints = np.vstack([effective_start_pose[np.newaxis], wps[1:]])

        # Pre-compute total SE3 poses for single allocation
        n = _MOVEP_SAMPLES_PER_SEGMENT
        n_segs = len(all_waypoints) - 1
        total = n_segs * n - (n_segs - 1)  # first segment full, rest skip junction
        cart_poses = np.empty((total, 4, 4), dtype=np.float64)
        cursor = 0

        for seg_idx in range(n_segs):
            wp_a = all_waypoints[seg_idx]
            wp_b = all_waypoints[seg_idx + 1]

            se3_from_rpy(
                wp_a[0] / 1000.0,
                wp_a[1] / 1000.0,
                wp_a[2] / 1000.0,
                np.radians(wp_a[3]),
                np.radians(wp_a[4]),
                np.radians(wp_a[5]),
                self._se3_buf_a,
            )
            se3_from_rpy(
                wp_b[0] / 1000.0,
                wp_b[1] / 1000.0,
                wp_b[2] / 1000.0,
                np.radians(wp_b[3]),
                np.radians(wp_b[4]),
                np.radians(wp_b[5]),
                self._se3_buf_b,
            )

            start_i = 0 if seg_idx == 0 else 1
            for i in range(start_i, n):
                s = i / (n - 1)
                se3_interp(self._se3_buf_a, self._se3_buf_b, s, cart_poses[cursor])
                cursor += 1

        logger.debug(
            "    Generated process move path with %d SE3 poses across %d segments",
            cursor,
            n_segs,
        )

        return cart_poses[:cursor]
