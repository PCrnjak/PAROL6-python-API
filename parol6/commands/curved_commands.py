"""
Smooth Geometry Commands

Commands for generating smooth geometric paths: circles, arcs, and splines.
These use the unified motion pipeline with TOPP-RA for time-optimal path parameterization.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from parol6.commands._collision_guard import guard_cartesian_path
from parol6.commands.base import TrajectoryMoveCommandBase, guard_homed
from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import CircularMotion, JointPath, SplineMotion, TrajectoryBuilder
from parol6.protocol.wire import (
    CmdType,
    MoveCCmd,
    MotionParamsMixin,
    MovePCmd,
    MoveSCmd,
)
from parol6.commands.cartesian_commands import (
    CartesianChainLink,
    pose6_to_se3,
    resolve_pose,
)
from parol6.motion.geometry import (
    ArcSegment,
    LineSegment,
    build_composite_cartesian_path,
    cartesian_path_knots,
    compute_circle_from_3_points,
)
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_se3
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError
from pinokin import se3_from_rpy, se3_rpy

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


#: Rotation weight in the combined pose metric [mm/rad]: a reorientation in
#: place still covers distance (par6's ``path_rot_weight_m_per_rad``).
PATH_ROT_WEIGHT_MM_PER_RAD: float = 150.0
#: A waypoint list's first entry stands in for the start pose within this.
WAYPOINT_SNAP_MM: float = 5.0

_dist_se3_a: np.ndarray = np.zeros((4, 4), dtype=np.float64)
_dist_se3_b: np.ndarray = np.zeros((4, 4), dtype=np.float64)


def _pose6_distance_mm(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance between two [x, y, z, rx, ry, rz] poses (mm, degrees) on the
    combined metric sqrt(translation² + (w·rotation)²)."""
    pose6_to_se3(a, _dist_se3_a)
    pose6_to_se3(b, _dist_se3_b)
    translation = (
        float(np.linalg.norm(_dist_se3_a[:3, 3] - _dist_se3_b[:3, 3])) * 1000.0
    )
    relative = _dist_se3_a[:3, :3].T @ _dist_se3_b[:3, :3]
    cos_angle = (float(np.trace(relative)) - 1.0) / 2.0
    rotation = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return float(np.hypot(translation, PATH_ROT_WEIGHT_MM_PER_RAD * rotation))


def _se3_chain(trajectory: np.ndarray) -> np.ndarray:
    """The generated geometry as an (N, 4, 4) SE3 chain: a spline comes
    back as [x, y, z, rx, ry, rz] rows (mm, degrees), an arc or a process
    path already as poses."""
    if trajectory.ndim == 3:
        return trajectory
    poses = np.empty((len(trajectory), 4, 4), dtype=np.float64)
    for row, out in zip(trajectory, poses, strict=True):
        pose6_to_se3(row, out)
    return poses


# =============================================================================
# Smooth Motion Command Base
# =============================================================================


class BaseSmoothMotionCommand(TrajectoryMoveCommandBase[_MP]):
    """Base class for smooth geometry commands (circle, arc, helix, spline).

    Subclasses implement generate_main_trajectory() to create Cartesian geometry.
    This base class handles IK conversion and trajectory building.
    """

    #: Hold the tool to one speed along the whole path (a process move)
    #: rather than as fast as the joints allow under the cartesian ceiling.
    constant_tool_speed: bool = False

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
        guard_homed(state)
        self.log_debug("  -> Preparing %s...", self.name)

        current_pose = self.get_current_pose(state)
        self.log_info(
            "  -> Generating %s from position: %s",
            self.name,
            [round(p, 1) for p in current_pose[:3]],
        )

        cartesian_trajectory = self.generate_main_trajectory(current_pose)
        if cartesian_trajectory is None or len(cartesian_trajectory) == 0:
            raise TrajectoryPlanningError(
                make_error(
                    ErrorCode.TRAJ_EMPTY_RESULT, detail="empty cartesian trajectory"
                )
            )
        cartesian_trajectory = _se3_chain(cartesian_trajectory)

        steps_to_rad(state.Position_in, self._q_rad_buf)

        try:
            joint_path = JointPath.from_poses(cartesian_trajectory, self._q_rad_buf)
        except IKError as e:
            self.log_error("  -> ERROR: IK failed during trajectory generation: %s", e)
            raise

        if joint_path.is_partial:
            assert joint_path.valid is not None
            n_valid = int(joint_path.valid.sum())
            n_total = len(joint_path)
            self.log_error(
                "  -> ERROR: Partial IK during trajectory generation (%d/%d valid)",
                n_valid,
                n_total,
            )
            raise TrajectoryPlanningError(
                make_error(
                    ErrorCode.IK_PARTIAL_PATH, valid=str(n_valid), total=str(n_total)
                )
            )

        guard_cartesian_path(joint_path)

        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=state.motion_profile,
            velocity_frac=self.p.resolved_speed,
            accel_frac=self.p.accel,
            duration=self.p.resolved_duration,
            dt=INTERVAL_S,
            cart_vel_limit=LIMITS.cart.hard.velocity.linear * self.p.resolved_speed,
            cart_acc_limit=LIMITS.cart.hard.acceleration.linear * self.p.accel,
            path_knots=cartesian_path_knots(cartesian_trajectory),
            constant_tool_speed=self.constant_tool_speed,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self.trajectory_rad = trajectory.positions_rad
        self._duration = trajectory.duration

        self.log_info(
            "  -> Trajectory prepared: %d steps, %.2fs duration",
            len(self.trajectory_steps),
            trajectory.duration,
        )

    def generate_main_trajectory(self, effective_start_pose) -> np.ndarray:
        """Override this in subclasses to generate the specific motion trajectory."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")


@register_command(CmdType.MOVEC)
class MoveCCommand(CartesianChainLink, BaseSmoothMotionCommand[MoveCCmd]):
    """Execute circular arc motion through current → via → end (3-point arc).

    Via and end resolve against the pose the move starts from: absolute in
    WRF, tool-frame offsets in TRF. With a blend radius the arc joins a
    cartesian blend chain and rounds into the move after it.
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

    def generate_main_trajectory(self, effective_start_pose) -> np.ndarray:
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

    def chain_segment(
        self, previous: np.ndarray, state: "ControllerState"
    ) -> tuple[LineSegment | ArcSegment, np.ndarray]:
        via = resolve_pose(previous, self.p.via, self.p.frame, False)
        end = resolve_pose(previous, self.p.end, self.p.frame, False)
        try:
            return ArcSegment(previous, via, end), end
        except ValueError as e:
            raise TrajectoryPlanningError(
                make_error(ErrorCode.COMM_VALIDATION_ERROR, detail=str(e))
            ) from e


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

    def generate_main_trajectory(self, effective_start_pose) -> np.ndarray:
        """Generate spline starting from actual position."""
        assert self._waypoints is not None

        wps = self._waypoints
        motion_gen = SplineMotion()

        first_wp_error = _pose6_distance_mm(wps[0], effective_start_pose)

        if first_wp_error > WAYPOINT_SNAP_MM:
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


#: Each interior corner of a process move is rounded with this fraction of
#: the shorter adjoining segment.
MOVEP_AUTO_BLEND_FRAC: float = 0.25
#: SE3 samples per straight segment of a process move.
_MOVEP_SAMPLES_PER_SEGMENT: int = 20


@register_command(CmdType.MOVEP)
class MovePCommand(BaseSmoothMotionCommand[MovePCmd]):
    """Process move — the waypoint list as straight segments with every
    interior corner rounded, run at one constant tool speed: the TCP sweeps
    the path without stopping at a single waypoint."""

    PARAMS_TYPE = MovePCmd
    constant_tool_speed = True

    __slots__ = ("_waypoints",)

    def __init__(self, p: MovePCmd) -> None:
        super().__init__(p)
        self._waypoints: np.ndarray | None = None

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if TRF, build trajectory with constant TCP speed."""
        self._waypoints = _transform_waypoints_trf_to_wrf(
            self.p.waypoints, self.p.frame, state
        )
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose) -> np.ndarray:
        """The polyline through the waypoints with each interior corner
        rounded by a quarter of the shorter adjoining segment."""
        assert self._waypoints is not None

        wps = self._waypoints

        first_wp_error = _pose6_distance_mm(wps[0], effective_start_pose)
        if first_wp_error > WAYPOINT_SNAP_MM:
            all_waypoints = np.vstack([effective_start_pose[np.newaxis], wps])
        else:
            all_waypoints = np.vstack([effective_start_pose[np.newaxis], wps[1:]])

        poses = list(_se3_chain(all_waypoints))
        lengths = [
            float(np.linalg.norm(poses[i + 1][:3, 3] - poses[i][:3, 3])) * 1000.0
            for i in range(len(poses) - 1)
        ]
        radii = [
            MOVEP_AUTO_BLEND_FRAC * min(lengths[i], lengths[i + 1])
            for i in range(len(lengths) - 1)
        ]
        cart_poses = build_composite_cartesian_path(
            poses, radii, samples_per_segment=_MOVEP_SAMPLES_PER_SEGMENT
        )

        logger.debug(
            "    Generated process move path with %d SE3 poses across %d segments",
            len(cart_poses),
            len(lengths),
        )

        return cart_poses
