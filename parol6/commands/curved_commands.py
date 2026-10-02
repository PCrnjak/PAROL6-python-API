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
from parol6.config import INTERVAL_S, LIMITS, PATH_SAMPLES, steps_to_rad
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.protocol.wire import (
    CmdType,
    MoveCCmd,
    MotionParamsMixin,
    MovePCmd,
    MoveSCmd,
)
from parol6.commands.cartesian_commands import (
    CartesianChainLink,
    resolve_pose,
)
from parol6.motion.geometry import (
    ArcSegment,
    LineSegment,
    build_blended_path,
    build_composite_cartesian_path,
    build_spline_path,
    pose_distance_m,
)
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_se3
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import IKError, TrajectoryPlanningError

_MP = TypeVar("_MP", bound=MotionParamsMixin)

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


#: A waypoint list's first entry stands in for the start pose within this,
#: on the combined translation and rotation metric [mm].
WAYPOINT_SNAP_MM: float = 5.0


def _waypoint_chain(
    start: np.ndarray, waypoints: Sequence[Sequence[float]], frame: str
) -> list[np.ndarray]:
    """The SE3 poses a waypoint list names, resolved against ``start`` as
    ``resolve_pose`` resolves a move's target, led by ``start`` itself: a
    first waypoint within ``WAYPOINT_SNAP_MM`` of it is the start."""
    poses = [start] + [resolve_pose(start, list(wp), frame, False) for wp in waypoints]
    if len(poses) > 1 and pose_distance_m(start, poses[1]) * 1000.0 <= WAYPOINT_SNAP_MM:
        del poses[1]
    return poses


# =============================================================================
# Smooth Motion Command Base
# =============================================================================


class BaseSmoothMotionCommand(TrajectoryMoveCommandBase[_MP]):
    """Base class for smooth geometry commands (arc, spline, process move).

    Subclasses implement generate_main_trajectory() to create Cartesian geometry.
    This base class handles IK conversion and trajectory building.
    """

    #: Hold the tool to one speed along the whole path (a process move)
    #: rather than as fast as the joints allow under the cartesian ceiling.
    constant_tool_speed: bool = False

    __slots__ = ()

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute trajectory from current position."""
        guard_homed(state)
        self.log_debug("  -> Preparing %s...", self.name)

        start = get_fkine_se3(state).copy()
        self.log_info(
            "  -> Generating %s from position: %s",
            self.name,
            [round(float(p) * 1000.0, 1) for p in start[:3, 3]],
        )

        cartesian_trajectory = self.generate_main_trajectory(start, state)
        if len(cartesian_trajectory) == 0:
            raise TrajectoryPlanningError(
                make_error(
                    ErrorCode.TRAJ_EMPTY_RESULT, detail="empty cartesian trajectory"
                )
            )

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
            path_knots=joint_path.knots,
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

    def generate_main_trajectory(
        self, start: np.ndarray, state: "ControllerState"
    ) -> np.ndarray:
        """The (N, 4, 4) SE3 poses the move runs through from ``start``."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")


@register_command(CmdType.MOVEC)
class MoveCCommand(CartesianChainLink, BaseSmoothMotionCommand[MoveCCmd]):
    """Execute circular arc motion through current → via → end (3-point arc).

    Via and end resolve against the pose the move starts from: absolute in
    WRF, tool-frame offsets in TRF. With a blend radius the arc joins a
    cartesian blend chain and rounds into the move after it.
    """

    PARAMS_TYPE = MoveCCmd

    __slots__ = ()

    def generate_main_trajectory(
        self, start: np.ndarray, state: "ControllerState"
    ) -> np.ndarray:
        """The arc from ``start`` through the via to the end."""
        arc, self.target_pose = self.chain_segment(start, state)
        return build_blended_path([arc], [], samples_per_segment=PATH_SAMPLES)

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

    __slots__ = ()

    def generate_main_trajectory(
        self, start: np.ndarray, state: "ControllerState"
    ) -> np.ndarray:
        """The spline from ``start`` through the waypoints, sampled by its
        length whatever duration times it: a duration too short to keep is
        stretched to one the arm can, never met by cutting the path."""
        poses = _waypoint_chain(start, self.p.waypoints, self.p.frame)
        trajectory = build_spline_path(poses)
        logger.debug("    Generated spline with %d poses", len(trajectory))
        return trajectory


#: Each interior corner of a process move is rounded with this fraction of
#: the shorter adjoining segment.
MOVEP_AUTO_BLEND_FRAC: float = 0.25


@register_command(CmdType.MOVEP)
class MovePCommand(BaseSmoothMotionCommand[MovePCmd]):
    """Process move — the waypoint list as straight segments with every
    interior corner rounded, run at one constant tool speed: the TCP sweeps
    the path without stopping at a single waypoint."""

    PARAMS_TYPE = MovePCmd
    constant_tool_speed = True

    __slots__ = ()

    def generate_main_trajectory(
        self, start: np.ndarray, state: "ControllerState"
    ) -> np.ndarray:
        """The polyline through the waypoints with each interior corner
        rounded by a quarter of the shorter adjoining segment, sampled by
        its length."""
        poses = _waypoint_chain(start, self.p.waypoints, self.p.frame)
        lengths = [
            float(np.linalg.norm(poses[i + 1][:3, 3] - poses[i][:3, 3])) * 1000.0
            for i in range(len(poses) - 1)
        ]
        radii = [
            MOVEP_AUTO_BLEND_FRAC * min(lengths[i], lengths[i + 1])
            for i in range(len(lengths) - 1)
        ]
        cart_poses = build_composite_cartesian_path(poses, radii)

        logger.debug(
            "    Generated process move path with %d SE3 poses across %d segments",
            len(cart_poses),
            len(lengths),
        )

        return cart_poses
