"""
Dry-run client that executes commands through the trajectory planner locally.

Delegates trajectory planning to TrajectoryPlanner (diagnostic=True) —
the same logic used by the real PlannerWorker subprocess. Jog commands
are simulated separately since the planner doesn't handle streaming.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from ..commands.base import MotionCommand
from ..commands.cartesian_commands import (
    JogLCommand,
    _CART_ANG_JOG_MAX_RAD,
    _CART_ANG_JOG_MIN_RAD,
    _CART_LIN_JOG_MAX_MS,
    _CART_LIN_JOG_MIN_MS,
    _linmap_frac,
)
from ..commands.basic_commands import JogJCommand
from ..config import (
    CONTROL_RATE_HZ,
    HOME_ANGLES_DEG,
    deg_to_steps,
    rad_to_steps,
    steps_to_rad,
)
from ..motion.geometry import joint_path_to_tcp_poses
from ..utils.ik import solve_ik
from pinokin import se3_from_rpy, se3_rpy
import re as _re

import parol6.protocol.wire as _wire
from ..protocol.wire import (
    HomeCmd,
    SelectToolCmd,
    SetTcpOffsetCmd,
    TeleportCmd,
    ToolActionCmd,
)
from ..server.command_registry import CommandRegistry
from ..server.motion_planner import (
    ErrorSegment,
    InlineSegment,
    Segment,
    TrajectoryPlanner,
    TrajectorySegment,
)
from ..server.state import ControllerState, get_fkine_se3
from ..utils.error_catalog import RobotError, make_error
from ..utils.error_codes import ErrorCode
from parol6.tools import get_registry


def _pascal_to_snake(name: str) -> str:
    """Convert PascalCase to snake_case: MoveJPose → move_j_pose"""
    s = _re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# Auto-derive method_name → struct_class from wire module.
# E.g. MoveJCmd → move_j, IsSimulatorCmd → is_simulator
_CMD_STRUCTS: dict[str, type] = {}
for _attr in dir(_wire):
    if _attr.endswith("Cmd") and isinstance(getattr(_wire, _attr), type):
        _CMD_STRUCTS[_pascal_to_snake(_attr.removesuffix("Cmd"))] = getattr(
            _wire, _attr
        )

_UPPER_FIELDS: frozenset[str] = frozenset({"tool_name", "tool_key", "profile"})


def build_cmd(name: str, *args: Any, **kwargs: Any) -> Any:
    """Build a command struct by method name."""
    struct_cls = _CMD_STRUCTS.get(name)
    if struct_cls is None:
        raise ValueError(f"Unknown command: {name}")
    struct_fields: tuple[str, ...] = getattr(struct_cls, "__struct_fields__", ())
    filtered = {}
    for k, v in kwargs.items():
        if v is None or k not in struct_fields:
            continue
        if k in _UPPER_FIELDS and isinstance(v, str):
            v = v.upper()
        filtered[k] = v
    return struct_cls(*args, **filtered)


logger = logging.getLogger(__name__)


@dataclass
class DryRunResult:
    """Result from a dry-run motion command."""

    tcp_poses: np.ndarray  # (N, 6) [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
    end_joints_rad: np.ndarray  # (6,) final joint angles
    duration: float  # trajectory duration in seconds
    error: RobotError | None = None
    valid: np.ndarray | None = None  # (N,) per-pose bool; None = all valid
    joint_trajectory_rad: np.ndarray | None = None  # (N, 6) full joint trajectory


def _error_result(error: RobotError) -> DryRunResult:
    """Build a DryRunResult for an error (empty trajectory)."""
    return DryRunResult(
        tcp_poses=np.empty((0, 6)),
        end_joints_rad=np.empty(6),
        duration=0.0,
        error=error,
        joint_trajectory_rad=None,
    )


def _build_result(radians: np.ndarray, duration: float) -> DryRunResult:
    """Build a DryRunResult from joint radians (N, 6) and duration.

    Converts joint radians → TCP poses in meters + radians.
    """
    tcp_poses = joint_path_to_tcp_poses(radians)
    tcp_poses[:, :3] /= 1000.0  # mm → m
    np.deg2rad(tcp_poses[:, 3:], out=tcp_poses[:, 3:])  # deg → rad
    return DryRunResult(
        tcp_poses=tcp_poses,
        end_joints_rad=radians[-1].copy(),
        duration=duration,
        joint_trajectory_rad=radians.copy(),
    )


class _DryRunTool:
    """Tool proxy for dry-run. Routes actions through the planner."""

    def __init__(self, client: DryRunRobotClient) -> None:
        self._client = client

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> DryRunResult | None:
            return self._client.tool_action(
                self._client._active_tool_key, name, list(args), **kwargs
            )

        return method


class DryRunRobotClient:
    """Runs commands through the trajectory planner without UDP/serial.

    Trajectory dispatch (including blend buffering and error handling) is
    delegated to TrajectoryPlanner in diagnostic mode. Jog commands are
    simulated separately since the planner doesn't handle streaming.

    Most methods are auto-dispatched via __getattr__ using CMD_MAP.
    Explicit methods exist only for angles/pose (read from state)
    and delay (no-op).
    """

    def __init__(
        self,
        initial_joints_deg: list[float] | None = None,
        max_snapshot_points: int = 200,
    ) -> None:
        # Reset tool transform — process pool workers persist across
        # invocations, so a previous run's select_tool() leaves a stale
        # TCP offset on the module-level robot singleton.
        PAROL6_ROBOT.apply_tool("NONE")

        self._state = ControllerState()
        init_deg = np.asarray(
            initial_joints_deg if initial_joints_deg is not None else HOME_ANGLES_DEG,
            dtype=np.float64,
        )
        deg_to_steps(init_deg, self._state.Position_in)

        self._planner = TrajectoryPlanner(diagnostic=True)
        self._planner.state.Position_in[:] = self._state.Position_in

        self._registry = CommandRegistry()
        self._q_rad_buf = np.zeros(6, dtype=np.float64)
        self._rpy_buf = np.zeros(3, dtype=np.float64)
        self._max_snapshot_points = max_snapshot_points
        self._active_tool_key: str = ""
        self._active_variant_key: str = ""
        self._tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._tool_proxy = _DryRunTool(self)

    @property
    def state(self) -> ControllerState:
        """Access the simulated controller state."""
        return self._state

    @property
    def tool(self) -> _DryRunTool:
        """Tool proxy that routes actions through the planner."""
        return self._tool_proxy

    def tcp_offset(self) -> list[float]:
        """Return current TCP offset in mm."""
        return [
            self._tcp_offset_m[0] * 1000.0,
            self._tcp_offset_m[1] * 1000.0,
            self._tcp_offset_m[2] * 1000.0,
        ]

    def flush(self) -> list[DryRunResult]:
        """Flush pending blend buffer. Call after script completion."""
        segments = self._planner.flush()
        self._state.Position_in[:] = self._planner.state.Position_in
        results: list[DryRunResult] = []
        for seg in segments:
            r = self._segment_to_result(seg)
            if r is not None:
                results.append(r)
        return results

    def _snap_to_angles(self, angles_deg: list[float]) -> DryRunResult:
        """Snap to angles instantly (no trajectory) — used by Home and Teleport."""
        self._planner.flush()
        deg = np.asarray(angles_deg, dtype=np.float64)
        deg_to_steps(deg, self._state.Position_in)
        self._planner.state.Position_in[:] = self._state.Position_in
        rad = np.radians(deg).reshape(1, -1)
        return _build_result(rad, duration=0.0)

    def _dispatch(self, params: Any) -> DryRunResult | None:
        """Route a command struct through the trajectory planner."""
        if isinstance(params, HomeCmd):
            return self._snap_to_angles(HOME_ANGLES_DEG)
        if isinstance(params, TeleportCmd):
            return self._snap_to_angles(params.angles)
        if isinstance(params, SelectToolCmd):
            self._active_tool_key = params.tool_name.strip().upper()
            self._active_variant_key = params.variant_key
            self._tcp_offset_m = (0.0, 0.0, 0.0)
        if isinstance(params, SetTcpOffsetCmd):
            self._tcp_offset_m = (
                params.x / 1000.0,
                params.y / 1000.0,
                params.z / 1000.0,
            )
            self._state._tcp_offset_m = self._tcp_offset_m
            PAROL6_ROBOT.apply_tool(
                self._active_tool_key or "NONE",
                variant_key=self._active_variant_key,
                tcp_offset_m=self._tcp_offset_m,
            )
        # Detect jog/servo commands — planner doesn't handle streaming.
        # Other non-trajectory MotionCommands (SelectTool, Home) fall through
        # to the planner which handles them as inline segments.
        cmd_cls = self._registry.get_command_for_struct(type(params))
        if cmd_cls is not None and issubclass(cmd_cls, (JogJCommand, JogLCommand)):
            # Flush blend buffer, sync state, simulate jog
            self._planner.flush()
            self._state.Position_in[:] = self._planner.state.Position_in
            cmd = cmd_cls(params)
            assert isinstance(cmd, MotionCommand)
            result = self._simulate_jog(cmd)
            self._planner.state.Position_in[:] = self._state.Position_in
            return result

        # Everything else → planner
        segments = self._planner.process(params)
        self._state.Position_in[:] = self._planner.state.Position_in

        results: list[DryRunResult] = []
        for seg in segments:
            r = self._segment_to_result(seg)
            if r is not None:
                results.append(r)

        if not results:
            return None
        if len(results) == 1:
            return results[0]
        return self._merge_results(results)

    def _segment_to_result(self, seg: Segment) -> DryRunResult | None:
        """Convert a planner segment to a DryRunResult."""
        if isinstance(seg, TrajectorySegment):
            return self._trajectory_segment_to_result(seg)
        if isinstance(seg, ErrorSegment):
            return self._error_segment_to_result(seg)
        if isinstance(seg, InlineSegment) and isinstance(seg.params, ToolActionCmd):
            return self._tool_action_segment_to_result(seg.params)
        # Other InlineSegments (SelectTool, Home, etc.) — no visualization
        return None

    def _trajectory_segment_to_result(self, seg: TrajectorySegment) -> DryRunResult:
        """Convert a TrajectorySegment to a DryRunResult."""
        steps = seg.trajectory_steps
        stride = max(1, len(steps) // self._max_snapshot_points)
        sampled = steps[::stride]
        if not np.array_equal(sampled[-1], steps[-1]):
            sampled = np.vstack([sampled, steps[-1:]])

        radians = np.empty((len(sampled), 6), dtype=np.float64)
        for i in range(len(sampled)):
            steps_to_rad(sampled[i], radians[i])

        return _build_result(radians, seg.duration)

    def _error_segment_to_result(self, seg: ErrorSegment) -> DryRunResult:
        """Convert an ErrorSegment to a DryRunResult with per-pose validity."""
        if seg.cartesian_path is not None and seg.ik_valid is not None:
            return DryRunResult(
                tcp_poses=seg.cartesian_path,
                end_joints_rad=np.zeros(6, dtype=np.float64),
                duration=0.0,
                error=seg.error,
                valid=seg.ik_valid,
                joint_trajectory_rad=None,
            )
        return _error_result(seg.error)

    def _tool_action_segment_to_result(self, cmd: ToolActionCmd) -> DryRunResult:
        """Return a single-point DryRunResult at the current TCP pose."""
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        duration = 0.0
        cfg = get_registry().get(cmd.tool_key.strip().upper())
        if cfg is not None:
            duration = cfg.estimate_duration(cmd.action, cmd.params)
        return _build_result(self._q_rad_buf[np.newaxis], duration)

    def _merge_results(self, results: list[DryRunResult]) -> DryRunResult:
        """Merge multiple DryRunResults into one (for multi-segment blends)."""
        non_empty = [r for r in results if r.tcp_poses.shape[0] > 0]
        first_error = next((r.error for r in results if r.error is not None), None)
        if not non_empty:
            if first_error is not None:
                return _error_result(first_error)
            return _error_result(make_error(ErrorCode.TRAJ_EMPTY_RESULT, detail=""))

        tcp_all = np.vstack([r.tcp_poses for r in non_empty])
        total_duration = sum(r.duration for r in results)
        last = non_empty[-1]

        has_any_valid = any(r.valid is not None for r in non_empty)
        if has_any_valid:
            valids = [
                r.valid
                if r.valid is not None
                else np.ones(r.tcp_poses.shape[0], dtype=np.bool_)
                for r in non_empty
            ]
            merged_valid = np.concatenate(valids)
        else:
            merged_valid = None

        has_any_joints = any(r.joint_trajectory_rad is not None for r in non_empty)
        if has_any_joints:
            joint_parts = [
                r.joint_trajectory_rad
                if r.joint_trajectory_rad is not None
                else np.broadcast_to(
                    r.end_joints_rad[np.newaxis, :],
                    (r.tcp_poses.shape[0], r.end_joints_rad.shape[0]),
                ).copy()
                for r in non_empty
            ]
            merged_joints = np.vstack(joint_parts)
        else:
            merged_joints = None

        return DryRunResult(
            tcp_poses=tcp_all,
            end_joints_rad=last.end_joints_rad,
            duration=total_duration,
            error=first_error,
            valid=merged_valid,
            joint_trajectory_rad=merged_joints,
        )

    # ---- Jog simulation (planner doesn't handle streaming) ----

    def _simulate_jog(self, cmd: MotionCommand) -> DryRunResult | None:
        """Simulate jog commands by computing linear displacement."""
        # Run do_setup so speeds_out / _axis_index / etc. are computed
        cmd.setup(self._state)

        if isinstance(cmd, JogJCommand):
            return self._simulate_joint_jog(cmd)
        if isinstance(cmd, JogLCommand):
            return self._simulate_cartesian_jog(cmd)
        return None

    def _simulate_joint_jog(self, cmd: JogJCommand) -> DryRunResult:
        """Simulate joint jog by computing linear displacement in joint space."""
        duration = cmd.p.duration
        n_points = min(
            self._max_snapshot_points,
            max(2, int(duration * CONTROL_RATE_HZ)),
        )

        # Compute total displacement (steps/tick * ticks_in_duration)
        ticks = duration * CONTROL_RATE_HZ
        displacements = cmd.speeds_out.astype(np.int64) * int(ticks)

        start_pos = self._state.Position_in.copy()
        fracs = np.arange(1, n_points + 1, dtype=np.float64) / n_points
        # trajectory shape (n_points, 6): start + fraction * displacement
        trajectory = start_pos[np.newaxis, :] + (
            fracs[:, np.newaxis] * displacements[np.newaxis, :]
        ).astype(np.int64)

        self._state.Position_in[:] = start_pos + displacements

        radians = np.empty((n_points, 6), dtype=np.float64)
        for i in range(n_points):
            steps_to_rad(trajectory[i], radians[i])

        return _build_result(radians, duration)

    def _simulate_cartesian_jog(self, cmd: JogLCommand) -> DryRunResult | None:
        """Simulate cartesian jog by displacing along a Cartesian axis and solving IK."""
        duration = cmd.p.duration
        n_points = min(
            self._max_snapshot_points,
            max(2, int(duration * CONTROL_RATE_HZ)),
        )

        # Current pose via FK
        current_se3 = get_fkine_se3(self._state)
        se3_rpy(current_se3, self._rpy_buf)
        # pose = [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
        pose = np.array(
            [
                current_se3[0, 3],
                current_se3[1, 3],
                current_se3[2, 3],
                self._rpy_buf[0],
                self._rpy_buf[1],
                self._rpy_buf[2],
            ],
            dtype=np.float64,
        )

        # Compute velocity along dominant axis using same mapping as production
        vels = cmd.p.velocities
        speed_mag = abs(vels[cmd._axis_index + (3 if cmd.is_rotation else 0)])
        if cmd.is_rotation:
            vel = _linmap_frac(speed_mag, _CART_ANG_JOG_MIN_RAD, _CART_ANG_JOG_MAX_RAD)
            total_disp = vel * cmd._axis_sign * duration
        else:
            vel = _linmap_frac(speed_mag, _CART_LIN_JOG_MIN_MS, _CART_LIN_JOG_MAX_MS)
            total_disp = vel * cmd._axis_sign * duration

        # Determine which component of the 6-element pose to displace
        pose_index = (3 + cmd._axis_index) if cmd.is_rotation else cmd._axis_index

        # Get current joint angles for IK seed
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        q_current = self._q_rad_buf.copy()
        steps_buf = np.zeros_like(self._state.Position_in)

        # Generate trajectory by interpolating and solving IK at each point
        radians = np.empty((n_points, 6), dtype=np.float64)
        last_valid_q = q_current.copy()
        target_se3 = np.zeros((4, 4), dtype=np.float64)

        for i in range(n_points):
            t = (i + 1) / n_points
            target_pose = pose.copy()
            target_pose[pose_index] += total_disp * t

            se3_from_rpy(
                target_pose[0],
                target_pose[1],
                target_pose[2],
                target_pose[3],
                target_pose[4],
                target_pose[5],
                target_se3,
            )
            ik_result = solve_ik(
                PAROL6_ROBOT.robot, target_se3, last_valid_q, quiet_logging=True
            )
            if ik_result.success:
                last_valid_q = ik_result.q.copy()

            radians[i] = last_valid_q

        # Update state to final position
        rad_to_steps(last_valid_q, steps_buf)
        self._state.Position_in[:] = steps_buf

        return _build_result(radians, duration)

    # ---- Explicit methods for state reads ----

    def angles(self) -> list[float]:
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        return np.degrees(self._q_rad_buf).tolist()

    def pose(self) -> list[float]:
        """Return [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        se3 = get_fkine_se3(self._state)
        se3_rpy(se3, self._rpy_buf)
        return [
            se3[0, 3] * 1000.0,
            se3[1, 3] * 1000.0,
            se3[2, 3] * 1000.0,
            float(np.degrees(self._rpy_buf[0])),
            float(np.degrees(self._rpy_buf[1])),
            float(np.degrees(self._rpy_buf[2])),
        ]

    def move_j(
        self,
        angles: list[float] | None = None,
        *,
        pose: list[float] | None = None,
        **kwargs: Any,
    ) -> DryRunResult | None:
        if pose is not None:
            return self._dispatch(build_cmd("move_j_pose", pose, **kwargs))
        return self._dispatch(build_cmd("move_j", angles or [], **kwargs))

    def servo_j(
        self,
        angles: list[float] | None = None,
        *,
        pose: list[float] | None = None,
        **kwargs: Any,
    ) -> DryRunResult | None:
        if pose is not None:
            return self._dispatch(build_cmd("servo_j_pose", pose, **kwargs))
        return self._dispatch(build_cmd("servo_j", angles or [], **kwargs))

    def delay(self, seconds: float = 0.0) -> None:
        pass

    def wait_motion(self, **kwargs: Any) -> None:
        self.flush()

    # ---- Auto-dispatch for everything else ----

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _CMD_STRUCTS:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        def method(*args: Any, **kwargs: Any) -> DryRunResult | None:
            cmd = build_cmd(name, *args, **kwargs)
            return self._dispatch(cmd)

        return method
