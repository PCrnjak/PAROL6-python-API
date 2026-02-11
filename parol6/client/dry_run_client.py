"""
Dry-run client that executes commands through the real command pipeline locally.

Uses a ControllerState with simulated Position_in. ALL commands go through
msgspec struct creation (validation) and command lookup. Motion commands
run do_setup() (same as real controller) to produce identical trajectories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from ..commands.base import (
    MotionCommand,
    SystemCommand,
    TrajectoryMoveCommandBase,
)
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
    MAX_BLEND_LOOKAHEAD,
    deg_to_steps,
    rad_to_steps,
    steps_to_rad,
)
from ..motion.geometry import joint_path_to_tcp_poses
from ..utils.ik import solve_ik
from pinokin import se3_from_rpy, se3_rpy
from ..protocol.wire import (
    CheckpointCmd,
    DelayCmd,
    ElectricGripperCmd,
    GetAnglesCmd,
    GetCurrentActionCmd,
    GetGripperCmd,
    GetIOCmd,
    GetLoopStatsCmd,
    GetPoseCmd,
    GetProfileCmd,
    GetQueueCmd,
    GetSpeedsCmd,
    GetStatusCmd,
    GetToolCmd,
    HaltCmd,
    HomeCmd,
    JogJCmd,
    JogLCmd,
    MoveCCmd,
    MoveJCmd,
    MoveJPoseCmd,
    MoveLCmd,
    MovePCmd,
    MoveSCmd,
    PingCmd,
    PneumaticGripperCmd,
    ResetCmd,
    ResetLoopStatsCmd,
    ResumeCmd,
    ServoJCmd,
    ServoJPoseCmd,
    ServoLCmd,
    SetIOCmd,
    SetPortCmd,
    SetProfileCmd,
    SetToolCmd,
    SimulatorCmd,
)
from ..server.command_registry import CommandRegistry
from ..server.state import ControllerState, get_fkine_se3

# Method name → (struct class, default kwargs applied before caller kwargs)
CMD_MAP: dict[str, tuple[type, dict[str, Any]]] = {
    "home": (HomeCmd, {}),
    "moveJ": (MoveJCmd, {}),
    "moveJ_pose": (MoveJPoseCmd, {}),
    "moveL": (MoveLCmd, {}),
    "moveC": (MoveCCmd, {"frame": "WRF"}),
    "moveS": (MoveSCmd, {"frame": "WRF"}),
    "moveP": (MovePCmd, {"frame": "WRF"}),
    "jogJ": (JogJCmd, {}),
    "jogL": (JogLCmd, {"frame": "WRF"}),
    "servoJ": (ServoJCmd, {}),
    "servoJ_pose": (ServoJPoseCmd, {}),
    "servoL": (ServoLCmd, {}),
    "checkpoint": (CheckpointCmd, {}),
    "delay": (DelayCmd, {}),
    "resume": (ResumeCmd, {}),
    "halt": (HaltCmd, {}),
    "reset": (ResetCmd, {}),
    "set_tool": (SetToolCmd, {}),
    "set_profile": (SetProfileCmd, {}),
    "set_io": (SetIOCmd, {}),
    "set_serial_port": (SetPortCmd, {}),
    "simulator_on": (SimulatorCmd, {"on": True}),
    "simulator_off": (SimulatorCmd, {"on": False}),
    "control_pneumatic_gripper": (PneumaticGripperCmd, {}),
    "control_electric_gripper": (ElectricGripperCmd, {}),
    "ping": (PingCmd, {}),
    "get_angles": (GetAnglesCmd, {}),
    "get_io": (GetIOCmd, {}),
    "get_gripper": (GetGripperCmd, {}),
    "get_speeds": (GetSpeedsCmd, {}),
    "get_pose": (GetPoseCmd, {}),
    "get_status": (GetStatusCmd, {}),
    "get_loop_stats": (GetLoopStatsCmd, {}),
    "reset_loop_stats": (ResetLoopStatsCmd, {}),
    "get_profile": (GetProfileCmd, {}),
    "get_tool": (GetToolCmd, {}),
    "get_current_action": (GetCurrentActionCmd, {}),
    "get_queue": (GetQueueCmd, {}),
}

# Client param names → struct field names (only applied when the struct
# has the target field, so "speed" won't rename on ElectricGripperCmd).
_FIELD_RENAMES: dict[str, str] = {
    "joint_angles": "angles",
    "joint_index": "joint",
    "index": "port_index",
    "program_lines": "lines",
}

_UPPER_FIELDS: frozenset[str] = frozenset({"tool_name", "profile"})


def build_cmd(name: str, *args: Any, **kwargs: Any) -> Any:
    """Build a command struct by method name with automatic param renaming."""
    entry = CMD_MAP.get(name)
    if entry is None:
        raise ValueError(f"Unknown command: {name}")
    struct_cls, defaults = entry
    struct_fields: tuple[str, ...] = getattr(struct_cls, "__struct_fields__", ())
    merged = dict(defaults)
    for k, v in kwargs.items():
        if v is None:
            continue
        renamed = _FIELD_RENAMES.get(k)
        field = renamed if renamed and renamed in struct_fields else k
        if field not in struct_fields:
            continue
        if k in _UPPER_FIELDS and isinstance(v, str):
            v = v.upper()
        merged[field] = v
    return struct_cls(*args, **merged)


logger = logging.getLogger(__name__)


@dataclass
class DryRunResult:
    """Result from a dry-run motion command."""

    tcp_poses: np.ndarray  # (N, 6) [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
    end_joints_rad: np.ndarray  # (6,) final joint angles
    duration: float  # trajectory duration in seconds
    error: str | None = None  # IK failure message etc.


def _error_result(msg: str) -> DryRunResult:
    """Build a DryRunResult for an error (empty trajectory)."""
    return DryRunResult(
        tcp_poses=np.empty((0, 6)),
        end_joints_rad=np.empty(6),
        duration=0.0,
        error=msg,
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
    )


class DryRunRobotClient:
    """Runs commands through the real command pipeline without UDP/serial.

    Uses a ControllerState with simulated Position_in. ALL commands go through
    msgspec struct creation (validation) and command lookup. Motion commands
    run do_setup() (same as real controller) to produce identical trajectories.

    Most methods are auto-dispatched via __getattr__ using CMD_MAP.
    Explicit methods exist only for get_angles/get_pose (read from state)
    and delay (no-op).
    """

    def __init__(
        self,
        initial_joints_deg: list[float] | None = None,
        max_snapshot_points: int = 50,
    ) -> None:
        self._state = ControllerState()
        init_deg = np.asarray(
            initial_joints_deg if initial_joints_deg is not None else HOME_ANGLES_DEG,
            dtype=np.float64,
        )
        deg_to_steps(init_deg, self._state.Position_in)

        self._registry = CommandRegistry()
        self._q_rad_buf = np.zeros(6, dtype=np.float64)
        self._rpy_buf = np.zeros(3, dtype=np.float64)
        self._max_snapshot_points = max_snapshot_points
        self._blend_buffer: list[TrajectoryMoveCommandBase] = []

    @property
    def state(self) -> ControllerState:
        """Access the simulated controller state."""
        return self._state

    def flush(self) -> DryRunResult | None:
        """Flush pending blend buffer. Call after script completion."""
        return self._flush_blend()

    def _setup_and_snapshot(
        self,
        cmd: TrajectoryMoveCommandBase,
        blend_cmds: list[TrajectoryMoveCommandBase] | None = None,
    ) -> DryRunResult:
        """Run setup (with optional blend), snapshot trajectory, update state.

        Shared by _flush_blend and _dispatch_trajectory for single-command dispatch.
        """
        try:
            if blend_cmds:
                cmd.do_setup_with_blend(self._state, blend_cmds)
            else:
                cmd.setup(self._state)
        except Exception as e:
            return _error_result(str(e))

        if len(cmd.trajectory_steps) == 0:
            return _error_result(cmd.error_message or "Empty trajectory")

        result = self._snapshot_trajectory(cmd)
        self._state.Position_in[:] = cmd.trajectory_steps[-1]
        return result

    def _flush_blend(self) -> DryRunResult | None:
        """Flush blend buffer: build composite trajectory from buffered commands."""
        if not self._blend_buffer:
            return None
        head = self._blend_buffer[0]
        rest = self._blend_buffer[1:]
        self._blend_buffer.clear()
        return self._setup_and_snapshot(head, rest if rest else None)

    def _dispatch(self, params: Any) -> DryRunResult | None:
        """Route a command struct through the real pipeline locally."""
        cmd_cls = self._registry.get_command_for_struct(type(params))
        if cmd_cls is None:
            logger.warning("No handler for %s", type(params).__name__)
            return None

        cmd = cmd_cls(params)

        if isinstance(cmd, MotionCommand):
            if isinstance(cmd, TrajectoryMoveCommandBase):
                return self._dispatch_trajectory(cmd)
            # Non-trajectory motion (jog): flush blend buffer first
            self._flush_blend()
            return self._simulate_jog(cmd)

        # System/Query: flush blend buffer first
        self._flush_blend()

        if isinstance(cmd, SystemCommand):
            try:
                cmd.setup(self._state)
                cmd.execute_step(self._state)
            except Exception as e:
                logger.debug("System command %s failed: %s", type(params).__name__, e)
            return None

        return None

    def _dispatch_trajectory(
        self, cmd: TrajectoryMoveCommandBase
    ) -> DryRunResult | None:
        """Dispatch a trajectory command, buffering if blend radius > 0."""
        if cmd.blend_radius > 0:
            self._blend_buffer.append(cmd)
            if len(self._blend_buffer) > MAX_BLEND_LOOKAHEAD:
                return self._flush_blend()
            return None

        if self._blend_buffer:
            # r=0 terminates the chain (included in composite, same as real executor)
            self._blend_buffer.append(cmd)
            return self._flush_blend()

        # No blending, single command dispatch
        return self._setup_and_snapshot(cmd)

    def _snapshot_trajectory(self, cmd: TrajectoryMoveCommandBase) -> DryRunResult:
        """Extract TCP poses from pre-computed trajectory steps."""
        steps = cmd.trajectory_steps
        stride = max(1, len(steps) // self._max_snapshot_points)
        sampled = steps[::stride]
        if not np.array_equal(sampled[-1], steps[-1]):
            sampled = np.vstack([sampled, steps[-1:]])

        radians = np.empty((len(sampled), 6), dtype=np.float64)
        for i in range(len(sampled)):
            steps_to_rad(sampled[i], radians[i])

        return _build_result(radians, cmd._duration)

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

    def get_angles(self) -> list[float]:
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        return np.degrees(self._q_rad_buf).tolist()

    def get_pose(self) -> list[float]:
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

    def delay(self, seconds: float = 0.0) -> None:
        pass

    # ---- Auto-dispatch for everything else ----

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in CMD_MAP:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        def method(*args: Any, **kwargs: Any) -> DryRunResult | None:
            cmd = build_cmd(name, *args, **kwargs)
            return self._dispatch(cmd)

        return method
