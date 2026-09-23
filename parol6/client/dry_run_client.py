"""
Dry-run client that executes commands through the trajectory planner locally.

Delegates trajectory planning to TrajectoryPlanner (diagnostic=True) —
the same logic used by the real PlannerWorker subprocess. Jog commands
are simulated separately since the planner doesn't handle streaming.

Every command a program issues becomes one block of a commanded
``TickIndex``: a motion at the planner's tick resolution, a delay as rows
holding the pose, a tool action as rows over its estimated travel, and a
command that plans nothing (a checkpoint, a tool selection, a refusal) as
a zero-row block that still carries its place and its error. ``plan()``
lays the blocks on one 50 Hz axis; ``simulate()`` returns the same record,
because a planner has no plant to predict with and the plan is the honest
answer to what the arm will do.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from waldoctl.execution import ExecutionSpeed, validate_execution_scale
from waldoctl.skills import UnresolvedPreview
from waldoctl.ticks import TickBlock, TickIndex

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from ..ack_policy import ARM_MOTION_CMD_TYPES
from ..commands.base import MotionCommand
from ..commands.cartesian_commands import JogLCommand, jog_twist
from ..commands.basic_commands import JogJCommand
from ..config import (
    CONTROL_RATE_HZ,
    HOME_ANGLES_DEG,
    INTERVAL_S,
    deg_to_steps,
    rad_to_steps,
    steps_to_rad,
)
from ..motion.geometry import joint_path_to_tcp_poses
from ..utils.ik import solve_ik
from pinokin import se3_rpy
from math import degrees, radians

import parol6.protocol.wire as _wire
from waldoctl.commands import CommandKind, command_table
from ..protocol.wire import (
    HomeCmd,
    SetShapesCmd,
    SelectToolCmd,
    SetTcpOffsetCmd,
    SetTcpTransformCmd,
    WriteIOCmd,
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
from ..utils.errors import TrajectoryPlanningError
from parol6.tools import (
    ElectricGripperConfig,
    PneumaticGripperConfig,
    get_registry,
    tool_action_refusal,
)
from waldoctl.tools import ToolType

if TYPE_CHECKING:
    from parol6.robot import Robot


# Auto-derive method_name → struct_class from wire module.
# E.g. MoveJCmd → move_j, IsSimulatorCmd → is_simulator
_CMD_STRUCTS: dict[str, type] = {}
for _attr in dir(_wire):
    if _attr.endswith("Cmd") and isinstance(getattr(_wire, _attr), type):
        _CMD_STRUCTS[_wire.pascal_to_snake(_attr.removesuffix("Cmd"))] = getattr(
            _wire, _attr
        )

_UPPER_FIELDS: frozenset[str] = frozenset({"tool_name", "tool_key", "profile"})

_COMMANDS = command_table()
_AXIS_INDEX: dict[str, int] = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}


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


def _twist_pose(
    start: np.ndarray, twist: np.ndarray, t: float, wrf: bool
) -> np.ndarray:
    """The pose a TCP driven at `twist` for `t` seconds from `start`
    reaches: the translation and the rotation each integrate on their own
    axis, in world axes when `wrf` else in the tool's."""
    omega = twist[3:] * t
    angle = float(np.linalg.norm(omega))
    if angle > 1e-12:
        k = omega / angle
        kx = np.array(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            dtype=np.float64,
        )
        rot = np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)
    else:
        rot = np.eye(3)
    out = np.eye(4, dtype=np.float64)
    r0 = start[:3, :3]
    if wrf:
        out[:3, :3] = rot @ r0
        out[:3, 3] = start[:3, 3] + twist[:3] * t
    else:
        out[:3, :3] = r0 @ rot
        out[:3, 3] = start[:3, 3] + r0 @ (twist[:3] * t)
    return out


#: Row spacing of the commanded record: the rate par6's engine keeps too,
#: so a host scrubs both backends on one axis.
_ROW_RATE_HZ = 50.0
_STRIDE = max(1, int(round(CONTROL_RATE_HZ / _ROW_RATE_HZ)))
_ROW_DT_S = _STRIDE * INTERVAL_S
_MOVE_TYPE: dict[str, str | None] = {
    name: spec.move_type for name, spec in _COMMANDS.items()
}


def _tcp_from_joints(q_rad: np.ndarray) -> np.ndarray:
    """``(N, 6)`` TCP ``[x, y, z, rx, ry, rz]`` in metres and radians for joint
    rows in radians, under the tool applied right now."""
    if q_rad.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float64)
    tcp = joint_path_to_tcp_poses(q_rad)
    tcp[:, :3] /= 1000.0
    np.deg2rad(tcp[:, 3:], out=tcp[:, 3:])
    return tcp


def _digest(joints: np.ndarray, tcp: np.ndarray) -> bytes:
    """Identity over what reaches the screen, quantised below what a display
    resolves, so two runs that paint the same picture hash the same."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.round(joints / 1e-4).astype(np.int64).tobytes())
    h.update(np.round(tcp / 1e-5).astype(np.int64).tobytes())
    return h.digest()


@dataclass(slots=True)
class _Chunk:
    """One program command at control-tick resolution, before the record
    decimates it onto the row axis. Zero ticks for a command that plans
    nothing: folded into a blend chain, refused, or state-only."""

    command: int
    method: str
    q_rad: np.ndarray
    tcp: np.ndarray
    tool_closed: np.ndarray
    valid: np.ndarray | None = None
    error: RobotError | None = None

    @property
    def ticks(self) -> int:
        return int(self.q_rad.shape[0])


def _truncated(record: TickIndex, max_seconds: float) -> TickIndex:
    limit = math.ceil(max_seconds / record.row_dt_s)
    if limit >= record.rows:
        return record
    blocks = tuple(
        TickBlock(
            command=b.command,
            start_row=min(b.start_row, limit),
            rows=max(0, min(b.rows, limit - b.start_row)),
            line_number=b.line_number,
            error=b.error,
            move_type=b.move_type,
        )
        for b in record.blocks
    )
    joints = record.joints_rad[:limit]
    tcp = record.tcp[:limit]
    return TickIndex(
        row_dt_s=record.row_dt_s,
        joints_rad=joints,
        tcp=tcp,
        tool_closed=record.tool_closed[:limit],
        tool_gripping=record.tool_gripping[:limit],
        blocks=blocks,
        stop="budget_exhausted",
        digest=_digest(joints, tcp),
        valid=None if record.valid is None else record.valid[:limit],
    )


class _DryRunTool:
    """Tool proxy for dry-run. Routes actions through the planner, spelling
    the ToolSpec methods as the live tools do: an electric gripper's
    ``open``/``close``/``set_position`` are a ``move`` at the tool's default
    current, ``release`` is ``idle``; a pneumatic ``set_position`` opens
    below 0.5 and closes at or above it."""

    def __init__(self, client: DryRunRobotClient) -> None:
        self._client = client

    @property
    def key(self) -> str:
        return self._client._active_tool_key

    @property
    def tool_type(self) -> str:
        spec = get_registry().get(self.key)
        return (
            ToolType.GRIPPER
            if isinstance(spec, (ElectricGripperConfig, PneumaticGripperConfig))
            else ToolType.NONE
        )

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> int:
            action, params = self._translate(name, list(args), kwargs)
            return self._client.tool_action(self.key, action, params, **kwargs)

        return method

    def _translate(
        self, name: str, args: list[Any], kwargs: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        cfg = get_registry().get(self.key)
        if isinstance(cfg, ElectricGripperConfig):
            if name in ("open", "close", "set_position"):
                position = (
                    0.0 if name == "open" else 1.0 if name == "close" else args[0]
                )
                speed = float(kwargs.pop("speed", 0.5))
                current = int(kwargs.pop("current", cfg.default_current))
                return "move", [position, speed, current]
            if name == "release":
                return "idle", []
        elif isinstance(cfg, PneumaticGripperConfig) and name == "set_position":
            return ("open" if args[0] < 0.5 else "close"), []
        return name, args


class DryRunRobotClient:
    """Runs commands through the trajectory planner without UDP/serial.

    Trajectory dispatch (including blend buffering and error handling) is
    delegated to TrajectoryPlanner in diagnostic mode. Jog commands are
    simulated separately since the planner doesn't handle streaming.

    Command methods answer as the live client does — a program index for
    queued work, a code for the rest — and the record of what they planned
    comes back from ``plan()``. Most methods are auto-dispatched via
    __getattr__ using CMD_MAP; execution controls change the planning
    clock; observations read local state.
    """

    _robot: Robot | None = None

    @property
    def robot(self) -> Robot:
        """The backend this preview stands in for, built on first read when
        the host constructed the client bare. A real descriptor on the class,
        so the read never reaches ``__getattr__``'s command dispatch."""
        if self._robot is None:
            from parol6.robot import Robot

            self._robot = Robot()
        return self._robot

    @robot.setter
    def robot(self, value: Robot | None) -> None:
        self._robot = value

    def __init__(
        self,
        initial_joints_deg: list[float] | None = None,
        initial_homed: bool = True,
        initial_gripper_calibrated: bool = False,
        robot: Robot | None = None,
    ) -> None:
        self._robot = robot
        # Reset tool transform — process pool workers persist across
        # invocations, so a previous run's select_tool() leaves a stale
        # TCP offset on the module-level robot singleton.
        PAROL6_ROBOT.apply_tool("NONE")

        # Spawn-mode subprocess: the tool registry is freshly imported with only
        # native tools, so plugin tools must be registered here too or
        # select_tool() of a plugin tool fails in apply_tool (mirrors the planner
        # worker).
        from parol6.tools import register_plugin_tools

        register_plugin_tools()

        self._state = ControllerState()
        # Mirror the live gate: an electric gripper's jaw move before a
        # calibrate is refused here exactly as the controller refuses it.
        self._state.gripper_calibrated = bool(initial_gripper_calibrated)
        init_deg = np.asarray(
            initial_joints_deg if initial_joints_deg is not None else HOME_ANGLES_DEG,
            dtype=np.float64,
        )
        deg_to_steps(init_deg, self._state.Position_in)

        self._planner = TrajectoryPlanner(diagnostic=True)
        self._planner.state.Position_in[:] = self._state.Position_in
        # Mirror the live gate: seeded from an unhomed robot, planned moves
        # are refused until the script homes (home()/teleport() establish
        # references — see _snap_to_angles).
        self._planner.state.Homed_in.fill(1 if initial_homed else 0)

        self._registry = CommandRegistry()
        self._q_rad_buf = np.zeros(6, dtype=np.float64)
        self._rpy_buf = np.zeros(3, dtype=np.float64)
        self._active_tool_key: str = "NONE"
        self._active_variant_key: str = ""
        self._tool_proxy = _DryRunTool(self)

        # The commanded record: one chunk per program command, filled as the
        # planner answers. Rows are recorded at submit time, under the tool
        # and execution speed in force then, so a later change cannot
        # rewrite an earlier command's path.
        self._chunks: list[_Chunk] = []
        self._tool_position = 0.0
        self._plan_cache: TickIndex | None = None

    @property
    def state(self) -> ControllerState:
        """Access the simulated controller state."""
        return self._state

    @property
    def tool(self) -> _DryRunTool:
        """Tool proxy that routes actions through the planner."""
        return self._tool_proxy

    @property
    def program_length(self) -> int:
        """Commands recorded so far — one block each in ``plan()``."""
        return len(self._chunks)

    def tcp_offset(self) -> list[float]:
        """Return current TCP offset in mm."""
        return [
            self._state.tcp_offset_m[0] * 1000.0,
            self._state.tcp_offset_m[1] * 1000.0,
            self._state.tcp_offset_m[2] * 1000.0,
        ]

    def tcp_transform(self) -> list[float]:
        return self.tcp_offset() + [degrees(v) for v in self._state.tcp_rotation_rad]

    # ---- The record ----

    def _open(self, method: str) -> int:
        """Reserve the next program index for *method* with an empty chunk."""
        idx = len(self._chunks)
        empty = np.empty((0, 6), dtype=np.float64)
        self._chunks.append(
            _Chunk(idx, method, empty, empty.copy(), np.empty(0, dtype=np.float64))
        )
        self._plan_cache = None
        return idx

    def _current_q(self) -> np.ndarray:
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        return self._q_rad_buf.copy()

    def _fill(
        self,
        idx: int,
        q_rad: np.ndarray,
        *,
        tcp: np.ndarray | None = None,
        valid: np.ndarray | None = None,
        tool_closed: np.ndarray | None = None,
        error: RobotError | None = None,
    ) -> None:
        """Give chunk *idx* its rows, FK'd under the tool applied now."""
        c = self._chunks[idx]
        c.q_rad = np.ascontiguousarray(q_rad, dtype=np.float64).reshape(-1, 6)
        c.tcp = tcp if tcp is not None else _tcp_from_joints(c.q_rad)
        c.tool_closed = (
            tool_closed
            if tool_closed is not None
            else np.full(c.ticks, self._tool_position, dtype=np.float64)
        )
        c.valid = valid
        c.error = error
        self._plan_cache = None

    def _hold(self, idx: int, ticks: int, *, error: RobotError | None = None) -> None:
        """Chunk *idx* holds the current pose for *ticks* control ticks."""
        q = np.repeat(self._current_q()[np.newaxis], max(0, ticks), axis=0)
        self._fill(idx, q, error=error)

    def _stretched(self, q_rad: np.ndarray) -> np.ndarray:
        """*q_rad* replayed at the execution speed in force: the segment player
        indexes the same waypoints at a scaled rate, so the path is the same
        and only the row count changes."""
        scale = self._state.execution_speed
        if scale == 1.0 or q_rad.shape[0] < 2:
            return q_rad
        ticks = max(1, int(round(q_rad.shape[0] / scale)))
        at = np.clip(
            np.round(np.arange(ticks) * scale).astype(np.intp), 0, q_rad.shape[0] - 1
        )
        return q_rad[at]

    def _tool_target(self, action: str, params: list) -> float:
        if action in ("open", "calibrate", "idle"):
            return 0.0
        if action == "close":
            return 1.0
        if action in ("move", "set_position") and params:
            return float(min(1.0, max(0.0, float(params[0]))))
        return self._tool_position

    def _fill_tool_action(self, idx: int, cmd: ToolActionCmd) -> None:
        """A tool action holds the arm for the tool's estimated travel while
        the jaws ramp to their target."""
        cfg = get_registry().get(cmd.tool_key.strip().upper())
        action = cmd.action.strip().lower()
        params = list(cmd.params)
        seconds = cfg.estimate_duration(action, params) if cfg is not None else 0.0
        ticks = int(round(seconds / INTERVAL_S))
        target = self._tool_target(action, params)
        if action == "calibrate":
            self._state.gripper_calibrated = True
        q = np.repeat(self._current_q()[np.newaxis], ticks, axis=0)
        closed = (
            np.linspace(self._tool_position, target, ticks, dtype=np.float64)
            if ticks
            else np.empty(0, dtype=np.float64)
        )
        self._fill(idx, q, tool_closed=closed)
        self._tool_position = target

    def _absorb(self, segments: list[Segment]) -> None:
        """Record what the planner produced, each segment under its own
        command. A blend chain lands under its head; the folded commands'
        chunks stay at zero ticks."""
        for seg in segments:
            if isinstance(seg, TrajectorySegment):
                self._fill(seg.command_index, self._stretched(seg.trajectory_rad))
            elif isinstance(seg, ErrorSegment):
                if seg.cartesian_path is not None and seg.ik_valid is not None:
                    path = np.asarray(seg.cartesian_path, dtype=np.float64)
                    q = np.repeat(self._current_q()[np.newaxis], path.shape[0], axis=0)
                    self._fill(
                        seg.command_index,
                        q,
                        tcp=path,
                        valid=np.asarray(seg.ik_valid, dtype=np.bool_),
                        error=seg.error,
                    )
                else:
                    self._fill(seg.command_index, np.empty((0, 6)), error=seg.error)
            elif isinstance(seg, InlineSegment) and isinstance(
                seg.params, ToolActionCmd
            ):
                self._fill_tool_action(seg.command_index, seg.params)
            # Any other InlineSegment (select_tool, checkpoint, write_io …)
            # plans nothing: its chunk keeps its place at zero ticks.

    def _assemble(self) -> TickIndex:
        joints_parts: list[np.ndarray] = []
        tcp_parts: list[np.ndarray] = []
        closed_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        blocks: list[TickBlock] = []
        any_valid = any(c.valid is not None for c in self._chunks)
        failed = False
        tick0 = 0
        row0 = 0
        for c in self._chunks:
            rows = 0
            if c.ticks:
                # Keep the control ticks that fall on the row grid, counted
                # from the program's first tick so blocks abut exactly.
                keep = np.arange((-tick0) % _STRIDE, c.ticks, _STRIDE)
                rows = int(keep.shape[0])
                if rows:
                    joints_parts.append(c.q_rad[keep])
                    tcp_parts.append(c.tcp[keep])
                    closed_parts.append(c.tool_closed[keep])
                    if any_valid:
                        valid_parts.append(
                            c.valid[keep]
                            if c.valid is not None
                            else np.ones(rows, dtype=np.bool_)
                        )
            blocks.append(
                TickBlock(
                    command=c.command,
                    start_row=row0,
                    rows=rows,
                    error=c.error,
                    move_type=_MOVE_TYPE.get(c.method),
                )
            )
            failed = failed or c.error is not None
            row0 += rows
            tick0 += c.ticks
        joints = (
            np.concatenate(joints_parts).astype(np.float32)
            if joints_parts
            else np.empty((0, 6), dtype=np.float32)
        )
        tcp = (
            np.concatenate(tcp_parts).astype(np.float32)
            if tcp_parts
            else np.empty((0, 6), dtype=np.float32)
        )
        closed = (
            np.concatenate(closed_parts).astype(np.float32)
            if closed_parts
            else np.empty(0, dtype=np.float32)
        )
        return TickIndex(
            row_dt_s=_ROW_DT_S,
            joints_rad=joints,
            tcp=tcp,
            tool_closed=closed,
            tool_gripping=np.zeros(row0, dtype=np.bool_),
            blocks=tuple(blocks),
            stop="failed" if failed else "completed",
            digest=_digest(joints, tcp),
            valid=np.concatenate(valid_parts) if valid_parts else None,
        )

    def plan(self, max_seconds: float | None = None) -> TickIndex:
        """The commanded record for everything submitted so far."""
        self.flush()
        if self._plan_cache is None:
            self._plan_cache = self._assemble()
        record = self._plan_cache
        return record if max_seconds is None else _truncated(record, max_seconds)

    def simulate(self, max_seconds: float | None = None) -> TickIndex:
        """The predicted record — the plan itself. A planner has no plant to
        drive, so where the controller would send the arm is the honest
        answer to where the arm goes."""
        return self.plan(max_seconds)

    # ---- Dispatch ----

    def flush(self) -> None:
        """Plan any pending blend chain. Call after script completion."""
        if self._planner._blend_buffer:
            self._require_running()
        self._absorb(self._planner.flush())
        self._state.Position_in[:] = self._planner.state.Position_in

    def _snap_to_angles(self, idx: int, angles_deg: list[float]) -> None:
        """Snap to angles instantly (no trajectory) — used by Home and Teleport.

        Both establish position references, so subsequent planned moves pass
        the homed gate. Blended moves still buffered in the planner are
        planned first, under their own commands — the live controller runs
        them before the snap — and the snap itself lands as one row at the
        new pose, so the record shows where the arm is once it is there."""
        self._absorb(self._planner.flush())
        deg = np.asarray(angles_deg, dtype=np.float64)
        deg_to_steps(deg, self._state.Position_in)
        self._planner.state.Position_in[:] = self._state.Position_in
        self._planner.state.Homed_in.fill(1)
        self._hold(idx, _STRIDE)

    def _dispatch(self, params: Any, method: str) -> int:
        """Route a command struct through the trajectory planner, recording
        it as the next program command. Returns its program index."""
        self._state.Homed_in[:] = self._planner.state.Homed_in
        cmd_cls = self._registry.get_command_for_struct(type(params))
        if (
            cmd_cls is not None
            and issubclass(cmd_cls, MotionCommand)
            and not cmd_cls.streamable
        ):
            self._require_running()
        if (
            not self._state.attachments_valid
            and _wire.STRUCT_TO_CMDTYPE.get(type(params)) in ARM_MOTION_CMD_TYPES
        ):
            raise ValueError("attachment context changed; reconcile and reapply")
        idx = self._open(method)
        if isinstance(params, _wire.StopCmd):
            # A stop discards the blends still buffered and lifts a pause,
            # as the controller's does.
            self._planner.cancel()
            self._state.execution_paused = False
            return idx
        if isinstance(params, ToolActionCmd):
            refusal = tool_action_refusal(
                params.tool_key,
                params.action,
                current_tool=self._state.current_tool,
                gripper_calibrated=self._state.gripper_calibrated,
            )
            if refusal is not None:
                self._fill(
                    idx,
                    np.empty((0, 6)),
                    error=make_error(ErrorCode.COMM_VALIDATION_ERROR, detail=refusal),
                )
                return idx
        if isinstance(params, (_wire.EstopCmd, _wire.ResetCmd)):
            self._state.invalidate_attachments()
            self._state.enabled = isinstance(params, _wire.ResetCmd)
            if not self._state.enabled:
                self._planner.cancel()
            return idx
        if isinstance(params, _wire.ResetStateCmd):
            self._planner.cancel()
            self._state.reset()
            self._planner.state.Position_in[:] = self._state.Position_in
            self._planner.state.Homed_in[:] = self._state.Homed_in
            return idx
        if isinstance(params, (_wire.SimulatorCmd, _wire.ConnectHardwareCmd)):
            self._state.invalidate_attachments()
            self._planner.cancel()
            self._state.Homed_in.fill(0)
            self._planner.state.Homed_in.fill(0)
            return idx
        if isinstance(params, SetShapesCmd):
            self._state.set_shapes(params.shapes)
        if isinstance(params, HomeCmd):
            if params.calibrate or not self._planner.state.Homed_in[:6].all():
                self._state.invalidate_attachments()
                self._snap_to_angles(idx, HOME_ANGLES_DEG)
                return idx
            # Already referenced → fall through: the planner fast-paths HOME
            # into a planned return move, so the preview renders the path.
        if isinstance(params, TeleportCmd):
            self._snap_to_angles(idx, params.angles)
            return idx
        if isinstance(params, (SelectToolCmd, SetTcpOffsetCmd, SetTcpTransformCmd)):
            # Resolve pending paths against their original TCP before changing it.
            self.flush()
        if isinstance(params, SelectToolCmd):
            self._active_tool_key = params.tool_name.strip().upper()
            self._active_variant_key = params.variant_key
            self._state.set_tool(self._active_tool_key, params.variant_key)
        if isinstance(params, (SetTcpOffsetCmd, SetTcpTransformCmd)):
            rotation = (
                (radians(params.roll), radians(params.pitch), radians(params.yaw))
                if isinstance(params, SetTcpTransformCmd)
                else (0.0, 0.0, 0.0)
            )
            self._state.set_tcp_transform(
                (params.x / 1000.0, params.y / 1000.0, params.z / 1000.0), rotation
            )
        # Detect jog/servo commands — planner doesn't handle streaming.
        # Other non-trajectory MotionCommands (SelectTool, Home) fall through
        # to the planner which handles them as inline segments.
        if cmd_cls is not None and issubclass(cmd_cls, (JogJCommand, JogLCommand)):
            self.flush()
            cmd = cmd_cls(params)
            assert isinstance(cmd, MotionCommand)
            try:
                path = self._simulate_jog(cmd)
            except TrajectoryPlanningError as refused:
                self._fill(idx, np.empty((0, 6)), error=refused.robot_error)
                return idx
            if path is not None:
                self._fill(idx, path)
            self._planner.state.Position_in[:] = self._state.Position_in
            return idx

        # Everything else → planner
        self._absorb(self._planner.process(params, command_index=idx))
        self._state.Position_in[:] = self._planner.state.Position_in
        return idx

    def _failed(self, idx: int) -> bool:
        return self._chunks[idx].error is not None

    # ---- Jog simulation (planner doesn't handle streaming) ----

    def _simulate_jog(self, cmd: MotionCommand) -> np.ndarray | None:
        """Simulate jog commands by computing linear displacement, one row
        per control tick."""
        # Run do_setup so speeds_out / _axis_index / etc. are computed
        cmd.setup(self._state)

        if isinstance(cmd, JogJCommand):
            return self._simulate_joint_jog(cmd)
        if isinstance(cmd, JogLCommand):
            return self._simulate_cartesian_jog(cmd)
        return None

    def _simulate_joint_jog(self, cmd: JogJCommand) -> np.ndarray:
        """Simulate joint jog by computing linear displacement in joint space."""
        duration = cmd.p.duration
        n_points = max(1, int(round(duration * CONTROL_RATE_HZ)))

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
        return radians

    def _simulate_cartesian_jog(self, cmd: JogLCommand) -> np.ndarray:
        """Simulate a cartesian jog by integrating its TCP twist over the
        duration and solving IK along the way."""
        duration = cmd.p.duration
        n_points = max(1, int(round(duration * CONTROL_RATE_HZ)))

        start_se3 = get_fkine_se3(self._state).copy()
        twist = np.zeros(6, dtype=np.float64)
        jog_twist(cmd.p.velocities, twist)
        wrf = cmd.p.frame == "WRF"

        # Get current joint angles for IK seed
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        last_valid_q = self._q_rad_buf.copy()
        steps_buf = np.zeros_like(self._state.Position_in)

        radians = np.empty((n_points, 6), dtype=np.float64)
        for i in range(n_points):
            t = duration * (i + 1) / n_points
            target_se3 = _twist_pose(start_se3, twist, t, wrf)
            ik_result = solve_ik(
                PAROL6_ROBOT.robot, target_se3, last_valid_q, quiet_logging=True
            )
            if ik_result.success:
                last_valid_q = ik_result.q.copy()

            radians[i] = last_valid_q

        rad_to_steps(last_valid_q, steps_buf)
        self._state.Position_in[:] = steps_buf
        return radians

    # ---- Explicit methods for state reads ----

    def angles(self) -> list[float]:
        steps_to_rad(self._state.Position_in, self._q_rad_buf)
        return np.degrees(self._q_rad_buf).tolist()

    def set_shapes(self, shapes: list) -> int:
        self._dispatch(SetShapesCmd(shapes=shapes), "set_shapes")
        return 1

    def shapes(self):
        """The preview's collision world by layer (mirrors the live query).

        Explicit so a script's readback never falls into the generic command
        dispatch, which has no query path.
        """
        from waldoctl import ShapeWorld

        return ShapeWorld(
            attachment_epoch=self._state.attachment_epoch,
            installation=tuple(PAROL6_ROBOT.installation_shapes()),
            program=tuple(PAROL6_ROBOT.program_shapes()),
        )

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

    # ---- Explicit command methods: the live client's signatures ----

    def home(self, **kwargs: Any) -> int:
        return self._dispatch(build_cmd("home", **kwargs), "home")

    def move_j(
        self,
        angles: list[float] | None = None,
        *,
        pose: list[float] | None = None,
        **kwargs: Any,
    ) -> int:
        if pose is not None:
            return self._dispatch(build_cmd("move_j_pose", pose, **kwargs), "move_j")
        return self._dispatch(build_cmd("move_j", angles or [], **kwargs), "move_j")

    def move_l(self, pose: list[float], **kwargs: Any) -> int:
        return self._dispatch(build_cmd("move_l", pose, **kwargs), "move_l")

    def servo_j(
        self,
        angles: list[float] | None = None,
        *,
        pose: list[float] | None = None,
        **kwargs: Any,
    ) -> int:
        if pose is not None:
            idx = self._dispatch(build_cmd("servo_j_pose", pose, **kwargs), "servo_j")
        else:
            idx = self._dispatch(
                build_cmd("servo_j", angles or [], **kwargs), "servo_j"
            )
        return -1 if self._failed(idx) else 1

    def checkpoint(self, label: str) -> int:
        return self._dispatch(build_cmd("checkpoint", label), "checkpoint")

    def write_io(self, index: int, value: int, *, timeout: float | None = None) -> int:
        if type(index) is not int or index not in (0, 1):
            raise ValueError("Output index must be 0 or 1")
        if type(value) not in (int, bool) or value not in (0, 1):
            raise ValueError("Digital output must be 0 or 1")
        return self._dispatch(
            WriteIOCmd(port_index=index + 2, value=int(value)), "write_io"
        )

    def delay(self, seconds: float) -> int:
        """Hold the pose for *seconds*: rows on the commanded record, so the
        timeline carries the wait as the controller will."""
        self._require_running()
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("delay needs a positive, finite number of seconds")
        # The live planner runs a pending blend chain before a delay; hold
        # the pose the chain ends at, not the one before it.
        self.flush()
        idx = self._open("delay")
        self._hold(idx, int(round(seconds / INTERVAL_S)))
        return idx

    def wait_command(self, command_index: int, timeout: float = 10.0) -> bool:
        """Whether the block for *command_index* planned without error."""
        self.flush()
        return 0 <= command_index < len(self._chunks) and not self._failed(
            command_index
        )

    def wait_motion(self, **kwargs: Any) -> bool:
        self.flush()
        return True

    def _require_running(self) -> None:
        if not self._state.enabled:
            raise ValueError("Controller disabled; reset before previewing motion")
        if self._state.execution_paused:
            raise UnresolvedPreview(
                "Queued execution is paused; preview needs an explicit resume "
                "before it can predict completion"
            )

    def set_execution_speed(self, scale: float, *, timeout: float = 3.0) -> int:
        self._state.execution_speed = validate_execution_scale(scale)
        self._open("set_execution_speed")
        return 1

    def execution_speed(self, *, timeout: float = 3.0) -> ExecutionSpeed:
        scale = self._state.execution_speed
        applied = 0.0 if self._state.execution_paused else scale
        return ExecutionSpeed(applied, applied, scale)

    def pause(self, *, timeout: float = 3.0) -> int:
        self._state.execution_paused = True
        self._open("pause")
        return 1

    def resume(self, *, timeout: float = 3.0) -> int:
        self._state.execution_paused = False
        self._open("resume")
        return 1

    def jog_j(
        self,
        joint: int = -1,
        speed: float = 0.0,
        duration: float = 0.1,
        *,
        joints: list[int] | None = None,
        speeds: list[float] | None = None,
        accel: float = 1.0,
    ) -> int:
        """The live client's signature, so a script's jog previews as written."""
        speed_arr = [0.0] * 6
        if joints is not None and speeds is not None:
            for j, s in zip(joints, speeds):
                speed_arr[j] = s
        elif joint >= 0:
            speed_arr[joint] = speed
        else:
            raise ValueError("jog_j requires either joint= or joints=/speeds=")
        idx = self._dispatch(
            _wire.JogJCmd(speeds=speed_arr, duration=duration, accel=accel), "jog_j"
        )
        return -1 if self._failed(idx) else 1

    def jog_l(
        self,
        frame: str,
        axis: str | None = None,
        speed: float = 0.0,
        duration: float = 0.1,
        *,
        axes: list[str] | None = None,
        speeds_list: list[float] | None = None,
        accel: float = 1.0,
    ) -> int:
        vel = [0.0] * 6
        if axes is not None and speeds_list is not None:
            for a, s in zip(axes, speeds_list):
                vel[_AXIS_INDEX[a]] = s
        elif axis is not None:
            vel[_AXIS_INDEX[axis]] = speed
        else:
            raise ValueError("jog_l requires either axis= or axes=/speeds_list=")
        idx = self._dispatch(
            _wire.JogLCmd(frame=frame, velocities=vel, duration=duration, accel=accel),
            "jog_l",
        )
        return -1 if self._failed(idx) else 1

    # ---- Auto-dispatch for everything else ----

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _CMD_STRUCTS:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        spec = _COMMANDS.get(name)
        if spec is None or spec.kind in (
            CommandKind.QUERY,
            CommandKind.OBSERVATION,
            CommandKind.SYNC,
        ):
            raise AttributeError(
                f"'{type(self).__name__}' previews no '{name}': the query reads "
                "live state the dry run does not keep"
            )

        def method(*args: Any, **kwargs: Any) -> int:
            idx = self._dispatch(build_cmd(name, *args, **kwargs), name)
            # Queued work answers with its program index; everything else
            # answers as the live client does: 1 when it applied, -1 when
            # the planner refused it.
            return idx if spec.mints_index else (-1 if self._failed(idx) else 1)

        return method
