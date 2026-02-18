"""Motion pipeline: async trajectory planning in a worker process.

The MotionPlanner offloads trajectory computation (TOPPRA, IK chains) from
the 100Hz control loop to a separate process.  Commands flow in via
``command_queue`` and computed segments flow back via ``segment_queue``.

Non-trajectory motion commands (Home, SetTool, Gripper, Checkpoint, Delay)
are forwarded as ``InlineSegment`` tokens so that the SegmentPlayer can
execute them in the control loop while preserving command ordering.

TrajectoryPlanner holds the shared planning logic used by both the real-time
PlannerWorker subprocess and the DryRunRobotClient (diagnostic mode).
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import signal
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union, cast

import numpy as np

from parol6.protocol.wire import HomeCmd, SetToolCmd
from parol6.utils.error_catalog import RobotError, extract_robot_error
from parol6.utils.error_codes import ErrorCode

from parol6.server.state import ControllerState

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType
    from parol6.commands.base import TrajectoryMoveCommandBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segment types (planner → player via segment_queue)
# ---------------------------------------------------------------------------


@dataclass
class TrajectorySegment:
    """Pre-computed trajectory waypoints ready for 100Hz playback."""

    command_index: int
    trajectory_steps: np.ndarray  # (M, 6) int32
    duration: float
    blend_consumed_indices: list[int] = field(default_factory=list)


@dataclass
class InlineSegment:
    """Forwarded non-trajectory command for execution in the control loop."""

    command_index: int
    params: object  # wire struct (msgspec.Struct — picklable)


@dataclass
class ErrorSegment:
    """Planning failure — surfaces error through the pipeline."""

    command_index: int
    error: RobotError
    cartesian_path: np.ndarray | None = None  # (N, 6) full TCP path
    ik_valid: np.ndarray | None = None  # (N,) per-pose bool


Segment = Union[TrajectorySegment, InlineSegment, ErrorSegment]

# ---------------------------------------------------------------------------
# Message types (main → planner via command_queue)
# ---------------------------------------------------------------------------


@dataclass
class PlanCommand:
    """Submit a motion command for planning or forwarding."""

    command_index: int
    params: object  # wire struct (MoveJCmd, SetToolCmd, HomeCmd, …)
    position_in: np.ndarray | None = None  # carries Position_in when sync needed


@dataclass
class SyncPosition:
    """Update the planner's internal position tracking."""

    position_in: np.ndarray


@dataclass
class SyncProfile:
    """Update the planner's motion profile setting."""

    profile: str


@dataclass
class SyncTool:
    """Update the planner's tool state (e.g. after E-stop cancel)."""

    tool_name: str


@dataclass
class CancelAll:
    """Clear the planner's internal state and discard pending work."""


PlannerMessage = Union[PlanCommand, SyncPosition, SyncProfile, SyncTool, CancelAll]


# ---------------------------------------------------------------------------
# Lightweight state for planner subprocess
# ---------------------------------------------------------------------------


@dataclass
class PlannerState:
    """Minimal state for trajectory computation.

    Carries the fields that trajectory ``do_setup()`` reads: joint position,
    motion profile, and FK cache.  Tool state is tracked as a string for
    change-detection; the actual tool transform lives on ``PAROL6_ROBOT.robot``.
    """

    Position_in: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    motion_profile: str = "TOPPRA"
    current_tool: str = "NONE"
    stop_on_failure: bool = True

    # Forward kinematics cache (same layout as ControllerState — needed by
    # get_fkine_se3/ensure_fkine_updated called from cartesian do_setup)
    _fkine_last_pos_in: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.int32)
    )
    _fkine_last_tool: str = ""
    _fkine_mat: np.ndarray = field(
        default_factory=lambda: np.asfortranarray(np.eye(4, dtype=np.float64))
    )
    _fkine_flat_mm: np.ndarray = field(
        default_factory=lambda: np.zeros(16, dtype=np.float64)
    )
    _fkine_q_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# TrajectoryPlanner — shared planning logic
# ---------------------------------------------------------------------------


class TrajectoryPlanner:
    """Core trajectory planning logic shared by PlannerWorker and DryRunRobotClient.

    Dispatches commands to trajectory or inline handlers, manages blend buffering,
    and emits ErrorSegment on failure instead of raising.

    Args:
        diagnostic: If True, sets stop_on_failure=False on PlannerState so that
            batch_ik solves all poses (needed for per-pose red/green visualization).
    """

    def __init__(self, diagnostic: bool = False) -> None:
        import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: N811
        from parol6.commands.base import TrajectoryMoveCommandBase
        from parol6.config import MAX_BLEND_LOOKAHEAD, deg_to_steps
        from parol6.server.command_registry import CommandRegistry

        self.state = PlannerState()
        if diagnostic:
            self.state.stop_on_failure = False
        self._diagnostic = diagnostic
        self._registry = CommandRegistry()
        self._trajectory_base: type[TrajectoryMoveCommandBase] = (
            TrajectoryMoveCommandBase
        )
        self._max_blend_lookahead = MAX_BLEND_LOOKAHEAD
        self._robot_module = PAROL6_ROBOT
        self._blend_buffer: list[tuple[int, TrajectoryMoveCommandBase]] = []
        self._output: list[Segment] = []

        # Pre-compute home position in steps
        self._home_steps = np.zeros(6, dtype=np.int32)
        _home_deg = np.array(PAROL6_ROBOT.joint.standby_deg, dtype=np.float64)
        deg_to_steps(_home_deg, self._home_steps)

    def process(self, params: object, command_index: int = 0) -> list[Segment]:
        """Plan a single command. Returns list of resulting segments."""
        self._output.clear()

        cmd_class = self._registry.get_command_for_struct(type(params))
        if cmd_class is not None and issubclass(cmd_class, self._trajectory_base):
            self._handle_trajectory(command_index, params, cmd_class)
        else:
            if self._blend_buffer:
                self._flush_blend()
            self._handle_inline(command_index, params)

        return list(self._output)

    def flush(self) -> list[Segment]:
        """Flush any pending blend buffer. Returns resulting segments."""
        self._output.clear()
        if self._blend_buffer:
            self._flush_blend()
        return list(self._output)

    def cancel(self) -> None:
        """Clear blend buffer."""
        self._blend_buffer.clear()

    def sync_tool(self, tool_name: str) -> None:
        """Sync tool state (e.g. after E-stop cancel)."""
        self.state.current_tool = tool_name
        self._robot_module.apply_tool(tool_name)

    # -- trajectory handling --

    def _handle_trajectory(
        self,
        command_index: int,
        params: object,
        cmd_class: type[TrajectoryMoveCommandBase],
    ) -> None:
        """Buffer for blending or compute trajectory immediately."""
        cmd = cmd_class(params)

        if cmd.blend_radius > 0:
            self._blend_buffer.append((command_index, cmd))
            if len(self._blend_buffer) > self._max_blend_lookahead:
                self._flush_blend()
            return

        if self._blend_buffer:
            self._blend_buffer.append((command_index, cmd))
            self._flush_blend()
            return

        # Single non-blended command
        state = cast(ControllerState, self.state)
        try:
            cmd.do_setup(state)
        except Exception as e:
            self._emit_error(command_index, cmd, e)
            return
        self._emit_trajectory(command_index, cmd)

    def _flush_blend(self) -> None:
        """Flush the blend buffer — either blend or single-command setup."""
        buf = self._blend_buffer
        if not buf:
            return

        state = cast(ControllerState, self.state)
        head_idx, head_cmd = buf[0]

        if len(buf) == 1:
            try:
                head_cmd.do_setup(state)
            except Exception as e:
                buf.clear()
                self._emit_error(head_idx, head_cmd, e)
                return
            self._emit_trajectory(head_idx, head_cmd)
        else:
            rest_cmds = [cmd for _, cmd in buf[1:]]
            try:
                consumed = head_cmd.do_setup_with_blend(state, rest_cmds)
            except Exception as e:
                buf.clear()
                self._emit_error(head_idx, head_cmd, e)
                return

            if consumed < len(rest_cmds):
                logger.warning(
                    "Blend zone degraded: requested %d segments, achieved %d",
                    len(rest_cmds),
                    consumed,
                )

            consumed_indices = [idx for idx, _ in buf[1 : 1 + consumed]]

            self._output.append(
                TrajectorySegment(
                    command_index=head_idx,
                    trajectory_steps=head_cmd.trajectory_steps.copy(),
                    duration=head_cmd._duration,
                    blend_consumed_indices=consumed_indices,
                )
            )
            self.state.Position_in[:] = head_cmd.trajectory_steps[-1]

            # Unconsumed tail commands: compute individually
            for i in range(1 + consumed, len(buf)):
                uc_idx, uc_cmd = buf[i]
                try:
                    uc_cmd.do_setup(state)
                except Exception as e:
                    buf.clear()
                    self._emit_error(uc_idx, uc_cmd, e)
                    return
                self._emit_trajectory(uc_idx, uc_cmd)

        buf.clear()

    def _emit_trajectory(
        self, command_index: int, cmd: TrajectoryMoveCommandBase
    ) -> None:
        """Append a TrajectorySegment to output and advance position."""
        self._output.append(
            TrajectorySegment(
                command_index=command_index,
                trajectory_steps=cmd.trajectory_steps.copy(),
                duration=cmd._duration,
            )
        )
        self.state.Position_in[:] = cmd.trajectory_steps[-1]

    def _emit_error(
        self, command_index: int, cmd: TrajectoryMoveCommandBase, exc: Exception
    ) -> None:
        """Append an ErrorSegment to output, with diagnostic data if available."""
        cartesian_path = None
        ik_valid = None
        if self._diagnostic:
            diag = getattr(cmd, "cartesian_diagnostic", None)
            if diag is not None:
                cartesian_path = diag.get("tcp_poses")
                ik_valid = diag.get("ik_valid")

        robot_error = extract_robot_error(
            exc, ErrorCode.MOTN_SETUP_FAILED, command_index, detail=str(exc)
        )
        self._output.append(
            ErrorSegment(
                command_index=command_index,
                error=robot_error,
                cartesian_path=cartesian_path,
                ik_valid=ik_valid,
            )
        )

    # -- inline command handling --

    def _handle_inline(self, command_index: int, params: object) -> None:
        """Emit an InlineSegment and predict state changes."""
        self._output.append(
            InlineSegment(
                command_index=command_index,
                params=params,
            )
        )

        # Predict state for subsequent trajectory planning
        if isinstance(params, SetToolCmd):
            self.state.current_tool = params.tool_name
            self._robot_module.apply_tool(params.tool_name)
        elif isinstance(params, HomeCmd):
            self.state.Position_in[:] = self._home_steps


# ---------------------------------------------------------------------------
# PlannerWorker — thin subprocess wrapper around TrajectoryPlanner
# ---------------------------------------------------------------------------


class PlannerWorker:
    """Wraps TrajectoryPlanner for use inside the planner subprocess.

    Receives PlanCommand messages, delegates to TrajectoryPlanner, and puts
    resulting segments on the segment queue.
    """

    def __init__(self, segment_queue: multiprocessing.Queue) -> None:
        self._segment_queue = segment_queue
        self._planner = TrajectoryPlanner(diagnostic=False)

    @property
    def state(self) -> PlannerState:
        return self._planner.state

    def process_command(self, msg: PlanCommand) -> None:
        """Route a PlanCommand through the planner and emit segments."""
        if msg.position_in is not None:
            self._planner.state.Position_in[:] = msg.position_in

        segments = self._planner.process(msg.params, msg.command_index)
        for seg in segments:
            self._segment_queue.put(seg)

    def flush_stale_blend(self) -> None:
        """Flush any pending blend buffer (called on queue timeout)."""
        segments = self._planner.flush()
        for seg in segments:
            self._segment_queue.put(seg)

    def cancel(self) -> None:
        """Clear blend buffer on CancelAll."""
        self._planner.cancel()

    def apply_tool(self, tool_name: str) -> None:
        """Sync tool state (e.g. after E-stop)."""
        self._planner.sync_tool(tool_name)


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def motion_planner_main(
    command_queue: multiprocessing.Queue,
    segment_queue: multiprocessing.Queue,
    shutdown_event: EventType,
) -> None:
    """Worker process main loop — compute trajectories and forward inline commands."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    worker = PlannerWorker(segment_queue)

    logger.info(
        "Motion planner subprocess started (PID %d)",
        multiprocessing.current_process().pid,
    )

    try:
        while not shutdown_event.is_set():
            try:
                msg = command_queue.get(timeout=0.1)
            except queue.Empty:
                worker.flush_stale_blend()
                continue

            if isinstance(msg, CancelAll):
                worker.cancel()
                continue

            if isinstance(msg, SyncPosition):
                worker.state.Position_in[:] = msg.position_in
                continue

            if isinstance(msg, SyncProfile):
                worker.state.motion_profile = msg.profile
                continue

            if isinstance(msg, SyncTool):
                worker.apply_tool(msg.tool_name)
                continue

            if isinstance(msg, PlanCommand):
                try:
                    worker.process_command(msg)
                except Exception:
                    logger.exception(
                        "Planner failed on command index=%d (%s)",
                        msg.command_index,
                        type(msg.params).__name__,
                    )
                    worker.cancel()
                    _drain_queue(command_queue)

    except Exception:
        logger.exception("Motion planner subprocess error")
    finally:
        logger.info("Motion planner subprocess exiting")


# ---------------------------------------------------------------------------
# MotionPlanner — main-process handle for the worker
# ---------------------------------------------------------------------------


class MotionPlanner:
    """Manages the trajectory planner subprocess.

    Provides a non-blocking interface for the controller to submit commands
    and poll for computed segments.
    """

    def __init__(self) -> None:
        self._command_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._segment_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._shutdown_event: EventType = multiprocessing.Event()
        self._process: multiprocessing.Process | None = None

    # -- lifecycle --

    def start(self) -> None:
        """Start the planner subprocess."""
        if self._process is not None and self._process.is_alive():
            return
        self._shutdown_event.clear()
        self._process = multiprocessing.Process(
            target=motion_planner_main,
            args=(self._command_queue, self._segment_queue, self._shutdown_event),
            daemon=True,
            name="MotionPlannerProcess",
        )
        self._process.start()
        logger.info("Motion planner started, PID: %s", self._process.pid)

    def stop(self) -> None:
        """Shut down the planner subprocess gracefully."""
        self._shutdown_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                logger.warning("Motion planner did not exit cleanly, terminating")
                self._process.terminate()
                self._process.join(timeout=1.0)
        # Drain queues to avoid BrokenPipeError on GC
        _drain_queue(self._command_queue)
        _drain_queue(self._segment_queue)
        logger.info("Motion planner stopped")

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # -- main → planner --

    def submit(self, msg: PlannerMessage) -> None:
        """Send a message to the planner (non-blocking)."""
        self._command_queue.put_nowait(msg)

    def sync_position(self, position_in: np.ndarray) -> None:
        """Update the planner's position tracking."""
        self.submit(SyncPosition(position_in=position_in))

    def sync_profile(self, profile: str) -> None:
        """Update the planner's motion profile."""
        self.submit(SyncProfile(profile=profile))

    def sync_tool(self, tool_name: str) -> None:
        """Update the planner's tool state."""
        self.submit(SyncTool(tool_name=tool_name))

    def cancel(self) -> None:
        """Cancel all pending work in the planner."""
        self.submit(CancelAll())

    # -- planner → main --

    def poll_segment(self) -> Segment | None:
        """Non-blocking poll for a computed segment. Returns None if empty."""
        try:
            return self._segment_queue.get_nowait()
        except queue.Empty:
            return None


def _drain_queue(q: multiprocessing.Queue) -> None:
    """Drain a queue, discarding all items."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
