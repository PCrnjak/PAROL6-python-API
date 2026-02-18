"""Segment player — consumes planned segments in the 100Hz control loop.

The SegmentPlayer is the execution-side counterpart of MotionPlanner.
It receives ``TrajectorySegment`` and ``InlineSegment`` objects from the
planner's output queue and executes them in order:

- **TrajectorySegment**: index into pre-computed waypoints at 100Hz
  (zero-allocation hot path, identical to the old execute_step()).
- **InlineSegment**: create the command object from its wire params and
  tick it in the control loop until completion (Home, Gripper, etc.).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from parol6.commands.base import CommandBase, ExecutionStatusCode
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import create_command_from_struct
from parol6.server.motion_planner import (
    ErrorSegment,
    InlineSegment,
    MotionPlanner,
    Segment,
    TrajectorySegment,
)
from parol6.utils.error_catalog import RobotError, make_error
from parol6.utils.error_codes import ErrorCode

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class SegmentPlayer:
    """Consumes segments from the planner and executes them at 100Hz.

    Handles both trajectory playback (zero-alloc waypoint indexing) and
    inline command execution (setup/tick lifecycle) while maintaining
    strict ordering.
    """

    __slots__ = (
        "_planner",
        "_active",
        "_step",
        "_buffer",
        "_inline_cmd",
        "_inline_activated",
    )

    def __init__(self, planner: MotionPlanner) -> None:
        self._planner = planner
        self._active: Segment | None = None
        self._step: int = 0
        self._buffer: deque[Segment] = deque()
        self._inline_cmd: CommandBase | None = None
        self._inline_activated: bool = False

    def tick(self, state: ControllerState) -> bool:
        """Execute one tick. Returns True if actively playing/executing.

        Called from the 100Hz control loop. For trajectory segments this is
        a zero-allocation hot path (array index + copy).
        """
        # Drain planner's output queue into local buffer (non-blocking)
        seg = self._planner.poll_segment()
        while seg is not None:
            self._buffer.append(seg)
            state.queued_segments += 1
            if isinstance(seg, TrajectorySegment):
                state.queued_duration += seg.duration
            seg = self._planner.poll_segment()

        # Process active segment or activate next
        max_immediate = 8  # prevent infinite recursion on back-to-back instant commands
        for _ in range(max_immediate):
            # Activate next segment if idle
            if self._active is None:
                if not self._buffer:
                    return False
                self._activate_next(state)

            active = self._active

            # --- Trajectory segment: index into waypoints ---
            if isinstance(active, TrajectorySegment):
                if self._step < len(active.trajectory_steps):
                    state.Position_out[:] = active.trajectory_steps[self._step]
                    state.Command_out = CommandCode.MOVE
                    self._step += 1
                    return True
                # Segment complete — try next immediately
                self._complete_segment(active, state)
                continue

            # --- Inline segment: tick the command ---
            if isinstance(active, InlineSegment):
                result = self._tick_inline(active, state)
                if result is None:
                    # Instant completion — try next immediately
                    continue
                return result

            # --- Error segment: halt advance run ---
            if isinstance(active, ErrorSegment):
                logger.error(
                    "Command %d failed: %s", active.command_index, active.error
                )
                state.error = active.error
                state.action_state = "ERROR"
                state.action_current = ""
                self._active = None
                # Halt: cancel all remaining planned work
                self._buffer.clear()
                self._planner.cancel()
                self._drain_planner_queue(state)
                return False

            # Unknown segment type
            logger.error("Unknown segment type: %s", type(active).__name__)
            self._active = None
            continue

        # Exhausted immediate iterations (unlikely)
        return self._active is not None

    def _activate_next(self, state: ControllerState) -> None:
        """Promote next buffered segment to active."""
        self._active = self._buffer.popleft()
        self._step = 0
        self._inline_cmd = None
        self._inline_activated = False
        state.executing_command_index = self._active.command_index
        state.action_state = "EXECUTING"

    def _tick_inline(self, seg: InlineSegment, state: ControllerState) -> bool | None:
        """Tick an inline command. Returns True (executing), False (failed), or None (completed)."""
        if self._inline_cmd is None:
            cmd, _, error_msg = create_command_from_struct(seg.params)
            if cmd is None:
                logger.error("Failed to create inline command: %s", error_msg)
                error = make_error(
                    ErrorCode.COMM_DECODE_ERROR,
                    seg.command_index,
                    detail=error_msg or "unknown command",
                )
                self._on_failure(seg, error, state)
                return False

            self._inline_cmd = cmd

        cmd = self._inline_cmd
        if not self._inline_activated:
            cmd.setup(state)
            state.action_current = type(cmd).__name__
            self._inline_activated = True

        code = cmd.tick(state)

        if code == ExecutionStatusCode.COMPLETED:
            self._complete_segment(seg, state)
            return None  # signal caller to try next immediately

        if code == ExecutionStatusCode.FAILED:
            logger.error(
                "Inline command failed: %s - %s",
                type(cmd).__name__,
                cmd.robot_error,
            )
            error = cmd.robot_error or make_error(
                ErrorCode.MOTN_TICK_FAILED, seg.command_index, detail=type(cmd).__name__
            )
            self._on_failure(seg, error, state)
            return False

        return True  # EXECUTING — continue next tick

    def _complete_segment(self, seg: Segment, state: ControllerState) -> None:
        """Mark segment as completed and update tracking indices."""
        final_idx = seg.command_index
        if isinstance(seg, TrajectorySegment):
            for idx in seg.blend_consumed_indices:
                if idx > final_idx:
                    final_idx = idx
            state.queued_duration -= seg.duration
        state.queued_segments -= 1
        state.completed_command_index = final_idx
        state.action_current = ""
        state.action_state = "IDLE"
        self._active = None

    def _on_failure(
        self, seg: Segment, error: RobotError, state: ControllerState
    ) -> None:
        """Handle inline command failure: set error state, clear buffer, cancel planner."""
        state.error = error
        state.action_current = ""
        state.action_state = "ERROR"
        self._active = None
        self._buffer.clear()
        self._planner.cancel()
        self._drain_planner_queue(state)

    def cancel(self, state: ControllerState) -> None:
        """Clear buffer, drain stale segments, and stop playback."""
        self._active = None
        self._step = 0
        self._inline_cmd = None
        self._inline_activated = False
        self._buffer.clear()
        self._planner.cancel()
        # Drain stale segments from planner output queue
        self._drain_planner_queue(state)

    def _drain_planner_queue(self, state: ControllerState) -> None:
        """Drain any remaining segments from the planner's output queue."""
        while self._planner.poll_segment() is not None:
            pass
        state.queued_segments = 0
        state.queued_duration = 0.0

    @property
    def active(self) -> bool:
        """True if playing a segment or has buffered segments."""
        return self._active is not None or len(self._buffer) > 0
