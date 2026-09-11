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

import numpy as np

from parol6.commands._collision_guard import guard_joint_path
from parol6.commands.base import CommandBase, ExecutionStatusCode
from parol6.config import (
    COLLISION_PATH_SAMPLES,
    EXECUTION_OVERRIDE_TRANSITION_S,
    INTERVAL_S,
    LIMITS,
    SETTLE_MAX_TICKS,
    rad_to_steps,
    steps_to_rad,
)
from parol6.protocol.wire import CommandCode, DelayCmd
from parol6.server.command_executor import _format_cmd_params
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
from parol6.utils.errors import TrajectoryPlanningError
from waldoctl import ActionState

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
        "_phase",
        "_position_rad",
        "_candidate_position_rad",
        "_velocity_rad_s",
        "_buffer",
        "_inline_cmd",
        "_inline_activated",
        "_settling",
        "_settle_ticks",
        "_settle_err",
        "_last_shapes_version",
    )

    def __init__(self, planner: MotionPlanner) -> None:
        self._planner = planner
        self._active: Segment | None = None
        self._step: int = 0
        self._phase = -1.0
        self._position_rad = np.zeros(6, dtype=np.float64)
        self._candidate_position_rad = np.zeros(6, dtype=np.float64)
        self._velocity_rad_s = np.zeros(6, dtype=np.float64)
        rad_to_steps(self._position_rad, np.empty(6, dtype=np.int32))
        self._buffer: deque[Segment] = deque()
        self._inline_cmd: CommandBase | None = None
        self._inline_activated: bool = False
        self._settling: bool = False
        self._settle_ticks: int = 0
        self._settle_err: int = -1
        self._last_shapes_version: int = 0

    @property
    def active(self) -> bool:
        """True if playing a segment or has buffered segments."""
        return self._active is not None or bool(self._buffer)

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

        # MoveIt-style invalidation: a world change (SET_SHAPES bumps
        # shapes_version) re-guards the streaming trajectory's remaining
        # waypoints. The unchanged path pays only this int compare.
        if state.shapes_version != self._last_shapes_version:
            self._last_shapes_version = state.shapes_version
            if isinstance(self._active, TrajectorySegment):
                start = self._step - 1 if self._step > 0 else 0
                if not self._world_guard(
                    self._active, self._active.trajectory_steps[start:], state
                ):
                    return False

        # Process active segment or activate next
        max_immediate = 8  # prevent infinite recursion on back-to-back instant commands
        for _ in range(max_immediate):
            # Activate next segment if idle
            if self._active is None:
                if not self._buffer:
                    state.execution_applied_speed = (
                        0.0 if state.execution_paused else state.execution_speed
                    )
                    return False
                if state.execution_paused and not isinstance(
                    self._buffer[0], ErrorSegment
                ):
                    state.execution_applied_speed = 0.0
                    state.Speed_out.fill(0)
                    return True
                self._activate_next(state)
                if self._active is None:
                    continue  # activation-time world guard rejected the segment

            active = self._active

            # --- Trajectory segment: index into waypoints ---
            if isinstance(active, TrajectorySegment):
                if state.execution_paused and (
                    state.execution_applied_speed == 0.0
                    or self._step >= len(active.trajectory_steps)
                ):
                    state.execution_applied_speed = 0.0
                    self._velocity_rad_s.fill(0.0)
                    state.Command_out = CommandCode.MOVE
                    state.Speed_out.fill(0)
                    return True
                if self._step < len(active.trajectory_steps):
                    old_scale = state.execution_applied_speed
                    target_scale = (
                        0.0 if state.execution_paused else state.execution_speed
                    )
                    # Steady state -- every tick of a playback nobody has
                    # overridden -- holds the scale, so the acceleration-window
                    # search and the admissibility bisection below are skipped:
                    # they exist to bound a change, and compute a rate that is
                    # forced to zero when there is none.
                    new_scale = old_scale
                    if target_scale != old_scale:
                        low = -1.0 / EXECUTION_OVERRIDE_TRANSITION_S
                        high = -low
                        for joint in range(6):
                            velocity = active.velocity_rad_s[self._step, joint]
                            if abs(velocity) > 1e-12:
                                base = (
                                    old_scale
                                    * old_scale
                                    * active.acceleration_rad_s2[self._step, joint]
                                )
                                limit = LIMITS.joint.hard.acceleration[joint]
                                first = (-limit - base) / velocity
                                second = (limit - base) / velocity
                                low = max(low, min(first, second))
                                high = min(high, max(first, second))
                        requested = (target_scale - old_scale) / INTERVAL_S
                        rate = min(high, max(low, requested)) if low <= high else 0.0
                        if rate * requested < 0.0:
                            rate = 0.0
                        new_scale = min(1.0, max(0.0, old_scale + rate * INTERVAL_S))
                        if abs(new_scale - target_scale) < 1e-12:
                            new_scale = target_scale
                    if new_scale != old_scale and not self._rate_is_admissible(
                        active, old_scale, new_scale
                    ):
                        allowed = 0.0
                        refused = 1.0
                        for _ in range(16):
                            fraction = 0.5 * (allowed + refused)
                            if self._rate_is_admissible(
                                active,
                                old_scale,
                                old_scale + fraction * (new_scale - old_scale),
                            ):
                                allowed = fraction
                            else:
                                refused = fraction
                        new_scale = old_scale + allowed * (new_scale - old_scale)
                    state.execution_applied_speed = new_scale
                    self._candidate_position(
                        active, self._phase + 0.5 * (old_scale + new_scale)
                    )
                    for joint in range(6):
                        self._velocity_rad_s[joint] = (
                            self._candidate_position_rad[joint]
                            - self._position_rad[joint]
                        ) / INTERVAL_S
                    self._position_rad[:] = self._candidate_position_rad
                    self._phase += 0.5 * (old_scale + new_scale)
                    if self._phase >= 1.0:
                        self._step += 1
                        self._phase -= 1.0
                    if self._phase <= 0.0 or self._step + 1 == len(
                        active.trajectory_steps
                    ):
                        state.Position_out[:] = active.trajectory_steps[self._step]
                    else:
                        # Preserve the planner's piecewise-linear joint path;
                        # rounding happens only at the firmware boundary.
                        rad_to_steps(self._position_rad, state.Position_out)
                    state.Command_out = CommandCode.MOVE
                    if (
                        self._step + 1 == len(active.trajectory_steps)
                        and self._phase >= 0.0
                    ):
                        self._step += 1
                    self._settling = False
                    return True
                # All waypoints sent — hold MOVE at target until Position_in
                # converges. The tick cap gates on stall, not elapsed time:
                # while the firmware is still closing on the target (e.g. it
                # fell behind the waypoint stream under CPU starvation) the
                # segment stays active, so completion is never reported with
                # the robot still in motion.
                target = active.trajectory_steps[-1]
                if not self._settling:
                    self._settling = True
                    self._settle_ticks = 0
                    self._settle_err = -1
                err = 0
                for i in range(6):
                    d = int(state.Position_in[i]) - int(target[i])
                    if d < 0:
                        d = -d
                    if d > err:
                        err = d
                if self._settle_err < 0 or err < self._settle_err:
                    self._settle_err = err
                    self._settle_ticks = 0
                self._settle_ticks += 1
                if err == 0 or self._settle_ticks > SETTLE_MAX_TICKS:
                    if err != 0:
                        logger.warning(
                            "Segment completed %d steps short of target "
                            "(no settle progress for %d ticks)",
                            err,
                            SETTLE_MAX_TICKS,
                        )
                    self._settling = False
                    self._complete_segment(active, state)
                    continue
                # Keep commanding the target so firmware continues tracking
                state.Position_out[:] = target
                state.Command_out = CommandCode.MOVE
                return True

            # --- Inline segment: tick the command ---
            if isinstance(active, InlineSegment):
                state.execution_applied_speed = (
                    0.0 if state.execution_paused else state.execution_speed
                )
                if state.execution_paused and isinstance(active.params, DelayCmd):
                    state.Speed_out.fill(0)
                    return True
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
                pairs = active.colliding_pairs
                state.collision_active = bool(pairs)
                state.collision_pairs = tuple(pairs) if pairs else ()
                state.action_state = ActionState.ERROR
                state.action_current = ""
                state.action_params = ""
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

    def _candidate_position(self, segment: TrajectorySegment, phase: float) -> None:
        step = self._step
        if phase >= 1.0:
            step += 1
            phase -= 1.0
        step = min(step, len(segment.trajectory_rad) - 1)
        following = min(step + 1, len(segment.trajectory_rad) - 1)
        phase = max(0.0, phase)
        for joint in range(6):
            left = segment.trajectory_rad[step, joint]
            right = segment.trajectory_rad[following, joint]
            self._candidate_position_rad[joint] = left + phase * (right - left)

    def _rate_is_admissible(
        self, segment: TrajectorySegment, old_scale: float, scale: float
    ) -> bool:
        # Central planner derivatives do not bound the second difference
        # of a fractional, piecewise-linear position command.
        self._candidate_position(segment, self._phase + 0.5 * (old_scale + scale))
        for joint in range(6):
            velocity = (
                self._candidate_position_rad[joint] - self._position_rad[joint]
            ) / INTERVAL_S
            acceleration = (velocity - self._velocity_rad_s[joint]) / INTERVAL_S
            if abs(acceleration) > LIMITS.joint.hard.acceleration[joint] * (1.0 + 1e-6):
                return False
        return True

    def _activate_next(self, state: ControllerState) -> None:
        """Promote next buffered segment to active.

        Trajectory segments are re-guarded against the *current* world first:
        their plan-time guard may predate a world change — the planner FIFO
        orders plans and shape syncs by submission, so a segment planned
        against the old world can arrive here after SET_SHAPES applied.
        """
        seg = self._buffer.popleft()
        if isinstance(seg, TrajectorySegment) and not self._world_guard(
            seg, seg.trajectory_steps, state
        ):
            return
        self._active = seg
        self._step = 0
        self._phase = -1.0
        self._inline_cmd = None
        self._inline_activated = False
        state.executing_command_index = self._active.command_index
        state.action_state = ActionState.EXECUTING
        # Populate action info for trajectory segments (inline segments set these later)
        if isinstance(self._active, TrajectorySegment):
            self._position_rad[:] = self._active.trajectory_rad[0]
            self._velocity_rad_s.fill(0.0)
            state.action_current = self._active.command_name
            state.action_params = self._active.action_params

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
            state.action_params = _format_cmd_params(seg.params)
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
        state.action_params = ""
        state.action_state = ActionState.IDLE
        self._active = None

    def _on_failure(
        self, seg: Segment, error: RobotError, state: ControllerState
    ) -> None:
        """Handle inline command failure: set error state, clear buffer, cancel planner."""
        state.error = error
        state.action_current = ""
        state.action_params = ""
        state.action_state = ActionState.ERROR
        self._active = None
        self._buffer.clear()
        self._planner.cancel()
        self._drain_planner_queue(state)

    def _world_guard(
        self, seg: Segment, steps: np.ndarray, state: ControllerState
    ) -> bool:
        """Validate trajectory waypoints (motor steps) against the current
        collision world; on violation, halt playback like an ErrorSegment.

        Runs only at segment activation and on a world change, never per-tick;
        rows are subsampled *before* the steps→rad conversion so the cost stays
        ~COLLISION_PATH_SAMPLES checker calls regardless of trajectory length.
        ``guard_joint_path`` keeps the escape semantics — a keep-out dropped
        onto the arm still permits the escaping remainder. Inline segments
        (Home/Gripper) are not trajectory playback and are not re-guarded.
        """
        n = len(steps)
        if n == 0:
            return True
        target = max(2, COLLISION_PATH_SAMPLES + 2)
        if n > target:
            idx = np.unique(np.linspace(0, n - 1, target).round().astype(int))
            steps = steps[idx]
        q = np.empty((len(steps), 6), dtype=np.float64)
        for i in range(len(steps)):
            steps_to_rad(steps[i], q[i])
        try:
            guard_joint_path(q)
        except TrajectoryPlanningError as exc:
            logger.error(
                "Command %d invalidated by world change: %s",
                seg.command_index,
                exc.robot_error,
            )
            state.error = exc.robot_error
            pairs = exc.colliding_pairs
            state.collision_active = bool(pairs)
            state.collision_pairs = tuple(pairs) if pairs else ()
            state.action_state = ActionState.ERROR
            state.action_current = ""
            state.action_params = ""
            self._active = None
            self._buffer.clear()
            self._planner.cancel()
            self._drain_planner_queue(state)
            return False
        return True

    def cancel(self, state: ControllerState) -> None:
        """Clear buffer, drain stale segments, and stop playback."""
        if self._active is not None:
            # Planned trajectories live here rather than in CommandExecutor.
            # Cancelling its command cannot clear this player's activity.
            state.action_current = ""
            state.action_params = ""
            state.action_state = ActionState.IDLE
        self._active = None
        self._step = 0
        self._phase = -1.0
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
