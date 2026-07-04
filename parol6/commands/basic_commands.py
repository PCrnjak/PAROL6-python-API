"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and SelectTool.
"""

import logging
from enum import Enum, auto
import numpy as np

from parol6.config import (
    COLLISION_JOG_LOOKAHEAD_S,
    JOG_MIN_STEPS,
    LIMITS,
    rad_to_steps,
    speed_steps_to_rad_scalar,
    steps_to_rad,
)
from parol6.protocol.wire import (
    CmdType,
    HomeCmd,
    JogJCmd,
    SelectToolCmd,
    TeleportCmd,
)
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState
from parol6.commands._collision_guard import collision_blocked
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.config import deg_to_steps
from parol6.server.transports.transport_factory import is_simulation_mode

import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: N811

from .base import (
    ExecutionStatusCode,
    MotionCommand,
)

logger = logging.getLogger(__name__)

_QLIM_ROWS: tuple[np.ndarray, np.ndarray] | None = None


def _qlim_rows() -> tuple[np.ndarray, np.ndarray]:
    """Joint-limit rows, fetched once per process — ``robot.qlim`` allocates a
    fresh matrix per access and this is consumed on the 100 Hz jog path."""
    global _QLIM_ROWS
    if _QLIM_ROWS is None:
        qlim = PAROL6_ROBOT.robot.qlim
        if qlim is None:
            _QLIM_ROWS = (np.full(6, -np.inf), np.full(6, np.inf))
        else:
            _QLIM_ROWS = (
                np.ascontiguousarray(qlim[0], dtype=np.float64),
                np.ascontiguousarray(qlim[1], dtype=np.float64),
            )
    return _QLIM_ROWS


def _limit_hit_mask(pos_steps: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    return ((speeds > 0) & (pos_steps >= LIMITS.joint.position.steps[:, 1])) | (
        (speeds < 0) & (pos_steps <= LIMITS.joint.position.steps[:, 0])
    )


class HomeState(Enum):
    """State machine states for the homing sequence."""

    START = auto()
    WAITING_FOR_UNHOMED = auto()
    WAITING_FOR_HOMED = auto()


@register_command(CmdType.HOME)
class HomeCommand(MotionCommand[HomeCmd]):
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """

    PARAMS_TYPE = HomeCmd

    __slots__ = (
        "state",
        "start_cmd_counter",
        "timeout_counter",
    )

    def __init__(self, p: HomeCmd):
        super().__init__(p)
        self.state = HomeState.START
        self.start_cmd_counter = 10
        self.timeout_counter = 4500

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Manages the homing command and monitors for completion using a state machine."""
        if self.state == HomeState.START:
            logger.debug(
                "  -> Sending home signal (100)... Countdown: %d",
                self.start_cmd_counter,
            )
            state.Command_out = CommandCode.HOME
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                self.state = HomeState.WAITING_FOR_UNHOMED
            return ExecutionStatusCode.EXECUTING

        if self.state == HomeState.WAITING_FOR_UNHOMED:
            state.Command_out = CommandCode.IDLE
            if np.any(state.Homed_in[:6] == 0):
                logger.info("  -> Homing sequence initiated by robot.")
                self.state = HomeState.WAITING_FOR_HOMED
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                self.fail(make_error(ErrorCode.MOTN_HOME_TIMEOUT))
                self.stop_and_idle(state)
                return ExecutionStatusCode.FAILED
            return ExecutionStatusCode.EXECUTING

        if self.state == HomeState.WAITING_FOR_HOMED:
            state.Command_out = CommandCode.IDLE
            if np.all(state.Homed_in[:6] == 1):
                self.log_info("Homing sequence complete. All joints reported home.")
                self.finish()
                self.stop_and_idle(state)
                return ExecutionStatusCode.COMPLETED
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                self.fail(make_error(ErrorCode.MOTN_HOME_TIMEOUT))
                self.stop_and_idle(state)
                return ExecutionStatusCode.FAILED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.JOGJ)
class JogJCommand(MotionCommand[JogJCmd]):
    """
    A non-blocking command to jog joints for a specific duration.
    Uses static 6-element speed array on the wire (all joints, zeros for inactive).
    """

    PARAMS_TYPE = JogJCmd
    streamable = True

    __slots__ = (
        "speeds_out",
        "_jog_initialized",
        "_jog_vel_rad",
        "_lookahead_buf",
    )

    def __init__(self, p: JogJCmd):
        super().__init__(p)
        self.speeds_out = np.zeros(6, dtype=np.int32)
        self._jog_initialized = False
        self._jog_vel_rad = np.zeros(6, dtype=np.float64)
        self._lookahead_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute step speeds and rad/s velocities for all 6 joints."""
        for i in range(6):
            s = self.p.speeds[i]
            if s == 0.0:
                self.speeds_out[i] = 0
                self._jog_vel_rad[i] = 0.0
            else:
                frac = min(abs(s), 1.0)
                step_speed = int(
                    JOG_MIN_STEPS
                    + (LIMITS.joint.jog.velocity_steps[i] - JOG_MIN_STEPS) * frac
                )
                self.speeds_out[i] = step_speed if s > 0 else -step_speed
                self._jog_vel_rad[i] = speed_steps_to_rad_scalar(step_speed, i) * (
                    1 if s > 0 else -1
                )
        self.start_timer(self.p.duration)
        self._jog_initialized = False

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of joint jogging via StreamingExecutor."""
        se = state.streaming_executor

        # Sync position on first tick
        if not self._jog_initialized:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            se.sync_position(self._q_rad_buf)
            self._jog_initialized = True

        stop_reason = self._check_stop_conditions(state)

        if stop_reason:
            self._jog_vel_rad.fill(0.0)
            se.set_jog_velocity(self._jog_vel_rad)
            pos_rad, vel, finished = se.tick()
            self._q_rad_buf[:] = pos_rad
            rad_to_steps(self._q_rad_buf, self._steps_buf)
            self.set_move_position(state, self._steps_buf)

            if finished or np.dot(vel, vel) < 1e-6:
                if stop_reason.startswith("Limit"):
                    logger.warning(stop_reason)
                else:
                    self.log_trace(stop_reason)
                se.active = False
                self.finish()
                return ExecutionStatusCode.COMPLETED
            return ExecutionStatusCode.EXECUTING

        se.set_jog_velocity(self._jog_vel_rad)
        pos_rad, _vel, _finished = se.tick()

        # Never stream a config that collides or approaches collision: the
        # streamed config itself is checked (catches anything inside the
        # lookahead window at jog start) plus a velocity-scaled horizon so
        # faster jogs stop further from contact — both compose with the
        # checker's fixed clearance. Exception: when the arm is ALREADY inside
        # (a keep-out placed over it), escaping motion is allowed, mirroring
        # the planner guard's start-in-collision semantics. An abrupt stop is
        # acceptable when the alternative is driving deeper. (The Cartesian jog
        # uses a graceful CSE-based stop; JogJ has no smoother, so it halts.)
        checker = PAROL6_ROBOT.collision
        if checker is not None:
            # In-place to keep the hot path allocation-free; clamped to joint
            # limits so a pose past the mechanical stop can't phantom-trip.
            la = self._lookahead_buf
            la[:] = self._jog_vel_rad
            la *= COLLISION_JOG_LOOKAHEAD_S
            la += pos_rad
            lo, hi = _qlim_rows()
            np.clip(la, lo, hi, out=la)
            if collision_blocked(checker, pos_rad, la):
                logger.warning("[JOGJ] collision predicted - stopping jog")
                # Allocate only here (the rare stop), never on the clean tick.
                state.collision_pairs = tuple(checker.colliding_pairs(la))
                state.collision_active = True
                se.active = False
                self.finish()
                return ExecutionStatusCode.COMPLETED

        self._q_rad_buf[:] = pos_rad
        rad_to_steps(self._q_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        return ExecutionStatusCode.EXECUTING

    def _check_stop_conditions(self, state: "ControllerState") -> str | None:
        """Check if jog should stop. Returns stop reason or None."""
        if self.timer_expired():
            return "Timed jog finished."

        limit_mask = _limit_hit_mask(state.Position_in, self.speeds_out)
        if np.any(limit_mask):
            return f"Limit reached on joint {int(np.argmax(limit_mask)) + 1}."

        return None


@register_command(CmdType.SELECT_TOOL)
class SelectToolCommand(MotionCommand[SelectToolCmd]):
    """
    Set the current end-effector tool configuration.
    """

    PARAMS_TYPE = SelectToolCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Set the tool in state and update robot kinematics."""
        tool_name = self.p.tool_name.strip().upper()
        variant_key = self.p.variant_key

        state.set_tool(tool_name, variant_key)
        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.TELEPORT)
class TeleportCommand(MotionCommand[TeleportCmd]):
    """Instantly set joint angles (simulator only, no trajectory)."""

    PARAMS_TYPE = TeleportCmd
    streamable = True

    __slots__ = ("_target_steps", "_deg_buf", "_sim_mode")

    def __init__(self, p: TeleportCmd):
        super().__init__(p)
        self._target_steps = np.empty(6, dtype=np.int32)
        self._deg_buf = np.empty(6, dtype=np.float64)
        self._sim_mode = False

    def do_setup(self, state: ControllerState) -> None:
        self._sim_mode = is_simulation_mode()
        self._deg_buf[:] = self.p.angles
        deg_to_steps(self._deg_buf, self._target_steps)

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        if not self._sim_mode:
            logger.warning("TELEPORT rejected: only allowed in simulator mode")
            self.finish()
            return ExecutionStatusCode.COMPLETED

        state.Position_out[:] = self._target_steps
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.TELEPORT

        if self.p.tool_positions:
            state.tool_teleport_pos = (
                max(0.0, min(1.0, self.p.tool_positions[0])) * 255.0
            )
            # Clear gripper command bits so write_frame's JIT doesn't re-arm the ramp
            state.Gripper_data_out[3] = 0

        self.finish()
        return ExecutionStatusCode.COMPLETED
