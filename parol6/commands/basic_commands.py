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

import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: N811

from .base import (
    ExecutionStatusCode,
    MotionCommand,
    SystemCommand,
    arm_homed,
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


# A jog stops this far short of a joint's limit, on top of its own
# stopping distance, so a joint that overshoots its ramp by a hair never
# touches the stop.
_JOG_LIMIT_MARGIN_RAD: float = 0.005
# The jerk-limited stopping distance is over-estimated by this factor.
_JOG_STOP_MARGIN: float = 1.05
# The measured position trails the commanded one by this many ticks
# (write, firmware, read back); the lookahead counts that travel too.
_JOG_LAG_TICKS: float = 2.0
# Joint travel the jog lookahead stops short of, read once: slicing the
# limits table every tick would allocate a view each time.
_POS_LO_RAD = LIMITS.joint.position.rad[:, 0]
_POS_HI_RAD = LIMITS.joint.position.rad[:, 1]


class HomeState(Enum):
    """State machine states for the homing sequence."""

    START = auto()
    WAITING_FOR_UNHOMED = auto()
    WAITING_FOR_HOMED = auto()


@register_command(CmdType.HOME)
class HomeCommand(MotionCommand[HomeCmd]):
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    Reached while the robot is unhomed, or on HOME(calibrate=True) from a
    referenced robot — the planner routes plain HOME from an already-referenced
    robot to a planned return move instead. The firmware clears the homed bits
    when the sequence starts, which WAITING_FOR_UNHOMED relies on.
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
        state.homing_step = self.state.value
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
                state.homing_step = 0
                self.fail(make_error(ErrorCode.MOTN_HOME_TIMEOUT))
                self.stop_and_idle(state)
                return ExecutionStatusCode.FAILED
            return ExecutionStatusCode.EXECUTING

        if self.state == HomeState.WAITING_FOR_HOMED:
            state.Command_out = CommandCode.IDLE
            if np.all(state.Homed_in[:6] == 1):
                self.log_info("Homing sequence complete. All joints reported home.")
                state.homing_step = 0
                self.finish()
                self.stop_and_idle(state)
                return ExecutionStatusCode.COMPLETED
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                state.homing_step = 0
                self.fail(make_error(ErrorCode.MOTN_HOME_TIMEOUT))
                self.stop_and_idle(state)
                return ExecutionStatusCode.FAILED

        return ExecutionStatusCode.EXECUTING


@register_command(CmdType.JOGJ)
class JogJCommand(MotionCommand[JogJCmd]):
    """
    A non-blocking command to jog joints for a specific duration.
    Uses static 6-element speed array on the wire (all joints, zeros for inactive).

    Each joint runs its own lookahead against its position limits: a joint
    whose stopping distance would reach its limit is ramped to rest there
    — that joint alone; the others carry on. The jog ends when its
    duration runs out or every commanded joint has been stopped by a limit.
    """

    PARAMS_TYPE = JogJCmd
    streamable = True

    __slots__ = (
        "speeds_out",
        "_jog_initialized",
        "_jog_vel_rad",
        "_target_vel",
        "_vel_prev",
        "_acc_prev",
        "_q_meas",
        "_blocked",
        "_lookahead_buf",
    )

    def __init__(self, p: JogJCmd):
        super().__init__(p)
        self.speeds_out = np.zeros(6, dtype=np.int32)
        self._jog_initialized = False
        self._jog_vel_rad = np.zeros(6, dtype=np.float64)
        self._target_vel = np.zeros(6, dtype=np.float64)
        self._vel_prev = np.zeros(6, dtype=np.float64)
        self._acc_prev = np.zeros(6, dtype=np.float64)
        self._q_meas = np.zeros(6, dtype=np.float64)
        # Per joint: the direction a limit has blocked (±1), or 0.
        self._blocked = np.zeros(6, dtype=np.int8)
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

    def _limit_lookahead(self, stopping: bool, dt: float) -> bool:
        """Fill the target velocities, zeroing any joint whose stopping
        distance reaches its limit. Returns True when nothing is left to
        drive: the timer ran out, or a limit stopped every commanded joint.

        The stopping distance is the jerk-limited ramp's: from the speed
        and acceleration the executor is at, the acceleration first
        reverses at the jerk limit, peaking the speed, then the ramp runs
        at the acceleration limit and rounds off at the jerk limit again.
        """
        lo = _POS_LO_RAD
        hi = _POS_HI_RAD
        accel = LIMITS.joint.hard.acceleration
        jerk = LIMITS.joint.hard.jerk
        driving = False
        for j in range(6):
            v_t = 0.0 if stopping else self._jog_vel_rad[j]
            v = self._vel_prev[j]
            probe = v if v != 0.0 else v_t
            if probe != 0.0:
                if probe > 0.0:
                    remaining = hi[j] - self._q_meas[j]
                    sgn = 1
                else:
                    remaining = self._q_meas[j] - lo[j]
                    sgn = -1
                a = accel[j] * self.p.accel
                jk = jerk[j]
                speed = abs(v)
                a0 = max(self._acc_prev[j] * sgn, 0.0)
                v_peak = speed + a0 * a0 / (2.0 * jk)
                stop = (
                    speed * a0 / jk
                    + a0 * a0 * a0 / (3.0 * jk * jk)
                    + v_peak * v_peak / (2.0 * a)
                    + v_peak * a / (2.0 * jk)
                )
                stop = _JOG_STOP_MARGIN * stop + _JOG_LAG_TICKS * speed * dt
                if stop + _JOG_LIMIT_MARGIN_RAD >= remaining:
                    self._blocked[j] = sgn
            if v_t != 0.0 and self._blocked[j] == (1 if v_t > 0.0 else -1):
                v_t = 0.0
            self._target_vel[j] = v_t
            if v_t != 0.0:
                driving = True
        return not driving

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of joint jogging via StreamingExecutor."""
        se = state.streaming_executor

        # A jog starting from rest syncs to the arm; one continued by the
        # next datagram keeps the motion it is in, and the lookahead keeps
        # the speed and acceleration it has measured of it.
        if not self._jog_initialized:
            if not se.active:
                steps_to_rad(state.Position_in, self._q_rad_buf)
                se.sync_position(self._q_rad_buf)
                self._vel_prev.fill(0.0)
                self._acc_prev.fill(0.0)
                self._blocked.fill(0)
            se.set_limits(1.0, self.p.accel)
            self._jog_initialized = True

        # The lookahead measures the remaining travel: the arm is what
        # approaches the limit, not the integrator.
        steps_to_rad(state.Position_in, self._q_meas)
        stopping = self.timer_expired()
        at_rest_wanted = self._limit_lookahead(stopping, se.dt)

        se.set_jog_velocity(self._target_vel)
        pos_rad, vel, finished = se.tick()
        np.subtract(vel, self._vel_prev, out=self._acc_prev)
        self._acc_prev /= se.dt
        self._vel_prev[:] = vel

        # Never stream a config that collides or approaches collision: the
        # streamed config itself is checked (catches anything inside the
        # lookahead window at jog start) plus a velocity-scaled horizon so
        # faster jogs stop further from contact — both compose with the
        # checker's fixed clearance. Exception: when the arm is ALREADY inside
        # (a keep-out placed over it), escaping motion is allowed, mirroring
        # the planner guard's start-in-collision semantics. An abrupt stop is
        # acceptable when the alternative is driving deeper. (The Cartesian jog
        # uses a graceful CSE-based stop; JogJ has no smoother, so it halts.)
        # An unreferenced arm's positions are not its own, so there is no
        # configuration to check: the jog that nudges it clear before it
        # can home runs unchecked, as does par6's.
        checker = PAROL6_ROBOT.collision
        if checker is not None and not at_rest_wanted and arm_homed(state):
            # In-place to keep the hot path allocation-free; clamped to joint
            # limits so a pose past the mechanical stop can't phantom-trip.
            la = self._lookahead_buf
            la[:] = self._target_vel
            la *= COLLISION_JOG_LOOKAHEAD_S
            la += pos_rad
            lo, hi = _qlim_rows()
            np.clip(la, lo, hi, out=la)
            if collision_blocked(checker, pos_rad, la):
                logger.warning("[JOGJ] collision predicted - stopping jog")
                # Allocate only here (the rare stop), never on the clean tick.
                state.collision_pairs = tuple(
                    PAROL6_ROBOT.display_pairs(checker.colliding_pairs(la))
                )
                state.collision_active = True
                se.active = False
                self.finish()
                return ExecutionStatusCode.COMPLETED

        self._q_rad_buf[:] = pos_rad
        rad_to_steps(self._q_rad_buf, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        if at_rest_wanted and (finished or np.dot(vel, vel) < 1e-6):
            if stopping:
                self.log_trace("Timed jog finished.")
            else:
                logger.warning(
                    "Limit reached on joint %d.", int(np.argmax(self._blocked != 0)) + 1
                )
            se.active = False
            self.finish()
            return ExecutionStatusCode.COMPLETED

        return ExecutionStatusCode.EXECUTING


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
class TeleportCommand(SystemCommand[TeleportCmd]):
    """Set the simulated arm's joint angles, and optionally its tool's
    positions, in one tick — no trajectory. The pose is exact afterwards,
    so the arm counts as homed. Refused on hardware, and when the tool
    positions do not match the fitted tool's degrees of freedom."""

    PARAMS_TYPE = TeleportCmd

    __slots__ = ("_target_steps",)

    def __init__(self, p: TeleportCmd):
        super().__init__(p)
        self._target_steps = np.empty(6, dtype=np.int32)

    def do_setup(self, state: ControllerState) -> None:
        # The controller refuses what the simulator cannot apply before
        # setup runs (off the simulator, tool positions the fitted tool has
        # no degrees of freedom for).
        deg_to_steps(np.asarray(self.p.angles, dtype=np.float64), self._target_steps)

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        state.Position_out[:] = self._target_steps
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.TELEPORT
        # The pose is exact: the arm is referenced there from this tick on,
        # and the simulator reports it so on the next frame.
        state.Homed_in[:6] = 1

        if self.p.tool_positions:
            state.tool_teleport_pos = self.p.tool_positions[0] * 255.0
            # Clear gripper command bits so write_frame's JIT doesn't re-arm the ramp
            state.Gripper_data_out[3] = 0

        self.finish()
        return ExecutionStatusCode.COMPLETED
