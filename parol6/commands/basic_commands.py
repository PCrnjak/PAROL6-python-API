"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and SelectTool.
"""

import logging
from enum import Enum, auto
import numpy as np
from numba import njit

from parol6.config import (
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
from parol6.commands._collision_guard import collision_blocked, stream_lookahead
from parol6.motion.streaming_executors import below_speed
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

# A jog stops this far short of a joint's limit, on top of its own
# stopping distance, so a joint that overshoots its ramp by a hair never
# touches the stop.
_JOG_LIMIT_MARGIN_RAD: float = 0.005
# The jerk-limited stopping distance is over-estimated by this factor. The
# limit itself is held by StreamingExecutor.hold_inside, whatever the
# lookahead predicted; this only decides how far short the ramp starts.
_JOG_STOP_MARGIN: float = 1.05
# The measured position trails the commanded one by this many ticks
# (write, firmware, read back); the lookahead counts that travel too.
_JOG_LAG_TICKS: float = 2.0
# Joint travel the jog stops short of, read once and contiguous: the
# per-tick kernels take them as they are.
_POS_LO_RAD = np.ascontiguousarray(LIMITS.joint.position.rad[:, 0])
_POS_HI_RAD = np.ascontiguousarray(LIMITS.joint.position.rad[:, 1])
_ACCEL_MAX = np.ascontiguousarray(LIMITS.joint.hard.acceleration, dtype=np.float64)
_JERK_MAX = np.ascontiguousarray(LIMITS.joint.hard.jerk, dtype=np.float64)


@njit(cache=True)
def _jog_lookahead_jit(
    jog_vel: np.ndarray,
    stopping: bool,
    vel: np.ndarray,
    acc: np.ndarray,
    q_meas: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    accel_max: np.ndarray,
    jerk_max: np.ndarray,
    accel_frac: float,
    dt: float,
    blocked: np.ndarray,
    target_vel: np.ndarray,
) -> int:
    """Fill the target velocities, latching the direction of any joint
    whose stopping distance reaches its limit and zeroing a joint driven
    into its latch. Returns bit 0 set while any joint is still driven, and
    bit ``j + 1`` for each joint latched this tick.

    The stopping distance is the jerk-limited ramp's: from the speed and
    acceleration the executor is at, the acceleration first reverses at the
    jerk limit, peaking the speed, then the ramp runs at the acceleration
    limit and rounds off at the jerk limit again. The remaining travel is
    measured: the arm is what approaches the limit, not the integrator.
    """
    status = 0
    for j in range(target_vel.shape[0]):
        v_t = 0.0 if stopping else jog_vel[j]
        v = vel[j]
        probe = v if v != 0.0 else v_t
        if probe != 0.0:
            if probe > 0.0:
                remaining = hi[j] - q_meas[j]
                sgn = 1
            else:
                remaining = q_meas[j] - lo[j]
                sgn = -1
            a = accel_max[j] * accel_frac
            jk = jerk_max[j]
            speed = abs(v)
            a0 = acc[j] * sgn
            if a0 < 0.0:
                a0 = 0.0
            v_peak = speed + a0 * a0 / (2.0 * jk)
            stop = (
                speed * a0 / jk
                + a0 * a0 * a0 / (3.0 * jk * jk)
                + v_peak * v_peak / (2.0 * a)
                + v_peak * a / (2.0 * jk)
            )
            stop = _JOG_STOP_MARGIN * stop + _JOG_LAG_TICKS * speed * dt
            if stop + _JOG_LIMIT_MARGIN_RAD >= remaining and blocked[j] != sgn:
                blocked[j] = sgn
                status |= 1 << (j + 1)
        if v_t != 0.0 and blocked[j] == (1 if v_t > 0.0 else -1):
            v_t = 0.0
        target_vel[j] = v_t
        if v_t != 0.0:
            status |= 1
    return status


@njit(cache=True)
def _track_rates_jit(
    vel: np.ndarray, vel_prev: np.ndarray, acc_prev: np.ndarray, dt: float
) -> None:
    """The executor's speed and acceleration as the lookahead reads them."""
    inv = 1.0 / dt
    for j in range(vel.shape[0]):
        acc_prev[j] = (vel[j] - vel_prev[j]) * inv
        vel_prev[j] = vel[j]


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
    — that joint alone; the others carry on — and held while the jog
    drives it that way. The hold is the joint's own: it lets go when the
    jog drives that joint the other way or stops driving it. Whatever the
    lookahead predicted, the commanded position never steps across a
    limit. The jog ends when its duration, counted in control ticks, runs
    out and the joints have come to rest.
    """

    PARAMS_TYPE = JogJCmd
    streamable = True

    __slots__ = (
        "speeds_out",
        "_synced",
        "_accel_applied",
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
        self._synced = False
        self._accel_applied = -1.0
        self._jog_vel_rad = np.zeros(6, dtype=np.float64)
        self._target_vel = np.zeros(6, dtype=np.float64)
        self._vel_prev = np.zeros(6, dtype=np.float64)
        self._acc_prev = np.zeros(6, dtype=np.float64)
        self._q_meas = np.zeros(6, dtype=np.float64)
        # Per joint: the direction a limit has blocked (±1), or 0.
        self._blocked = np.zeros(6, dtype=np.int8)
        self._lookahead_buf = np.zeros(6, dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute step speeds and rad/s velocities for all 6 joints,
        releasing the limit hold of a joint this datagram stops driving or
        drives the other way."""
        for i in range(6):
            s = self.p.speeds[i]
            held = self._blocked[i]
            if held != 0 and (
                (s == 0.0 and self._jog_vel_rad[i] != 0.0)
                or (s != 0.0 and (1 if s > 0.0 else -1) != held)
            ):
                self._blocked[i] = 0
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
        self.start_tick_timer(self.p.duration)

    def execute_step(self, state: "ControllerState") -> ExecutionStatusCode:
        """Execute one tick of joint jogging via StreamingExecutor."""
        se = state.streaming_executor

        # A new jog starts from the arm at rest; one continued by the next
        # datagram keeps the motion it is in, and the lookahead the speed
        # and acceleration it has measured of it.
        if not self._synced:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            se.sync_position(self._q_rad_buf)
            self._vel_prev.fill(0.0)
            self._acc_prev.fill(0.0)
            self._synced = True
        if self.p.accel != self._accel_applied:
            se.set_limits(1.0, self.p.accel)
            self._accel_applied = self.p.accel

        steps_to_rad(state.Position_in, self._q_meas)
        stopping = self.tick_timer_expired()
        status = _jog_lookahead_jit(
            self._jog_vel_rad,
            stopping,
            self._vel_prev,
            self._acc_prev,
            self._q_meas,
            _POS_LO_RAD,
            _POS_HI_RAD,
            _ACCEL_MAX,
            _JERK_MAX,
            self.p.accel,
            se.dt,
            self._blocked,
            self._target_vel,
        )
        if status > 1:
            for j in range(6):
                if status & (1 << (j + 1)):
                    logger.info("[JOGJ] joint %d stopping short of its limit", j + 1)

        se.set_jog_velocity(self._target_vel)
        pos_rad, vel, finished = se.tick()
        # _q_rad_buf still holds the position commanded last tick.
        se.hold_inside(self._q_rad_buf, self._q_meas, _POS_LO_RAD, _POS_HI_RAD)
        _track_rates_jit(vel, self._vel_prev, self._acc_prev, se.dt)

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
        if checker is not None and status & 1 and arm_homed(state):
            la = stream_lookahead(pos_rad, self._target_vel, self._lookahead_buf)
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

        if stopping and (finished or below_speed(vel, 1e-6)):
            self.log_trace("Timed jog finished.")
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
    positions are not as many as status reports for the fitted tool."""

    PARAMS_TYPE = TeleportCmd

    __slots__ = ("_target_steps",)

    def __init__(self, p: TeleportCmd):
        super().__init__(p)
        self._target_steps = np.empty(6, dtype=np.int32)

    def do_setup(self, state: ControllerState) -> None:
        # The controller refuses what the simulator cannot apply before
        # setup runs (off the simulator, tool positions other than the ones
        # status reports for the fitted tool).
        deg_to_steps(np.asarray(self.p.angles, dtype=np.float64), self._target_steps)

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        state.Position_out[:] = self._target_steps
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.TELEPORT
        # The pose is exact: the arm is referenced there from this tick on,
        # and the simulator reports it so on the next frame. A move read
        # after the teleport, in the same batch, is planned from the landing.
        state.Position_in[:] = self._target_steps
        state.Homed_in[:6] = 1

        if self.p.tool_positions:
            state.tool_teleport_pos = self.p.tool_positions[0] * 255.0
            # Clear gripper command bits so write_frame's JIT doesn't re-arm the ramp
            state.Gripper_data_out[3] = 0

        self.finish()
        return ExecutionStatusCode.COMPLETED
