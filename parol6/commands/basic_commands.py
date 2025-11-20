"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and MultiJog
"""

import logging
from math import ceil

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import INTERVAL_S
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState
from parol6.tools import TOOL_CONFIGS, list_tools

from .base import (
    ExecutionStatus,
    MotionCommand,
    csv_floats,
    csv_ints,
    parse_float,
    parse_int,
)

logger = logging.getLogger(__name__)


@register_command("HOME")
class HomeCommand(MotionCommand):
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """

    __slots__ = (
        "state",
        "start_cmd_counter",
        "timeout_counter",
    )

    def __init__(self):
        super().__init__()
        # State machine: START -> WAIT_FOR_UNHOMED -> WAIT_FOR_HOMED -> FINISHED
        self.state = "START"
        # Counter to send the home command for multiple cycles
        self.start_cmd_counter = 10  # Send command 100 for 10 cycles (0.1s)
        # Safety timeout (20 seconds at 0.01s interval)
        self.timeout_counter = 2000

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse HOME command (no parameters).

        Format: HOME
        """
        if len(parts) != 1:
            return (False, "HOME command takes no parameters")
        self.log_trace("Parsed HOME command")
        return (True, None)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """
        Manages the homing command and monitors for completion using a state machine.
        """
        # --- State: START ---
        # On the first few executions, continuously send the 'home' (100) command.
        if self.state == "START":
            logger.debug(
                f"  -> Sending home signal (100)... Countdown: {self.start_cmd_counter}"
            )
            state.Command_out = CommandCode.HOME
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                # Once sent for enough cycles, move to the next state
                self.state = "WAITING_FOR_UNHOMED"
            return ExecutionStatus.executing("Homing: start")

        # --- State: WAITING_FOR_UNHOMED ---
        # The robot's firmware should reset the homed status. We wait to see that happen.
        # During this time, we send 'idle' (255) to let the robot's controller take over.
        if self.state == "WAITING_FOR_UNHOMED":
            state.Command_out = CommandCode.IDLE
            # Homing sequence initiated detection
            if np.any(state.Homed_in[:6] == 0):
                logger.info("  -> Homing sequence initiated by robot.")
                self.state = "WAITING_FOR_HOMED"
            # Homing timeout protection
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                raise TimeoutError(
                    "Timeout waiting for robot to start homing sequence."
                )
            return ExecutionStatus.executing("Homing: waiting for unhomed")

        # --- State: WAITING_FOR_HOMED ---
        # Now we wait for all joints to report that they are homed (all flags are 1).
        if self.state == "WAITING_FOR_HOMED":
            state.Command_out = CommandCode.IDLE
            # Homing completion verification
            if np.all(state.Homed_in[:6] == 1):
                self.log_info("Homing sequence complete. All joints reported home.")
                self.is_finished = True
                self.stop_and_idle(state)
                return ExecutionStatus.completed("Homing complete")

            return ExecutionStatus.executing("Homing: waiting for homed")

        return ExecutionStatus.executing("Homing")


@register_command("JOG")
class JogCommand(MotionCommand):
    """
    A non-blocking command to jog a joint for a specific duration or distance.
    It performs all safety and validity checks upon initialization.
    """

    streamable = True  # Can be replaced in stream mode

    __slots__ = (
        "mode",
        "command_step",
        "joint",
        "speed_percentage",
        "duration",
        "distance_deg",
        "direction",
        "joint_index",
        "speed_out",
        "command_len",
        "target_position",
    )

    def __init__(self):
        """
        Initializes the jog command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()
        self.mode = None
        self.command_step = 0

        # Parameters (set in match())
        self.joint = None
        self.speed_percentage = None
        self.duration = None
        self.distance_deg = None

        # Calculated values
        self.direction = 1
        self.joint_index = 0
        self.speed_out = 0
        self.target_position = 0

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse JOG command parameters.

        Format: JOG|joint|speed_pct|duration|distance
        Example: JOG|0|50|2.0|None

        Args:
            parts: Pre-split message parts

        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 5:
            return (
                False,
                "JOG requires 4 parameters: joint, speed, duration, distance",
            )

        # Parse parameters using utilities
        self.joint = parse_int(parts[1])
        self.speed_percentage = parse_float(parts[2])
        self.duration = parse_float(parts[3])
        self.distance_deg = parse_float(parts[4])

        if self.joint is None or self.speed_percentage is None:
            return (False, "Joint and speed percentage are required")

        # Determine mode
        if self.duration and self.distance_deg:
            self.mode = "distance"
            self.log_trace(
                "Parsed Jog: Joint %s, Distance %s deg, Duration %ss.",
                self.joint,
                self.distance_deg,
                self.duration,
            )
        elif self.duration:
            self.mode = "time"
            self.log_trace(
                "Parsed Jog: Joint %s, Speed %s%%, Duration %ss.",
                self.joint,
                self.speed_percentage,
                self.duration,
            )
        elif self.distance_deg:
            self.mode = "distance"
            self.log_trace(
                "Parsed Jog: Joint %s, Speed %s%%, Distance %s deg.",
                self.joint,
                self.speed_percentage,
                self.distance_deg,
            )
        else:
            return (False, "JOG requires either duration or distance")

        self.is_valid = True
        return (True, None)

    def setup(self, state):
        """Pre-computes speeds and target positions using live data."""
        # Validate joint is set
        if self.joint is None:
            raise RuntimeError("Joint index not set")

        # Joint direction and index mapping
        self.direction = 1 if 0 <= self.joint <= 5 else -1
        self.joint_index = self.joint if self.direction == 1 else self.joint - 6

        lims = self.LIMS_STEPS[self.joint_index]
        min_limit, max_limit = lims[0], lims[1]

        distance_steps = 0
        if self.distance_deg is not None:
            distance_steps = int(
                PAROL6_ROBOT.ops.deg_to_steps(abs(self.distance_deg), self.joint_index)
            )
            self.target_position = state.Position_in[self.joint_index] + (
                distance_steps * self.direction
            )

            if not (min_limit <= self.target_position <= max_limit):
                # Convert to degrees for clearer error message
                target_deg = PAROL6_ROBOT.ops.steps_to_deg(
                    self.target_position, self.joint_index
                )
                min_deg = PAROL6_ROBOT.ops.steps_to_deg(min_limit, self.joint_index)
                max_deg = PAROL6_ROBOT.ops.steps_to_deg(max_limit, self.joint_index)
                raise ValueError(
                    f"Target position {target_deg:.2f}° is out of joint limits ({min_deg:.2f}°, {max_deg:.2f}°)."
                )

        # Motion timing calculations
        jog_min = self.JOG_MIN[self.joint_index]
        jog_max = self.JOG_MAX[self.joint_index]

        if self.mode == "distance" and self.duration:
            speed_steps_per_sec = (
                int(distance_steps / self.duration) if self.duration > 0 else 0
            )
            if speed_steps_per_sec > jog_max:
                raise ValueError(
                    f"Required speed ({speed_steps_per_sec} steps/s) exceeds joint's max jog speed ({jog_max} steps/s)."
                )
            # Ensure speed is at least the minimum jog speed if not zero
            if speed_steps_per_sec > 0:
                speed_steps_per_sec = max(speed_steps_per_sec, jog_min)
        else:
            if self.speed_percentage is None:
                raise ValueError(
                    "'speed_percentage' must be provided if not calculating automatically."
                )
            speed_steps_per_sec = int(
                self.linmap_pct(abs(self.speed_percentage), jog_min, jog_max)
            )

        self.speed_out = speed_steps_per_sec * self.direction

        # Start timer for time-based mode
        if self.mode == "time" and self.duration and self.duration > 0:
            self.start_timer(self.duration)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """This is the EXECUTION phase. It runs on every loop cycle."""

        # Type guard to ensure joint_index is valid
        if self.joint_index is None or not isinstance(self.joint_index, int):
            raise RuntimeError("Invalid joint index in execute_step")

        stop_reason = None
        cur = state.Position_in[self.joint_index]

        if self.mode == "time" and self.timer_expired():
            stop_reason = "Timed jog finished."
        elif self.mode == "distance" and (
            (self.direction == 1 and cur >= self.target_position)
            or (self.direction == -1 and cur <= self.target_position)
        ):
            stop_reason = "Distance jog finished."

        if not stop_reason:
            # Use base class limit_hit_mask helper
            speeds_array = np.zeros(6)
            speeds_array[self.joint_index] = self.speed_out
            limit_mask = self.limit_hit_mask(state.Position_in, speeds_array)
            if limit_mask[self.joint_index]:
                stop_reason = f"Limit reached on joint {self.joint_index + 1}."

        if stop_reason:
            if stop_reason.startswith("Limit"):
                logger.warning(stop_reason)
            else:
                self.log_trace(stop_reason)

            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed(stop_reason)

        state.Speed_out.fill(0)
        state.Speed_out[self.joint_index] = self.speed_out
        state.Command_out = CommandCode.JOG
        self.command_step += 1
        return ExecutionStatus.executing("Jogging")


@register_command("MULTIJOG")
class MultiJogCommand(MotionCommand):
    """
    A non-blocking command to jog multiple joints simultaneously for a specific duration.
    It performs all safety and validity checks upon initialization.
    """

    streamable = True  # Can be replaced in stream mode

    __slots__ = (
        "command_step",
        "joints",
        "speed_percentages",
        "duration",
        "command_len",
        "speeds_out",
        "_lims_steps",
    )

    def __init__(self):
        """
        Initializes the multi-jog command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()
        self.command_step = 0

        # Parameters (set in do_match())
        self.joints = None
        self.speed_percentages = None
        self.duration = None
        self.command_len = 0

        # Calculated values
        self.speeds_out = np.zeros(6, dtype=np.int32)
        self._lims_steps = PAROL6_ROBOT.joint.limits.steps

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MULTIJOG command parameters.

        Format: MULTIJOG|joints_csv|speeds_csv|duration
        Example: MULTIJOG|0,1,2|50,75,100|3.0

        Args:
            parts: Pre-split message parts

        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 4:
            return (False, "MULTIJOG requires 3 parameters: joints, speeds, duration")

        # Parse parameters using utilities
        self.joints = csv_ints(parts[1])
        self.speed_percentages = csv_floats(parts[2])
        self.duration = parse_float(parts[3]) or 0.0

        # Validate
        if len(self.joints) != len(self.speed_percentages):
            return (False, "Number of joints must match number of speeds")

        if self.duration <= 0:
            return (False, "Duration must be positive")

        # Conflict detection on base joints
        base = set()
        for j in self.joints:
            b = j % 6
            if b in base:
                return (False, f"Conflicting commands for Joint {b + 1}")
            base.add(b)

        self.log_trace(
            "Parsed MultiJog for joints %s with speeds %s%% for %ss.",
            self.joints,
            self.speed_percentages,
            self.duration,
        )

        self.command_len = ceil(self.duration / INTERVAL_S)
        self.is_valid = True
        return (True, None)

    def setup(self, state):
        """Pre-computes the speeds for each joint."""
        # Validate joints and speed_percentages are set
        if self.joints is None or self.speed_percentages is None:
            raise ValueError("Joints or speed percentages not set")

        # Vectorized computation for all joints
        joints_arr = np.asarray(self.joints, dtype=int)
        speeds_pct = np.asarray(self.speed_percentages, dtype=float)

        # Map to base joint index (0-5) and direction (+/-)
        direction = np.where((joints_arr >= 0) & (joints_arr <= 5), 1, -1)
        joint_index = np.where(direction == 1, joints_arr, joints_arr - 6)

        # Validate indices
        invalid_mask = (joint_index < 0) | (joint_index >= 6)
        if np.any(invalid_mask):
            bad = joint_index[invalid_mask]
            raise ValueError(f"Invalid joint indices {bad.tolist()}")

        pct = np.clip(np.abs(speeds_pct) / 100.0, 0.0, 1.0)
        for i, idx in enumerate(joint_index):
            self.speeds_out[idx] = (
                int(
                    self.linmap_pct(
                        pct[i] * 100.0, self.JOG_MIN[idx], self.JOG_MAX[idx]
                    )
                )
                * direction[i]
            )

        # Start timer if duration is specified
        if self.duration and self.duration > 0:
            self.start_timer(self.duration)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """This is the EXECUTION phase. It runs on every loop cycle."""
        # Stop if the duration has elapsed (check both timer and step count)
        if self.timer_expired() or self.command_step >= self.command_len:
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MultiJog")

        # Use base class helper for limit checks
        limit_mask = self.limit_hit_mask(state.Position_in, self.speeds_out)
        if np.any(limit_mask):
            i = np.argmax(limit_mask)  # first violating joint
            logger.warning(f"Limit reached on joint {i + 1}. Stopping jog.")
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed(f"Limit reached on J{i + 1}")

        # Apply self.speeds_out
        np.copyto(state.Speed_out, self.speeds_out, casting="no")
        state.Command_out = CommandCode.JOG
        self.command_step += 1
        return ExecutionStatus.executing("MultiJog")


@register_command("SET_TOOL")
class SetToolCommand(MotionCommand):
    """
    Set the current end-effector tool configuration.
    Changes the tool transform used for forward/inverse kinematics.
    """

    __slots__ = ("tool_name",)

    def __init__(self):
        super().__init__()
        self.tool_name = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse SET_TOOL command parameters.

        Format: SET_TOOL|tool_name
        Example: SET_TOOL|PNEUMATIC
        """
        if len(parts) != 2:
            return (False, "SET_TOOL requires 1 parameter: tool_name")

        self.tool_name = parts[1].strip().upper()

        # Validate tool name during parsing
        if self.tool_name not in TOOL_CONFIGS:
            available = list_tools()
            return (False, f"Unknown tool '{self.tool_name}'. Available: {available}")

        self.log_trace(f"Parsed SET_TOOL command: {self.tool_name}")
        return (True, None)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Set the tool in state and update robot kinematics."""
        # Type guard
        if self.tool_name is None:
            raise RuntimeError("Tool name not set")

        # Update server state - property setter handles tool application and cache invalidation
        state.current_tool = self.tool_name

        self.log_info(f"Tool set to: {self.tool_name}")
        self.is_finished = True
        return ExecutionStatus.completed(f"Tool set: {self.tool_name}")
