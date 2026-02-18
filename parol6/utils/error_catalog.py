"""Structured robot error types and error catalog.

RobotError carries KUKA-style structured fields: code, title, cause, effect, remedy.
The catalog maps ErrorCode → template; make_error() instantiates with runtime params.
"""

from __future__ import annotations

from dataclasses import dataclass

from .error_codes import ErrorCode


@dataclass(frozen=True)
class RobotError:
    """Structured error with code, title, cause, effect, and remedy."""

    command_index: int
    code: int
    title: str
    cause: str
    effect: str
    remedy: str

    def to_wire(self) -> list:
        """Serialize to a list for ormsgpack packing."""
        return [
            self.command_index,
            self.code,
            self.title,
            self.cause,
            self.effect,
            self.remedy,
        ]

    @staticmethod
    def from_wire(data: list) -> RobotError:
        """Reconstruct from a wire-format list."""
        return RobotError(
            command_index=data[0],
            code=data[1],
            title=data[2],
            cause=data[3],
            effect=data[4],
            remedy=data[5],
        )

    def __str__(self) -> str:
        return f"[{self.code}] {self.title}: {self.cause}"


@dataclass(frozen=True)
class _ErrorTemplate:
    title: str
    cause: str  # may contain {placeholders}
    effect: str
    remedy: str


_CATALOG: dict[int, _ErrorTemplate] = {
    # -- IK --
    ErrorCode.IK_TARGET_UNREACHABLE: _ErrorTemplate(
        title="IK: target unreachable",
        cause="Target pose has no valid IK solution. {detail}",
        effect="Motion command rejected. Pipeline halted.",
        remedy="Verify target is within workspace. Try different orientation.",
    ),
    ErrorCode.IK_PARTIAL_PATH: _ErrorTemplate(
        title="IK: partial path failure",
        cause="Only {valid}/{total} poses along the path are reachable.",
        effect="Motion command rejected. Pipeline halted.",
        remedy="Shorten the move, add intermediate waypoints, or adjust orientation.",
    ),
    # -- Trajectory --
    ErrorCode.TRAJ_EMPTY_RESULT: _ErrorTemplate(
        title="Trajectory: empty result",
        cause="Trajectory generation returned no waypoints. {detail}",
        effect="Motion command rejected.",
        remedy="Check motion parameters. Start and end may be too close.",
    ),
    ErrorCode.TRAJ_NO_STEPS: _ErrorTemplate(
        title="Trajectory: no steps",
        cause="Trajectory calculation produced zero steps. {detail}",
        effect="Motion command rejected.",
        remedy="Increase duration or reduce speed fraction.",
    ),
    # -- Motion execution --
    ErrorCode.MOTN_HOME_TIMEOUT: _ErrorTemplate(
        title="Homing timeout",
        cause="Robot did not start homing sequence within timeout.",
        effect="Home command aborted.",
        remedy="Check serial connection and robot power. Ensure E-stop is released.",
    ),
    ErrorCode.MOTN_GRIPPER_TIMEOUT: _ErrorTemplate(
        title="Gripper timeout",
        cause="Gripper command timed out in state {state}.",
        effect="Gripper command aborted.",
        remedy="Check gripper connection and calibration.",
    ),
    ErrorCode.MOTN_GRIPPER_UNKNOWN: _ErrorTemplate(
        title="Gripper: unknown state",
        cause="Gripper entered an unknown internal state.",
        effect="Gripper command aborted.",
        remedy="Reset the controller and recalibrate the gripper.",
    ),
    ErrorCode.MOTN_SETUP_FAILED: _ErrorTemplate(
        title="Command setup failed",
        cause="Command could not be initialized. {detail}",
        effect="Command rejected. Pipeline halted.",
        remedy="Check command parameters and robot state.",
    ),
    ErrorCode.MOTN_TICK_FAILED: _ErrorTemplate(
        title="Command execution error",
        cause="Unexpected error during execution. {detail}",
        effect="Command aborted. Robot stopped.",
        remedy="Check robot state. May need to re-home.",
    ),
    # -- Communication --
    ErrorCode.COMM_QUEUE_FULL: _ErrorTemplate(
        title="Command queue full",
        cause="Motion queue at maximum capacity.",
        effect="Command rejected.",
        remedy="Wait for current motions to complete.",
    ),
    ErrorCode.COMM_UNKNOWN_COMMAND: _ErrorTemplate(
        title="Unknown command",
        cause="No handler for received command type.",
        effect="Command ignored.",
        remedy="Check client library version matches server.",
    ),
    ErrorCode.COMM_DECODE_ERROR: _ErrorTemplate(
        title="Command decode error",
        cause="Failed to decode command. {detail}",
        effect="Command ignored.",
        remedy="Check command encoding. Possible version mismatch.",
    ),
    ErrorCode.COMM_VALIDATION_ERROR: _ErrorTemplate(
        title="Command validation error",
        cause="Invalid parameters. {detail}",
        effect="Command rejected.",
        remedy="Check parameter ranges and types.",
    ),
    # -- System / safety --
    ErrorCode.SYS_CONTROLLER_DISABLED: _ErrorTemplate(
        title="Controller disabled",
        cause="Motion command sent while controller is disabled. {detail}",
        effect="Command rejected.",
        remedy="Call resume() to re-enable the controller.",
    ),
    ErrorCode.SYS_ESTOP_ACTIVE: _ErrorTemplate(
        title="E-stop active",
        cause="Emergency stop is currently engaged.",
        effect="All motion halted. Queue cleared.",
        remedy="Release the E-stop button and call resume().",
    ),
    ErrorCode.SYS_PORT_SAVE_FAILED: _ErrorTemplate(
        title="Serial port save failed",
        cause="Could not save serial port configuration.",
        effect="Port may not persist across restarts.",
        remedy="Check file permissions and disk space.",
    ),
    ErrorCode.SYS_PROFILE_INVALID: _ErrorTemplate(
        title="Invalid motion profile",
        cause="Unrecognized motion profile: {detail}",
        effect="Profile not changed.",
        remedy="Use one of: TOPPRA, RUCKIG, QUINTIC, TRAPEZOID, LINEAR.",
    ),
}


def make_error(
    code: ErrorCode, command_index: int = -1, **params: object
) -> RobotError:
    """Create a RobotError from the catalog, formatting placeholders in title/cause."""
    tmpl = _CATALOG[code]
    return RobotError(
        command_index=command_index,
        code=int(code),
        title=tmpl.title.format_map(params) if params else tmpl.title,
        cause=tmpl.cause.format_map(params) if params else tmpl.cause,
        effect=tmpl.effect,
        remedy=tmpl.remedy,
    )


def extract_robot_error(
    exc: Exception, fallback_code: ErrorCode, command_index: int = -1, **params: object
) -> RobotError:
    """Extract a RobotError from an exception, falling back to a catalog error.

    If the exception carries a ``robot_error`` attribute (e.g. IKError,
    TrajectoryPlanningError), return it directly.  Otherwise, construct a
    new RobotError from the catalog using *fallback_code* and *params*.
    """
    robot_error: RobotError | None = getattr(exc, "robot_error", None)
    if robot_error is not None:
        return robot_error
    return make_error(fallback_code, command_index, **params)
