"""The error catalog: waldoctl's RobotError plus this backend's templates.

RobotError carries KUKA-style structured fields: code, title, cause, effect, remedy.
The catalog maps ErrorCode → template; make_error() instantiates with runtime params.
"""

from __future__ import annotations

from dataclasses import dataclass

from waldoctl.errors import RobotError as _RobotError

from .error_codes import ErrorCode


# The refusal type is the contract's, not this backend's: a frontend
# represents a refused command the same way whichever arm raised it, and
# the six fields and the wire list are identical in both directions. An
# exception rather than a dataclass, so a client can raise it as-is.
RobotError = _RobotError


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
    ErrorCode.MOTN_NOT_HOMED: _ErrorTemplate(
        title="Robot not homed",
        cause=(
            "Planned motion requested while the robot is not homed — "
            "reported joint positions are unreferenced until homing."
        ),
        effect="Motion command rejected before dispatch.",
        remedy="Run home() first. Jogging remains available.",
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
    ErrorCode.SYS_STATUS_RATE_INVALID: _ErrorTemplate(
        title="Status rate not achievable",
        cause="Status is broadcast every Nth control tick, so {requested} Hz "
        "does not divide the {control} Hz control loop evenly.",
        effect="Broadcast rate unchanged.",
        remedy="Pick a rate that divides {control} Hz: {allowed}.",
    ),
    ErrorCode.SYS_CONTROLLER_DISABLED: _ErrorTemplate(
        title="Controller disabled",
        cause="Motion command sent while controller is disabled. {detail}",
        effect="Command rejected.",
        remedy="Call reset() to re-enable the controller.",
    ),
    ErrorCode.SYS_ESTOP_ACTIVE: _ErrorTemplate(
        title="E-stop active",
        cause="Emergency stop is currently engaged.",
        effect="All motion stopped. Queue cleared.",
        remedy="Release the E-stop button and call reset().",
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
    ErrorCode.SYS_SELF_COLLISION: _ErrorTemplate(
        title="Self-collision predicted",
        cause="Planned configuration would self-collide at sample {sample} of {total}: {pairs}",
        effect="Motion command rejected before dispatch.",
        remedy="Choose a different target, add intermediate waypoints, or disable via PAROL6_COLLISION_CHECK=0.",
    ),
}


def make_error(
    code: ErrorCode, command_index: int = -1, **params: object
) -> RobotError:
    """Create a RobotError from the catalog, formatting its placeholders.

    Every field is formatted, not just the ones that usually carry a
    placeholder: a remedy is where a refusal says what would have worked, so
    a template that puts its parameter there must not reach the client with
    the placeholder still in it.
    """
    tmpl = _CATALOG[code]
    return RobotError(
        command_index=command_index,
        code=int(code),
        title=tmpl.title.format_map(params) if params else tmpl.title,
        cause=tmpl.cause.format_map(params) if params else tmpl.cause,
        effect=tmpl.effect.format_map(params) if params else tmpl.effect,
        remedy=tmpl.remedy.format_map(params) if params else tmpl.remedy,
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
