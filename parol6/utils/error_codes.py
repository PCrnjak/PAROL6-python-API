"""Error codes for structured robot error reporting.

Subsystem ranges (10 codes each):
    10-19  IK (inverse kinematics)
    20-29  Trajectory planning
    30-39  Motion execution
    40-49  Communication / protocol
    50-59  System / safety
"""

from enum import IntEnum


class ErrorCode(IntEnum):
    # IK subsystem
    IK_TARGET_UNREACHABLE = 10
    IK_PARTIAL_PATH = 11

    # Trajectory planning
    TRAJ_EMPTY_RESULT = 20
    TRAJ_NO_STEPS = 21

    # Motion execution
    MOTN_HOME_TIMEOUT = 30
    MOTN_GRIPPER_TIMEOUT = 31
    MOTN_GRIPPER_UNKNOWN = 32
    MOTN_SETUP_FAILED = 33
    MOTN_TICK_FAILED = 34

    # Communication / protocol
    COMM_QUEUE_FULL = 40
    COMM_UNKNOWN_COMMAND = 41
    COMM_DECODE_ERROR = 42
    COMM_VALIDATION_ERROR = 43

    # System / safety
    SYS_CONTROLLER_DISABLED = 50
    SYS_ESTOP_ACTIVE = 51
    SYS_PORT_SAVE_FAILED = 52
    SYS_PROFILE_INVALID = 53
