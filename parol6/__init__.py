"""
PAROL6 Python Package

A unified library for controlling PAROL6 robot arms with async-first UDP client,
optional sync wrapper, and server management capabilities.

Key components:
- Robot: Unified entry point — lifecycle, configuration, kinematics, factories
- AsyncRobotClient: Async UDP client for robot operations
- RobotClient: Sync wrapper with automatic event loop handling
"""

from . import PAROL6_ROBOT
from ._version import __version__
from .client.async_client import AsyncRobotClient
from .client.dry_run_client import DryRunRobotClient
from .client.sync_client import RobotClient
from .protocol.wire import (
    CurrentActionResultStruct,
    LoopStatsResultStruct,
    StatusResultStruct,
    ToolResultStruct,
)
from .protocol.types import PingResult
from .robot import Robot
from .utils.error_catalog import RobotError, extract_robot_error, make_error
from .utils.error_codes import ErrorCode
from .utils.errors import MotionError

# Type aliases for backward compatibility
CurrentActionResult = CurrentActionResultStruct
LoopStatsResult = LoopStatsResultStruct
StatusResult = StatusResultStruct
ToolResult = ToolResultStruct

__all__ = [
    "__version__",
    "Robot",
    "AsyncRobotClient",
    "DryRunRobotClient",
    "RobotClient",
    "PAROL6_ROBOT",
    # Result types (msgspec structs)
    "CurrentActionResultStruct",
    "LoopStatsResultStruct",
    "StatusResultStruct",
    "ToolResultStruct",
    # Backward-compatible aliases
    "CurrentActionResult",
    "LoopStatsResult",
    "StatusResult",
    "ToolResult",
    # Other types
    "PingResult",
    # Errors
    "ErrorCode",
    "RobotError",
    "extract_robot_error",
    "make_error",
    "MotionError",
]
