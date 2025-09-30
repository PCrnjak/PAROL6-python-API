"""
Type definitions for PAROL6 protocol.

Defines enums, TypedDicts, and dataclasses used across the public API.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, TypedDict


# Stream mode state enum
class StreamModeState(Enum):
    """Stream mode state for jog commands."""
    OFF = 0  # Stream mode disabled (default FIFO queueing)
    ON = 1   # Stream mode enabled (latest-wins for jog commands)


# Frame literals
Frame = Literal['WRF', 'TRF']

# Axis literals  
Axis = Literal['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-', 'RX+', 'RX-', 'RY+', 'RY-', 'RZ+', 'RZ-']

# Acknowledgment status literals
AckStatus = Literal['SENT', 'QUEUED', 'EXECUTING', 'COMPLETED', 'FAILED', 'INVALID', 'CANCELLED', 'TIMEOUT', 'NO_TRACKING']


class IOStatus(TypedDict):
    """Digital I/O status."""
    in1: int
    in2: int
    out1: int
    out2: int
    estop: int


class GripperStatus(TypedDict):
    """Electric gripper status."""
    id: int
    position: int
    speed: int
    current: int
    status_byte: int
    object_detect: int


class StatusAggregate(TypedDict):
    """Aggregate robot status."""
    pose: list[float]  # 4x4 transformation matrix flattened (len=16)
    angles: list[float]  # 6 joint angles in degrees
    io: IOStatus | list[int]  # Back-compat with existing server format
    gripper: GripperStatus | list[int]


class TrackingStatus(TypedDict):
    """Command tracking status."""
    command_id: str | None
    status: AckStatus
    details: str
    completed: bool
    ack_time: datetime | None


class SendResult(TypedDict):
    """Standardized result for command-sending APIs."""
    command_id: str | None
    status: AckStatus
    details: str
    completed: bool
    ack_time: datetime | None


class WireResponse(TypedDict):
    """Typed wrapper for parsed wire responses."""
    type: Literal['PONG','POSE','ANGLES','IO','GRIPPER','SPEEDS','STATUS','GCODE_STATUS','SERVER_STATE']
    payload: dict | list | str
