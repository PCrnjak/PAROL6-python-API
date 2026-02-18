"""
Wire protocol for PAROL6 robot communication.

This module contains all protocol definitions:
- Binary serial frame packing/unpacking (firmware communication)
- Msgpack message types and structs (UDP communication)
- Command/response encoding and decoding

Wire format uses msgpack arrays with integer type codes:
- OK:       MsgType.OK (just the integer)
- ERROR:    [MsgType.ERROR, message]
- STATUS:   [MsgType.STATUS, pose, angles, speeds, io, gripper, action_current, action_state, joint_en, cart_en_wrf, cart_en_trf, executing_index, completed_index, last_checkpoint, error, queued_segments, queued_duration]
- RESPONSE: [MsgType.RESPONSE, query_type, value]
- COMMAND:  [CmdType.XXX, ...params]
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Annotated, Any, TypeAlias, Union, cast

import msgspec
import numpy as np
import ormsgpack
from numba import njit  # type: ignore[import-untyped]

from parol6.config import LIMITS
from parol6.tools import TOOL_CONFIGS, list_tools
from parol6.utils.error_catalog import RobotError, make_error
from parol6.utils.error_codes import ErrorCode

logger = logging.getLogger(__name__)


# =============================================================================
# Numpy msgpack encoding hooks
# =============================================================================


def _enc_hook(obj: object) -> object:
    """Custom encoder hook for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalar to Python native type
    raise NotImplementedError(f"Cannot encode {type(obj)}")


# Module-level encoder with numpy support (thread-safe, reusable)
_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)

# Module-level decoder for generic msgpack
_decoder = msgspec.msgpack.Decoder()


# =============================================================================
# Message Types
# =============================================================================


class MsgType(IntEnum):
    """Message type codes for responses."""

    OK = auto()
    ERROR = auto()
    STATUS = auto()
    RESPONSE = auto()


class QueryType(IntEnum):
    """Query type codes for responses."""

    PING = auto()
    STATUS = auto()
    ANGLES = auto()
    POSE = auto()
    IO = auto()
    GRIPPER = auto()
    SPEEDS = auto()
    TOOL = auto()
    QUEUE = auto()
    CURRENT_ACTION = auto()
    LOOP_STATS = auto()
    PROFILE = auto()


class CmdType(IntEnum):
    """Command type codes for incoming commands.

    Wire format: [CmdType.XXX, ...params]
    """

    # Query commands (immediate, read-only)
    PING = auto()
    GET_STATUS = auto()
    GET_ANGLES = auto()
    GET_POSE = auto()
    GET_IO = auto()
    GET_GRIPPER = auto()
    GET_SPEEDS = auto()
    GET_TOOL = auto()
    GET_QUEUE = auto()
    GET_CURRENT_ACTION = auto()
    GET_LOOP_STATS = auto()
    GET_PROFILE = auto()

    # System commands (execute regardless of enable state)
    RESUME = auto()
    HALT = auto()
    SET_IO = auto()
    SET_PORT = auto()
    SIMULATOR = auto()
    SET_PROFILE = auto()
    RESET = auto()
    RESET_LOOP_STATS = auto()

    # Motion commands — queued, pre-computed trajectory
    HOME = auto()
    MOVEJ = auto()
    MOVEJ_POSE = auto()
    MOVEL = auto()
    MOVEC = auto()
    MOVES = auto()
    MOVEP = auto()
    SET_TOOL = auto()
    DELAY = auto()
    CHECKPOINT = auto()

    # Streaming commands — position (servo) and velocity (jog)
    SERVOJ = auto()
    SERVOJ_POSE = auto()
    SERVOL = auto()
    JOGJ = auto()
    JOGL = auto()

    # Gripper commands
    PNEUMATICGRIPPER = auto()
    ELECTRICGRIPPER = auto()


# =============================================================================
# Command Structs - Tagged Union for single-pass decode
# Wire format: [CmdType.XXX, ...fields]
# =============================================================================


def _check_speed_accel(speed: float, accel: float, *, signed: bool = False) -> None:
    """Validate speed/accel are in the expected fractional range."""
    lo = -1.0 if signed else 0.0
    if not (lo <= speed <= 1.0):
        raise ValueError(
            f"speed={speed} is out of range [{lo}, 1.0]. "
            "Speed is a fraction of max velocity, not a percentage."
        )
    if not (0.0 <= accel <= 1.0):
        raise ValueError(
            f"accel={accel} is out of range [0.0, 1.0]. "
            "Accel is a fraction of max acceleration, not a percentage."
        )


class MotionParamsMixin:
    """Mixin providing resolved motion parameters for wire structs.

    Handles both sentinel patterns:
    - Move commands use 0.0 as "not specified"
    - Curved/spline commands use None as "not specified"

    Field declarations live on the concrete Struct subclasses, not here,
    to avoid Pylance invariance errors on override.
    """

    __slots__ = ()

    accel: float

    @property
    def resolved_duration(self) -> float | None:
        """Duration in seconds, or None for velocity-based timing."""
        d = cast("float | None", getattr(self, "duration"))
        return d if d is not None and d > 0.0 else None

    @property
    def resolved_speed(self) -> float:
        """Velocity fraction 0-1, defaults to 1.0 (full speed)."""
        s = cast("float | None", getattr(self, "speed"))
        return s if s is not None and s > 0.0 else 1.0


# -- Queued move commands (pre-computed trajectory) --


class MoveJCmd(
    MotionParamsMixin,
    msgspec.Struct,
    tag=int(CmdType.MOVEJ),
    array_like=True,
    frozen=True,
    gc=False,
):
    """MOVEJ: joint-space move to target angles (degrees)."""

    angles: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    r: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    rel: bool = False

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEJ requires either duration > 0 or speed > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEJ requires only one of duration or speed")
        _check_speed_accel(self.speed, self.accel)
        if not self.rel:
            for i in range(6):
                if not (
                    LIMITS.joint.position.deg[i, 0]
                    <= self.angles[i]
                    <= LIMITS.joint.position.deg[i, 1]
                ):
                    raise ValueError(
                        f"Joint {i + 1} target ({self.angles[i]:.1f} deg) is out of range"
                    )


class MoveJPoseCmd(
    MotionParamsMixin,
    msgspec.Struct,
    tag=int(CmdType.MOVEJ_POSE),
    array_like=True,
    frozen=True,
    gc=False,
):
    """MOVEJ_POSE: joint-space move to a Cartesian pose (IK at target)."""

    pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    r: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEJ_POSE requires either duration > 0 or speed > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEJ_POSE requires only one of duration or speed")
        _check_speed_accel(self.speed, self.accel)


class MoveLCmd(
    MotionParamsMixin,
    msgspec.Struct,
    tag=int(CmdType.MOVEL),
    array_like=True,
    frozen=True,
    gc=False,
):
    """MOVEL: linear Cartesian move to target pose."""

    pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")] = "WRF"
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    r: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    rel: bool = False

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEL requires either duration > 0 or speed > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEL requires only one of duration or speed")
        _check_speed_accel(self.speed, self.accel)


class MoveCCmd(
    MotionParamsMixin,
    msgspec.Struct,
    tag=int(CmdType.MOVEC),
    array_like=True,
    frozen=True,
    gc=False,
):
    """MOVEC: circular arc through current → via → end."""

    via: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    end: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")] = "WRF"
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] | None = None
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    r: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0

    def __post_init__(self) -> None:
        has_duration = self.duration is not None and self.duration > 0.0
        has_speed = self.speed is not None and self.speed > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEC requires either duration > 0 or speed > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEC requires only one of duration or speed")
        if self.speed is not None:
            _check_speed_accel(self.speed, self.accel)


class MoveSCmd(
    MotionParamsMixin,
    msgspec.Struct,
    tag=int(CmdType.MOVES),
    array_like=True,
    frozen=True,
    gc=False,
):
    """MOVES: cubic spline through waypoints."""

    waypoints: Annotated[list[list[float]], msgspec.Meta(min_length=2)]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")] = "WRF"
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] | None = None
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0

    def __post_init__(self) -> None:
        has_duration = self.duration is not None and self.duration > 0.0
        has_speed = self.speed is not None and self.speed > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVES requires either duration > 0 or speed > 0")
        if has_duration and has_speed:
            raise ValueError("MOVES requires only one of duration or speed")
        if self.speed is not None:
            _check_speed_accel(self.speed, self.accel)
        waypoints = self.waypoints
        for i in range(len(waypoints)):
            if len(waypoints[i]) != 6:
                raise ValueError(f"Waypoint {i} must have 6 values (x,y,z,rx,ry,rz)")


class MovePCmd(
    MotionParamsMixin,
    msgspec.Struct,
    tag=int(CmdType.MOVEP),
    array_like=True,
    frozen=True,
    gc=False,
):
    """MOVEP: process move — constant TCP speed with auto-blending at corners."""

    waypoints: Annotated[list[list[float]], msgspec.Meta(min_length=2)]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")] = "WRF"
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] | None = None
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0

    def __post_init__(self) -> None:
        has_duration = self.duration is not None and self.duration > 0.0
        has_speed = self.speed is not None and self.speed > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEP requires either duration > 0 or speed > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEP requires only one of duration or speed")
        waypoints = self.waypoints
        for i in range(len(waypoints)):
            if len(waypoints[i]) != 6:
                raise ValueError(f"Waypoint {i} must have 6 values (x,y,z,rx,ry,rz)")


class CheckpointCmd(
    msgspec.Struct,
    tag=int(CmdType.CHECKPOINT),
    array_like=True,
    frozen=True,
    gc=False,
):
    """CHECKPOINT: queue marker for progress tracking."""

    label: Annotated[str, msgspec.Meta(min_length=1, max_length=128)]


# -- Streaming commands: servo (position) --


class ServoJCmd(
    msgspec.Struct,
    tag=int(CmdType.SERVOJ),
    array_like=True,
    frozen=True,
    gc=False,
):
    """SERVOJ: streaming joint position target (degrees)."""

    target: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0


class ServoJPoseCmd(
    msgspec.Struct,
    tag=int(CmdType.SERVOJ_POSE),
    array_like=True,
    frozen=True,
    gc=False,
):
    """SERVOJ_POSE: streaming joint position target via Cartesian pose (IK)."""

    pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0


class ServoLCmd(
    msgspec.Struct,
    tag=int(CmdType.SERVOL),
    array_like=True,
    frozen=True,
    gc=False,
):
    """SERVOL: streaming linear Cartesian position target."""

    pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0


# -- Streaming commands: jog (velocity) --


class JogJCmd(
    msgspec.Struct,
    tag=int(CmdType.JOGJ),
    array_like=True,
    frozen=True,
    gc=False,
):
    """JOGJ: streaming joint velocity. Static 6-element signed speed fractions."""

    speeds: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(gt=0.0)]
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0

    def __post_init__(self) -> None:
        for i in range(6):
            if not (-1.0 <= self.speeds[i] <= 1.0):
                raise ValueError(
                    f"Speed[{i}]={self.speeds[i]} out of range [-1.0, 1.0]"
                )


class JogLCmd(
    msgspec.Struct,
    tag=int(CmdType.JOGL),
    array_like=True,
    frozen=True,
    gc=False,
):
    """JOGL: streaming Cartesian velocity. Static 6-element [vx,vy,vz,wx,wy,wz]."""

    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")]
    velocities: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(gt=0.0)]
    accel: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 1.0

    def __post_init__(self) -> None:
        for i in range(6):
            if not (-1.0 <= self.velocities[i] <= 1.0):
                raise ValueError(
                    f"Velocity[{i}]={self.velocities[i]} out of range [-1.0, 1.0]"
                )


class HomeCmd(
    msgspec.Struct, tag=int(CmdType.HOME), array_like=True, frozen=True, gc=False
):
    """HOME: [CmdType.HOME]"""

    pass


class ResumeCmd(
    msgspec.Struct, tag=int(CmdType.RESUME), array_like=True, frozen=True, gc=False
):
    """RESUME: [CmdType.RESUME] — re-enable the controller."""

    pass


class HaltCmd(
    msgspec.Struct, tag=int(CmdType.HALT), array_like=True, frozen=True, gc=False
):
    """HALT: [CmdType.HALT] — stop all motion and disable."""

    pass


class ResetCmd(
    msgspec.Struct, tag=int(CmdType.RESET), array_like=True, frozen=True, gc=False
):
    """RESET: [CmdType.RESET]"""

    pass


class ResetLoopStatsCmd(
    msgspec.Struct,
    tag=int(CmdType.RESET_LOOP_STATS),
    array_like=True,
    frozen=True,
    gc=False,
):
    """RESET_LOOP_STATS: [CmdType.RESET_LOOP_STATS]

    Reset timing statistics (min/max/overrun counts) without affecting controller state.
    """

    pass


class SetIOCmd(
    msgspec.Struct, tag=int(CmdType.SET_IO), array_like=True, frozen=True, gc=False
):
    """SET_IO: [CmdType.SET_IO, port_index, value]

    port_index: 0-7 (8-bit I/O port)
    value: 0 or 1
    """

    port_index: Annotated[int, msgspec.Meta(ge=0, le=7)]
    value: Annotated[int, msgspec.Meta(ge=0, le=1)]


class SetPortCmd(
    msgspec.Struct, tag=int(CmdType.SET_PORT), array_like=True, frozen=True, gc=False
):
    """SET_PORT: [CmdType.SET_PORT, port_str]"""

    port_str: Annotated[str, msgspec.Meta(min_length=1, max_length=256)]


class SimulatorCmd(
    msgspec.Struct, tag=int(CmdType.SIMULATOR), array_like=True, frozen=True, gc=False
):
    """SIMULATOR: [CmdType.SIMULATOR, on]"""

    on: bool


class DelayCmd(
    msgspec.Struct, tag=int(CmdType.DELAY), array_like=True, frozen=True, gc=False
):
    """DELAY: [CmdType.DELAY, seconds]"""

    seconds: Annotated[float, msgspec.Meta(gt=0.0)]


class SetToolCmd(
    msgspec.Struct, tag=int(CmdType.SET_TOOL), array_like=True, frozen=True, gc=False
):
    """SET_TOOL: [CmdType.SET_TOOL, tool_name]"""

    tool_name: Annotated[str, msgspec.Meta(min_length=1, max_length=64)]

    def __post_init__(self) -> None:
        name = self.tool_name.strip().upper()
        if name not in TOOL_CONFIGS:
            raise ValueError(f"Unknown tool '{name}'. Available: {list_tools()}")


class SetProfileCmd(
    msgspec.Struct, tag=int(CmdType.SET_PROFILE), array_like=True, frozen=True, gc=False
):
    """SET_PROFILE: [CmdType.SET_PROFILE, profile]"""

    profile: Annotated[str, msgspec.Meta(min_length=1, max_length=32)]


class PneumaticGripperCmd(
    msgspec.Struct,
    tag=int(CmdType.PNEUMATICGRIPPER),
    array_like=True,
    frozen=True,
    gc=False,
):
    """PNEUMATICGRIPPER: [CmdType.PNEUMATICGRIPPER, action, port]"""

    action: Annotated[str, msgspec.Meta(pattern=r"^(open|close)$")]
    port: Annotated[int, msgspec.Meta(ge=1, le=2)]  # Output port 1 or 2


class ElectricGripperCmd(
    msgspec.Struct,
    tag=int(CmdType.ELECTRICGRIPPER),
    array_like=True,
    frozen=True,
    gc=False,
):
    """ELECTRICGRIPPER: [CmdType.ELECTRICGRIPPER, action, position, speed, current]"""

    action: Annotated[str, msgspec.Meta(pattern=r"^(move|calibrate)$")]
    position: Annotated[float, msgspec.Meta(ge=0.0, le=1.0)] = 0.0
    speed: Annotated[float, msgspec.Meta(gt=0.0, le=1.0)] = 0.5
    current: Annotated[int, msgspec.Meta(ge=100, le=1000)] = 500


# Query commands (no params, just the tag)
class PingCmd(
    msgspec.Struct, tag=int(CmdType.PING), array_like=True, frozen=True, gc=False
):
    """PING: [CmdType.PING]"""

    pass


class GetStatusCmd(
    msgspec.Struct, tag=int(CmdType.GET_STATUS), array_like=True, frozen=True, gc=False
):
    """GET_STATUS: [CmdType.GET_STATUS]"""

    pass


class GetAnglesCmd(
    msgspec.Struct, tag=int(CmdType.GET_ANGLES), array_like=True, frozen=True, gc=False
):
    """GET_ANGLES: [CmdType.GET_ANGLES]"""

    pass


class GetPoseCmd(
    msgspec.Struct, tag=int(CmdType.GET_POSE), array_like=True, frozen=True, gc=False
):
    """GET_POSE: [CmdType.GET_POSE, frame]"""

    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")] | None = None


class GetIOCmd(
    msgspec.Struct, tag=int(CmdType.GET_IO), array_like=True, frozen=True, gc=False
):
    """GET_IO: [CmdType.GET_IO]"""

    pass


class GetGripperCmd(
    msgspec.Struct, tag=int(CmdType.GET_GRIPPER), array_like=True, frozen=True, gc=False
):
    """GET_GRIPPER: [CmdType.GET_GRIPPER]"""

    pass


class GetSpeedsCmd(
    msgspec.Struct, tag=int(CmdType.GET_SPEEDS), array_like=True, frozen=True, gc=False
):
    """GET_SPEEDS: [CmdType.GET_SPEEDS]"""

    pass


class GetToolCmd(
    msgspec.Struct, tag=int(CmdType.GET_TOOL), array_like=True, frozen=True, gc=False
):
    """GET_TOOL: [CmdType.GET_TOOL]"""

    pass


class GetQueueCmd(
    msgspec.Struct, tag=int(CmdType.GET_QUEUE), array_like=True, frozen=True, gc=False
):
    """GET_QUEUE: [CmdType.GET_QUEUE]"""

    pass


class GetCurrentActionCmd(
    msgspec.Struct,
    tag=int(CmdType.GET_CURRENT_ACTION),
    array_like=True,
    frozen=True,
    gc=False,
):
    """GET_CURRENT_ACTION: [CmdType.GET_CURRENT_ACTION]"""

    pass


class GetLoopStatsCmd(
    msgspec.Struct,
    tag=int(CmdType.GET_LOOP_STATS),
    array_like=True,
    frozen=True,
    gc=False,
):
    """GET_LOOP_STATS: [CmdType.GET_LOOP_STATS]"""

    pass


class GetProfileCmd(
    msgspec.Struct, tag=int(CmdType.GET_PROFILE), array_like=True, frozen=True, gc=False
):
    """GET_PROFILE: [CmdType.GET_PROFILE]"""

    pass


# =============================================================================
# Auto-generated Command union and STRUCT_TO_CMDTYPE
# =============================================================================


def _collect_command_structs() -> list[type]:
    """Collect all command struct classes from this module."""
    import sys

    module = sys.modules[__name__]
    structs = []
    for name, cls in vars(module).items():
        if not name.endswith("Cmd"):
            continue
        if not isinstance(cls, type):
            continue
        if not issubclass(cls, msgspec.Struct):
            continue
        # Check for tag in struct config
        config = getattr(cls, "__struct_config__", None)
        if config is not None and config.tag is not None:
            structs.append(cls)
    return structs


def _build_struct_to_cmdtype(structs: list[type]) -> dict[type, CmdType]:
    """Auto-generate struct -> CmdType mapping from tagged structs."""
    mapping: dict[type, CmdType] = {}
    for struct_cls in structs:
        config = getattr(struct_cls, "__struct_config__", None)
        if config is None:
            continue
        tag = getattr(config, "tag", None)
        if tag is None:
            continue
        try:
            cmd_type = CmdType(tag)
            mapping[struct_cls] = cmd_type
        except ValueError:
            pass  # Not a valid CmdType tag
    return mapping


# Build at import time
_COMMAND_STRUCTS = _collect_command_structs()
STRUCT_TO_CMDTYPE: dict[type, CmdType] = _build_struct_to_cmdtype(_COMMAND_STRUCTS)

# Build Command union dynamically from collected structs
Command: TypeAlias = Union[tuple(_COMMAND_STRUCTS)]  # type: ignore[valid-type]

# Module-level decoder for single-pass command decode
_command_decoder = msgspec.msgpack.Decoder(Command)


def decode_command(data: bytes) -> Command:
    """Decode raw bytes to typed command struct.

    Args:
        data: Raw msgpack-encoded command bytes

    Returns:
        Typed command struct

    Raises:
        msgspec.ValidationError: If data is invalid or doesn't match any command type
    """
    return _command_decoder.decode(data)


def encode_command(cmd: Command) -> bytes:
    """Encode a typed command struct to bytes.

    Args:
        cmd: Typed command struct

    Returns:
        Raw msgpack-encoded bytes
    """
    return _encoder.encode(cmd)


def encode_command_into(cmd: Command, buf: bytearray) -> bytearray:
    """Encode a typed command struct into a pre-allocated bytearray.

    The buffer is resized to exactly fit the encoded output.
    Reuses the same bytearray object across calls to avoid per-send
    ``bytes`` allocations on fire-and-forget paths.

    Args:
        cmd: Typed command struct
        buf: Pre-allocated bytearray (will be resized in-place)

    Returns:
        The same *buf* object, now containing the encoded bytes.
    """
    _encoder.encode_into(cmd, buf)
    return buf


# =============================================================================
# Response Structs - Tagged Union for single-pass decode
# Wire format: [MsgType.RESPONSE, QueryType.XXX, ...fields]
# =============================================================================


class StatusResultStruct(
    msgspec.Struct, tag=int(QueryType.STATUS), array_like=True, frozen=True, gc=False
):
    """Aggregate robot status."""

    pose: list[float]
    angles: list[float]
    speeds: list[float]
    io: list[int]
    gripper: list[int]


class LoopStatsResultStruct(
    msgspec.Struct,
    tag=int(QueryType.LOOP_STATS),
    array_like=True,
    frozen=True,
    gc=False,
):
    """Control loop runtime metrics."""

    target_hz: float
    loop_count: int
    overrun_count: int
    mean_period_s: float
    std_period_s: float
    min_period_s: float
    max_period_s: float
    p95_period_s: float
    p99_period_s: float
    mean_hz: float


class ToolResultStruct(
    msgspec.Struct, tag=int(QueryType.TOOL), array_like=True, frozen=True, gc=False
):
    """Tool configuration."""

    tool: str
    available: list[str]


class CurrentActionResultStruct(
    msgspec.Struct,
    tag=int(QueryType.CURRENT_ACTION),
    array_like=True,
    frozen=True,
    gc=False,
):
    """Current executing action."""

    current: str
    state: str
    next: str


class PingResultStruct(
    msgspec.Struct, tag=int(QueryType.PING), array_like=True, frozen=True, gc=False
):
    """Ping response with serial connectivity status."""

    serial_connected: int  # 0 or 1


class AnglesResultStruct(
    msgspec.Struct, tag=int(QueryType.ANGLES), array_like=True, frozen=True, gc=False
):
    """Joint angles response."""

    angles: list[float]


class PoseResultStruct(
    msgspec.Struct, tag=int(QueryType.POSE), array_like=True, frozen=True, gc=False
):
    """Pose response."""

    pose: list[float]


class IOResultStruct(
    msgspec.Struct, tag=int(QueryType.IO), array_like=True, frozen=True, gc=False
):
    """I/O status response."""

    io: list[int]


class GripperResultStruct(
    msgspec.Struct, tag=int(QueryType.GRIPPER), array_like=True, frozen=True, gc=False
):
    """Gripper status response."""

    gripper: list[int]


class SpeedsResultStruct(
    msgspec.Struct, tag=int(QueryType.SPEEDS), array_like=True, frozen=True, gc=False
):
    """Speeds response."""

    speeds: list[float]


class ProfileResultStruct(
    msgspec.Struct, tag=int(QueryType.PROFILE), array_like=True, frozen=True, gc=False
):
    """Motion profile response."""

    profile: str


class QueueResultStruct(
    msgspec.Struct, tag=int(QueryType.QUEUE), array_like=True, frozen=True, gc=False
):
    """Queue status response."""

    queue: list


# Tagged Union for responses
Response = (
    StatusResultStruct
    | LoopStatsResultStruct
    | ToolResultStruct
    | CurrentActionResultStruct
    | PingResultStruct
    | AnglesResultStruct
    | PoseResultStruct
    | IOResultStruct
    | GripperResultStruct
    | SpeedsResultStruct
    | ProfileResultStruct
    | QueueResultStruct
)


# Typed message classes for parsed responses


class OkMsg(
    msgspec.Struct,
    tag=int(MsgType.OK),
    array_like=True,
    frozen=True,
    gc=False,
):
    """OK response, optionally carrying a command index for queued commands."""

    index: int | None = None


class ErrorMsg(
    msgspec.Struct,
    tag=int(MsgType.ERROR),
    array_like=True,
    frozen=True,
    gc=False,
):
    """Error response carrying a RobotError wire representation."""

    message: list


class ResponseMsg(
    msgspec.Struct,
    tag=int(MsgType.RESPONSE),
    array_like=True,
    frozen=True,
    gc=False,
):
    """Query response with type and value."""

    query_type: QueryType
    value: Any


# Tagged union for single-pass decode of server replies
Message: TypeAlias = Union[OkMsg, ErrorMsg, ResponseMsg]
_message_decoder = msgspec.msgpack.Decoder(Message)


def decode_message(data: bytes) -> Message:
    """Decode raw msgpack bytes into a typed Message.

    Raises:
        msgspec.ValidationError: If data doesn't match any message type.
    """
    return _message_decoder.decode(data)


# =============================================================================
# Generic msgpack encode/decode functions
# =============================================================================


def encode(obj: object) -> bytes:
    """Encode any msgspec struct or Python object to bytes with numpy support."""
    return _encoder.encode(obj)


def decode(data: bytes) -> object:
    """Decode msgpack bytes to a Python object."""
    return _decoder.decode(data)


# Pre-packed common responses (avoid repeated packing)
OK_PACKED = _encoder.encode(OkMsg())

# Cache for common error responses (3x faster for repeated errors)
_UNKNOWN_CMD_ERROR = make_error(ErrorCode.COMM_UNKNOWN_COMMAND)
_QUEUE_FULL_ERROR = make_error(ErrorCode.COMM_QUEUE_FULL)
_ERROR_CACHE: dict[int, bytes] = {
    ErrorCode.COMM_UNKNOWN_COMMAND: _encoder.encode(
        ErrorMsg(_UNKNOWN_CMD_ERROR.to_wire())
    ),
    ErrorCode.COMM_QUEUE_FULL: _encoder.encode(ErrorMsg(_QUEUE_FULL_ERROR.to_wire())),
}


def pack_ok() -> bytes:
    """Pack an OK response (no command index)."""
    return OK_PACKED


def pack_ok_index(index: int) -> bytes:
    """Pack an OK response with a command index for queued commands."""
    return _encoder.encode(OkMsg(index=index))


def pack_error(error: RobotError) -> bytes:
    """Pack an error response: [ERROR, [command_index, code, title, cause, effect, remedy]].

    Common errors are cached by ErrorCode for performance.
    """
    cached = _ERROR_CACHE.get(error.code)
    if cached is not None:
        return cached
    return _encoder.encode(ErrorMsg(error.to_wire()))


def pack_response(query_type: QueryType, value: Any) -> bytes:
    """Pack a query response: [RESPONSE, query_type, value]."""
    return _encoder.encode(ResponseMsg(query_type, value))


def pack_status(
    pose: np.ndarray,
    angles: np.ndarray,
    speeds: np.ndarray,
    io: np.ndarray,
    gripper: np.ndarray,
    action_current: str,
    action_state: str,
    joint_en: np.ndarray,
    cart_en_wrf: np.ndarray,
    cart_en_trf: np.ndarray,
    executing_index: int = -1,
    completed_index: int = -1,
    last_checkpoint: str = "",
    error: RobotError | None = None,
    queued_segments: int = 0,
    queued_duration: float = 0.0,
) -> bytes:
    """Pack a status broadcast message.

    Uses ormsgpack with OPT_SERIALIZE_NUMPY for ~80x fewer allocations
    compared to msgspec with enc_hook (reads numpy buffers directly via C API).
    """
    return ormsgpack.packb(
        (
            MsgType.STATUS,
            pose,
            angles,
            speeds,
            io,
            gripper,
            action_current,
            action_state,
            joint_en,
            cart_en_wrf,
            cart_en_trf,
            executing_index,
            completed_index,
            last_checkpoint,
            error.to_wire() if error is not None else None,
            queued_segments,
            queued_duration,
        ),
        option=ormsgpack.OPT_SERIALIZE_NUMPY,
    )


# =============================================================================
# Status Buffer (for zero-allocation status parsing)
# =============================================================================


@dataclass
class StatusBuffer:
    """Preallocated buffer for zero-allocation status parsing.

    All numeric arrays are numpy for cache-friendly access and potential numba use.
    Use decode_status_bin_into() to fill this buffer without allocating new objects.
    """

    pose: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float64))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    speeds: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    io: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.int32))
    gripper: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    joint_en: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.int32))
    cart_en_wrf: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.int32))
    cart_en_trf: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.int32))
    action_current: str = ""
    action_state: str = ""
    executing_index: int = -1
    completed_index: int = -1
    last_checkpoint: str = ""
    error: RobotError | None = None
    queued_segments: int = 0
    queued_duration: float = 0.0

    @property
    def cart_en(self) -> dict[str, np.ndarray]:
        """Frame name → (12,) int32 Cartesian enable envelope."""
        return {"WRF": self.cart_en_wrf, "TRF": self.cart_en_trf}

    def copy(self) -> "StatusBuffer":
        """Return a deep copy with all arrays copied."""
        return StatusBuffer(
            pose=self.pose.copy(),
            angles=self.angles.copy(),
            speeds=self.speeds.copy(),
            io=self.io.copy(),
            gripper=self.gripper.copy(),
            joint_en=self.joint_en.copy(),
            cart_en_wrf=self.cart_en_wrf.copy(),
            cart_en_trf=self.cart_en_trf.copy(),
            action_current=self.action_current,
            action_state=self.action_state,
            executing_index=self.executing_index,
            completed_index=self.completed_index,
            last_checkpoint=self.last_checkpoint,
            error=self.error,
            queued_segments=self.queued_segments,
            queued_duration=self.queued_duration,
        )


def decode_status_bin_into(data: bytes, buf: StatusBuffer) -> bool:
    """Zero-allocation decode of STATUS message into preallocated buffer.

    Message format: [MsgType.STATUS, pose, angles, speeds, io, gripper,
                     action_current, action_state, joint_en, cart_en_wrf, cart_en_trf,
                     executing_index, completed_index, last_checkpoint,
                     error, queued_segments, queued_duration]

    Args:
        data: Raw msgpack bytes
        buf: Preallocated StatusBuffer to fill

    Returns:
        True if valid STATUS message, False otherwise.
    """
    try:
        msg = _decoder.decode(data)
        if (
            not isinstance(msg, (list, tuple))
            or len(msg) < 17
            or msg[0] != MsgType.STATUS
        ):
            return False

        buf.pose[:] = msg[1]
        buf.angles[:] = msg[2]
        buf.speeds[:] = msg[3]
        buf.io[:] = msg[4]
        buf.gripper[:] = msg[5]
        buf.action_current = msg[6]
        buf.action_state = msg[7]
        buf.joint_en[:] = msg[8]
        buf.cart_en_wrf[:] = msg[9]
        buf.cart_en_trf[:] = msg[10]
        buf.executing_index = msg[11]
        buf.completed_index = msg[12]
        buf.last_checkpoint = msg[13]
        raw_error = msg[14]
        buf.error = RobotError.from_wire(raw_error) if raw_error is not None else None
        buf.queued_segments = msg[15]
        buf.queued_duration = msg[16]

        return True
    except Exception:
        return False


# =============================================================================
# Binary serial frame packing/unpacking (firmware communication)
# =============================================================================


class CommandCode(IntEnum):
    """Unified command codes for firmware interface."""

    HOME = 100
    ENABLE = 101
    DISABLE = 102
    JOG = 123
    MOVE = 156
    IDLE = 255


START = b"\xff\xff\xff"
END = b"\x01\x02"
PAYLOAD_LEN = 52  # matches existing firmware expectation


@njit(cache=True)
def split_to_3_bytes(n: int) -> tuple[int, int, int]:
    """
    Convert int to signed 24-bit big-endian (two's complement) encoded bytes (b0,b1,b2).
    """
    n24 = n & 0xFFFFFF
    return ((n24 >> 16) & 0xFF, (n24 >> 8) & 0xFF, n24 & 0xFF)


@njit(cache=True)
def fuse_3_bytes(b0: int, b1: int, b2: int) -> int:
    """
    Convert 3 bytes (big-endian) into a signed 24-bit integer.
    """
    val = (b0 << 16) | (b1 << 8) | b2
    return val - 0x1000000 if (val & 0x800000) else val


@njit(cache=True)
def fuse_2_bytes(b0: int, b1: int) -> int:
    """
    Convert 2 bytes (big-endian) into a signed 16-bit integer.
    """
    val = (b0 << 8) | b1
    return val - 0x10000 if (val & 0x8000) else val


@njit(cache=True)
def _pack_positions(
    out: np.ndarray | memoryview, values: np.ndarray, offset: int
) -> None:
    for i in range(6):
        v = int(values[i]) & 0xFFFFFF
        j = offset + i * 3
        out[j] = (v >> 16) & 0xFF
        out[j + 1] = (v >> 8) & 0xFF
        out[j + 2] = v & 0xFF


@njit(cache=True)
def _unpack_positions(data: np.ndarray | memoryview, out: np.ndarray) -> None:
    for i in range(6):
        j = i * 3
        val = (int(data[j]) << 16) | (int(data[j + 1]) << 8) | int(data[j + 2])
        if val >= 0x800000:
            val -= 0x1000000
        out[i] = val


@njit(cache=True)
def _pack_bitfield(arr: np.ndarray) -> int:
    """Pack 8-element array into a single byte (MSB first)."""
    return (
        (int(arr[0] != 0) << 7)
        | (int(arr[1] != 0) << 6)
        | (int(arr[2] != 0) << 5)
        | (int(arr[3] != 0) << 4)
        | (int(arr[4] != 0) << 3)
        | (int(arr[5] != 0) << 2)
        | (int(arr[6] != 0) << 1)
        | int(arr[7] != 0)
    )


@njit(cache=True)
def _unpack_bitfield(byte_val: int, out: np.ndarray) -> None:
    """Unpack a byte into 8 bits (MSB first) into output array."""
    out[0] = (byte_val >> 7) & 1
    out[1] = (byte_val >> 6) & 1
    out[2] = (byte_val >> 5) & 1
    out[3] = (byte_val >> 4) & 1
    out[4] = (byte_val >> 3) & 1
    out[5] = (byte_val >> 2) & 1
    out[6] = (byte_val >> 1) & 1
    out[7] = byte_val & 1


@njit(cache=True)
def pack_tx_frame_into(
    out: memoryview,
    position_out: np.ndarray,
    speed_out: np.ndarray,
    command_code: int,
    affected_joint_out: np.ndarray,
    inout_out: np.ndarray,
    timeout_out: int,
    gripper_data_out: np.ndarray,
) -> None:
    """
    Pack a full TX frame into the provided memoryview without allocations.

    Expects 'out' to be a writable buffer of length >= 56 bytes.
    """
    # Header: 0xFF 0xFF 0xFF + payload length
    out[0] = 0xFF
    out[1] = 0xFF
    out[2] = 0xFF
    out[3] = 52

    # Positions and speeds: JIT-compiled packing
    _pack_positions(out, position_out, 4)
    _pack_positions(out, speed_out, 22)

    # Command
    out[40] = command_code

    # Bitfields
    out[41] = _pack_bitfield(affected_joint_out)
    out[42] = _pack_bitfield(inout_out)

    # Timeout
    out[43] = int(timeout_out) & 0xFF

    # Gripper: position, speed, current as 2 bytes each (big-endian)
    g0 = int(gripper_data_out[0]) & 0xFFFF
    g1 = int(gripper_data_out[1]) & 0xFFFF
    g2 = int(gripper_data_out[2]) & 0xFFFF
    out[44] = (g0 >> 8) & 0xFF
    out[45] = g0 & 0xFF
    out[46] = (g1 >> 8) & 0xFF
    out[47] = g1 & 0xFF
    out[48] = (g2 >> 8) & 0xFF
    out[49] = g2 & 0xFF

    # Gripper command, mode, id
    out[50] = int(gripper_data_out[3]) & 0xFF
    out[51] = int(gripper_data_out[4]) & 0xFF
    out[52] = int(gripper_data_out[5]) & 0xFF

    # CRC placeholder byte (0xE4) — fixed value, not computed
    out[53] = 228

    # End bytes
    out[54] = 0x01
    out[55] = 0x02


@njit(cache=True)
def unpack_rx_frame_into(
    data: memoryview,
    pos_out: np.ndarray,
    spd_out: np.ndarray,
    homed_out: np.ndarray,
    io_out: np.ndarray,
    temp_out: np.ndarray,
    poserr_out: np.ndarray,
    timing_out: np.ndarray,
    grip_out: np.ndarray,
) -> bool:
    """
    Zero-allocation decode of a 52-byte RX frame payload (memoryview) directly into numpy arrays.
    Expects:
      - pos_out, spd_out: shape (6,), dtype=int32
      - homed_out, io_out, temp_out, poserr_out: shape (8,), dtype=uint8
      - timing_out: shape (1,), dtype=int32
      - grip_out: shape (6,), dtype=int32 [device_id, pos, spd, cur, status, obj]
    """
    if len(data) < 52:
        return False

    _unpack_positions(data, pos_out)
    _unpack_positions(data[18:], spd_out)

    _unpack_bitfield(int(data[36]), homed_out)
    _unpack_bitfield(int(data[37]), io_out)
    _unpack_bitfield(int(data[38]), temp_out)
    _unpack_bitfield(int(data[39]), poserr_out)

    timing_out[0] = fuse_3_bytes(0, int(data[40]), int(data[41]))

    device_id = int(data[44])
    grip_pos = fuse_2_bytes(int(data[45]), int(data[46]))
    grip_spd = fuse_2_bytes(int(data[47]), int(data[48]))
    grip_cur = fuse_2_bytes(int(data[49]), int(data[50]))
    status_byte = int(data[51])

    obj_detection = ((status_byte >> 3) & 1) << 1 | ((status_byte >> 2) & 1)

    grip_out[0] = device_id
    grip_out[1] = grip_pos
    grip_out[2] = grip_spd
    grip_out[3] = grip_cur
    grip_out[4] = status_byte
    grip_out[5] = obj_detection

    return True


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "MsgType",
    "QueryType",
    "CmdType",
    "CommandCode",
    # Command structs — motion (queued)
    "MoveJCmd",
    "MoveJPoseCmd",
    "MoveLCmd",
    "MoveCCmd",
    "MoveSCmd",
    "MovePCmd",
    "HomeCmd",
    "CheckpointCmd",
    # Command structs — streaming (servo/jog)
    "ServoJCmd",
    "ServoJPoseCmd",
    "ServoLCmd",
    "JogJCmd",
    "JogLCmd",
    # Command structs — system/query/other
    "ResumeCmd",
    "HaltCmd",
    "ResetCmd",
    "ResetLoopStatsCmd",
    "SetIOCmd",
    "SetPortCmd",
    "SimulatorCmd",
    "DelayCmd",
    "SetToolCmd",
    "SetProfileCmd",
    "PneumaticGripperCmd",
    "ElectricGripperCmd",
    "PingCmd",
    "GetStatusCmd",
    "GetAnglesCmd",
    "GetPoseCmd",
    "GetIOCmd",
    "GetGripperCmd",
    "GetSpeedsCmd",
    "GetToolCmd",
    "GetQueueCmd",
    "GetCurrentActionCmd",
    "GetLoopStatsCmd",
    "GetProfileCmd",
    "Command",
    # Mixin
    "MotionParamsMixin",
    # Response structs
    "StatusResultStruct",
    "LoopStatsResultStruct",
    "ToolResultStruct",
    "CurrentActionResultStruct",
    "PingResultStruct",
    "AnglesResultStruct",
    "PoseResultStruct",
    "IOResultStruct",
    "GripperResultStruct",
    "SpeedsResultStruct",
    "ProfileResultStruct",
    "QueueResultStruct",
    "Response",
    # Message types
    "OkMsg",
    "ErrorMsg",
    "ResponseMsg",
    "Message",
    # Encode/decode
    "decode_command",
    "encode_command",
    "STRUCT_TO_CMDTYPE",
    "decode_message",
    "encode",
    "decode",
    "pack_ok",
    "pack_ok_index",
    "pack_error",
    "pack_response",
    "pack_status",
    # Status buffer
    "StatusBuffer",
    "decode_status_bin_into",
    # Binary frame protocol
    "START",
    "END",
    "PAYLOAD_LEN",
    "split_to_3_bytes",
    "fuse_3_bytes",
    "fuse_2_bytes",
    "pack_tx_frame_into",
    "unpack_rx_frame_into",
]
