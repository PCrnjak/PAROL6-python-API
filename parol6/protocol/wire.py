"""
Wire protocol helpers for UDP encoding/decoding.

This module centralizes encoding of command strings and decoding of common
response payloads used by the headless controller.
"""

import logging
from collections.abc import Sequence

# Centralized binary wire protocol helpers (pack/unpack + codes)
from enum import IntEnum
from typing import Literal, cast

import numpy as np

from .types import Axis, Frame, StatusAggregate

logger = logging.getLogger(__name__)

# Precomputed bit-unpack lookup table for 0..255 (MSB..LSB)
# Using NumPy ensures fast vectorized selection without per-call allocations.
_BIT_UNPACK = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="big")
START = b"\xff\xff\xff"
END = b"\x01\x02"
PAYLOAD_LEN = 52  # matches existing firmware expectation

__all__ = [
    "CommandCode",
    "pack_tx_frame_into",
    "unpack_rx_frame_into",
    "encode_move_joint",
    "encode_move_pose",
    "encode_move_cartesian",
    "encode_move_cartesian_rel_trf",
    "encode_jog_joint",
    "encode_cart_jog",
    "encode_gcode",
    "encode_gcode_program_inline",
    "decode_simple",
    "decode_status",
    "split_to_3_bytes",
    "fuse_3_bytes",
    "fuse_2_bytes",
]


class CommandCode(IntEnum):
    """Unified command codes for firmware interface."""

    HOME = 100
    ENABLE = 101
    DISABLE = 102
    JOG = 123
    MOVE = 156
    IDLE = 255


def split_bitfield(byte_val: int) -> list[int]:
    """Split an 8-bit integer into a big-endian list of bits (MSB..LSB)."""
    return [(byte_val >> i) & 1 for i in range(7, -1, -1)]


def fuse_bitfield_2_bytearray(bits: list[int] | Sequence[int]) -> bytes:
    """
    Fuse a big-endian list of 8 bits (MSB..LSB) into a single byte.
    Any truthy value is treated as 1.
    """
    number = 0
    for b in bits[:8]:
        number = (number << 1) | (1 if b else 0)
    return bytes([number])


def split_to_3_bytes(n: int) -> tuple[int, int, int]:
    """
    Convert int to signed 24-bit big-endian (two's complement) encoded bytes (b0,b1,b2).
    """
    n24 = n & 0xFFFFFF
    return ((n24 >> 16) & 0xFF, (n24 >> 8) & 0xFF, n24 & 0xFF)


def fuse_3_bytes(b0: int, b1: int, b2: int) -> int:
    """
    Convert 3 bytes (big-endian) into a signed 24-bit integer.
    """
    val = (b0 << 16) | (b1 << 8) | b2
    return val - 0x1000000 if (val & 0x800000) else val


def fuse_2_bytes(b0: int, b1: int) -> int:
    """
    Convert 2 bytes (big-endian) into a signed 16-bit integer.
    """
    val = (b0 << 8) | b1
    return val - 0x10000 if (val & 0x8000) else val


def _get_array_value(arr: np.ndarray | memoryview, index: int, default: int = 0) -> int:
    """
    Safely get value from array-like object with bounds checking.
    Optimized for zero-copy access when possible.
    """
    try:
        if index < len(arr):
            return int(arr[index])
        return default
    except (IndexError, TypeError):
        return default


def pack_tx_frame_into(
    out: memoryview,
    position_out: np.ndarray,
    speed_out: np.ndarray,
    command_code: int | CommandCode,
    affected_joint_out: np.ndarray,
    inout_out: np.ndarray,
    timeout_out: int,
    gripper_data_out: np.ndarray,
) -> None:
    """
    Pack a full TX frame into the provided memoryview without allocations.

    Expects 'out' to be a writable buffer of length >= 56 bytes:
      - 3 start bytes + 1 length byte + 52-byte payload

    Layout of the 52-byte payload:
      - 6x position (3 bytes each) = 18
      - 6x speed (3 bytes each)    = 18
      - 1 byte command
      - 1 byte affected joint bitfield
      - 1 byte in/out bitfield
      - 1 byte timeout
      - 2 bytes reserved (legacy)
      - 2 bytes gripper position
      - 2 bytes gripper speed
      - 2 bytes gripper current
      - 1 byte gripper command
      - 1 byte gripper mode
      - 1 byte gripper id
      - 1 byte CRC (placeholder 228)
      - 2 bytes end markers (0x01, 0x02)
    """
    # Header
    out[0:3] = START
    out[3] = PAYLOAD_LEN
    offset = 4

    # Positions: 6 * 3 bytes
    for i in range(6):
        val = _get_array_value(position_out, i, 0)
        b0, b1, b2 = split_to_3_bytes(val)
        out[offset] = b0
        out[offset + 1] = b1
        out[offset + 2] = b2
        offset += 3

    # Speeds: 6 * 3 bytes
    for i in range(6):
        val = _get_array_value(speed_out, i, 0)
        b0, b1, b2 = split_to_3_bytes(val)
        out[offset] = b0
        out[offset + 1] = b1
        out[offset + 2] = b2
        offset += 3

    # Command
    out[offset] = int(command_code)
    offset += 1

    # Affected joints as bitfield byte
    bitfield_val = 0
    for i in range(8):
        if _get_array_value(affected_joint_out, i, 0):
            bitfield_val |= 1 << (7 - i)
    out[offset] = bitfield_val
    offset += 1

    # In/Out as bitfield byte
    bitfield_val = 0
    for i in range(8):
        if _get_array_value(inout_out, i, 0):
            bitfield_val |= 1 << (7 - i)
    out[offset] = bitfield_val
    offset += 1

    # Timeout
    out[offset] = int(timeout_out) & 0xFF
    offset += 1

    # Gripper: position, speed, current as 2 bytes each (big-endian)
    for idx in range(3):
        v = _get_array_value(gripper_data_out, idx, 0) & 0xFFFF
        out[offset] = (v >> 8) & 0xFF
        out[offset + 1] = v & 0xFF
        offset += 2

    # Gripper command, mode, id
    out[offset] = _get_array_value(gripper_data_out, 3, 0) & 0xFF
    out[offset + 1] = _get_array_value(gripper_data_out, 4, 0) & 0xFF
    out[offset + 2] = _get_array_value(gripper_data_out, 5, 0) & 0xFF
    offset += 3

    # CRC (placeholder - unchanged from legacy)
    out[offset] = 228
    offset += 1

    # End bytes
    out[offset] = 0x01
    out[offset + 1] = 0x02


def unpack_rx_frame_into(
    data: memoryview,
    *,
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
    try:
        if len(data) < 52:
            logger.warning(f"unpack_rx_frame_into: payload too short ({len(data)} bytes)")
            return False

        mv = memoryview(data)

        # Positions (0..17) and speeds (18..35), 3 bytes per value, big-endian signed 24-bit
        b = np.frombuffer(mv[:36], dtype=np.uint8).reshape(12, 3)
        pos3 = b[:6]
        spd3 = b[6:]

        pos = (
            (pos3[:, 0].astype(np.int32) << 16)
            | (pos3[:, 1].astype(np.int32) << 8)
            | pos3[:, 2].astype(np.int32)
        )
        spd = (
            (spd3[:, 0].astype(np.int32) << 16)
            | (spd3[:, 1].astype(np.int32) << 8)
            | spd3[:, 2].astype(np.int32)
        )

        # Sign-correct 24-bit to int32
        pos[pos >= (1 << 23)] -= 1 << 24
        spd[spd >= (1 << 23)] -= 1 << 24

        np.copyto(pos_out, pos, casting="no")
        np.copyto(spd_out, spd, casting="no")

        homed_byte = mv[36]
        io_byte = mv[37]
        temp_err_byte = mv[38]
        pos_err_byte = mv[39]
        timing_b0 = mv[40]
        timing_b1 = mv[41]
        # indices 42..43 exist in some variants (timeout/xtr), legacy code ignores

        device_id = mv[44]
        grip_pos_b0, grip_pos_b1 = mv[45], mv[46]
        grip_spd_b0, grip_spd_b1 = mv[47], mv[48]
        grip_cur_b0, grip_cur_b1 = mv[49], mv[50]
        status_byte = mv[51]

        # Bitfields (MSB..LSB) via LUT (no per-call Python loops)
        homed_out[:] = _BIT_UNPACK[int(homed_byte)]
        io_out[:] = _BIT_UNPACK[int(io_byte)]
        temp_out[:] = _BIT_UNPACK[int(temp_err_byte)]
        poserr_out[:] = _BIT_UNPACK[int(pos_err_byte)]

        # Timing (legacy semantics: fuse_3_bytes(0, b0, b1))
        timing_val = fuse_3_bytes(0, int(timing_b0), int(timing_b1))
        timing_out[0] = int(timing_val)

        # Gripper values
        grip_pos = fuse_2_bytes(int(grip_pos_b0), int(grip_pos_b1))
        grip_spd = fuse_2_bytes(int(grip_spd_b0), int(grip_spd_b1))
        grip_cur = fuse_2_bytes(int(grip_cur_b0), int(grip_cur_b1))

        sbits = _BIT_UNPACK[int(status_byte)]
        obj_detection = (int(sbits[4]) << 1) | int(sbits[5])

        grip_out[0] = int(device_id)
        grip_out[1] = int(grip_pos)
        grip_out[2] = int(grip_spd)
        grip_out[3] = int(grip_cur)
        grip_out[4] = int(status_byte)
        grip_out[5] = int(obj_detection)

        return True
    except Exception as e:
        logger.error(f"unpack_rx_frame_into: exception {e}")
        return False


# =========================
# Encoding helpers
# =========================


def _opt(value: object | None, none_token: str = "NONE") -> str:
    """Format an optional value as a string, using none_token for None."""
    return none_token if value is None else str(value)


def encode_move_joint(
    angles: Sequence[float],
    duration: float | None,
    speed: float | None,
) -> str:
    """
    MOVEJOINT|j1|j2|j3|j4|j5|j6|DUR|SPD
    Use "NONE" for omitted duration/speed.
    Note: Validation (requiring one of duration/speed) is left to caller.
    """
    angles_str = "|".join(str(a) for a in angles)
    return f"MOVEJOINT|{angles_str}|{_opt(duration)}|{_opt(speed)}"


def encode_move_pose(
    pose: Sequence[float],
    duration: float | None,
    speed: float | None,
) -> str:
    """
    MOVEPOSE|x|y|z|rx|ry|rz|DUR|SPD
    Use "NONE" for omitted duration/speed.
    """
    pose_str = "|".join(str(v) for v in pose)
    return f"MOVEPOSE|{pose_str}|{_opt(duration)}|{_opt(speed)}"


def encode_move_cartesian(
    pose: Sequence[float],
    duration: float | None,
    speed: float | None,
) -> str:
    """
    MOVECART|x|y|z|rx|ry|rz|DUR|SPD
    Use "NONE" for omitted duration/speed.
    """
    pose_str = "|".join(str(v) for v in pose)
    return f"MOVECART|{pose_str}|{_opt(duration)}|{_opt(speed)}"


def encode_move_cartesian_rel_trf(
    deltas: Sequence[float],  # [dx, dy, dz, rx, ry, rz] in mm/deg relative to TRF
    duration: float | None,
    speed: float | None,
    accel: int | None,
    profile: str | None,
    tracking: str | None,
) -> str:
    """
    MOVECARTRELTRF|dx|dy|dz|rx|ry|rz|DUR|SPD|ACC|PROFILE|TRACKING
    Non-required fields should use "NONE".
    """
    delta_str = "|".join(str(v) for v in deltas)
    prof_str = (profile or "NONE").upper()
    track_str = (tracking or "NONE").upper()
    return (
        f"MOVECARTRELTRF|{delta_str}|{_opt(duration)}|{_opt(speed)}|"
        f"{_opt(accel)}|{prof_str}|{track_str}"
    )


def encode_jog_joint(
    joint_index: int,
    speed_percentage: int,
    duration: float | None,
    distance_deg: float | None,
) -> str:
    """
    JOG|joint_index|speed_pct|DUR|DIST
    duration and distance_deg are optional; use "NONE" if omitted.
    """
    return f"JOG|{joint_index}|{speed_percentage}|{_opt(duration)}|{_opt(distance_deg)}"


def encode_cart_jog(
    frame: Frame,
    axis: Axis,
    speed_percentage: int,
    duration: float,
) -> str:
    """
    CARTJOG|FRAME|AXIS|speed_pct|duration
    """
    return f"CARTJOG|{frame}|{axis}|{speed_percentage}|{duration}"


def encode_gcode(line: str) -> str:
    """
    GCODE|<single_line>
    The caller should ensure that '|' is not present in the line.
    """
    return f"GCODE|{line}"


def encode_gcode_program_inline(lines: Sequence[str]) -> str:
    """
    GCODE_PROGRAM|INLINE|line1;line2;...
    The caller should ensure that '|' is not present inside any line.
    """
    program_str = ";".join(lines)
    return f"GCODE_PROGRAM|INLINE|{program_str}"


# =========================
# Decoding helpers
# =========================
def decode_simple(
    resp: str, expected_prefix: Literal["ANGLES", "IO", "GRIPPER", "SPEEDS", "POSE"]
) -> list[float] | list[int] | None:
    """
    Decode simple prefixed payloads like:
      ANGLES|a0,a1,a2,a3,a4,a5
      IO|in1,in2,out1,out2,estop
      GRIPPER|id,pos,spd,cur,status,obj
      SPEEDS|s0,s1,s2,s3,s4,s5
      POSE|p0,p1,...,p15

    Returns list[float] or list[int] depending on the expected_prefix.
    """
    if not resp:
        logger.debug(f"decode_simple: Empty response for expected prefix '{expected_prefix}'")
        return None
    parts = resp.strip().split("|", 1)
    if len(parts) != 2 or parts[0] != expected_prefix:
        logger.warning(
            f"decode_simple: Invalid response format. Expected '{expected_prefix}|...' but got '{resp}'"
        )
        return None
    payload = parts[1]
    tokens = [t for t in payload.split(",") if t != ""]

    # IO and GRIPPER are integer-based; others default to float
    if expected_prefix in ("IO", "GRIPPER"):
        try:
            return [int(t) for t in tokens]
        except ValueError as e:
            logger.error(
                f"decode_simple: Failed to parse integers for {expected_prefix}. Payload: '{payload}', Error: {e}"
            )
            return None
    else:
        try:
            return [float(t) for t in tokens]
        except ValueError as e:
            logger.error(
                f"decode_simple: Failed to parse floats for {expected_prefix}. Payload: '{payload}', Error: {e}"
            )
            return None


def decode_status(resp: str) -> StatusAggregate | None:
    """
    Decode aggregate status:
      STATUS|POSE=p0,p1,...,p15|ANGLES=a0,...,a5|SPEEDS=s0,...,s5|IO=in1,in2,out1,out2,estop|GRIPPER=id,pos,spd,cur,status,obj|
             ACTION_CURRENT=...|ACTION_STATE=...

    Returns a dict matching StatusAggregate or None on parse failure.
    """
    if not resp or not resp.startswith("STATUS|"):
        return None

    # Split top-level sections after "STATUS|"
    sections = resp.split("|")[1:]
    result: dict[str, object] = {
        "pose": None,
        "angles": None,
        "speeds": None,
        "io": None,
        "gripper": None,
        "action_current": None,
        "action_state": None,
    }
    for sec in sections:
        if sec.startswith("POSE="):
            vals = [float(x) for x in sec[len("POSE=") :].split(",") if x]
            result["pose"] = vals
        elif sec.startswith("ANGLES="):
            vals = [float(x) for x in sec[len("ANGLES=") :].split(",") if x]
            result["angles"] = vals
        elif sec.startswith("SPEEDS="):
            vals = [float(x) for x in sec[len("SPEEDS=") :].split(",") if x]
            result["speeds"] = vals
        elif sec.startswith("IO="):
            vals = [int(x) for x in sec[len("IO=") :].split(",") if x]
            result["io"] = vals
        elif sec.startswith("GRIPPER="):
            vals = [int(x) for x in sec[len("GRIPPER=") :].split(",") if x]
            result["gripper"] = vals
        elif sec.startswith("ACTION_CURRENT="):
            result["action_current"] = sec[len("ACTION_CURRENT=") :]
        elif sec.startswith("ACTION_STATE="):
            result["action_state"] = sec[len("ACTION_STATE=") :]

    # Basic validation: accept if at least one of the core groups is present
    if (
        result["pose"] is None
        and result["angles"] is None
        and result["io"] is None
        and result["gripper"] is None
        and result["action_current"] is None
    ):
        return None

    return cast(StatusAggregate, result)
