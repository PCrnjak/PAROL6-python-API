"""
Wire protocol helpers for UDP encoding/decoding.

This module centralizes encoding of command strings and decoding of common
response payloads used by the headless controller.
"""
import json
import logging
from typing import List, Literal, Sequence

from .types import Frame, Axis, StatusAggregate
# Centralized binary wire protocol helpers (pack/unpack + codes)
from enum import IntEnum

logger = logging.getLogger(__name__)


class CommandCode(IntEnum):
    """Unified command codes for firmware interface."""
    IDLE = 255
    HOME = 100
    JOG = 123
    MOVE = 156


def split_bitfield(byte_val: int) -> list[int]:
    """Split an 8-bit integer into a big-endian list of bits (MSB..LSB)."""
    return [(byte_val >> i) & 1 for i in range(7, -1, -1)]


def fuse_bitfield_2_bytearray(bits: list[int]) -> bytes:
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
    Convert an int to 24-bit big-endian (two's complement) and return 3 bytes.
    This mirrors the existing Split_2_3_bytes semantics from headless_commander.
    """
    n24 = n & 0xFFFFFF  # two's complement 24-bit
    b = n24.to_bytes(4, "big", signed=False)
    return (b[1], b[2], b[3])


def fuse_3_bytes(b0: int, b1: int, b2: int) -> int:
    """
    Convert 3 bytes (big-endian) into a signed 24-bit integer.
    Matches the existing Fuse_3_bytes semantics.
    """
    val = int.from_bytes(bytes([0, b0, b1, b2]), "big", signed=False)
    if val >= (1 << 23):
        val -= (1 << 24)
    return val


def fuse_2_bytes(b0: int, b1: int) -> int:
    """
    Convert 2 bytes (big-endian) into a signed 16-bit integer.
    Matches the existing Fuse_2_bytes semantics.
    """
    val = int.from_bytes(bytes([0, 0, b0, b1]), "big", signed=False)
    if val >= (1 << 15):
        val -= (1 << 16)
    return val


def pack_tx_frame(
    position_out: list[int],
    speed_out: list[int],
    command_code: int | CommandCode,
    affected_joint_out: list[int],
    inout_out: list[int],
    timeout_out: int,
    gripper_data_out: list[int],
) -> bytes:
    """
    Pack a full TX frame to firmware.

    Layout (excluding 3 start bytes and 1 length byte, total payload len = 52):
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
    START = b"\xff\xff\xff"
    END = b"\x01\x02"
    PAYLOAD_LEN = 52  # matches existing firmware expectation

    # Safety clamps and conversions
    cmd = int(command_code)
    # Build payload
    out = bytearray()
    out += START
    out += bytes([PAYLOAD_LEN])

    # Positions: 6 * 3 bytes
    for i in range(6):
        b0, b1, b2 = split_to_3_bytes(int(position_out[i]))
        out += bytes([b0, b1, b2])

    # Speeds: 6 * 3 bytes
    for i in range(6):
        b0, b1, b2 = split_to_3_bytes(int(speed_out[i]))
        out += bytes([b0, b1, b2])

    # Command
    out += bytes([cmd])

    # Affected joints as bitfield byte
    out += fuse_bitfield_2_bytearray(list(affected_joint_out[:8]) + [0] * (8 - len(affected_joint_out[:8])))

    # In/Out as bitfield byte
    out += fuse_bitfield_2_bytearray(list(inout_out[:8]) + [0] * (8 - len(inout_out[:8])))

    # Timeout
    out += bytes([int(timeout_out) & 0xFF])

    # Reserved/legacy bytes to match firmware payload length
    out += b"\x00\x00"

    # Gripper: position, speed, current as 2 bytes each (big-endian)
    for idx in range(3):
        v = int(gripper_data_out[idx]) & 0xFFFF
        out += bytes([(v >> 8) & 0xFF, v & 0xFF])

    # Gripper command, mode, id
    out += bytes([
        int(gripper_data_out[3]) & 0xFF,
        int(gripper_data_out[4]) & 0xFF,
        int(gripper_data_out[5]) & 0xFF,
    ])

    # CRC (placeholder - unchanged from legacy)
    out += bytes([228])

    # End bytes
    out += END
    return bytes(out)


def unpack_rx_frame(data: bytes) -> dict | None:
    """
    Unpack a full RX frame payload (expected 52 bytes: data buffer after len).
    Mirrors the existing Unpack_data logic to produce the same structures.
    Returns dict with keys:
      Position_in, Speed_in, Homed_in, InOut_in, Temperature_error_in, Position_error_in,
      Timeout_error, Timing_data_in, Gripper_data_in
    """
    try:
        # Basic validation (minimum structure)
        if len(data) < 52:
            logger.warning(f"unpack_rx_frame: payload too short ({len(data)} bytes)")
            return None

        # Positions (0..17) and speeds (18..35)
        pos_in = [0] * 6
        spd_in = [0] * 6
        for i in range(6):
            off = i * 3
            pos_in[i] = fuse_3_bytes(data[off + 0], data[off + 1], data[off + 2])
        for i in range(6):
            off = 18 + i * 3
            spd_in[i] = fuse_3_bytes(data[off + 0], data[off + 1], data[off + 2])

        homed_byte = data[36]
        io_byte = data[37]
        temp_err_byte = data[38]
        pos_err_byte = data[39]
        timing_b0 = data[40]
        timing_b1 = data[41]
        # indices 42..43 exist in some variants (timeout/xtr), legacy code ignores
        device_id = data[44]
        grip_pos_b0, grip_pos_b1 = data[45], data[46]
        grip_spd_b0, grip_spd_b1 = data[47], data[48]
        grip_cur_b0, grip_cur_b1 = data[49], data[50]
        status_byte = data[51]
        # Optional: data[52] object detection (legacy ignored here)
        # data[53] CRC, data[54..55] end markers

        homed = split_bitfield(homed_byte)
        io_bits = split_bitfield(io_byte)
        temp_bits = split_bitfield(temp_err_byte)
        pos_bits = split_bitfield(pos_err_byte)
        timing_val = fuse_3_bytes(0, timing_b0, timing_b1)

        # Gripper values
        grip_pos = fuse_2_bytes(grip_pos_b0, grip_pos_b1)
        grip_spd = fuse_2_bytes(grip_spd_b0, grip_spd_b1)
        grip_cur = fuse_2_bytes(grip_cur_b0, grip_cur_b1)

        status_bits = split_bitfield(status_byte)
        # Combine bits 3 and 2 (big-endian list indices 4 and 5)
        obj_detection = ((status_bits[4] << 1) | status_bits[5]) if len(status_bits) >= 6 else 0

        gripper_data_in = [int(device_id), int(grip_pos), int(grip_spd), int(grip_cur), int(status_byte), int(obj_detection)]

        return {
            "Position_in": pos_in,
            "Speed_in": spd_in,
            "Homed_in": homed,
            "InOut_in": io_bits,
            "Temperature_error_in": temp_bits,
            "Position_error_in": pos_bits,
            "Timeout_error": 0,  # legacy not provided here
            "Timing_data_in": [timing_val],
            "Gripper_data_in": gripper_data_in,
        }
    except Exception as e:
        logger.error(f"unpack_rx_frame: exception {e}")
        return None


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
def decode_simple(resp: str, expected_prefix: Literal["ANGLES", "IO", "GRIPPER", "SPEEDS", "POSE"]) -> List[float] | List[int] | None:
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
        logger.warning(f"decode_simple: Invalid response format. Expected '{expected_prefix}|...' but got '{resp}'")
        return None
    payload = parts[1]
    tokens = [t for t in payload.split(",") if t != ""]

    # IO and GRIPPER are integer-based; others default to float
    if expected_prefix in ("IO", "GRIPPER"):
        try:
            return [int(t) for t in tokens]
        except ValueError as e:
            logger.error(f"decode_simple: Failed to parse integers for {expected_prefix}. Payload: '{payload}', Error: {e}")
            return None
    else:
        try:
            return [float(t) for t in tokens]
        except ValueError as e:
            logger.error(f"decode_simple: Failed to parse floats for {expected_prefix}. Payload: '{payload}', Error: {e}")
            return None


def decode_status(resp: str) -> StatusAggregate | None:
    """
    Decode aggregate status:
      STATUS|POSE=p0,p1,...,p15|ANGLES=a0,...,a5|IO=in1,in2,out1,out2,estop|GRIPPER=id,pos,spd,cur,status,obj

    Returns a dict matching StatusAggregate or None on parse failure.
    """
    if not resp or not resp.startswith("STATUS|"):
        return None

    # Split top-level sections after "STATUS|"
    sections = resp.split("|")[1:]
    result: dict[str, object] = {
        "pose": None,
        "angles": None,
        "io": None,
        "gripper": None,
    }
    for sec in sections:
        if sec.startswith("POSE="):
            vals = [float(x) for x in sec[len("POSE="):].split(",") if x]
            result["pose"] = vals
        elif sec.startswith("ANGLES="):
            vals = [float(x) for x in sec[len("ANGLES="):].split(",") if x]
            result["angles"] = vals
        elif sec.startswith("IO="):
            vals = [int(x) for x in sec[len("IO="):].split(",") if x]
            result["io"] = vals
        elif sec.startswith("GRIPPER="):
            vals = [int(x) for x in sec[len("GRIPPER="):].split(",") if x]
            result["gripper"] = vals

    # Basic validation
    if result["pose"] is None and result["angles"] is None and result["io"] is None and result["gripper"] is None:
        return None

    return result

def parse_server_state(resp: str) -> dict | None:
    """
    Parse server state JSON from:
      SERVER_STATE|{"ready": true, ...}
    Returns dict or None.
    """
    if not resp or not resp.startswith("SERVER_STATE|"):
        logger.debug(f"parse_server_state: Invalid response format. Expected 'SERVER_STATE|...' but got '{resp}'")
        return None
    _, json_part = resp.split("|", 1)
    try:
        return json.loads(json_part)
    except json.JSONDecodeError as e:
        logger.error(f"parse_server_state: Failed to parse JSON. JSON part: '{json_part}', Error: {e}")
        return None
