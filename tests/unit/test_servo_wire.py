"""A streamed servo target is validated where the controller decodes it, as
the planned moves are: a non-finite value, or a joint angle outside its
range, never reaches a stream."""

import math

import msgspec
import pytest

from parol6.config import LIMITS
from parol6.protocol.wire import (
    CmdType,
    ServoJCmd,
    ServoJPoseCmd,
    ServoLCmd,
    decode_command,
    encode,
)

STANDBY = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
POSE = [236.8, 0.0, 334.0, -180.0, -90.0, -180.0]


def _with(values: list[float], index: int, value: float) -> list[float]:
    out = list(values)
    out[index] = value
    return out


@pytest.mark.parametrize(
    ("tag", "cls", "valid"),
    [
        (CmdType.SERVOJ, ServoJCmd, STANDBY),
        (CmdType.SERVOJ_POSE, ServoJPoseCmd, POSE),
        (CmdType.SERVOL, ServoLCmd, POSE),
    ],
)
def test_a_servo_target_decodes_only_when_finite(tag, cls, valid):
    assert decode_command(encode([tag, valid, 0.5, 0.5])) == cls(valid, 0.5, 0.5)
    for i in range(6):
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(msgspec.ValidationError):
                decode_command(encode([tag, _with(valid, i, bad), 0.5, 0.5]))


def test_a_servo_j_target_decodes_only_inside_the_joint_range():
    lo, hi = LIMITS.joint.position.deg[:, 0], LIMITS.joint.position.deg[:, 1]
    for i in range(6):
        for edge in (float(lo[i]), float(hi[i])):
            angles = _with(STANDBY, i, edge)
            assert decode_command(encode([CmdType.SERVOJ, angles])) == ServoJCmd(angles)
        for outside in (float(lo[i]) - 0.5, float(hi[i]) + 0.5):
            with pytest.raises(msgspec.ValidationError, match=f"Joint {i + 1}"):
                decode_command(encode([CmdType.SERVOJ, _with(STANDBY, i, outside)]))
