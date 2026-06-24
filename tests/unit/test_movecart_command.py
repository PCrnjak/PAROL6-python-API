"""Tests for MoveLCommand parsing via msgspec structs.

Commands now use msgspec Structs for parameter validation:
Format: MoveLCmd(pose, frame, duration, speed, accel, r, rel)
"""

import msgspec
import pytest

from parol6.protocol.wire import MoveLCmd


class TestMoveLCommandParsing:
    """Test MoveLCmd struct parsing and validation."""

    def test_validation_requires_duration_or_speed(self):
        """Must have either duration > 0 or speed > 0."""
        with pytest.raises((ValueError, msgspec.ValidationError)):
            # Both zero (default) should fail __post_init__
            MoveLCmd(pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_validation_rejects_both_duration_and_speed(self):
        """Cannot have both duration > 0 and speed > 0."""
        with pytest.raises((ValueError, msgspec.ValidationError)):
            MoveLCmd(
                pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                duration=2.0,
                speed=0.5,
            )

    def test_validation_pose_length(self):
        """Pose must have exactly 6 elements when decoded from wire format."""
        from parol6.protocol.wire import CmdType, decode_command

        # Validation happens during decode from msgpack (wire format)
        import msgspec

        encoder = msgspec.msgpack.Encoder()
        # Wire format: [tag, pose, frame, duration, speed, accel, r, rel]
        raw = encoder.encode(
            [int(CmdType.MOVEL), [0.0, 0.0, 0.0], "WRF", 0.0, 0.5, 1.0, 0.0, False]
        )

        with pytest.raises(msgspec.ValidationError):
            decode_command(raw)
