"""Tests for MoveLCommand parsing via msgspec structs.

Commands now use msgspec Structs for parameter validation:
Format: MoveLCmd(pose, frame, duration, speed, accel, r, rel)
"""

import msgspec
import pytest

from parol6.commands.cartesian_commands import MoveLCommand
from parol6.protocol.wire import MoveLCmd


class TestMoveLCommandParsing:
    """Test MoveLCmd struct parsing and validation."""

    def test_parse_with_speed(self):
        """Parse with explicit speed."""
        # Create params struct
        params = MoveLCmd(
            pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            speed=0.5,
            accel=0.75,
        )

        cmd = MoveLCommand(params)

        assert cmd.p.pose == [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
        assert cmd.p.duration == 0.0  # default
        assert cmd.p.speed == 0.5
        assert cmd.p.accel == 0.75

    def test_parse_accel_default(self):
        """Default acceleration should be 1.0."""
        params = MoveLCmd(
            pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            speed=0.5,
        )

        cmd = MoveLCommand(params)

        assert cmd.p.accel == 1.0  # default

    def test_parse_with_duration(self):
        """Parse with duration instead of speed."""
        params = MoveLCmd(
            pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            duration=2.5,
            accel=0.8,
        )

        cmd = MoveLCommand(params)

        assert cmd.p.duration == 2.5
        assert cmd.p.speed == 0.0  # default
        assert cmd.p.accel == 0.8

    def test_parse_full_accel_range(self):
        """Test acceleration values at boundaries."""
        # Min accel (must be > 0)
        params1 = MoveLCmd(
            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed=0.5,
            accel=0.001,
        )
        cmd1 = MoveLCommand(params1)
        assert cmd1.p.accel == 0.001

        # Max accel
        params2 = MoveLCmd(
            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed=0.5,
            accel=1.0,
        )
        cmd2 = MoveLCommand(params2)
        assert cmd2.p.accel == 1.0

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

    def test_command_init(self):
        """Test that MoveLCommand initializes correctly."""
        params = MoveLCmd(
            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed=0.5,
        )
        cmd = MoveLCommand(params)

        assert cmd.p is params
        assert not cmd.is_finished
