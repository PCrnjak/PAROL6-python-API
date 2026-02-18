"""
Unit tests for binary protocol message helpers.
"""

import numpy as np
import pytest
import msgspec

from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.protocol.wire import (
    ErrorMsg,
    MsgType,
    OkMsg,
    QueryType,
    ResponseMsg,
    decode,
    decode_message,
    encode,
    pack_error,
    pack_ok,
    pack_ok_index,
    pack_response,
    pack_status,
)


class TestPackUnpack:
    """Test packing and unpacking roundtrips via decode_message."""

    def test_pack_ok(self):
        msg = decode_message(pack_ok())
        assert isinstance(msg, OkMsg)
        assert msg.index is None

    def test_pack_ok_index(self):
        msg = decode_message(pack_ok_index(42))
        assert isinstance(msg, OkMsg)
        assert msg.index == 42

    def test_pack_error(self):
        error = make_error(
            ErrorCode.COMM_VALIDATION_ERROR, detail="Something went wrong"
        )
        msg = decode_message(pack_error(error))
        assert isinstance(msg, ErrorMsg)
        assert isinstance(msg.message, list)
        from parol6.utils.error_catalog import RobotError

        recovered = RobotError.from_wire(msg.message)
        assert recovered.code == ErrorCode.COMM_VALIDATION_ERROR
        assert "Something went wrong" in recovered.cause

    def test_pack_response(self):
        msg = decode_message(
            pack_response(QueryType.ANGLES, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )
        assert isinstance(msg, ResponseMsg)
        assert msg.query_type == QueryType.ANGLES
        assert msg.value == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_pack_response_with_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        msg = decode_message(pack_response(QueryType.POSE, arr))
        assert isinstance(msg, ResponseMsg)
        assert msg.value == [1.0, 2.0, 3.0]

    def test_pack_status_roundtrip(self):
        """Status broadcast (uses separate decode path, not decode_message)."""
        pose = np.arange(16, dtype=np.float64)
        angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        speeds = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)
        io = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        gripper = np.array([1, 255, 150, 500, 3, 1], dtype=np.int32)
        joint_en = np.ones(12, dtype=np.uint8)
        cart_en_wrf = np.ones(12, dtype=np.uint8)
        cart_en_trf = np.ones(12, dtype=np.uint8)

        packed = pack_status(
            pose,
            angles,
            speeds,
            io,
            gripper,
            "MoveJCommand",
            "EXECUTING",
            joint_en,
            cart_en_wrf,
            cart_en_trf,
        )
        unpacked = decode(packed)
        assert unpacked[0] == MsgType.STATUS
        assert unpacked[1] == list(pose)
        assert unpacked[2] == list(angles)
        assert unpacked[6] == "MoveJCommand"
        assert unpacked[7] == "EXECUTING"

    def test_invalid_data_raises(self):
        with pytest.raises(msgspec.ValidationError):
            decode_message(encode(["not", "a", "valid", "message"]))
