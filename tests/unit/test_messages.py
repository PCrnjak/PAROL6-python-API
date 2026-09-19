"""
Unit tests for binary protocol message helpers.
"""

import numpy as np
import pytest
import msgspec

from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from waldoctl import ActionState, ToolStatus
from waldoctl.tools import ToolState

from parol6.protocol.wire import (
    AnglesResultStruct,
    ErrorMsg,
    MsgType,
    OkMsg,
    PoseResultStruct,
    ResponseMsg,
    StatusBuffer,
    decode,
    decode_message,
    decode_status_bin_into,
    encode,
    pack_error,
    pack_ok,
    pack_ok_index,
    pack_response,
    pack_status,
)
from parol6.protocol.wire import PROTO_VERSION, ProtocolVersionError


class TestPackUnpack:
    """Test packing and unpacking roundtrips via decode_message."""

    def test_pack_ok(self):
        msg = decode_message(pack_ok(7))
        assert isinstance(msg, OkMsg)
        assert msg.req_id == 7
        assert msg.index is None

    def test_pack_ok_index(self):
        msg = decode_message(pack_ok_index(42, 7))
        assert isinstance(msg, OkMsg)
        assert (msg.req_id, msg.index) == (7, 42)

    def test_pack_error(self):
        error = make_error(
            ErrorCode.COMM_VALIDATION_ERROR, detail="Something went wrong"
        )
        msg = decode_message(pack_error(error, 7))
        assert isinstance(msg, ErrorMsg)
        assert msg.req_id == 7
        assert isinstance(msg.message, list)
        from parol6.utils.error_catalog import RobotError

        recovered = RobotError.from_wire(msg.message)
        assert recovered.code == ErrorCode.COMM_VALIDATION_ERROR
        assert "Something went wrong" in recovered.cause

    def test_pack_response(self):
        msg = decode_message(
            pack_response(AnglesResultStruct(angles=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 7)
        )
        assert isinstance(msg, ResponseMsg)
        assert msg.req_id == 7
        assert isinstance(msg.result, AnglesResultStruct)
        assert msg.result.angles == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_pack_response_with_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        msg = decode_message(pack_response(PoseResultStruct(pose=arr), 7))
        assert isinstance(msg, ResponseMsg)
        assert isinstance(msg.result, PoseResultStruct)
        assert msg.result.pose == [1.0, 2.0, 3.0]

    def test_pack_status_roundtrip(self):
        """Status broadcast (uses separate decode path, not decode_message)."""
        pose = np.arange(16, dtype=np.float64)
        angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        speeds = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)
        io = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        joint_en = np.ones(12, dtype=np.uint8)
        cart_en_wrf = np.ones(12, dtype=np.uint8)
        cart_en_trf = np.ones(12, dtype=np.uint8)
        tool_status = ToolStatus(
            key="ssg48",
            state=ToolState.ACTIVE,
            engaged=True,
            part_detected=True,
            fault_code=0,
            positions=(0.75, 0.25),
            channels=(5.5, 3.14),
        )

        packed = pack_status(
            pose,
            angles,
            speeds,
            io,
            "MoveJCommand",
            ActionState.EXECUTING,
            joint_en,
            cart_en_wrf,
            cart_en_trf,
            action_params="speed=50 acc=100",
            tool_status=tool_status,
            tcp_speed=123.456,
        )
        unpacked = decode(packed)
        assert unpacked[0] == MsgType.STATUS
        assert unpacked[1] == PROTO_VERSION
        assert unpacked[2] == list(pose)
        assert unpacked[3] == list(angles)
        assert unpacked[6] == "MoveJCommand"
        assert unpacked[7] == ActionState.EXECUTING

        # action_params at index 17
        assert unpacked[17] == "speed=50 acc=100"

        # The optional variant follows the original seven tool-status fields.
        ts = unpacked[18]
        assert ts[0] == "ssg48"  # key
        assert ts[1] == 2  # state (ToolState.ACTIVE)
        assert ts[2] is True  # engaged
        assert ts[3] is True  # part_detected
        assert ts[4] == 0  # fault_code
        assert ts[5] == [0.75, 0.25]  # positions (tuple -> list via msgpack)
        assert ts[6] == [5.5, 3.14]  # channels (tuple -> list via msgpack)

        # tcp_speed at index 19
        assert unpacked[19] == pytest.approx(123.456)

    def test_pack_decode_status_bin_roundtrip(self):
        """pack_status -> decode_status_bin_into preserves all tool status fields."""
        pose = np.eye(4, dtype=np.float64).ravel()
        angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)
        speeds = np.zeros(6, dtype=np.float64)
        io = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
        joint_en = np.ones(12, dtype=np.uint8)
        cart_en_wrf = np.ones(12, dtype=np.uint8)
        cart_en_trf = np.ones(12, dtype=np.uint8)
        tool_status = ToolStatus(
            key="electric_gripper",
            variant_key="pinch",
            state=ToolState.IDLE,
            engaged=False,
            part_detected=True,
            fault_code=42,
            positions=(0.5,),
            channels=(1.2, 3.4),
        )

        packed = pack_status(
            pose,
            angles,
            speeds,
            io,
            "HomeCommand",
            ActionState.IDLE,
            joint_en,
            cart_en_wrf,
            cart_en_trf,
            tool_status=tool_status,
            tcp_speed=55.5,
        )

        buf = StatusBuffer()
        assert decode_status_bin_into(packed, buf) is True

        ts = buf.tool_status
        assert ts.key == "electric_gripper"
        assert ts.state == ToolState.IDLE
        assert isinstance(ts.state, ToolState)
        assert ts.engaged is False
        assert ts.part_detected is True
        assert ts.fault_code == 42
        assert ts.positions == (0.5,)
        assert ts.channels == (1.2, 3.4)
        assert buf.tcp_speed == pytest.approx(55.5)
        assert ts.variant_key == "pinch"
        assert buf.copy().tool_status.variant_key == "pinch"
        legacy = decode(packed)
        legacy[18] = legacy[18][:7]
        assert decode_status_bin_into(encode(legacy), buf)
        assert buf.tool_status.variant_key == "", (
            "legacy status retained a stale variant"
        )
        for invalid in (False, 42, None, "x" * 129):
            bad = decode(packed)
            bad[18][7] = invalid
            assert not decode_status_bin_into(encode(bad), buf)

        # A producer speaking another protocol version is named, not decoded
        # as silence: a consumer that saw nothing would report a dead
        # controller and send an operator looking at cables.
        other = decode(packed)
        other[1] = PROTO_VERSION + 1
        with pytest.raises(ProtocolVersionError, match=str(PROTO_VERSION + 1)):
            decode_status_bin_into(encode(other), buf)

    def test_invalid_data_raises(self):
        with pytest.raises(msgspec.ValidationError):
            decode_message(encode(["not", "a", "valid", "message"]))
