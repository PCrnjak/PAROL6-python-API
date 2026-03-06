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
            pack_response(AnglesResultStruct(angles=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        )
        assert isinstance(msg, ResponseMsg)
        assert isinstance(msg.result, AnglesResultStruct)
        assert msg.result.angles == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_pack_response_with_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        msg = decode_message(pack_response(PoseResultStruct(pose=arr)))
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
        assert unpacked[1] == list(pose)
        assert unpacked[2] == list(angles)
        assert unpacked[5] == "MoveJCommand"
        assert unpacked[6] == ActionState.EXECUTING

        # action_params at index 16
        assert unpacked[16] == "speed=50 acc=100"

        # tool_status at index 17 is a 7-element tuple:
        # (key, state, engaged, part_detected, fault_code, positions, channels)
        ts = unpacked[17]
        assert ts[0] == "ssg48"        # key
        assert ts[1] == 2              # state (ToolState.ACTIVE)
        assert ts[2] is True           # engaged
        assert ts[3] is True           # part_detected
        assert ts[4] == 0              # fault_code
        assert ts[5] == [0.75, 0.25]   # positions (tuple -> list via msgpack)
        assert ts[6] == [5.5, 3.14]    # channels (tuple -> list via msgpack)

        # tcp_speed at index 18
        assert unpacked[18] == pytest.approx(123.456)

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
            state=ToolState.IDLE,
            engaged=False,
            part_detected=True,
            fault_code=42,
            positions=(0.5,),
            channels=(1.2, 3.4),
        )

        packed = pack_status(
            pose, angles, speeds, io,
            "HomeCommand",
            ActionState.IDLE,
            joint_en, cart_en_wrf, cart_en_trf,
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

    def test_invalid_data_raises(self):
        with pytest.raises(msgspec.ValidationError):
            decode_message(encode(["not", "a", "valid", "message"]))
