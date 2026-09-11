"""Attachment contexts and scoped contacts survive the command/reply codecs."""

import msgspec
import pytest
from waldoctl import Sphere

from parol6.protocol.wire import (
    CmdType,
    MsgType,
    QueryType,
    decode_command,
    decode_message,
    encode,
)


def test_attachment_wire_roundtrip_and_hostile_contexts():
    part = Sphere(name="part", radius=0.02).attach(
        flange_pose=(0, 0, 0.25, 0, 0, 0),
        epoch=2**64 - 1,
        allowed_contacts=("shape:fixture",),
    )
    wire = list(part.to_wire())
    command = [CmdType.SET_SHAPES, [wire]]
    assert msgspec.msgpack.decode(
        encode(decode_command(encode(command)))
    ) == msgspec.msgpack.decode(encode(command))
    # The reply carries the request id it answers.
    reply = [MsgType.RESPONSE, 7, [QueryType.SHAPES, [], [wire], 2, 2**64 - 1]]
    assert msgspec.msgpack.decode(
        encode(decode_message(encode(reply)))
    ) == msgspec.msgpack.decode(encode(reply))
    for binding in (
        [0, []],
        [-1, []],
        [True, []],
        [1.5, []],
        [1, "arm"],
        [1, ["*"]],
        [1, ["x", "x"]],
        [1, [""]],
        [1, [str(i) for i in range(33)]],
        [1],
        [1, [], 4],
    ):
        wire[-1] = binding
        with pytest.raises(msgspec.ValidationError):
            decode_command(encode([CmdType.SET_SHAPES, [wire]]))
        with pytest.raises(msgspec.ValidationError):
            decode_message(
                encode([MsgType.RESPONSE, [QueryType.SHAPES, [], [wire], 2, 1]])
            )
