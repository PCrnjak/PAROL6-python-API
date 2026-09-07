"""Execution controls reject invalid values and malformed wire frames."""

import math

import msgspec
import pytest

from parol6.protocol.wire import (
    CmdType,
    ExecutionSpeedCmd,
    ExecutionSpeedResultStruct,
    MsgType,
    PauseCmd,
    QueryType,
    ResponseMsg,
    SetExecutionSpeedCmd,
    decode_command,
    decode_message,
    encode,
)


def test_execution_controls_roundtrip_and_reject_malformed_frames():
    for command in (
        SetExecutionSpeedCmd(0.1),
        SetExecutionSpeedCmd(1),
        PauseCmd(True),
        PauseCmd(False),
        ExecutionSpeedCmd(),
    ):
        assert decode_command(encode(command)) == command
    for values in ((0.5, 0.7, 0.5), (0, 0.03, 0.6), (0, 0, 1)):
        response = ResponseMsg(ExecutionSpeedResultStruct(*values))
        assert decode_message(encode(response)) == response

    invalid_commands = [
        [CmdType.SET_EXECUTION_SPEED],
        [CmdType.SET_EXECUTION_SPEED, 0.5, 1],
        [CmdType.PAUSE],
        [CmdType.PAUSE, True, False],
        [CmdType.EXECUTION_SPEED, 0],
        *(
            [CmdType.SET_EXECUTION_SPEED, value]
            for value in (
                True,
                0,
                -1,
                0.09,
                1.01,
                2,
                math.inf,
                -math.inf,
                math.nan,
                "0.5",
            )
        ),
        *([CmdType.PAUSE, value] for value in (0, 1, "true", None)),
    ]
    for wire in invalid_commands:
        with pytest.raises(msgspec.ValidationError):
            decode_command(encode(wire))
    for values in (
        (0.5, 0.7),
        (0.5, 0.7, 0.5, 1),
        (0.5, 0.7, 0.6),
        (0, -0.1, 0.5),
        (0, 1.1, 0.5),
        (0, math.nan, 0.5),
        (0, 0, 0),
        (True, 0, 1),
    ):
        with pytest.raises(msgspec.ValidationError):
            decode_message(
                encode([MsgType.RESPONSE, [QueryType.EXECUTION_SPEED, *values]])
            )
