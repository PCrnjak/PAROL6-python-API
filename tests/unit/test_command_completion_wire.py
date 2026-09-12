"""Exact completion readback remains bounded and rejects malformed indices."""

import math

import msgspec
import pytest

from parol6.protocol.wire import (
    CmdType,
    CommandCompletionCmd,
    MsgType,
    QueryType,
    decode_command,
    decode_message,
    encode,
)
from parol6.server.command_registry import create_command
from parol6.server.state import ControllerState


def test_completion_history_is_exact_expires_and_resets_through_wire_query():
    state = ControllerState()

    def completed(index):
        command, _, error = create_command(encode(CommandCompletionCmd(index)))
        assert command is not None, error
        command.setup(state)
        result = decode_message(command.compute(state)).result
        assert result.command_index == index
        assert result.session_id == state.status_session_id
        return result.completed

    state.record_completion(10)
    assert completed(10) and not completed(9) and not completed(11)
    # Completion order can differ from acceptance order.
    state.record_completion(9)
    assert completed(9) and completed(10)
    for index in range(11, 1034):
        state.record_completion(index)
    assert not completed(10) and completed(9) and completed(1033)
    state.record_completion(1034)
    assert not completed(9) and completed(1034)
    state.reset()
    assert not completed(1034)


def test_completion_packets_reject_invalid_indices_sessions_and_verdicts():
    for value in (0, 2**63 - 1):
        command = CommandCompletionCmd(value)
        assert decode_command(encode(command)) == command
    invalid = [
        [CmdType.COMMAND_COMPLETION],
        [CmdType.COMMAND_COMPLETION, 0, 1],
        *(
            [CmdType.COMMAND_COMPLETION, value]
            for value in (-1, 2**63, True, "1", 0.5, math.nan, math.inf, None)
        ),
    ]
    for packet in invalid:
        with pytest.raises(msgspec.ValidationError):
            decode_command(encode(packet))
    for values in (
        (-1, 1, True),
        (2**63, 1, True),
        (1, 0, True),
        (1, -1, True),
        (1, True, True),
        (1, 1, 1),
        (1, 1),
        (1, 1, True, 0),
    ):
        with pytest.raises(msgspec.ValidationError):
            decode_message(
                encode([MsgType.RESPONSE, [QueryType.COMMAND_COMPLETION, *values]])
            )
