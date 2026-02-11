"""
Unit tests for action-related query commands.

Tests GET_CURRENT_ACTION and GET_QUEUE query commands without requiring a running server.
Uses minimal state objects to test command logic in isolation.
"""

from types import SimpleNamespace

import msgspec.msgpack

from parol6.commands.query_commands import GetCurrentActionCommand, GetQueueCommand
from parol6.protocol.wire import GetCurrentActionCmd, GetQueueCmd, MsgType, QueryType

_decode = msgspec.msgpack.Decoder().decode


def _unpack_response(data: bytes):
    """Unpack [MsgType.RESPONSE, query_type, value] from packed bytes."""
    msg_type, qt, value = _decode(data)
    assert msg_type == MsgType.RESPONSE
    return qt, value


def test_get_current_action_command_init():
    """Test that GET_CURRENT_ACTION command initializes correctly."""
    cmd = GetCurrentActionCommand(GetCurrentActionCmd())

    assert not cmd.is_finished
    assert cmd.PARAMS_TYPE is not None
    assert cmd.QUERY_TYPE == QueryType.CURRENT_ACTION


def test_get_current_action_returns_details():
    """Test that GET_CURRENT_ACTION compute() returns correct data."""
    state = SimpleNamespace(
        action_current="MoveJPoseCommand",
        action_state="EXECUTING",
        action_next="HomeCommand",
    )

    cmd = GetCurrentActionCommand(GetCurrentActionCmd())
    cmd.setup(state)
    qt, value = _unpack_response(cmd.compute(state))

    assert qt == QueryType.CURRENT_ACTION
    current, action_state, next_ = value
    assert current == "MoveJPoseCommand"
    assert action_state == "EXECUTING"
    assert next_ == "HomeCommand"


def test_get_current_action_with_idle_state():
    """Test GET_CURRENT_ACTION when robot is idle."""
    state = SimpleNamespace(action_current="", action_state="IDLE", action_next="")

    cmd = GetCurrentActionCommand(GetCurrentActionCmd())
    cmd.setup(state)
    qt, value = _unpack_response(cmd.compute(state))

    current, action_state, next_ = value
    assert current == ""
    assert action_state == "IDLE"
    assert next_ == ""


def test_get_queue_command_init():
    """Test that GET_QUEUE command initializes correctly."""
    cmd = GetQueueCommand(GetQueueCmd())

    assert not cmd.is_finished
    assert cmd.PARAMS_TYPE is not None
    assert cmd.QUERY_TYPE == QueryType.QUEUE


def test_get_queue_returns_details():
    """Test that GET_QUEUE compute() returns correct data."""
    state = SimpleNamespace(
        queue_nonstreamable=["MoveJPoseCommand", "HomeCommand", "MoveJCommand"]
    )

    cmd = GetQueueCommand(GetQueueCmd())
    cmd.setup(state)
    qt, value = _unpack_response(cmd.compute(state))

    assert qt == QueryType.QUEUE
    assert value == ["MoveJPoseCommand", "HomeCommand", "MoveJCommand"]


def test_get_queue_with_empty_queue():
    """Test GET_QUEUE when queue is empty."""
    state = SimpleNamespace(queue_nonstreamable=[])

    cmd = GetQueueCommand(GetQueueCmd())
    cmd.setup(state)
    _qt, value = _unpack_response(cmd.compute(state))

    assert value == []


def test_get_queue_excludes_streamable():
    """Test that queue only contains non-streamable commands (by design)."""
    state = SimpleNamespace(queue_nonstreamable=["MoveJPoseCommand", "HomeCommand"])

    cmd = GetQueueCommand(GetQueueCmd())
    cmd.setup(state)
    _qt, value = _unpack_response(cmd.compute(state))

    assert "MoveJPoseCommand" in value
    assert "HomeCommand" in value
    assert len(value) == 2
