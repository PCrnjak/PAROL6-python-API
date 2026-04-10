"""
Unit tests for action-related query commands.

Tests ACTIVITY and QUEUE query commands without requiring a running server.
Uses minimal state objects to test command logic in isolation.
"""

from types import SimpleNamespace

from waldoctl import ActionState

from parol6.commands.query_commands import ActivityCommand, QueueCommand
from parol6.protocol.wire import (
    CurrentActionResultStruct,
    ActivityCmd,
    QueueCmd,
    QueryType,
    QueueResultStruct,
    ResponseMsg,
    decode_message,
)


def _unpack_response(data: bytes):
    """Decode packed bytes into a typed result struct."""
    msg = decode_message(data)
    assert isinstance(msg, ResponseMsg)
    return msg.result


def test_activity_command_init():
    """Test that ACTIVITY command initializes correctly."""
    cmd = ActivityCommand(ActivityCmd())

    assert not cmd.is_finished
    assert cmd.PARAMS_TYPE is not None
    assert cmd.QUERY_TYPE == QueryType.CURRENT_ACTION


def test_activity_returns_details():
    """Test that ACTIVITY compute() returns correct data."""
    state = SimpleNamespace(
        action_current="MoveJPoseCommand",
        action_state=ActionState.EXECUTING,
        action_next="HomeCommand",
        action_params="angles=[10,20,30,40,50,60]",
    )

    cmd = ActivityCommand(ActivityCmd())
    cmd.setup(state)
    result = _unpack_response(cmd.compute(state))

    assert isinstance(result, CurrentActionResultStruct)
    assert result.current == "MoveJPoseCommand"
    assert result.state == "EXECUTING"
    assert result.next == "HomeCommand"
    assert result.params == "angles=[10,20,30,40,50,60]"


def test_activity_with_idle_state():
    """Test ACTIVITY when robot is idle."""
    state = SimpleNamespace(
        action_current="",
        action_state=ActionState.IDLE,
        action_next="",
        action_params="",
    )

    cmd = ActivityCommand(ActivityCmd())
    cmd.setup(state)
    result = _unpack_response(cmd.compute(state))

    assert isinstance(result, CurrentActionResultStruct)
    assert result.current == ""
    assert result.state == "IDLE"
    assert result.next == ""
    assert result.params == ""


def test_queue_command_init():
    """Test that QUEUE command initializes correctly."""
    cmd = QueueCommand(QueueCmd())

    assert not cmd.is_finished
    assert cmd.PARAMS_TYPE is not None
    assert cmd.QUERY_TYPE == QueryType.QUEUE


def test_queue_returns_details():
    """Test that QUEUE compute() returns correct data."""
    state = SimpleNamespace(
        queue_nonstreamable=["MoveJPoseCommand", "HomeCommand", "MoveJCommand"],
        executing_command_index=1,
        completed_command_index=0,
        last_checkpoint="cp1",
        queued_duration=3.5,
    )

    cmd = QueueCommand(QueueCmd())
    cmd.setup(state)
    result = _unpack_response(cmd.compute(state))

    assert isinstance(result, QueueResultStruct)
    assert result.queue == ["MoveJPoseCommand", "HomeCommand", "MoveJCommand"]
    assert result.executing_index == 1
    assert result.completed_index == 0
    assert result.last_checkpoint == "cp1"
    assert result.queued_duration == 3.5


def test_queue_with_empty_queue():
    """Test QUEUE when queue is empty."""
    state = SimpleNamespace(
        queue_nonstreamable=[],
        executing_command_index=-1,
        completed_command_index=-1,
        last_checkpoint="",
        queued_duration=0.0,
    )

    cmd = QueueCommand(QueueCmd())
    cmd.setup(state)
    result = _unpack_response(cmd.compute(state))

    assert isinstance(result, QueueResultStruct)
    assert result.queue == []
    assert result.executing_index == -1
    assert result.completed_index == -1


def test_queue_excludes_streamable():
    """Test that queue only contains non-streamable commands (by design)."""
    state = SimpleNamespace(
        queue_nonstreamable=["MoveJPoseCommand", "HomeCommand"],
        executing_command_index=2,
        completed_command_index=1,
        last_checkpoint="",
        queued_duration=1.0,
    )

    cmd = QueueCommand(QueueCmd())
    cmd.setup(state)
    result = _unpack_response(cmd.compute(state))

    assert isinstance(result, QueueResultStruct)
    assert "MoveJPoseCommand" in result.queue
    assert "HomeCommand" in result.queue
    assert len(result.queue) == 2
