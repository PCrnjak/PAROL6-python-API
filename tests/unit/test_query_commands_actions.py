"""
Unit tests for action-related query commands.

Tests the ACTIVITY query command without requiring a running server.
Uses minimal state objects to test command logic in isolation.
"""

from types import SimpleNamespace

from waldoctl import ActionState

from parol6.commands.query_commands import ActivityCommand
from parol6.protocol.wire import (
    ActivityCmd,
    CurrentActionResultStruct,
    ResponseMsg,
    decode_message,
)


def _unpack_response(data: bytes):
    """Decode packed bytes into a typed result struct."""
    msg = decode_message(data)
    assert isinstance(msg, ResponseMsg)
    return msg.result


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
