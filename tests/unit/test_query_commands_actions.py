"""
Unit tests for action-related query commands.

Tests the ACTIVITY query command without requiring a running server.
Uses minimal state objects to test command logic in isolation.
"""

from waldoctl import ActionState

from parol6.commands.query_commands import ActivityCommand
from parol6.server.state import ControllerState
from parol6.protocol.wire import (
    ActivityCmd,
    CurrentActionResultStruct,
)


def test_activity_returns_details():
    """Test that ACTIVITY compute() returns correct data."""
    state = ControllerState(
        action_current="move_j_pose",
        action_state=ActionState.EXECUTING,
        action_next="home",
        action_params="angles=[10,20,30,40,50,60]",
    )

    cmd = ActivityCommand(ActivityCmd())
    cmd.setup(state)
    result = cmd.compute(state)

    assert isinstance(result, CurrentActionResultStruct)
    assert result.current == "move_j_pose"
    assert result.state == "EXECUTING"
    assert result.next == "home"
    assert result.params == "angles=[10,20,30,40,50,60]"


def test_activity_with_idle_state():
    """Test ACTIVITY when robot is idle."""
    state = ControllerState(
        action_current="",
        action_state=ActionState.IDLE,
        action_next="",
        action_params="",
    )

    cmd = ActivityCommand(ActivityCmd())
    cmd.setup(state)
    result = cmd.compute(state)

    assert isinstance(result, CurrentActionResultStruct)
    assert result.current == ""
    assert result.state == "IDLE"
    assert result.next == ""
    assert result.params == ""
