"""
Unit tests for wire protocol action field parsing.

Tests that decode_status correctly parses ACTION_CURRENT, ACTION_STATE, and ACTION_NEXT fields.
"""

import pytest
from parol6.protocol import wire


def test_decode_status_with_action_fields():
    """Test that decode_status parses ACTION_* fields from status string."""
    # Build status string with action fields
    resp = (
        "STATUS|"
        "POSE=" + ",".join(str(i) for i in range(1, 17)) + "|"
        "ANGLES=0,10,20,30,40,50|"
        "IO=1,1,0,0,1|"
        "GRIPPER=1,20,150,500,3,1|"
        "ACTION_CURRENT=MovePoseCommand|"
        "ACTION_STATE=EXECUTING|"
        "ACTION_NEXT=HomeCommand"
    )

    result = wire.decode_status(resp)

    assert result is not None
    assert isinstance(result, dict)

    # Verify traditional fields still work
    assert len(result["pose"]) == 16
    assert result["angles"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert result["io"] == [1, 1, 0, 0, 1]
    assert result["gripper"] == [1, 20, 150, 500, 3, 1]

    # Verify new action fields
    assert result["action_current"] == "MovePoseCommand"
    assert result["action_state"] == "EXECUTING"


def test_decode_status_with_empty_action_fields():
    """Test parsing when action fields are present but empty."""
    resp = (
        "STATUS|"
        "POSE=" + ",".join(str(i) for i in range(1, 17)) + "|"
        "ANGLES=0,0,0,0,0,0|"
        "IO=1,1,0,0,1|"
        "GRIPPER=0,0,0,0,0,0|"
        "ACTION_CURRENT=|"
        "ACTION_STATE=IDLE|"
        "ACTION_NEXT="
    )

    result = wire.decode_status(resp)

    assert result is not None
    assert result["action_current"] == ""
    assert result["action_state"] == "IDLE"


def test_decode_status_backward_compatible_without_actions():
    """Test that status without ACTION_* fields still decodes (backward compat)."""
    # Old-style status without action fields
    resp = (
        "STATUS|"
        "POSE=" + ",".join(str(i) for i in range(1, 17)) + "|"
        "ANGLES=0,10,20,30,40,50|"
        "IO=1,1,0,0,1|"
        "GRIPPER=1,20,150,500,3,1"
    )

    result = wire.decode_status(resp)

    assert result is not None
    # Traditional fields should work
    assert result["angles"] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    # Action fields should be None when not present
    assert result.get("action_current") is None
    assert result.get("action_state") is None


def test_decode_status_with_various_action_states():
    """Test parsing with different action state values."""
    states = ["IDLE", "EXECUTING", "COMPLETED", "FAILED"]

    for state_value in states:
        resp = (
            "STATUS|"
            "POSE=" + ",".join("0" for _ in range(16)) + "|"
            "ANGLES=0,0,0,0,0,0|"
            "IO=1,1,0,0,1|"
            "GRIPPER=0,0,0,0,0,0|"
            f"ACTION_CURRENT=TestCommand|"
            f"ACTION_STATE={state_value}|"
            f"ACTION_NEXT=NextCommand"
        )

        result = wire.decode_status(resp)
        assert result is not None
        assert result["action_state"] == state_value
