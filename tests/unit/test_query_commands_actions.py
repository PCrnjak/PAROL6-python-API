"""
Unit tests for action-related query commands.

Tests GET_CURRENT_ACTION and GET_QUEUE query commands without requiring a running server.
Uses stub UDP transport and minimal state objects to test command logic in isolation.
"""

import json
from types import SimpleNamespace

import pytest
from parol6.commands.base import CommandContext
from parol6.commands.query_commands import GetCurrentActionCommand, GetQueueCommand


class StubUDPTransport:
    """Stub UDP transport that captures sent responses."""

    def __init__(self):
        self.sent = []

    def send_response(self, message: str, addr: tuple):
        """Capture sent responses for verification."""
        self.sent.append((message, addr))


def test_get_current_action_command_match():
    """Test that GET_CURRENT_ACTION command matches correctly."""
    cmd = GetCurrentActionCommand()

    # Should match
    can_handle, error = cmd.do_match(["GET_CURRENT_ACTION"])
    assert can_handle
    assert error is None

    # Should not match other commands
    can_handle, error = cmd.do_match(["GET_QUEUE"])
    assert not can_handle

    can_handle, error = cmd.do_match(["UNKNOWN"])
    assert not can_handle


def test_get_current_action_replies_json():
    """Test that GET_CURRENT_ACTION returns correct JSON response."""
    # Setup
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Create minimal state with action tracking fields
    state = SimpleNamespace(
        action_current="MovePoseCommand", action_state="EXECUTING", action_next="HomeCommand"
    )

    # Execute command
    cmd = GetCurrentActionCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    status = cmd.tick(state)

    # Verify response sent
    assert len(udp.sent) == 1
    message, addr = udp.sent[0]

    # Verify message format
    assert message.startswith("ACTION|")

    # Parse and verify JSON payload
    json_str = message.split("|", 1)[1]
    payload = json.loads(json_str)

    assert payload["current"] == "MovePoseCommand"
    assert payload["state"] == "EXECUTING"
    assert payload["next"] == "HomeCommand"

    # Verify command completed
    assert status.code.value == "COMPLETED"
    assert cmd.is_finished


def test_get_current_action_with_idle_state():
    """Test GET_CURRENT_ACTION when robot is idle."""
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Idle state - no current action
    state = SimpleNamespace(action_current="", action_state="IDLE", action_next="")

    cmd = GetCurrentActionCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    cmd.tick(state)

    # Verify response
    assert len(udp.sent) == 1
    message, _ = udp.sent[0]
    json_str = message.split("|", 1)[1]
    payload = json.loads(json_str)

    assert payload["current"] == ""
    assert payload["state"] == "IDLE"
    assert payload["next"] == ""


def test_get_queue_command_match():
    """Test that GET_QUEUE command matches correctly."""
    cmd = GetQueueCommand()

    # Should match
    can_handle, error = cmd.do_match(["GET_QUEUE"])
    assert can_handle
    assert error is None

    # Should not match other commands
    can_handle, error = cmd.do_match(["GET_CURRENT_ACTION"])
    assert not can_handle


def test_get_queue_replies_json():
    """Test that GET_QUEUE returns correct JSON response."""
    # Setup
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Create state with queued commands
    state = SimpleNamespace(
        queue_nonstreamable=["MovePoseCommand", "HomeCommand", "MoveJointCommand"]
    )

    # Execute command
    cmd = GetQueueCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    status = cmd.tick(state)

    # Verify response sent
    assert len(udp.sent) == 1
    message, addr = udp.sent[0]

    # Verify message format
    assert message.startswith("QUEUE|")

    # Parse and verify JSON payload
    json_str = message.split("|", 1)[1]
    payload = json.loads(json_str)

    assert payload["non_streamable"] == ["MovePoseCommand", "HomeCommand", "MoveJointCommand"]
    assert payload["size"] == 3

    # Verify command completed
    assert status.code.value == "COMPLETED"
    assert cmd.is_finished


def test_get_queue_with_empty_queue():
    """Test GET_QUEUE when queue is empty."""
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Empty queue
    state = SimpleNamespace(queue_nonstreamable=[])

    cmd = GetQueueCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    cmd.tick(state)

    # Verify response
    assert len(udp.sent) == 1
    message, _ = udp.sent[0]
    json_str = message.split("|", 1)[1]
    payload = json.loads(json_str)

    assert payload["non_streamable"] == []
    assert payload["size"] == 0


def test_get_queue_excludes_streamable():
    """Test that queue only contains non-streamable commands (by design)."""
    # This test verifies the API contract - the queue_nonstreamable field
    # should already have streamable commands filtered out by the controller

    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # State should only contain non-streamable commands
    # (streamable commands like JogJointCommand are filtered by controller)
    state = SimpleNamespace(queue_nonstreamable=["MovePoseCommand", "HomeCommand"])

    cmd = GetQueueCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    cmd.tick(state)

    message, _ = udp.sent[0]
    json_str = message.split("|", 1)[1]
    payload = json.loads(json_str)

    # Verify only non-streamable commands in response
    assert "MovePoseCommand" in payload["non_streamable"]
    assert "HomeCommand" in payload["non_streamable"]
    assert payload["size"] == 2
