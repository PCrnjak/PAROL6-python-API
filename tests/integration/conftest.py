"""Integration test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def clean_state(server_proc, client):
    """
    Reset controller state before each integration test for isolation.

    Uses RESET command to instantly reset positions, queues, tool, errors.
    Sets LINEAR motion profile for faster test execution.
    Depends on server_proc to ensure server is ready before resetting.
    """
    client.reset()
    client.set_profile("LINEAR")
    idx = client.home()
    assert idx >= 0, "Home command failed to send"
    assert client.wait_command_complete(idx, timeout=5.0), "Home did not complete"
    return client
