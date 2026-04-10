"""Test multicast status reception from live server."""

import asyncio

import pytest

from parol6 import AsyncRobotClient


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multicast_status_reception(server_proc, ports):
    """Verify that multicast status is actually received from a running server."""
    received_count = 0

    async with AsyncRobotClient(port=ports.server_port) as client:

        async def receive_status():
            nonlocal received_count
            async for _ in client.stream_status_shared():
                received_count += 1
                if received_count >= 3:
                    return

        # Should receive at least 3 status messages within 5 seconds
        try:
            await asyncio.wait_for(receive_status(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

    assert received_count >= 3, (
        f"Expected at least 3 status messages, got {received_count}"
    )
