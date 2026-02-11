import asyncio

import pytest

from parol6.client.async_client import AsyncRobotClient


@pytest.mark.asyncio
@pytest.mark.integration
async def test_status_listener_shuts_down_on_close(ports, server_proc):
    """AsyncRobotClient.close() should close the status UDP transport.

    This test uses the real server process (server_proc), the real AsyncRobotClient,
    and the real status listener. It only relies on the existing test fixtures to
    start the FAKE_SERIAL controller on the auto-detected test ports.
    """

    client = AsyncRobotClient(
        host=ports.server_ip, port=ports.server_port, timeout=1.0, retries=0
    )

    try:
        # Ensure the controller is responsive before starting the status listener
        await client.wait_ready(timeout=5.0)

        # Force endpoint and status listener creation
        await client._ensure_endpoint()
        transport = client._status_transport

        assert transport is not None, "Status transport should be created"

        # Invoke graceful shutdown
        await client.close()

        # After close(), the transport should be cleared
        assert client._status_transport is None, (
            "Status transport should be cleared after close()"
        )
    finally:
        # close() is idempotent; ensure cleanup even if assertions fail earlier
        await client.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_status_stream_terminates_on_close(ports, server_proc):
    """status_stream consumers should terminate when AsyncRobotClient.close() is called.

    This test exercises the real server process and the real status listener
    to ensure that any background tasks consuming status_stream() are
    unblocked and finished by the time close() completes.
    """

    client = AsyncRobotClient(
        host=ports.server_ip, port=ports.server_port, timeout=1.0, retries=0
    )

    async def consumer() -> None:
        # Consume a few status messages until the client is closed.
        # The loop should terminate automatically when close() is invoked.
        async for _ in client.status_stream():
            # Yield control so we don't spin too fast in tests
            await asyncio.sleep(0)

    try:
        # Ensure the controller is responsive before starting the status listener
        await client.wait_ready(timeout=5.0)

        # Start the consumer task; this will internally trigger _ensure_endpoint()
        consumer_task = asyncio.create_task(consumer())

        # Wait briefly to allow the status listener and status stream to start
        await asyncio.sleep(0.5)

        # Closing the client should cause the consumer to exit its async-for loop
        await client.close()

        # The consumer task should complete promptly after close()
        await asyncio.wait_for(consumer_task, timeout=5.0)

        assert consumer_task.done(), (
            "status_stream consumer should be finished after close()"
        )
    finally:
        # Ensure cleanup even if assertions fail earlier
        await client.close()
