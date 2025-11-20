import asyncio

import pytest

from parol6.client.async_client import AsyncRobotClient


@pytest.mark.asyncio
@pytest.mark.integration
async def test_multicast_listener_shuts_down_on_close(ports, server_proc):
    """AsyncRobotClient.close() should cancel and clean up the real multicast listener.

    This test uses the real server process (server_proc), the real AsyncRobotClient,
    and the real multicast subscriber stack. It only relies on the existing
    test fixtures to start the FAKE_SERIAL controller on the auto-detected
    test ports.
    """

    client = AsyncRobotClient(
        host=ports.server_ip, port=ports.server_port, timeout=1.0, retries=0
    )

    try:
        # Ensure the controller is responsive before starting the multicast listener
        await client.wait_for_server_ready(timeout=5.0)

        # Force endpoint and multicast listener creation; this will invoke the
        # real _start_multicast_listener, which uses subscribe_status and
        # the underlying multicast socket implementation.
        await client._ensure_endpoint()
        task = client._multicast_task

        assert task is not None, "Multicast listener task should be created"
        assert not task.done(), (
            "Multicast listener task should be running before close()"
        )

        # Invoke graceful shutdown
        await client.close()

        # After close(), the task should be finished and cleared
        assert client._multicast_task is None, (
            "Multicast listener reference should be cleared after close()"
        )
        assert task.done(), "Multicast listener task should be completed after close()"
    finally:
        # close() is idempotent; ensure cleanup even if assertions fail earlier
        await client.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_status_stream_terminates_on_close(ports, server_proc):
    """status_stream consumers should terminate when AsyncRobotClient.close() is called.

    This test exercises the real server process and the real multicast subscriber
    stack to ensure that any background tasks consuming status_stream() are
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
        # Ensure the controller is responsive before starting the multicast listener
        await client.wait_for_server_ready(timeout=5.0)

        # Start the consumer task; this will internally trigger _ensure_endpoint()
        consumer_task = asyncio.create_task(consumer())

        # Wait briefly to allow the multicast listener and status stream to start
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
