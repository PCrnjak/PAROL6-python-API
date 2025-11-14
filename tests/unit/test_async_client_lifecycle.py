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

    client = AsyncRobotClient(host=ports.server_ip, port=ports.server_port, timeout=1.0, retries=0)

    try:
        # Ensure the controller is responsive before starting the multicast listener
        await client.wait_for_server_ready(timeout=5.0)

        # Force endpoint and multicast listener creation; this will invoke the
        # real _start_multicast_listener, which uses subscribe_status and
        # the underlying multicast socket implementation.
        await client._ensure_endpoint()
        task = client._multicast_task

        assert task is not None, "Multicast listener task should be created"
        assert not task.done(), "Multicast listener task should be running before close()"

        # Invoke graceful shutdown
        await client.close()

        # After close(), the task should be finished and cleared
        assert client._multicast_task is None, "Multicast listener reference should be cleared after close()"
        assert task.done(), "Multicast listener task should be completed after close()"
    finally:
        # close() is idempotent; ensure cleanup even if assertions fail earlier
        await client.close()
