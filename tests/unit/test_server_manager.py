
import pytest

from parol6.client.manager import ServerManager, is_server_running, manage_server


@pytest.mark.asyncio
async def test_is_server_running_false_when_no_server(monkeypatch):
    # Use an unlikely high port to avoid collisions
    assert not await is_server_running(host="127.0.0.1", port=59999, timeout=0.2)


@pytest.mark.asyncio
async def test_manage_server_starts_and_reports_running(monkeypatch):
    # Choose a high-numbered UDP port to minimize collision risk
    host = "127.0.0.1"
    port = 59998

    # Ensure no server is running before we start
    assert not await is_server_running(host=host, port=port, timeout=0.2)

    manager: ServerManager | None = None
    try:
        manager = await manage_server(host=host, port=port, com_port=None, extra_env=None, normalize_logs=False)
        assert isinstance(manager, ServerManager)

        # After manage_server, the UDP endpoint should respond to PING
        assert await is_server_running(host=host, port=port, timeout=1.0)
    finally:
        if manager is not None:
            await manager.stop_controller()


@pytest.mark.asyncio
async def test_manage_server_fast_fails_when_already_running(monkeypatch):
    host = "127.0.0.1"
    port = 59997

    manager: ServerManager | None = None
    try:
        # First start a server
        manager = await manage_server(host=host, port=port, com_port=None, extra_env=None, normalize_logs=False)
        assert await is_server_running(host=host, port=port, timeout=1.0)

        # Second attempt should raise RuntimeError because the port is taken by an existing server
        with pytest.raises(RuntimeError):
            await manage_server(host=host, port=port, com_port=None, extra_env=None, normalize_logs=False)
    finally:
        if manager is not None:
            await manager.stop_controller()
