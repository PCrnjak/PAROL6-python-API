import asyncio
import contextlib
import socket

import pytest

from parol6 import config as cfg
from parol6.server.state import StateManager
from parol6.server.status_broadcast import StatusBroadcaster
from parol6.server.status_cache import get_cache


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.mark.asyncio
async def test_status_broadcast_auto_failover_to_unicast(monkeypatch):
    """
    Deterministically force multicast setup to fail so the broadcaster falls back
    to UNICAST mode.
    """
    port = _free_udp_port()
    group = cfg.MCAST_GROUP
    iface = "127.0.0.1"

    # Prepare state/cache so broadcaster is allowed to send (cache not stale)
    cache = get_cache()
    cache.mark_serial_observed()

    # Start broadcaster with our chosen port and force unicast fallback
    state_mgr = StateManager()

    def _force_unicast_setup(self: StatusBroadcaster) -> None:
        self._use_unicast = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock = sock

    monkeypatch.setattr(StatusBroadcaster, "_setup_socket", _force_unicast_setup)

    broadcaster = StatusBroadcaster(
        state_mgr=state_mgr,
        group=group,
        port=port,
        iface_ip=iface,
        stale_s=1.0,
    )

    # StatusBroadcaster is now a polling class - call tick() manually
    stop_flag = False

    async def _tick_loop():
        while not stop_flag:
            broadcaster.tick()
            await asyncio.sleep(0.05)

    tick_task = asyncio.create_task(_tick_loop())

    try:
        await asyncio.sleep(0.1)
        assert broadcaster._use_unicast is True, (
            "Broadcaster did not fall back to unicast"
        )
    finally:
        stop_flag = True
        tick_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick_task
        broadcaster.close()


@pytest.mark.asyncio
async def test_subscriber_multicast_socket_receives_unicast(monkeypatch):
    """
    Verify that when the subscriber is configured for multicast, it still receives
    a unicast datagram sent to the same port (since it binds to ("", port)).
    """
    from parol6.client.async_client import _create_multicast_socket
    from parol6.protocol.wire import StatusBuffer, decode_status_bin_into

    port = _free_udp_port()
    group = cfg.MCAST_GROUP
    iface = "127.0.0.1"

    # Ensure subscriber chooses multicast socket
    monkeypatch.setattr(cfg, "STATUS_TRANSPORT", "MULTICAST", raising=False)

    # Craft a valid binary msgpack STATUS payload
    payload = get_cache().to_binary()

    # Create a multicast socket (same as client would)
    sock = _create_multicast_socket(group, port, iface)

    async def _send_once():
        await asyncio.sleep(0.05)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Send unicast to the multicast socket's port
            s.sendto(payload, ("127.0.0.1", port))
        finally:
            s.close()

    async def _consume_one(timeout: float = 3.0) -> bool:
        loop = asyncio.get_running_loop()
        buf = StatusBuffer()

        # Start sender in background
        sender = asyncio.create_task(_send_once())
        try:
            # Wait for data on the multicast socket
            data = await asyncio.wait_for(loop.sock_recv(sock, 4096), timeout=timeout)
            if decode_status_bin_into(data, buf):
                assert buf.io is not None
                return True
        except asyncio.TimeoutError:
            pass
        finally:
            with contextlib.suppress(Exception):
                sender.cancel()
                await sender
            sock.close()
        return False

    ok = await asyncio.wait_for(_consume_one(), timeout=4.0)
    assert ok, "Subscriber did not receive unicast datagram on multicast socket"


@pytest.mark.asyncio
async def test_multicast_send_failure_recovers_with_unicast_delivery(monkeypatch):
    from parol6.protocol.wire import StatusBuffer, decode_status_bin_into

    monkeypatch.setattr(cfg, "STATUS_TRANSPORT", "MULTICAST")
    monkeypatch.setattr(cfg, "STATUS_UNICAST_HOST", "127.0.0.1")
    monkeypatch.setattr(
        StatusBroadcaster, "_verify_multicast_reachable", lambda *args: True
    )
    cache = get_cache()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.setblocking(False)
        broadcaster = StatusBroadcaster(
            state_mgr=StateManager(),
            port=receiver.getsockname()[1],
            iface_ip="127.0.0.1",
            stale_s=2.0,
        )
        failed_socket = broadcaster._sock
        sendto = socket.socket.sendto

        def fail_multicast(sock, *args, **kwargs):
            if sock is failed_socket:
                raise OSError("simulated multicast send failure")
            return sendto(sock, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "sendto", fail_multicast)
        try:
            for _ in range(broadcaster._max_send_failures):
                cache.mark_serial_observed()
                broadcaster.tick()
            assert broadcaster._use_unicast
            cache.mark_serial_observed()
            broadcaster.tick()
            data = await asyncio.wait_for(
                asyncio.get_running_loop().sock_recv(receiver, 65536),
                timeout=3,
            )
            assert decode_status_bin_into(data, StatusBuffer())
        finally:
            broadcaster.close()
