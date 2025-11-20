import asyncio
import contextlib
import socket
import time

import pytest

from parol6 import config as cfg
from parol6.server.state import StateManager
from parol6.server.status_broadcast import StatusBroadcaster
from parol6.server.status_cache import get_cache
from parol6.client.status_subscriber import subscribe_status


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


@pytest.mark.asyncio
async def test_status_broadcast_auto_failover_receives_frame(monkeypatch):
    """
    Deterministically force multicast setup to fail so the broadcaster falls back
    to UNICAST and verify that a multicast-configured subscriber still receives
    frames on the same port.
    """
    port = _free_udp_port()
    group = cfg.MCAST_GROUP
    iface = "127.0.0.1"

    # Ensure subscriber uses multicast socket (which also accepts unicast to port)
    monkeypatch.setattr(cfg, "STATUS_TRANSPORT", "MULTICAST", raising=False)
    monkeypatch.setattr(cfg, "STATUS_UNICAST_HOST", "127.0.0.1", raising=False)

    # Prepare state/cache so broadcaster is allowed to send (cache not stale)
    cache = get_cache()
    cache.mark_serial_observed()

    # Start broadcaster with our chosen port and force unicast fallback
    state_mgr = StateManager()

    def _force_unicast_setup(self: StatusBroadcaster) -> None:  # type: ignore[no-redef]
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
        rate_hz=20.0,
        stale_s=1.0,
    )

    broadcaster.start()

    try:
        # Give broadcaster a tiny moment to initialize
        await asyncio.sleep(0.05)
        assert broadcaster._use_unicast is True, "Broadcaster did not fall back to unicast"

        async def _consume_one(timeout: float = 3.0) -> bool:
            deadline = time.time() + timeout
            async for status in subscribe_status(group=group, port=port, iface_ip=iface):
                assert isinstance(status, dict)
                assert "angles" in status
                return True
                if time.time() > deadline:
                    break
            return False

        ok = await asyncio.wait_for(_consume_one(), timeout=4.0)
        assert ok, "Did not receive a status frame within timeout"

    finally:
        broadcaster.stop()
        with contextlib.suppress(Exception):
            broadcaster.join(timeout=1.0)


@pytest.mark.asyncio
async def test_subscriber_multicast_socket_receives_unicast(monkeypatch):
    """
    Verify that when the subscriber is configured for multicast, it still receives
    a unicast datagram sent to the same port (since it binds to ("", port)).
    """
    port = _free_udp_port()
    group = cfg.MCAST_GROUP
    iface = "127.0.0.1"

    # Ensure subscriber chooses multicast socket
    monkeypatch.setattr(cfg, "STATUS_TRANSPORT", "MULTICAST", raising=False)

    # Craft a valid STATUS payload (defaults are acceptable for parsing)
    payload = get_cache().to_ascii().encode("ascii", errors="ignore")

    async def _send_once():
        await asyncio.sleep(0.05)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(payload, ("127.0.0.1", port))
        finally:
            s.close()

    async def _consume_one(timeout: float = 3.0) -> bool:
        # Start sender in background
        sender = asyncio.create_task(_send_once())
        try:
            async for status in subscribe_status(group=group, port=port, iface_ip=iface):
                assert isinstance(status, dict)
                assert "io" in status
                return True
        finally:
            with contextlib.suppress(Exception):
                sender.cancel()
                await sender
        # If we exit the loop without receiving, signal failure
        return False

    ok = await asyncio.wait_for(_consume_one(), timeout=4.0)
    assert ok, "Subscriber did not receive unicast datagram on multicast socket"


def _raise_sendto(*args, **kwargs):  # helper for monkeypatching socket.sendto
    raise OSError("simulated send failure")


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_multicast_send_errors_should_trigger_fallback_but_currently_do_not(monkeypatch):
    """
    Demonstrate the bug: if multicast setup succeeds but subsequent send() calls fail,
    the broadcaster should fall back to UNICAST. Current implementation does not,
    so this test is expected to FAIL until the logic is improved.
    """
    port = _free_udp_port()
    # Ensure we attempt multicast path
    monkeypatch.setattr(cfg, "STATUS_TRANSPORT", "MULTICAST", raising=False)

    cache = get_cache()
    cache.mark_serial_observed()

    state_mgr = StateManager()
    broadcaster = StatusBroadcaster(state_mgr=state_mgr, port=port, iface_ip="127.0.0.1", rate_hz=20.0, stale_s=2.0)
    broadcaster.start()
    try:
        # Allow setup to complete and at least one send to work
        await asyncio.sleep(0.1)

        # From now on, every sendto should fail
        monkeypatch.setattr(socket.socket, "sendto", _raise_sendto)

        # Give it a few cycles to "detect" and hypothetically fall back
        await asyncio.sleep(0.3)

        # The desired behavior would be to switch to unicast after persistent errors.
        # Current code does not, so this assertion should FAIL, making the problem visible.
        assert broadcaster._use_unicast is True, "Broadcaster did not fall back to unicast on repeated send errors"
    finally:
        broadcaster.stop()
        with contextlib.suppress(Exception):
            broadcaster.join(timeout=1.0)
