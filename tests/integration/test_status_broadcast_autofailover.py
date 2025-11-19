import asyncio
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
async def test_status_broadcast_auto_failover_receives_frame():
    """
    Verify that a StatusBroadcaster transmits on the configured port and that
    the subscriber receives at least one frame, regardless of whether multicast
    works on the host. On environments where multicast is unavailable (e.g.,
    some CI runners), the broadcaster should automatically fall back to unicast
    and the subscriber should still receive frames on the same port.
    """
    port = _free_udp_port()
    group = cfg.MCAST_GROUP
    iface = "127.0.0.1"

    # Prepare state/cache so broadcaster is allowed to send (cache not stale)
    cache = get_cache()
    cache.mark_serial_observed()

    # Start broadcaster with our chosen port
    state_mgr = StateManager()
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

        # Consume a single status frame (multicast or unicast) with timeout
        async def _consume_one(timeout: float = 3.0):
            start = time.time()
            async for status in subscribe_status(group=group, port=port, iface_ip=iface):
                # Basic sanity checks on parsed payload
                assert isinstance(status, dict)
                assert "angles" in status
                assert "io" in status
                return True
                if time.time() - start > timeout:
                    break
            return False

        ok = await asyncio.wait_for(_consume_one(), timeout=4.0)
        assert ok, "Did not receive a status frame within timeout"

    finally:
        broadcaster.stop()
        # Best-effort join
        try:
            broadcaster.join(timeout=1.0)
        except Exception:
            pass
