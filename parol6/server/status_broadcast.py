from __future__ import annotations

import os
import socket
import struct
import threading
import time
import logging
from typing import Optional

from parol6.server.state import StateManager
from parol6.server.status_cache import get_cache
from parol6 import config as cfg

logger = logging.getLogger(__name__)


class StatusUpdater(threading.Thread):
    """
    Periodically updates the status cache from the controller state.

    Rate controlled by config.STATUS_RATE_HZ (default 50).
    """

    def __init__(self, state_mgr: StateManager, rate_hz: float = 50.0) -> None:
        super().__init__(daemon=True)
        self._state_mgr = state_mgr
        self._rate_hz = rate_hz
        self._running = threading.Event()
        self._running.set()

    def run(self) -> None:
        cache = get_cache()
        period = 1.0 / max(self._rate_hz, 1.0)
        while self._running.is_set():
            state = self._state_mgr.get_state()
            cache.update_from_state(state)
            time.sleep(period)

    def stop(self) -> None:
        self._running.clear()


class StatusBroadcaster(threading.Thread):
    """
    Broadcasts ASCII STATUS frames via UDP multicast.

    Config:
      - cfg.MCAST_GROUP (default "239.255.0.101")
      - cfg.MCAST_PORT (default 50510)
      - cfg.MCAST_TTL  (default 1)
      - cfg.MCAST_IF   (default "127.0.0.1")
      - cfg.STATUS_RATE_HZ (default 50)
      - cfg.STATUS_STALE_S (default 0.2) -> skip broadcast if cache is stale
    """

    def __init__(
        self,
        group: str = "239.255.0.101",
        port: int = 50510,
        ttl: int = 1,
        iface_ip: str = "127.0.0.1",
        rate_hz: float = 50.0,
        stale_s: float = 0.2,
    ) -> None:
        super().__init__(daemon=True)
        self.group = group
        self.port = port
        self.ttl = ttl
        self.iface_ip = iface_ip
        self._period = 1.0 / max(rate_hz, 1.0)
        self._stale_s = stale_s

        self._sock: Optional[socket.socket] = None
        self._running = threading.Event()
        self._running.set()

    def _setup_socket(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self.iface_ip))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock = sock

    def run(self) -> None:
        self._setup_socket()
        cache = get_cache()
        dest = (self.group, self.port)
        sock = self._sock
        if sock is None:
            logger.error("StatusBroadcaster socket not initialized")
            return

        while self._running.is_set():
            # Skip broadcast if cache is stale (e.g., serial disconnected)
            if cache.age_s() <= self._stale_s:
                payload = cache.to_ascii().encode("ascii", errors="ignore")
                # memoryview avoids an extra copy in some implementations
                sock.sendto(memoryview(payload), dest)
            time.sleep(self._period)

    def stop(self) -> None:
        self._running.clear()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass


def start_status_services(state_mgr: StateManager) -> tuple[StatusUpdater, StatusBroadcaster]:
    """
    Helper to start updater and broadcaster using central config.
    """
    rate_hz = cfg.STATUS_RATE_HZ
    updater = StatusUpdater(state_mgr, rate_hz=rate_hz)

    group = cfg.MCAST_GROUP
    port = cfg.MCAST_PORT
    ttl = cfg.MCAST_TTL
    iface = cfg.MCAST_IF
    stale_s = cfg.STATUS_STALE_S

    logger.info(f"StatusBroadcaster config: group={group} port={port} ttl={ttl} iface={iface} rate_hz={rate_hz} stale_s={stale_s}")
    broadcaster = StatusBroadcaster(group=group, port=port, ttl=ttl, iface_ip=iface, rate_hz=rate_hz, stale_s=stale_s)

    updater.start()
    broadcaster.start()
    return updater, broadcaster
