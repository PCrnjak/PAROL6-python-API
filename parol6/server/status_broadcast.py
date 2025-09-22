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
        state_mgr: StateManager,
        group: str = "239.255.0.101",
        port: int = 50510,
        ttl: int = 1,
        iface_ip: str = "127.0.0.1",
        rate_hz: float = 20.0,
        stale_s: float = 0.2,
    ) -> None:
        super().__init__(daemon=True)
        self._state_mgr = state_mgr
        self.group = group
        self.port = port
        self.ttl = ttl
        self.iface_ip = iface_ip
        self._period = 1.0 / max(rate_hz, 1.0)
        self._stale_s = stale_s

        self._sock: Optional[socket.socket] = None
        self._running = threading.Event()
        self._running.set()
        
        # EMA rate tracking for multicast TX
        self._tx_count = 0
        self._tx_last_time = time.monotonic()
        self._tx_ema_period = 1.0 / rate_hz  # Initialize with expected period
        self._tx_last_log_time = time.monotonic()  # For 3-second logging interval

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

        # Deadline-based timing to maintain consistent rate
        next_deadline = time.monotonic() + self._period
        
        while self._running.is_set():
            # Always refresh cache from latest state before considering broadcast
            try:
                state = self._state_mgr.get_state()
                cache.update_from_state(state)
            except Exception as e:
                logger.debug("StatusBroadcaster cache refresh failed: %s", e)

            # Skip broadcast if cache is stale (e.g., serial disconnected)
            if cache.age_s() <= self._stale_s:
                payload = cache.to_ascii().encode("ascii", errors="ignore")
                # memoryview avoids an extra copy in some implementations
                sock.sendto(memoryview(payload), dest)
                
                # Track multicast TX rate with EMA
                now = time.monotonic()
                if self._tx_count > 0:  # Skip first sample for period calculation
                    period = now - self._tx_last_time
                    if period > 0:
                        # EMA update: 0.1 * new + 0.9 * old
                        self._tx_ema_period = 0.1 * period + 0.9 * self._tx_ema_period
                self._tx_last_time = now
                self._tx_count += 1
                
                # Log rate every 3 seconds
                if now - self._tx_last_log_time >= 3.0 and self._tx_ema_period > 0:
                    tx_hz = 1.0 / self._tx_ema_period
                    logger.debug(f"Multicast TX: {tx_hz:.1f} Hz (count={self._tx_count})")
                    self._tx_last_log_time = now
            
            # Sleep until next deadline (compensates for work time)
            sleep_time = next_deadline - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_deadline += self._period

    def stop(self) -> None:
        self._running.clear()
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
