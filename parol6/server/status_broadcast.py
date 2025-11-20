from __future__ import annotations

import logging
import socket
import threading
import time

from parol6 import config as cfg
from parol6.server.state import StateManager
from parol6.server.status_cache import get_cache

logger = logging.getLogger(__name__)


class StatusBroadcaster(threading.Thread):
    """
    Broadcasts ASCII STATUS frames via UDP.

    Transport:
      - cfg.STATUS_TRANSPORT: "MULTICAST" (default) or "UNICAST"

    Multicast Config (used when STATUS_TRANSPORT == MULTICAST):
      - cfg.MCAST_GROUP (default "239.255.0.101")
      - cfg.MCAST_PORT  (default 50510)
      - cfg.MCAST_TTL   (default 1)
      - cfg.MCAST_IF    (default "127.0.0.1")

    Unicast Config (used when STATUS_TRANSPORT == UNICAST):
      - cfg.STATUS_UNICAST_HOST (default "127.0.0.1")

    General:
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
        rate_hz: float = cfg.STATUS_RATE_HZ,
        stale_s: float = cfg.STATUS_STALE_S,
    ) -> None:
        super().__init__(daemon=True)
        self._state_mgr = state_mgr
        self.group = group
        self.port = port
        self.ttl = ttl
        self.iface_ip = iface_ip
        self._period = 1.0 / max(rate_hz, 1.0)
        self._stale_s = stale_s

        # Negotiated transport (can be forced via env or auto-fallback at runtime)
        self._use_unicast: bool = cfg.STATUS_TRANSPORT == "UNICAST"

        self._sock: socket.socket | None = None
        self._running = threading.Event()
        self._running.set()

        # EMA rate tracking for TX
        self._tx_count = 0
        self._tx_last_time = time.monotonic()
        self._tx_ema_period = 1.0 / rate_hz  # Initialize with expected period
        self._tx_last_log_time = time.monotonic()  # For 3-second logging interval

        # Failure tracking for runtime fallback
        self._send_failures = 0
        self._max_send_failures = 3

    def _detect_primary_ip(self) -> str:
        tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            tmp.connect(("1.1.1.1", 80))
            return tmp.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            try:
                tmp.close()
            except Exception:
                pass

    def _verify_multicast_reachable(
        self, sock: socket.socket, timeout: float = 0.1
    ) -> bool:
        """
        Attempt a loopback reachability check for multicast by joining the group on a
        temporary receiver and sending a probe. Returns True if the probe is received.
        """
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
            recv_sock.bind(("", self.port))
            mreq = socket.inet_aton(self.group) + socket.inet_aton(self.iface_ip)
            recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            recv_sock.settimeout(timeout)

            token = b"PAROL6_MCAST_PROBE"
            try:
                sock.sendto(token, (self.group, self.port))
            except OSError as e:
                logger.debug(f"Multicast probe send failed: {e}")
                return False
            try:
                data, _ = recv_sock.recvfrom(2048)
                return data == token
            except Exception:
                return False
        finally:
            try:
                recv_sock.close()
            except Exception:
                pass

    def _switch_to_unicast(self) -> None:
        """Close current socket and switch to unicast transport."""
        try:
            if self._sock:
                self._sock.close()
        except Exception as e:
            logger.debug(f"Error closing multicast socket during fallback: {e}")
        usock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        usock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock = usock
        self._use_unicast = True
        self._send_failures = 0
        logger.info(
            f"StatusBroadcaster (UNICAST-FALLBACK) -> dest={cfg.STATUS_UNICAST_HOST}:{self.port}"
        )

    def _setup_socket(self) -> None:
        # UNICAST: simple UDP socket without multicast options
        if self._use_unicast:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
            self._sock = sock
            logger.info(
                f"StatusBroadcaster (UNICAST) -> dest={cfg.STATUS_UNICAST_HOST}:{self.port}"
            )
            return

        # MULTICAST: configure multicast TTL/IF with verification and fallback
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

        try:
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(self.iface_ip),
            )
            if not self._verify_multicast_reachable(sock):
                raise RuntimeError(
                    f"Initial multicast reachability check failed on iface {self.iface_ip}"
                )
        except Exception as e:
            logger.warning(
                f"StatusBroadcaster: interface {self.iface_ip} failed verification: {e}"
            )
            try:
                primary_ip = self._detect_primary_ip()
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_MULTICAST_IF,
                    socket.inet_aton(primary_ip),
                )
                logger.info(
                    f"StatusBroadcaster: fallback IP_MULTICAST_IF to {primary_ip}"
                )
                if not self._verify_multicast_reachable(sock):
                    raise RuntimeError("Fallback multicast reachability failed")
            except Exception as e2:
                logger.warning(
                    f"StatusBroadcaster: failed to set IP_MULTICAST_IF: {e2}"
                )
                # As a last resort, switch to UNICAST
                try:
                    sock.close()
                except Exception:
                    pass
                self._switch_to_unicast()
                return

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock = sock
        logger.info(
            f"StatusBroadcaster (MULTICAST) -> group={self.group} port={self.port} iface={self.iface_ip} ttl={self.ttl}"
        )

    def run(self) -> None:
        self._setup_socket()
        cache = get_cache()

        # Validate socket exists
        if self._sock is None:
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
                # Refresh socket and destination each loop in case we switched transports
                sock = self._sock
                if sock is None:
                    # Socket disappeared unexpectedly; try to switch to unicast and continue
                    self._switch_to_unicast()
                    sock = self._sock
                dest = (
                    (cfg.STATUS_UNICAST_HOST, self.port)
                    if self._use_unicast
                    else (self.group, self.port)
                )
                try:
                    sock.sendto(memoryview(payload), dest)  # type: ignore[arg-type]
                except OSError as e:
                    self._send_failures += 1
                    # Log occasionally to avoid flooding
                    if time.monotonic() - self._tx_last_log_time >= 5.0:
                        logger.warning(f"StatusBroadcaster send failed: {e}")
                        self._tx_last_log_time = time.monotonic()
                    # If too many failures and we are on multicast, fall back to unicast
                    if (
                        not self._use_unicast
                        and self._send_failures >= self._max_send_failures
                    ):
                        logger.info(
                            f"StatusBroadcaster: {self._send_failures} consecutive send errors; switching to UNICAST"
                        )
                        self._switch_to_unicast()
                else:
                    # Reset failure count on success
                    self._send_failures = 0
                    # Track TX rate with EMA
                    now = time.monotonic()
                    if self._tx_count > 0:  # Skip first sample for period calculation
                        period = now - self._tx_last_time
                        if period > 0:
                            # EMA update: 0.1 * new + 0.9 * old
                            self._tx_ema_period = (
                                0.1 * period + 0.9 * self._tx_ema_period
                            )
                    self._tx_last_time = now
                    self._tx_count += 1

                    # Log rate every 3 seconds
                    if now - self._tx_last_log_time >= 3.0 and self._tx_ema_period > 0:
                        tx_hz = 1.0 / self._tx_ema_period
                        logger.debug(
                            f"Status TX: {tx_hz:.1f} Hz (count={self._tx_count})"
                        )
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
