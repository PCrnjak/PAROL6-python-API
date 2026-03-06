from __future__ import annotations

import logging
import socket
import sys
import time

from parol6 import config as cfg
from parol6.server.state import StateManager
from parol6.server.status_cache import get_cache

logger = logging.getLogger(__name__)


class StatusBroadcaster:
    """
    Broadcasts binary msgpack STATUS frames via UDP. Called from main loop.

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
        group: str = cfg.MCAST_GROUP,
        port: int = cfg.MCAST_PORT,
        ttl: int = cfg.MCAST_TTL,
        iface_ip: str = cfg.MCAST_IF,
        rate_hz: float = cfg.STATUS_RATE_HZ,
        stale_s: float = cfg.STATUS_STALE_S,
    ) -> None:
        self._state_mgr = state_mgr
        self.group = group
        self.port = port
        self.ttl = ttl
        self.iface_ip = iface_ip
        self._stale_s = stale_s

        # Negotiated transport (can be forced via env or auto-fallback at runtime)
        self._use_unicast: bool = cfg.STATUS_TRANSPORT == "UNICAST"

        self._sock: socket.socket | None = None

        # Failure tracking for runtime fallback
        self._send_failures = 0
        self._max_send_failures = 3
        self._last_fail_log_time = 0.0

        # Setup socket on construction
        self._setup_socket()

    def _detect_primary_ip(self) -> str:
        tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            tmp.connect(("1.1.1.1", 80))
            return tmp.getsockname()[0]
        except Exception as e:
            logger.debug("IP detect: %s", e)
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
            sock.setblocking(False)
            self._sock = sock
            logger.info(
                f"StatusBroadcaster (UNICAST) -> dest={cfg.STATUS_UNICAST_HOST}:{self.port}"
            )
            return

        # MULTICAST: configure multicast TTL/IF with verification and fallback
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.ttl)
        # macOS requires loopback enabled for multicast to work on localhost
        if sys.platform == "darwin":
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        sock.setblocking(False)

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

    def tick(self) -> None:
        """Broadcast status if cache is fresh. Called from main loop."""
        cache = get_cache()
        try:
            state = self._state_mgr.get_state()
            cache.update_from_state(state)
        except Exception as e:
            logger.debug("StatusBroadcaster cache refresh failed: %s", e)
            return

        if cache.age_s() > self._stale_s:
            return

        payload = cache.to_binary()
        sock = self._sock
        if sock is None:
            self._switch_to_unicast()
            sock = self._sock
            assert sock is not None  # _switch_to_unicast always sets _sock

        dest = (
            (cfg.STATUS_UNICAST_HOST, self.port)
            if self._use_unicast
            else (self.group, self.port)
        )

        try:
            sock.sendto(payload, dest)
            self._send_failures = 0

        except BlockingIOError:
            pass  # Transient kernel buffer pressure — not a multicast failure
        except OSError as e:
            self._handle_send_failure(e)

    def _handle_send_failure(self, e: OSError) -> None:
        """Handle send failure with logging and fallback."""
        self._send_failures += 1
        now = time.monotonic()
        if now - self._last_fail_log_time >= 5.0:
            logger.warning(f"StatusBroadcaster send failed: {e}")
            self._last_fail_log_time = now
        if not self._use_unicast and self._send_failures >= self._max_send_failures:
            logger.info(
                f"StatusBroadcaster: {self._send_failures} consecutive send errors; switching to UNICAST"
            )
            self._switch_to_unicast()

    def close(self) -> None:
        """Close socket."""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
