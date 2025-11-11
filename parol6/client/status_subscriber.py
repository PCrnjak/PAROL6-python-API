import asyncio
import contextlib
import logging
import socket
import struct
import time
from collections.abc import AsyncIterator

from parol6 import config as cfg
from parol6.protocol.types import StatusAggregate
from parol6.protocol.wire import decode_status

logger = logging.getLogger(__name__)


class MulticastProtocol(asyncio.DatagramProtocol):
    """Protocol handler for multicast UDP datagrams that works with uvloop."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.transport = None
        self.receive_count = 0
        self.last_log_time = time.time()

        # EMA rate tracking for multicast RX
        self._rx_count = 0
        self._rx_last_time = time.monotonic()
        self._rx_ema_period = 0.05  # Initialize with 20 Hz expected
        self._rx_last_log_time = time.monotonic()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        # Track multicast RX rate with EMA
        now = time.monotonic()
        if self._rx_count > 0:  # Skip first sample for period calculation
            period = now - self._rx_last_time
            if period > 0:
                # EMA update: 0.1 * new + 0.9 * old
                self._rx_ema_period = 0.1 * period + 0.9 * self._rx_ema_period
        self._rx_last_time = now
        self._rx_count += 1

        # Log rate every 3 seconds
        if now - self._rx_last_log_time >= 3.0 and self._rx_ema_period > 0:
            rx_hz = 1.0 / self._rx_ema_period
            logger.debug(f"Multicast RX: {rx_hz:.1f} Hz (count={self._rx_count})")
            self._rx_last_log_time = now

        try:
            self.queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            # Drop oldest packet if queue is full
            try:
                self.queue.get_nowait()
                self.queue.put_nowait((data, addr))
            except:
                pass

    def error_received(self, exc):
        logger.error(f"Error received: {exc}")

    def connection_lost(self, exc):
        logger.info(f"Connection lost: {exc}")


def _create_multicast_socket(group: str, port: int, iface_ip: str) -> socket.socket:
    """Create and configure a multicast socket with loopback-first semantics and robust joins."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple listeners on same port; prefer SO_REUSEPORT where available
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except Exception:
        # Not available or not permitted on this platform; continue
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    # Bind to port (try wildcard first, then iface_ip)
    try:
        sock.bind(("", port))
    except OSError:
        sock.bind((iface_ip, port))

    # Helper to determine active NIC IP (no packets sent)
    def _detect_primary_ip() -> str:
        tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            tmp.connect(("1.1.1.1", 80))
            return tmp.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            with contextlib.suppress(Exception):
                tmp.close()

    # Join multicast group on specified interface (loopback preferred), with fallbacks
    try:
        mreq = struct.pack("=4s4s", socket.inet_aton(group), socket.inet_aton(iface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except Exception:
        # Retry using primary NIC IP
        try:
            primary_ip = _detect_primary_ip()
            mreq = struct.pack("=4s4s", socket.inet_aton(group), socket.inet_aton(primary_ip))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            # Final fallback: INADDR_ANY variant
            mreq_any = struct.pack("=4sl", socket.inet_aton(group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq_any)

    # Non-blocking for asyncio
    sock.setblocking(False)
    return sock


async def subscribe_status(
    group: str | None = None, port: int | None = None, iface_ip: str | None = None
) -> AsyncIterator[StatusAggregate]:
    """
    Async generator that yields decoded STATUS dicts from the UDP multicast broadcaster.

    Uses create_datagram_endpoint for uvloop compatibility.

    Usage:
        async for status in subscribe_status():
            # status is a dict with keys pose, angles, io, gripper (or None on parse failure)
            ...

    Notes:
    - Uses loopback multicast by default (cfg.MCAST_* values).
    - Yields only messages that decode successfully via decode_status; otherwise skips.
    """
    group = group or cfg.MCAST_GROUP
    port = port or cfg.MCAST_PORT
    iface_ip = iface_ip or cfg.MCAST_IF

    logger.info(f"subscribe_status starting: group={group}, port={port}, iface_ip={iface_ip}")

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=100)  # type: ignore

    # Create the socket with multicast configuration
    sock = _create_multicast_socket(group, port, iface_ip)

    # Create the datagram endpoint with our protocol
    transport = None
    try:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: MulticastProtocol(queue), sock=sock
        )

        while True:
            try:
                # Wait for data with timeout
                data, addr = await asyncio.wait_for(queue.get(), timeout=2.0)
                text = data.decode("ascii", errors="ignore")

                parsed = decode_status(text)
                if parsed is not None:
                    yield parsed

            except TimeoutError:
                logger.warning(f"No multicast received for 2s on {group}:{port} (iface={iface_ip})")
                continue

    except Exception as e:
        logger.error(f"Error in subscribe_status: {e}", exc_info=True)
        raise
    finally:
        if transport:
            logger.info("Closing transport...")
            transport.close()
        try:
            sock.close()
        except:
            pass
