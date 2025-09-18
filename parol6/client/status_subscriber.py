from __future__ import annotations

import asyncio
import socket
import struct
import time
import logging
from typing import AsyncIterator

from parol6 import config as cfg
from parol6.protocol.wire import decode_status
from parol6.protocol.types import StatusAggregate

logger = logging.getLogger(__name__)


class MulticastProtocol(asyncio.DatagramProtocol):
    """Protocol handler for multicast UDP datagrams that works with uvloop."""
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.transport = None
        self.receive_count = 0
        self.last_log_time = time.time()
        
    def connection_made(self, transport):
        self.transport = transport
        
    def datagram_received(self, data, addr):
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
    """Create and configure a multicast socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    
    # Allow multiple listeners on same port
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind to port
    try:
        sock.bind(("", port))
    except OSError as e:
        sock.bind((iface_ip, port))
    
    # Join multicast group on the specified interface
    mreq = struct.pack("=4s4s", socket.inet_aton(group), socket.inet_aton(iface_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    # Non-blocking for asyncio
    sock.setblocking(False)
    
    return sock


async def subscribe_status(
    group: str | None = None,
    port: int | None = None,
    iface_ip: str | None = None
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
    queue = asyncio.Queue(maxsize=100)
    
    # Create the socket with multicast configuration
    sock = _create_multicast_socket(group, port, iface_ip)
    
    # Create the datagram endpoint with our protocol
    transport = None
    try:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: MulticastProtocol(queue),
            sock=sock
        )
        
        while True:
            try:
                # Wait for data with timeout
                data, addr = await asyncio.wait_for(queue.get(), timeout=2.0)
                text = data.decode("ascii", errors="ignore")
                    
                parsed = decode_status(text)
                if parsed is not None:
                    yield parsed
                        
            except asyncio.TimeoutError:
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
