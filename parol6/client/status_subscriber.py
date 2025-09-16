from __future__ import annotations

import asyncio
import socket
import struct
from typing import AsyncIterator, Callable, Optional

from parol6 import config as cfg
from parol6.protocol.wire import decode_status
from parol6.protocol.types import StatusAggregate


def _join_multicast_group(sock: socket.socket, group_ip: str, iface_ip: str) -> None:
    """
    Join an IPv4 multicast group on a specific interface.
    """
    mreq = struct.pack("=4s4s", socket.inet_aton(group_ip), socket.inet_aton(iface_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)


async def subscribe_status(
    group: str | None = None,
    port: int | None = None,
    iface_ip: str | None = None,
    bufsize: int = 8192,
) -> AsyncIterator[StatusAggregate]:
    """
    Async generator that yields decoded STATUS dicts from the UDP multicast broadcaster.

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

    loop = asyncio.get_running_loop()

    # Create a UDP socket and join multicast group
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # Allow multiple listeners on same port
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
    except OSError:
        # Fallback to binding to the interface explicitly
        sock.bind((iface_ip, port))
    # Join multicast group on the specified interface
    _join_multicast_group(sock, group, iface_ip)
    # Non-blocking for asyncio
    sock.setblocking(False)

    try:
        while True:
            data, _ = await loop.sock_recvfrom(sock, bufsize)
            text = data.decode("ascii", errors="ignore")
            parsed = decode_status(text)
            if parsed is not None:
                yield parsed
    finally:
        try:
            sock.close()
        except Exception:
            pass
