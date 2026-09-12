"""A status producer speaking another protocol version is named, not silence.

The client and the controller are released separately, and a field added to
the status layout shifts every slot after it. Before the version travelled on
the wire, an older controller's status simply failed to decode and the client
reported nothing at all — which reads as an unplugged arm and sends an
operator looking at cables instead of at versions.
"""

from __future__ import annotations

import asyncio
import socket

import numpy as np
import pytest
from waldoctl import ActionState

from parol6 import config as cfg
from parol6.client.async_client import AsyncRobotClient
from parol6.protocol.wire import (
    PROTO_VERSION,
    MsgType,
    ProtocolVersionError,
    decode,
    encode,
    pack_status,
)


def _status(version: int) -> bytes:
    """A well-formed status broadcast, relabelled with *version*."""
    packed = pack_status(
        np.eye(4, dtype=np.float64).ravel(),
        np.zeros(6, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.zeros(5, dtype=np.uint8),
        "",
        ActionState.IDLE,
        np.ones(12, dtype=np.uint8),
        np.ones(12, dtype=np.uint8),
        np.ones(12, dtype=np.uint8),
    )
    if version == PROTO_VERSION:
        return packed
    fields = decode(packed)
    assert fields[0] == MsgType.STATUS
    fields[1] = version
    return encode(fields)


def test_a_status_from_another_protocol_version_reaches_the_caller(monkeypatch):
    monkeypatch.setattr(cfg, "STATUS_TRANSPORT", "UNICAST")
    monkeypatch.setattr(cfg, "STATUS_UNICAST_HOST", "127.0.0.1")

    async def scenario() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            status_port = probe.getsockname()[1]
        monkeypatch.setattr(cfg, "MCAST_PORT", status_port)
        # No controller: this client only listens for the status broadcast.
        client = AsyncRobotClient(port=status_port + 1, timeout=0.05, retries=0)
        try:
            await client._ensure_endpoint()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as producer:
                producer.sendto(_status(PROTO_VERSION), ("127.0.0.1", status_port))
                assert await client.wait_status(
                    lambda s: s.action_state == ActionState.IDLE, timeout=2
                ), "a status of this version is read normally"

                producer.sendto(_status(PROTO_VERSION + 1), ("127.0.0.1", status_port))
                with pytest.raises(ProtocolVersionError, match="update the older side"):
                    await client.wait_status(lambda s: False, timeout=2)
                # It keeps saying so: a program cannot mistake the mismatch for
                # a level that has not arrived yet.
                with pytest.raises(ProtocolVersionError):
                    await client.wait_status(lambda s: False, timeout=0.1)
        finally:
            await client.close()

    asyncio.run(scenario())
