"""Digital I/O uses logical output indices and per-call reply deadlines."""

import asyncio
import socket
import time

import pytest
from waldoctl.skills import skill

from parol6 import AsyncRobotClient


def test_digital_io_readback_and_missing_peer_deadlines(client, server_proc):
    before = client.io(timeout=2)
    assert before is not None
    try:
        assert client.write_io(0, 1 - before[2], timeout=2) >= 0
        assert client.wait_status(lambda s: s.io[2] == 1 - before[2], timeout=2)
        assert client.io(timeout=2)[2] == 1 - before[2]
    finally:
        client.write_io(0, before[2], timeout=2)

    async def missing_peer():
        @skill(id="test.io_deadline", version="1.0.0")
        async def query(rbt):
            return await rbt.io(timeout=0.05)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as silent:
            silent.bind(("127.0.0.1", 0))
            async with AsyncRobotClient(
                port=silent.getsockname()[1], timeout=5, retries=3
            ) as absent:
                start = time.monotonic()
                assert await query.async_call(absent) is None
                assert time.monotonic() - start < 1.0, (
                    "query ignored its per-call deadline"
                )
                start = time.monotonic()
                with pytest.raises(TimeoutError):
                    await absent.write_io(0, 1, timeout=0.05)
                assert time.monotonic() - start < 1.0, (
                    "write ignored its per-call deadline"
                )
                for invalid in (0, -1, float("nan"), float("inf"), True):
                    with pytest.raises(ValueError):
                        await absent.io(timeout=invalid)
                    with pytest.raises(ValueError):
                        await absent.write_io(0, 1, timeout=invalid)

    asyncio.run(missing_peer())
