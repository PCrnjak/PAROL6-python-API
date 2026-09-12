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


def test_late_replies_never_answer_the_next_request(ports, server_proc):
    """A reply that lands after its caller's deadline expired is not served to
    the next request: it carries the abandoned request's id, so the client
    drops it instead of answering the wrong query and leaving every later one
    a reply behind."""
    from parol6.protocol.wire import IOResultStruct, pack_ok, pack_response

    async def scenario():
        async with AsyncRobotClient(
            host=ports.server_ip, port=ports.server_port, timeout=5.0
        ) as rbt:
            await rbt._ensure_endpoint()
            peer = (ports.server_ip, ports.server_port)
            abandoned = 10_000  # an id no live request will be given
            rbt._rx_queue.put_nowait(
                (pack_response(IOResultStruct(io=[0, 0, 0, 0, 1]), abandoned), peer)
            )
            pose = await rbt.pose()
            assert pose is not None and len(pose) == 6
            assert await rbt.angles() is not None
            rbt._rx_queue.put_nowait((pack_ok(abandoned), peer))
            index = await rbt.delay(0.1)
            assert index >= 1, "a stale index-less OK must not stand in for the ack"
            assert await rbt.wait_command(index, timeout=5)
            # A deadline that lapses mid-flight: the reply lands afterwards
            # and must be dropped before the next query goes out.
            assert await rbt.io(timeout=1e-4) is None
            await asyncio.sleep(0.1)
            pose = await rbt.pose()
            assert pose is not None and len(pose) == 6

    asyncio.run(scenario())
