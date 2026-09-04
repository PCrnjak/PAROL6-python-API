"""The control loop's own health rides the status broadcast.

A display that wants to say whether the loop is keeping up should not have
to poll ``loop_stats()`` for it: the period tail and the deadline-miss
count are on every STATUS frame, and they are the live numbers rather than
a boot-time snapshot.
"""

import asyncio

import pytest

from parol6 import AsyncRobotClient


@pytest.mark.asyncio
@pytest.mark.integration
async def test_status_carries_the_loops_own_health(server_proc, ports):
    """STATUS reports the loop percentile and overrun count, and they agree
    with what the LOOP_STATS query answers about the same loop."""
    async with AsyncRobotClient(port=ports.server_port) as client:
        assert await client.wait_ready(timeout=10.0)

        # The overrun count only ever climbs, and comparing one sample of
        # it against another taken at a different moment says nothing
        # unless the order of the two is known. So: read it, then wait for
        # a frame that has caught up to that reading, then read it again.
        # The frame is then known to sit between the two, and a loaded
        # runner missing a deadline in between is the loop being honest
        # rather than the broadcast disagreeing with the query.
        before = await client.loop_stats()
        assert before is not None

        seen: dict = {}

        async def collect() -> None:
            async for status in client.stream_status_shared():
                health = dict(getattr(status, "loop_health", {}) or {})
                # The percentile needs a full sampling window before it
                # means anything, and the shared buffer may still hold a
                # frame from before the reading above — wait past both
                # rather than taking whichever frame arrives first.
                if (
                    health.get("p99_period_s", 0.0) > 0.0
                    and health.get("overruns", -1) >= before.overrun_count
                ):
                    seen.update(health)
                    return

        try:
            await asyncio.wait_for(collect(), timeout=15.0)
        except asyncio.TimeoutError:
            pytest.fail("no loop health ever arrived on STATUS")

        after = await client.loop_stats()
        assert after is not None
        assert abs(seen["p99_period_s"] - after.p99_period_s) < 5e-3, (
            f"STATUS says p99 {seen['p99_period_s']}, "
            f"the query says {after.p99_period_s}"
        )
        assert seen["overruns"] <= after.overrun_count, (
            f"STATUS says {seen['overruns']} overruns, past the "
            f"{after.overrun_count} a query taken after it reports — the "
            "broadcast is not reading the same counter"
        )
