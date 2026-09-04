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

        # The overrun count only ever climbs, so the broadcast sample is
        # bracketed by a query on either side of it rather than compared
        # to one taken later: a loaded runner misses a deadline or two in
        # between, and that is the loop being honest, not a disagreement.
        before = await client.loop_stats()
        assert before is not None

        seen: dict = {}

        async def collect() -> None:
            async for status in client.stream_status_shared():
                health = dict(getattr(status, "loop_health", {}) or {})
                # The percentile needs a full sampling window before it
                # means anything, so wait for it rather than taking
                # whichever frame arrives first.
                if health.get("p99_period_s", 0.0) > 0.0:
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
        assert before.overrun_count <= seen["overruns"] <= after.overrun_count, (
            f"STATUS says {seen['overruns']} overruns, outside the "
            f"{before.overrun_count}..{after.overrun_count} the query "
            "bracketed it with — the broadcast is not reading the same counter"
        )
