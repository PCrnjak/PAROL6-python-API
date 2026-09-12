"""The generic skill contract over the real fake-serial controller."""

import asyncio

import pytest
from waldoctl.skills import observe_skills, skill

from parol6 import AsyncRobotClient
from parol6.protocol.wire import ActionState


@skill(
    id="test.nudge",
    version="1.0.0",
    requires=frozenset({"motion.joint", "backend.parol6"}),
)
async def nudge(rbt: AsyncRobotClient, *, degrees: float) -> list[float]:
    target = await rbt.angles()
    assert target is not None
    target[0] += degrees
    index = await rbt.move_j(target, speed=0.5)
    assert index >= 0 and await rbt.wait_command(index, timeout=20.0)
    observed = await rbt.angles()
    assert observed is not None
    return observed


def test_sync_skill_uses_the_supplied_connection_and_preserves_results(
    client, server_proc
):
    before = client.angles()
    events = []
    with observe_skills(events.append):
        after = nudge(client, degrees=-5.0)
    assert after[0] == pytest.approx(before[0] - 5.0, abs=0.5)
    assert [event.phase for event in events] == ["started", "completed"]


def test_async_cancel_stops_the_controller_and_prevents_the_next_move(
    client, server_proc, ports
):
    async def scenario():
        async with AsyncRobotClient(
            host=ports.server_ip, port=ports.server_port
        ) as rbt:
            start = await rbt.angles()
            assert start is not None
            moving = asyncio.Event()

            @skill(id="test.cancel", version="1.0.0")
            async def sequence(rbt: AsyncRobotClient) -> None:
                target = list(start)
                target[0] -= 20
                index = await rbt.move_j(target, duration=5.0)
                assert index >= 0
                moving.set()
                try:
                    await rbt.wait_command(index, timeout=15.0)
                except asyncio.CancelledError:
                    # Catching cancellation must not allow another command.
                    await rbt.move_j(start, speed=0.5)
                    pytest.fail("a cancelled invocation issued another command")

            events = []
            with observe_skills(events.append):
                task = asyncio.create_task(sequence.async_call(rbt))
                await asyncio.wait_for(moving.wait(), timeout=10.0)
                assert await rbt.wait_status(
                    lambda s: s.angles[0] < start[0] - 1.0, timeout=10.0
                )
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert events[-1].phase == "cancelled"
            assert events[-1].stop_confirmed is True
            assert await rbt.wait_status(
                lambda s: s.action_state == ActionState.IDLE and s.queued_segments == 0,
                timeout=3.0,
            ), await rbt.activity()
            assert await nudge.async_call(rbt, degrees=-2.0) is not None

    asyncio.run(scenario())
