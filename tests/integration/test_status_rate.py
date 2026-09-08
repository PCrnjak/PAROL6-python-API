"""The status broadcast rate is a session knob, not a boot constant.

Raising it is how a capture or a tuning run gets resolution the default
50 Hz cannot give. Status is emitted every Nth control tick, so the rates a
controller can serve are the divisors of its control rate — reported as the
control rate itself, so a caller computes the set rather than probing for
it by rejection.
"""

import asyncio
import time

import pytest

from parol6 import AsyncRobotClient
from parol6.server.state import ControllerState
from parol6.server.status_cache import StatusCache
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import MotionError


async def _observed_hz(client: AsyncRobotClient, frames: int = 40) -> float:
    """Measure arrival rate over *frames* distinct broadcasts."""
    seen = 0
    start = 0.0
    async for _ in client.stream_status():
        if seen == 0:
            start = time.perf_counter()
        seen += 1
        if seen > frames:
            break
    return frames / max(time.perf_counter() - start, 1e-9)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_the_rate_reports_the_loop_it_divides(server_proc, ports):
    """``control_hz`` is what makes the constraint computable by a caller:
    every rate it implies must actually be accepted."""
    async with AsyncRobotClient(port=ports.server_port) as client:
        assert await client.wait_ready(timeout=10.0)

        rate = await client.status_rate()
        assert rate is not None
        assert rate.control_hz > 0.0
        assert rate.hz > 0.0
        assert rate.control_hz % rate.hz == 0.0, (
            f"the controller is broadcasting at {rate.hz} Hz, which does not "
            f"divide its own {rate.control_hz} Hz loop"
        )

        # Everything achievable() derives from control_hz must be accepted;
        # that is the whole contract of reporting the loop rate instead of a
        # list, so it is checked rather than assumed.
        for candidate in rate.achievable():
            assert await client.set_status_rate(candidate) > 0, (
                f"{candidate} Hz divides {rate.control_hz} Hz but was refused"
            )
        assert await client.set_status_rate(rate.hz) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_raising_the_rate_delivers_more_frames(server_proc, ports):
    """The point of the knob is resolution, so the change has to show up in
    the arrival rate rather than only in the readback."""
    async with AsyncRobotClient(port=ports.server_port) as client:
        assert await client.wait_ready(timeout=10.0)
        original = await client.status_rate()
        assert original is not None

        low = original.control_hz / 10
        high = original.control_hz / 2
        try:
            assert await client.set_status_rate(low) > 0
            await asyncio.sleep(0.3)
            slow = await _observed_hz(client)

            assert await client.set_status_rate(high) > 0
            back = await client.status_rate()
            assert back is not None and back.hz == high
            await asyncio.sleep(0.3)
            fast = await _observed_hz(client)
        finally:
            await client.set_status_rate(original.hz)

        assert fast > slow * 2, (
            f"asked for {high} Hz after {low} Hz but saw {fast:.1f} vs {slow:.1f}"
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_an_unachievable_rate_is_refused_with_the_rule(server_proc, ports):
    """Refused, never rounded to a neighbour: a capture taken at a rate nobody
    asked for is wrong in a way nothing reports. The refusal has to reach the
    caller carrying the rates that would have worked, since that is the whole
    of what an operator needs — including for the rates whose arithmetic the
    check itself cannot survive: 0.5 Hz floors to a zero divisor, and NaN
    cannot be made an int at all, so a validator that divides before it
    screens turns a refusal into a crash.
    """
    async with AsyncRobotClient(port=ports.server_port) as client:
        assert await client.wait_ready(timeout=10.0)
        before = await client.status_rate()
        assert before is not None
        achievable = before.achievable()

        for bogus in (0.0, -50.0, 0.5, 62.5, float("nan"), float("inf")):
            assert bogus not in achievable
            with pytest.raises(MotionError) as caught:
                await client.set_status_rate(bogus)

            refusal = caught.value.robot_error
            assert refusal.code == ErrorCode.SYS_STATUS_RATE_INVALID, (
                f"{bogus} Hz came back as {refusal.title!r} rather than as an "
                f"unservable rate: {refusal.cause}"
            )
            unnamed = [hz for hz in achievable if str(int(hz)) not in refusal.remedy]
            assert not unnamed, (
                f"refusing {bogus} Hz has to say what would work instead, but "
                f"{unnamed} are missing from {refusal.remedy!r}"
            )

        after = await client.status_rate()
        assert after is not None and after.hz == before.hz, (
            "a refused rate must leave the broadcast alone"
        )


@pytest.mark.integration
def test_the_speed_derivative_follows_the_rate_it_was_sampled_at():
    """TCP speed is a difference over the broadcast period, so the period the
    cache divides by has to be the one the controller is actually broadcasting
    at — and the sample that straddles a rate change spans the period it was
    taken at, not the one that has just replaced it. Get either wrong and a
    steady arm appears to change speed the moment somebody changes the rate.

    Only J1 moves, so equal step increments are equal chords of one circle
    about the base axis: the displacement is the same every sample, and any
    change in the reported speed is the period alone.
    """
    cache = StatusCache()
    try:
        state = ControllerState()

        def advance() -> float:
            state.Position_in[0] += 200
            cache.update_from_state(state)
            return cache.tcp_speed

        advance()  # first difference has nothing to difference against
        at_50 = advance()
        assert at_50 > 0.0, "a moving arm has to report a speed"

        state.status_rate_hz = 25.0
        straddling = advance()
        settled = advance()

        assert straddling == pytest.approx(at_50, rel=1e-3), (
            "the sample taken before the rate changed spans the old period"
        )
        assert settled == pytest.approx(at_50 / 2, rel=1e-3), (
            "half the broadcast rate is twice the period, so the same "
            f"movement per frame is half the speed: {settled} vs {at_50}"
        )
    finally:
        cache.close()
