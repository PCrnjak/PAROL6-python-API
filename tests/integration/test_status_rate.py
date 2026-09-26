"""The status broadcast rate is a session knob, not a boot constant.

Raising it is how a capture or a tuning run gets resolution the default
50 Hz cannot give. Status is emitted every Nth control tick, so the rates a
controller can serve are the divisors of its control rate — reported as the
control rate itself, so a caller computes the set rather than probing for
it by rejection.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from parol6 import AsyncRobotClient
from parol6.server import status_cache
from parol6.server.state import ControllerState
from parol6.server.status_cache import _TCP_SPEED_WINDOW, StatusCache
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import MotionError


async def _observed_hz(client: AsyncRobotClient, frames: int = 40) -> float:
    """Measure arrival rate over *frames* distinct broadcasts."""
    seen = 0
    start = 0.0
    previous = None
    async for status in client.stream_status():
        assert status.session_id > 0 and status.mono_time_ns > 0
        if previous is not None:
            assert status.session_id == previous.session_id
            assert status.seq > previous.seq
            assert status.mono_time_ns > previous.mono_time_ns
        previous = status
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
        assert before.servable, (
            "the controller knows its divisor set -- it formats it into the "
            "refusal -- so the query has to report it rather than leaving the "
            "client to re-derive one backend's rule"
        )
        assert before.achievable() == before.servable
        assert before.hz in before.servable
        assert before.control_hz == max(before.servable)
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
            unnamed = [hz for hz in achievable if f"{hz:g}" not in refusal.remedy]
            assert not unnamed, (
                f"refusing {bogus} Hz has to say what would work instead, but "
                f"{unnamed} are missing from {refusal.remedy!r}"
            )

        after = await client.status_rate()
        assert after is not None and after.hz == before.hz, (
            "a refused rate must leave the broadcast alone"
        )


@pytest.mark.integration
def test_the_speed_derivative_follows_the_frames_it_was_sampled_from(monkeypatch):
    """TCP speed is displacement over the time between the serial frames the
    samples came from: not over the broadcast period, which a query
    refreshing the cache between broadcasts would halve, and not over the
    refresh interval, which the loop's jitter would scatter.

    Only J1 moves, so equal step increments are equal chords of one circle
    about the base axis: the displacement is the same every frame, and any
    change in the reported speed is timing alone.
    """
    clock = [100.0]
    monkeypatch.setattr(
        status_cache,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=time.time, sleep=time.sleep),
    )
    cache = StatusCache()
    try:
        state = ControllerState()

        def frame(dt: float, steps: int = 200) -> float:
            clock[0] += dt
            state.Position_in[0] += steps
            cache.mark_serial_observed()
            cache.update_from_state(state)
            return cache.tcp_speed

        for _ in range(2 * _TCP_SPEED_WINDOW):
            started = frame(0.01)
        assert started > 0.0, "a moving arm has to report a speed"

        # A refresh between frames (a query) sees the frame the broadcast
        # saw: no new information, and the speed stands.
        cache.update_from_state(state)
        assert cache.tcp_speed == started

        # Frames at half the rate carry the same chord over twice the
        # time: half the speed, once the window has turned over.
        for _ in range(2 * _TCP_SPEED_WINDOW):
            settled = frame(0.02)
        assert settled == pytest.approx(started / 2, rel=1e-3), (
            f"the same movement per frame over twice the time is half the speed: "
            f"{settled} vs {started}"
        )

        # A window of frames that leave the arm where it is: at rest.
        for _ in range(_TCP_SPEED_WINDOW):
            frame(0.02, steps=0)
        assert cache.tcp_speed == 0.0

        # At a 5 Hz broadcast the cache is refreshed every twentieth frame.
        # The first refresh after the arm stops finds its last movement a
        # fifth of a second old: at rest, not a window of refreshes later.
        for _ in range(3):
            moving = frame(0.2, steps=4000)
        assert moving > 0.0, "a moving arm has to report a speed at 5 Hz too"
        assert frame(0.2, steps=0) == 0.0, (
            "the speed outlived the motion by a window of 5 Hz refreshes"
        )
    finally:
        monkeypatch.undo()
        cache.close()
