"""The control loop keeps its ticks on a machine whose sleeps wake late."""

import time

from parol6.server import loop_timer
from parol6.server.loop_timer import LoopTimer


def _run(timer: LoopTimer, ticks: int) -> None:
    timer.start()
    for _ in range(ticks):
        timer.wait_for_next_tick()


def test_sleeps_that_wake_late_are_spun_through(monkeypatch):
    real_sleep = time.sleep
    # An oversubscribed VM: every sleep wakes 20 ms after it was asked to.
    monkeypatch.setattr(loop_timer.time, "sleep", lambda s: real_sleep(s + 0.02))
    timer = LoopTimer(0.01, busy_threshold_s=0.001)
    _run(timer, 200)
    assert timer.metrics.overrun_count <= 10, timer.metrics.overrun_count


def test_accurate_sleep_keeps_the_configured_busy_window(monkeypatch):
    def exact(seconds: float) -> None:
        until = time.perf_counter() + seconds
        while time.perf_counter() < until:
            pass

    monkeypatch.setattr(loop_timer.time, "sleep", exact)
    timer = LoopTimer(0.01, busy_threshold_s=0.001)
    _run(timer, 100)
    assert timer._busy_threshold < 0.0015, timer._busy_threshold
