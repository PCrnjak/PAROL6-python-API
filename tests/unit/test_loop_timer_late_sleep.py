"""The control loop keeps its ticks on a machine whose sleeps wake late."""

from parol6.server import loop_timer
from parol6.server.loop_timer import LoopTimer


class _Clock:
    """A clock the timer reads and sleeps on: each read moves it a
    microsecond, and a sleep lasts what it was asked plus ``late``."""

    def __init__(self, late: float) -> None:
        self.now = 0.0
        self.late = late

    def perf_counter(self) -> float:
        self.now += 1e-6
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds + self.late


def _run(monkeypatch, late: float, ticks: int) -> LoopTimer:
    clock = _Clock(late)
    monkeypatch.setattr(loop_timer.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(loop_timer.time, "sleep", clock.sleep)
    timer = LoopTimer(0.01, busy_threshold_s=0.001)
    timer.start()
    for _ in range(ticks):
        timer.wait_for_next_tick()
    return timer


def test_sleeps_that_wake_late_are_spun_through(monkeypatch):
    # An oversubscribed VM: every sleep wakes 20 ms after it was asked to.
    timer = _run(monkeypatch, late=0.02, ticks=500)
    assert timer.metrics.overrun_count <= 25, timer.metrics.overrun_count


def test_accurate_sleep_keeps_the_configured_busy_window(monkeypatch):
    timer = _run(monkeypatch, late=0.0, ticks=500)
    assert timer.metrics.overrun_count == 0
    assert timer._busy_threshold < 0.0011, timer._busy_threshold
