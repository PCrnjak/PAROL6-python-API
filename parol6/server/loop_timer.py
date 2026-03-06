"""Loop timing with hybrid sleep + busy-loop for precise deadline scheduling."""

import gc
import time
from typing import TYPE_CHECKING

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from parol6 import config as cfg

if TYPE_CHECKING:
    from typing import Self


# =============================================================================
# Constants for power-of-2 buffer operations
# =============================================================================
# ~5 seconds of data, rounded up to next power of 2 for fast modulo via bitmask
_TARGET_BUFFER_SECONDS = 5.0
_raw_size = int(cfg.CONTROL_RATE_HZ * _TARGET_BUFFER_SECONDS)
BUFFER_SIZE = 1 << (_raw_size - 1).bit_length()  # Next power of 2
BUFFER_MASK = BUFFER_SIZE - 1


# =============================================================================
# Numba-accelerated statistics functions (cached to disk)
# =============================================================================


@njit(cache=True)
def _quickselect_partition(arr: np.ndarray, left: int, right: int) -> int:
    """Partition array around last element as pivot. Returns pivot index."""
    pivot = arr[right]
    i = left - 1
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


@njit(cache=True)
def _quickselect(arr: np.ndarray, k: int) -> float:
    """Find k-th smallest element in-place. Modifies arr."""
    left = 0
    right = len(arr) - 1
    while left < right:
        pivot_idx = _quickselect_partition(arr, left, right)
        if pivot_idx == k:
            return arr[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1
    return arr[k]


@njit(cache=True)
def _compute_phase_stats(
    samples: np.ndarray, scratch: np.ndarray, n: int
) -> tuple[float, float, float]:
    """Compute phase stats: mean, max, p99. Uses pre-allocated scratch buffer."""
    if n == 0:
        return 0.0, 0.0, 0.0

    # Compute mean and max in single pass
    total = 0.0
    max_val = samples[0]
    for i in range(n):
        total += samples[i]
        if samples[i] > max_val:
            max_val = samples[i]
    mean = total / n

    # p99 via quickselect (copy to scratch first to preserve original)
    if n >= 20:
        for i in range(n):
            scratch[i] = samples[i]
        k = int(n * 0.99)
        p99 = _quickselect(scratch[:n], k)
    else:
        p99 = max_val

    return mean, max_val, p99


@njit(cache=True)
def _compute_loop_stats(
    samples: np.ndarray, scratch: np.ndarray, n: int
) -> tuple[float, float, float, float, float, float]:
    """Compute loop stats using single-pass Welford's algorithm for mean+std.

    Uses pre-allocated scratch buffer for percentile computation.
    Only one copy to scratch (p99 first, then p95 on same data).
    """
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Single-pass Welford's algorithm for mean and variance + min/max
    mean = 0.0
    m2 = 0.0  # Sum of squared differences
    min_val = samples[0]
    max_val = samples[0]

    for i in range(n):
        x = samples[i]
        # Welford's online algorithm
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        m2 += delta * delta2
        # Track min/max
        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x

    std = np.sqrt(m2 / n) if n > 0 else 0.0

    # p95 and p99 via quickselect with single copy
    if n >= 20:
        # Copy to scratch once
        for i in range(n):
            scratch[i] = samples[i]

        # Compute p99 first (higher index)
        k99 = int(n * 0.99)
        p99 = _quickselect(scratch[:n], k99)

        # Compute p95 on same array (works because k95 < k99)
        k95 = int(n * 0.95)
        p95 = _quickselect(scratch[:n], k95)
    else:
        p95 = max_val
        p99 = max_val

    return mean, std, min_val, max_val, p95, p99


# =============================================================================
# EventRateMetrics - track rate of sporadic events
# =============================================================================


@njit(cache=True)
def _compute_event_rate(
    buffer: np.ndarray, count: int, now: float, max_age_s: float
) -> float:
    """Compute event rate from timestamps in buffer.

    Args:
        buffer: Circular buffer of event timestamps
        count: Number of valid entries in buffer
        now: Current timestamp
        max_age_s: Only count events within this window

    Returns:
        Events per second, or 0.0 if insufficient data.
    """
    if count < 2:
        return 0.0

    cutoff = now - max_age_s
    events_in_window = 0
    oldest = now
    newest = 0.0

    for i in range(count):
        ts = buffer[i]
        if ts >= cutoff:
            events_in_window += 1
            if ts < oldest:
                oldest = ts
            if ts > newest:
                newest = ts

    if events_in_window < 2:
        return 0.0

    time_span = newest - oldest
    if time_span < 0.001:
        return 0.0

    return (events_in_window - 1) / time_span


class EventRateMetrics:
    """Track rate of sporadic events (e.g., commands) with proper decay.

    Unlike LoopMetrics which measures fixed-rate loop periods, this tracks
    events that may arrive in bursts or not at all. Returns 0 when no
    recent events.

    Uses a circular buffer of timestamps and numba-accelerated rate calculation.
    """

    __slots__ = (
        "_buffer",
        "_buffer_idx",
        "_buffer_count",
        "_buffer_mask",
        "_last_event_time",
        "event_count",
    )

    def __init__(self, buffer_size: int = 64) -> None:
        """Initialize with a timestamp buffer.

        Args:
            buffer_size: Number of timestamps to retain. Rounded up to power of 2.
        """
        # Round up to power of 2 for fast modulo via bitmask
        size = 1
        while size < buffer_size:
            size <<= 1
        self._buffer = np.zeros(size, dtype=np.float64)
        self._buffer_mask = size - 1
        self._buffer_idx = 0
        self._buffer_count = 0
        self._last_event_time = 0.0
        self.event_count = 0

    def record(self, now: float) -> None:
        """Record an event at the given timestamp."""
        self._last_event_time = now
        self.event_count += 1
        self._buffer[self._buffer_idx] = now
        self._buffer_idx = (self._buffer_idx + 1) & self._buffer_mask
        if self._buffer_count < len(self._buffer):
            self._buffer_count += 1

    def rate_hz(self, now: float, max_age_s: float = 3.0) -> float:
        """Calculate event rate from recent events.

        Args:
            now: Current timestamp (perf_counter)
            max_age_s: Only count events within this window. Also used to
                determine staleness - returns 0 if no events within max_age_s.

        Returns:
            Events per second, or 0.0 if no recent events.
        """
        # Fast path: check staleness before calling into numba
        if self._buffer_count < 2 or now - self._last_event_time > max_age_s:
            return 0.0
        return _compute_event_rate(self._buffer, self._buffer_count, now, max_age_s)

    def reset(self) -> None:
        """Reset all state."""
        self._buffer[:] = 0.0
        self._buffer_idx = 0
        self._buffer_count = 0
        self._last_event_time = 0.0
        self.event_count = 0


class GCTracker:
    """Track garbage collection frequency and duration, with optional deferred collection.

    Registers a callback with gc.callbacks to record GC events.
    Provides rate (collections/sec) and duration statistics.

    When ``take_control()`` is called, automatic GC is disabled and the caller
    is responsible for calling ``collect_deferred()`` each loop iteration.
    Collections are scheduled during slack time to avoid disrupting the hot path.
    """

    __slots__ = (
        "_timestamps",
        "_durations",
        "_idx",
        "_count",
        "_buffer_mask",
        "_gc_start",
        "_last_duration",
        "_controlled",
        "total_count",
        "total_time",
    )

    def __init__(self, buffer_size: int = 64) -> None:
        # Round up to power of 2
        size = 1
        while size < buffer_size:
            size <<= 1
        self._timestamps = np.zeros(size, dtype=np.float64)
        self._durations = np.zeros(size, dtype=np.float64)
        self._buffer_mask = size - 1
        self._idx = 0
        self._count = 0
        self._gc_start = 0.0
        self._last_duration = 0.0
        self._controlled = False
        self.total_count = 0
        self.total_time = 0.0
        gc.callbacks.append(self._callback)

    def _callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_start = time.perf_counter()
        elif phase == "stop":
            now = time.perf_counter()
            duration = now - self._gc_start
            self._last_duration = duration
            self.total_count += 1
            self.total_time += duration
            idx = self._idx & self._buffer_mask
            self._timestamps[idx] = now
            self._durations[idx] = duration
            self._idx += 1
            if self._count < len(self._timestamps):
                self._count += 1

    def stats(self, now: float, max_age_s: float = 3.0) -> tuple[float, float]:
        """Get windowed GC stats: (rate_hz, mean_duration_ms).

        Both values are computed from events within max_age_s window.
        Returns (0.0, 0.0) if no recent GC events.
        """
        if self._count == 0:
            return 0.0, 0.0
        last_ts = self._timestamps[(self._idx - 1) & self._buffer_mask]
        if now - last_ts > max_age_s:
            return 0.0, 0.0

        # Compute rate and windowed mean duration
        cutoff = now - max_age_s
        total_dur = 0.0
        count_in_window = 0
        for i in range(self._count):
            ts = self._timestamps[i]
            if ts >= cutoff:
                total_dur += self._durations[i]
                count_in_window += 1

        if count_in_window < 2:
            return 0.0, 0.0

        rate = _compute_event_rate(self._timestamps, self._count, now, max_age_s)
        mean_ms = (total_dur / count_in_window) * 1000.0
        return rate, mean_ms

    def recent_duration_ms(self) -> float:
        """Duration of most recent GC in milliseconds."""
        return self._last_duration * 1000.0

    def take_control(self) -> None:
        """Disable automatic GC. Caller must call collect_deferred() each iteration."""
        gc.disable()
        gc.collect()  # clean slate before entering controlled mode
        self._controlled = True

    def release_control(self) -> None:
        """Re-enable automatic GC and run a full collection."""
        if self._controlled:
            self._controlled = False
            gc.enable()
            gc.collect()

    def collect_deferred(self, slack_s: float, tick_count: int) -> None:
        """Run a deferred GC collection if there is enough slack time.

        Call once per loop iteration after all work phases complete.

        Strategy:
          - gen-0 every tick if slack > 0.5ms  (~50-100µs typical)
          - gen-1 every 1000 ticks (~10s) if slack > 2ms
          - gen-2 every 10000 ticks (~100s) if slack > 3ms
        """
        if not self._controlled or slack_s < 0.0005:
            return
        if tick_count % 10000 == 0 and slack_s > 0.003:
            gc.collect(2)
        elif tick_count % 1000 == 0 and slack_s > 0.002:
            gc.collect(1)
        else:
            gc.collect(0)

    def shutdown(self) -> None:
        """Remove callback and re-enable GC on shutdown."""
        self.release_control()
        try:
            gc.callbacks.remove(self._callback)
        except ValueError:
            pass


# =============================================================================
# PhaseMetrics - regular Python class (no jitclass overhead)
# =============================================================================


class PhaseMetrics:
    """Rolling statistics for a single phase.

    Uses pre-allocated numpy arrays and calls @njit helper functions
    for the heavy computation.
    """

    __slots__ = (
        "_buffer",
        "_scratch",
        "_buffer_idx",
        "_buffer_count",
        "last_s",
        "mean_s",
        "max_s",
        "p99_s",
    )

    def __init__(self) -> None:
        self._buffer = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self._scratch = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self._buffer_idx = 0
        self._buffer_count = 0
        self.last_s = 0.0
        self.mean_s = 0.0
        self.max_s = 0.0
        self.p99_s = 0.0

    def record(self, duration: float) -> None:
        """Record a duration sample."""
        self.last_s = duration
        self._buffer[self._buffer_idx] = duration
        self._buffer_idx = (self._buffer_idx + 1) & BUFFER_MASK
        if self._buffer_count < BUFFER_SIZE:
            self._buffer_count += 1

    def compute_stats(self) -> None:
        """Compute statistics from buffer."""
        if self._buffer_count == 0:
            return
        mean, max_val, p99 = _compute_phase_stats(
            self._buffer, self._scratch, self._buffer_count
        )
        self.mean_s = mean
        self.max_s = max_val
        self.p99_s = p99


class PhaseTimer:
    """Timer for measuring durations of multiple named phases in a loop.

    Usage:
        timer = PhaseTimer(["read", "execute", "write"])
        while True:
            with timer.phase("read"):
                do_read()
            with timer.phase("execute"):
                do_execute()
            timer.tick()  # Compute stats periodically
    """

    def __init__(self, phase_names: list[str], stats_interval: int = 50):
        self._phases: dict[str, PhaseMetrics] = {
            name: PhaseMetrics() for name in phase_names
        }
        self._stats_interval = stats_interval
        self._tick_count = 0
        self._current_phase: str | None = None
        self._phase_start: float = 0.0

    @property
    def phases(self) -> dict[str, PhaseMetrics]:
        """Access phase metrics by name."""
        return self._phases

    def start(self, phase: str) -> None:
        """Start timing a phase."""
        self._current_phase = phase
        self._phase_start = time.perf_counter()

    def stop(self) -> float:
        """Stop timing current phase and record duration. Returns duration."""
        if self._current_phase is None:
            return 0.0
        duration = time.perf_counter() - self._phase_start
        self._phases[self._current_phase].record(duration)
        self._current_phase = None
        return duration

    def phase(self, name: str) -> "PhaseContext":
        """Context manager for timing a phase."""
        return PhaseContext(self, name)

    def tick(self) -> None:
        """Call once per loop iteration to compute stats periodically."""
        self._tick_count += 1
        if self._tick_count % self._stats_interval == 0:
            for p in self._phases.values():
                p.compute_stats()

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all phases as dict."""
        return {
            name: {
                "mean_ms": p.mean_s * 1000,
                "max_ms": p.max_s * 1000,
                "p99_ms": p.p99_s * 1000,
            }
            for name, p in self._phases.items()
        }


class PhaseContext:
    """Context manager for timing a phase."""

    __slots__ = ("_timer", "_phase")

    def __init__(self, timer: PhaseTimer, phase: str):
        self._timer = timer
        self._phase = phase

    def __enter__(self) -> "Self":
        self._timer.start(self._phase)
        return self

    def __exit__(self, *args: object) -> None:
        self._timer.stop()


# =============================================================================
# LoopMetrics
# =============================================================================


class LoopMetrics:
    """Metrics tracked by the loop timer with rolling statistics.

    Provides unified timing, logging, and degradation checking across subsystems.
    Use configure() to set target period, then call tick() each iteration.

    Uses pre-allocated numpy arrays and calls @njit helper functions
    for the heavy computation.
    """

    __slots__ = (
        "loop_count",
        "overrun_count",
        "mean_period_s",
        "std_period_s",
        "min_period_s",
        "max_period_s",
        "p95_period_s",
        "p99_period_s",
        # Overshoot tracking (how much we miss the deadline by)
        "mean_overshoot_s",
        "max_overshoot_s",
        "p99_overshoot_s",
        "_buffer",
        "_scratch",
        "_buffer_idx",
        "_buffer_count",
        "_overshoot_buffer",
        "_overshoot_idx",
        "_overshoot_count",
        "_target_period_s",
        "_prev_time",
        "_last_log_time",
        "_last_warn_time",
        "_stats_interval",
        "_start_time",
        "_grace_period_s",
    )

    def __init__(self) -> None:
        self.loop_count = 0
        self.overrun_count = 0
        self.mean_period_s = 0.0
        self.std_period_s = 0.0
        self.min_period_s = 0.0
        self.max_period_s = 0.0
        self.p95_period_s = 0.0
        self.p99_period_s = 0.0
        # Overshoot stats
        self.mean_overshoot_s = 0.0
        self.max_overshoot_s = 0.0
        self.p99_overshoot_s = 0.0
        self._buffer = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self._scratch = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self._buffer_idx = 0
        self._buffer_count = 0
        self._overshoot_buffer = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self._overshoot_idx = 0
        self._overshoot_count = 0
        self._target_period_s = 0.0
        self._prev_time = 0.0
        self._last_log_time = 0.0
        self._last_warn_time = 0.0
        self._stats_interval = 50
        self._start_time = 0.0
        self._grace_period_s = 10.0

    def configure(
        self, target_period_s: float, stats_interval: int, grace_period_s: float = 15.0
    ) -> None:
        """Configure target period, stats interval, and startup grace period.

        Args:
            target_period_s: Target loop period in seconds
            stats_interval: How often to compute rolling statistics (in loop iterations)
            grace_period_s: Duration after mark_started() during which overbudget
                warnings are suppressed. Defaults to 15s to allow for JIT warmup
                and process pool initialization.
        """
        self._target_period_s = target_period_s
        self._stats_interval = stats_interval
        self._grace_period_s = grace_period_s

    def mark_started(self, now: float) -> None:
        """Mark the start time for grace period calculation."""
        self._start_time = now

    def tick(self, now: float) -> None:
        """Call once per iteration. Auto-records period and computes stats periodically."""
        if self._prev_time > 0:
            self.record_period(now - self._prev_time)
        self._prev_time = now
        self.loop_count += 1
        if self.loop_count % self._stats_interval == 0:
            self.compute_stats()

    def should_log(self, now: float, interval: float) -> bool:
        """Returns True and updates timestamp if interval has passed."""
        if now - self._last_log_time >= interval:
            self._last_log_time = now
            return True
        return False

    def check_degraded(
        self, now: float, threshold: float, rate_limit: float
    ) -> tuple[bool, float]:
        """Check if p99 exceeds target by threshold. Returns (should_warn, degradation_pct).

        Rate-limited to once per rate_limit seconds. Suppressed during startup grace period.
        """
        if self._target_period_s <= 0 or self.p99_period_s <= 0:
            return False, 0.0
        # Suppress warnings during startup grace period
        if self._start_time > 0 and (now - self._start_time) < self._grace_period_s:
            return False, 0.0
        if now - self._last_warn_time < rate_limit:
            return False, 0.0
        if self.p99_period_s > self._target_period_s * (1.0 + threshold):
            degradation = (self.p99_period_s / self._target_period_s - 1.0) * 100.0
            self._last_warn_time = now
            return True, degradation
        return False, 0.0

    def record_period(self, period: float) -> None:
        """Record a period sample into the circular buffer."""
        self._buffer[self._buffer_idx] = period
        self._buffer_idx = (self._buffer_idx + 1) & BUFFER_MASK
        if self._buffer_count < BUFFER_SIZE:
            self._buffer_count += 1

    def record_overshoot(self, overshoot: float) -> None:
        """Record busy-loop overshoot (time past deadline) into buffer."""
        self._overshoot_buffer[self._overshoot_idx] = overshoot
        self._overshoot_idx = (self._overshoot_idx + 1) & BUFFER_MASK
        if self._overshoot_count < BUFFER_SIZE:
            self._overshoot_count += 1

    def compute_stats(self) -> None:
        """Compute statistics from buffers."""
        if self._buffer_count > 0:
            mean, std, min_val, max_val, p95, p99 = _compute_loop_stats(
                self._buffer, self._scratch, self._buffer_count
            )
            self.mean_period_s = mean
            self.std_period_s = std
            self.min_period_s = min_val
            self.max_period_s = max_val
            self.p95_period_s = p95
            self.p99_period_s = p99

        # Compute overshoot stats using the simpler phase stats function
        if self._overshoot_count > 0:
            mean, max_val, p99 = _compute_phase_stats(
                self._overshoot_buffer, self._scratch, self._overshoot_count
            )
            self.mean_overshoot_s = mean
            self.max_overshoot_s = max_val
            self.p99_overshoot_s = p99

    def reset_stats(self, include_counters: bool = False) -> None:
        """Reset rolling statistics.

        Args:
            include_counters: If True, also reset overrun_count (loop_count is preserved).
        """
        self._buffer.fill(0.0)
        self._buffer_idx = 0
        self._buffer_count = 0
        self.mean_period_s = 0.0
        self.std_period_s = 0.0
        self.min_period_s = 0.0
        self.max_period_s = 0.0
        self.p95_period_s = 0.0
        self.p99_period_s = 0.0
        # Reset overshoot stats
        self._overshoot_buffer.fill(0.0)
        self._overshoot_idx = 0
        self._overshoot_count = 0
        self.mean_overshoot_s = 0.0
        self.max_overshoot_s = 0.0
        self.p99_overshoot_s = 0.0
        if include_counters:
            self.overrun_count = 0


def format_hz_summary(m: LoopMetrics) -> str:
    """Format metrics as 'XXX.XHz σ=X.XXms p99=X.XXms'."""
    if m.mean_period_s <= 0:
        return "0.0Hz σ=0.00ms p99=0.00ms"
    hz = 1.0 / m.mean_period_s
    return (
        f"{hz:.1f}Hz σ={m.std_period_s * 1000:.2f}ms p99={m.p99_period_s * 1000:.2f}ms"
    )


class LoopTimer:
    """Deadline-based loop timing with hybrid sleep + busy-loop.

    Uses time.sleep() for most of the wait time to reduce CPU usage,
    then switches to a busy-loop for the final portion to achieve
    precise timing without OS scheduling jitter.
    """

    def __init__(
        self,
        interval_s: float,
        busy_threshold_s: float | None = None,
        stats_interval: int = 50,
    ):
        """Initialize the loop timer.

        Args:
            interval_s: Target loop interval in seconds.
            busy_threshold_s: Time before deadline to switch from sleep to busy-loop.
                             Default from PAROL6_BUSY_THRESHOLD_MS env var (1ms).
            stats_interval: Compute stats every N loops (default 50 = 5Hz at 250Hz loop).
        """
        self._interval = interval_s
        self._busy_threshold = (
            busy_threshold_s
            if busy_threshold_s is not None
            else cfg.BUSY_THRESHOLD_MS / 1000.0
        )
        self._stats_interval = stats_interval
        self._next_deadline = 0.0
        self._prev_t = 0.0
        self.metrics = LoopMetrics()
        self.metrics.configure(interval_s, stats_interval)

    @property
    def interval(self) -> float:
        """Target loop interval in seconds."""
        return self._interval

    def start(self) -> None:
        """Initialize timing at loop start. Call once before entering the loop."""
        now = time.perf_counter()
        self._next_deadline = now
        self._prev_t = now

    def time_to_next_deadline(self) -> float:
        """Remaining seconds until the next tick deadline.

        Call after work phases complete, before ``wait_for_next_tick()``.
        Positive means ahead of schedule (slack time available).
        """
        return (self._next_deadline + self._interval) - time.perf_counter()

    def wait_for_next_tick(self) -> None:
        """Wait until next deadline using hybrid sleep + busy-loop.

        Updates metrics (loop_count, period, stats) and handles overruns.
        Call at the end of each loop iteration.
        """
        self.metrics.loop_count += 1

        # Compute stats periodically (not every loop)
        if self.metrics.loop_count % self._stats_interval == 0:
            self.metrics.compute_stats()

        # Advance deadline
        self._next_deadline += self._interval
        sleep_time = self._next_deadline - time.perf_counter()

        if sleep_time > self._busy_threshold:
            # Sleep for most of the time, leaving headroom for busy-loop
            time.sleep(sleep_time - self._busy_threshold)

        if sleep_time > 0:
            # Busy-loop for remaining time (precise timing)
            while time.perf_counter() < self._next_deadline:
                pass
            # Track how much we overshot the deadline
            now = time.perf_counter()
            self.metrics.record_overshoot(now - self._next_deadline)
        else:
            # Overrun - reset deadline to avoid perpetual catch-up
            self.metrics.overrun_count += 1
            now = time.perf_counter()
            self._next_deadline = now

        # Measure period from deadline-to-deadline (not affected by work jitter)
        if self._prev_t > 0:
            self.metrics.record_period(now - self._prev_t)
        self._prev_t = now
