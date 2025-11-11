"""
Shared trajectory planning utilities.
"""

from collections.abc import Sequence

import numpy as np

from parol6.config import CONTROL_RATE_HZ


def _samples_for_duration(duration: float, sample_rate: float) -> int:
    if duration <= 0:
        return 2
    n = int(round(duration * sample_rate)) + 1
    return max(2, n)


def plan_linear_quintic(
    start: Sequence[float],
    end: Sequence[float],
    duration: float,
    sample_rate: float | None = None,
) -> np.ndarray:
    """
    Quintic time-scaling with zero velocity and acceleration at endpoints.
    Applies the scalar blend S(τ) = 10τ^3 - 15τ^4 + 6τ^5 to each dimension.

    Returns: array of shape (N, D)
    """
    sr = CONTROL_RATE_HZ if sample_rate is None else float(sample_rate)
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    if start_arr.shape != end_arr.shape:
        raise ValueError("start and end must have the same shape")

    if duration <= 0:
        return np.vstack([start_arr, end_arr])

    n = _samples_for_duration(duration, sr)
    tau = np.linspace(0.0, 1.0, n)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5  # [0..1]
    s = s.reshape(-1, 1)
    delta = (end_arr - start_arr).reshape(1, -1)
    traj = start_arr.reshape(1, -1) + s * delta
    return traj


def plan_linear_cubic(
    start: Sequence[float],
    end: Sequence[float],
    duration: float,
    sample_rate: float | None = None,
) -> np.ndarray:
    """
    Cubic time-scaling with zero velocity at endpoints.
    Applies S(τ) = 3τ^2 - 2τ^3 to each dimension.

    Returns: array of shape (N, D)
    """
    sr = CONTROL_RATE_HZ if sample_rate is None else float(sample_rate)
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    if start_arr.shape != end_arr.shape:
        raise ValueError("start and end must have the same shape")

    if duration <= 0:
        return np.vstack([start_arr, end_arr])

    n = _samples_for_duration(duration, sr)
    tau = np.linspace(0.0, 1.0, n)
    s = 3 * tau**2 - 2 * tau**3
    s = s.reshape(-1, 1)
    delta = (end_arr - start_arr).reshape(1, -1)
    traj = start_arr.reshape(1, -1) + s * delta
    return traj


def _trapezoid_timings(
    distance: float, v_max: float, a_max: float
) -> tuple[float, float, float, float, bool]:
    """
    Compute trapezoid or triangular profile timing.

    Returns: (T, t_a, t_c, v_peak, triangular)
      - T: total time
      - t_a: accel time
      - t_c: constant velocity time (0 for triangular)
      - v_peak: peak velocity reached
      - triangular: True if triangular profile (no cruise), else False
    """
    if distance <= 0 or v_max <= 0 or a_max <= 0:
        return 0.0, 0.0, 0.0, 0.0, True

    t_a = v_max / a_max
    s_a = 0.5 * a_max * t_a**2  # distance covered during accel

    if 2 * s_a < distance:
        # Trapezoidal: accel, cruise, decel
        s_c = distance - 2 * s_a
        t_c = s_c / v_max
        T = 2 * t_a + t_c
        return T, t_a, t_c, v_max, False
    else:
        # Triangular: peak velocity determined by distance
        v_peak = np.sqrt(a_max * distance)
        t_a = v_peak / a_max
        T = 2 * t_a
        return T, t_a, 0.0, v_peak, True


def plan_trapezoid_position_1d(
    start: float,
    end: float,
    v_max: float,
    a_max: float,
    sample_rate: float | None = None,
) -> np.ndarray:
    """
    Generate 1D position samples following a trapezoidal (or triangular) velocity profile.
    Returns positions of shape (N,), including start and end.

    Notes:
    - start and end can be any floats; profile is computed along the line with correct sign.
    """
    sr = CONTROL_RATE_HZ if sample_rate is None else float(sample_rate)
    d = float(end) - float(start)
    sign = 1.0 if d >= 0 else -1.0
    L = abs(d)

    if L == 0 or v_max <= 0 or a_max <= 0:
        return np.array([start, end], dtype=float)

    T, t_a, t_c, v_peak, triangular = _trapezoid_timings(L, v_max, a_max)
    if T <= 0:
        return np.array([start, end], dtype=float)

    n = _samples_for_duration(T, sr)
    t = np.linspace(0.0, T, n)

    # Piecewise position function along positive axis (0..L)
    pos = np.zeros_like(t)
    if triangular:
        # accel then decel with peak v_peak
        for i, ti in enumerate(t):
            if ti <= t_a:
                pos[i] = 0.5 * a_max * ti**2
            else:
                # decel phase mirrored
                td = ti - t_a
                pos[i] = (0.5 * a_max * t_a**2) + v_peak * td - 0.5 * a_max * td**2
    else:
        # trapezoid: accel (0..t_a), cruise (t_a..t_a+t_c), decel (t_a+t_c..T)
        for i, ti in enumerate(t):
            if ti <= t_a:
                pos[i] = 0.5 * a_max * ti**2
            elif ti <= (t_a + t_c):
                pos[i] = (0.5 * a_max * t_a**2) + v_max * (ti - t_a)
            else:
                td = ti - (t_a + t_c)
                pos[i] = (0.5 * a_max * t_a**2) + v_max * t_c + v_peak * td - 0.5 * a_max * td**2

    # Clamp last sample to exact L to avoid drift
    pos[-1] = L
    # Apply direction and offset
    world_pos = float(start) + sign * pos
    return world_pos.astype(float)
