import math

import numpy as np
import pytest
from parol6.config import CONTROL_RATE_HZ
from parol6.utils import trajectory as traj


def approx_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_plan_linear_quintic_endpoints_and_shape():
    start = [0.0, 0.0]
    end = [10.0, 20.0]
    duration = 1.0  # seconds

    path = traj.plan_linear_quintic(start, end, duration)
    # Expected sample count: duration * rate + 1
    expected_n = int(round(duration * CONTROL_RATE_HZ)) + 1
    assert path.shape == (expected_n, 2)

    # Endpoints
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], end)

    # Monotonic progression on each axis
    diffs = np.diff(path, axis=0)
    assert np.all(diffs[:, 0] >= -1e-9)
    assert np.all(diffs[:, 1] >= -1e-9)


def test_plan_linear_cubic_endpoints_and_shape():
    start = [5.0, -5.0, 2.5]
    end = [6.0, 0.0, 4.5]
    duration = 0.5

    path = traj.plan_linear_cubic(start, end, duration)
    expected_n = int(round(duration * CONTROL_RATE_HZ)) + 1
    assert path.shape == (expected_n, 3)

    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], end)

    diffs = np.diff(path, axis=0)
    # Allow tiny numerical noise
    assert np.all(diffs >= -1e-9)


def _timings(distance, v_max, a_max):
    # Mirror of _trapezoid_timings logic for test-side expectation
    if distance <= 0 or v_max <= 0 or a_max <= 0:
        return 0.0, 0.0, 0.0, 0.0, True
    t_a = v_max / a_max
    s_a = 0.5 * a_max * t_a**2
    if 2 * s_a < distance:
        s_c = distance - 2 * s_a
        t_c = s_c / v_max
        T = 2 * t_a + t_c
        return T, t_a, t_c, v_max, False
    else:
        v_peak = math.sqrt(a_max * distance)
        t_a = v_peak / a_max
        T = 2 * t_a
        return T, t_a, 0.0, v_peak, True


@pytest.mark.parametrize(
    "start,end,v_max,a_max",
    [
        (0.0, 1.0, 0.5, 1.0),  # trapezoidal (2*s_a < distance)
        (0.0, 0.02, 1.0, 10.0),  # triangular (cannot reach cruise)
        (10.0, 0.0, 0.5, 1.0),  # reverse direction trapezoidal
    ],
)
def test_plan_trapezoid_position_1d_shapes_and_endpoints(start, end, v_max, a_max):
    positions = traj.plan_trapezoid_position_1d(start, end, v_max, a_max)
    assert positions.ndim == 1
    assert approx_equal(positions[0], start, tol=1e-9)
    assert approx_equal(positions[-1], end, tol=1e-9)

    # Monotonic in the proper direction
    diffs = np.diff(positions)
    if end >= start:
        assert np.all(diffs >= -1e-9)
    else:
        assert np.all(diffs <= 1e-9)

    # Sample count should match expected duration discretization
    dist = abs(end - start)
    T, _ta, _tc, _vp, _tri = _timings(dist, v_max, a_max)
    if T > 0:
        expected_n = int(round(T * CONTROL_RATE_HZ)) + 1
        assert len(positions) == expected_n
    else:
        assert len(positions) == 2
