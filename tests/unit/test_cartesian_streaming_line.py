"""The Cartesian streaming executor has to keep the TCP on its line.

That is what MOVECART and SERVOL promise, and both drive the executor
the same way a servo stream does: the same target, repeated at the tick
rate.
"""

import math

import numpy as np
import pytest

from parol6.config import LIMITS
from parol6.motion.streaming_executors import CartesianStreamingExecutor

DT = 0.01
# A diagonal in all three axes: a move along one axis alone cannot tell a
# straight path from a bowed one.
DELTA_M = np.array([0.12, -0.09, 0.06])


def _pose(xyz):
    m = np.eye(4)
    m[:3, 3] = xyz
    return m


def _sweep(retarget_every_tick: bool):
    """Drive a MOVECART to completion, returning (worst off-line m, peak
    TCP speed m/s, miss distance m)."""
    cse = CartesianStreamingExecutor(dt=DT)
    start = _pose([0.35, 0.10, 0.20])
    cse.sync_pose(start)
    goal = _pose(start[:3, 3] + DELTA_M)

    a = start[:3, 3].copy()
    b = goal[:3, 3].copy()
    unit = (b - a) / np.linalg.norm(b - a)

    cse.set_pose_target(goal)
    worst_off, peak = 0.0, 0.0
    prev = a.copy()
    for _ in range(20_000):
        if retarget_every_tick:
            cse.set_pose_target(goal)
        pose, _vel, finished = cse.tick()
        p = pose[:3, 3]
        rel = p - a
        worst_off = max(worst_off, float(np.linalg.norm(rel - (rel @ unit) * unit)))
        peak = max(peak, float(np.linalg.norm(p - prev)) / DT)
        prev = p.copy()
        if finished:
            break
    else:
        pytest.fail("the move never finished")
    return worst_off, peak, float(np.linalg.norm(prev - b))


@pytest.mark.unit
def test_cartesian_stream_holds_its_line_and_its_tcp_speed():
    """Two things the executor owes a Cartesian move.

    The path is straight, whether the target is set once or repeated
    every tick as a servo stream does — Ruckig only holds the six tangent
    components to one shared profile under phase synchronization, and a
    re-plan drops out of phase unless the target is left alone once set.

    The speed ceiling is a TCP speed, not a per-axis one. Ruckig bounds
    each component separately, so an isotropic envelope lets a diagonal
    run the resultant up to sqrt(3) times the configured limit.
    """
    ceiling = LIMITS.cart.jog.velocity.linear

    for repeated in (False, True):
        off, peak, miss = _sweep(retarget_every_tick=repeated)
        how = "repeated every tick" if repeated else "set once"
        assert off < 1e-6, f"target {how}: TCP bowed {off * 1000:.3f} mm off the line"
        assert peak <= ceiling * 1.01, (
            f"target {how}: TCP ran at {peak:.4f} m/s over a {ceiling:.4f} m/s ceiling"
        )
        assert miss < 1e-9, f"target {how}: stopped {miss * 1000:.4f} mm short"


@pytest.mark.unit
def test_cartesian_stream_rations_a_mixed_move_between_both_ceilings():
    """A move that turns spends its budget on both halves at once, so
    neither the linear nor the angular ceiling may be exceeded."""
    cse = CartesianStreamingExecutor(dt=DT)
    start = _pose([0.30, 0.05, 0.25])
    cse.sync_pose(start)

    goal = start.copy()
    goal[:3, 3] = start[:3, 3] + DELTA_M
    # A rotation about Z, well clear of the pi wrap.
    angle = 0.6
    c, s = math.cos(angle), math.sin(angle)
    goal[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    cse.set_pose_target(goal)
    peak_lin, peak_ang = 0.0, 0.0
    prev_p = start[:3, 3].copy()
    prev_R = start[:3, :3].copy()
    for _ in range(20_000):
        cse.set_pose_target(goal)
        pose, _vel, finished = cse.tick()
        peak_lin = max(peak_lin, float(np.linalg.norm(pose[:3, 3] - prev_p)) / DT)
        # Rotation angle between consecutive orientations.
        dR = prev_R.T @ pose[:3, :3]
        cos = (np.trace(dR) - 1.0) / 2.0
        peak_ang = max(peak_ang, math.acos(min(1.0, max(-1.0, cos))) / DT)
        prev_p = pose[:3, 3].copy()
        prev_R = pose[:3, :3].copy()
        if finished:
            break
    else:
        pytest.fail("the move never finished")

    assert peak_lin <= LIMITS.cart.jog.velocity.linear * 1.01, (
        f"linear TCP speed {peak_lin:.4f} m/s over its ceiling"
    )
    assert peak_ang <= LIMITS.cart.jog.velocity.angular * 1.01, (
        f"angular TCP speed {peak_ang:.4f} rad/s over its ceiling"
    )
