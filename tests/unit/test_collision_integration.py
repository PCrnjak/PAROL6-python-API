"""Unit tests for PAROL6's pinokin collision integration.

Covers:
- Singleton checker initialization in PAROL6_ROBOT
- SRDF disabled-pair count
- Public Robot.in_collision / colliding_pairs / min_distance / check_trajectory
- guard_joint_path raising MotionError on a colliding sample
"""

from __future__ import annotations

import numpy as np
import pytest

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
import parol6.config  # noqa: F401 - imports trigger collision-checker init
from parol6 import Robot
from parol6.commands._collision_guard import guard_joint_path
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import MotionError


def test_singleton_checker_initialized():
    assert PAROL6_ROBOT.collision is not None
    assert PAROL6_ROBOT.collision.num_geometry_objects > 0
    assert PAROL6_ROBOT.collision.num_collision_pairs > 0


def test_srdf_disabled_pairs_reduce_pair_count():
    # Without SRDF: 7 link geometries -> 21 pairs minus 6 parent/child adjacent = 15.
    # The bundled SRDF disables 7 more pairs (base<->L4/L5/L6, L1<->L4/L5/L6, L2<->L4).
    assert PAROL6_ROBOT.collision.num_collision_pairs == 8


def test_home_is_clear():
    q = np.zeros(PAROL6_ROBOT.robot.nq)
    assert PAROL6_ROBOT.collision.in_collision(q) is False


def test_robot_in_collision_method():
    r = Robot()
    try:
        q = np.zeros(6)
        assert r.in_collision(q) is False
    finally:
        del r


def test_robot_min_distance_positive_at_home():
    r = Robot()
    try:
        q = np.zeros(6)
        d = r.min_distance(q)
        assert d > 0.0 and d != float("inf")
    finally:
        del r


def test_robot_check_trajectory_clear():
    r = Robot()
    try:
        # Tiny perturbation around home - definitely clear.
        q_path = np.linspace(np.zeros(6), 0.01 * np.ones(6), 5)
        assert r.check_trajectory(q_path) == -1
    finally:
        del r


def test_guard_joint_path_clear_returns_none():
    positions = np.zeros((10, 6))
    # No exception means no collision detected.
    guard_joint_path(positions)


def test_guard_joint_path_raises_on_explicit_collision(monkeypatch):
    """Force a fake collision by patching the singleton checker temporarily."""
    real = PAROL6_ROBOT.collision

    class FakeChecker:
        def in_collision(self, q):
            return True

        def check_path(self, q):
            return 2

        def colliding_pairs(self, q):
            return [("ssg48_body_simplified.stl", "L4_0")]

    monkeypatch.setattr(PAROL6_ROBOT, "collision", FakeChecker())

    positions = np.zeros((5, 6))
    with pytest.raises(MotionError) as exc_info:
        guard_joint_path(positions)
    err = exc_info.value.robot_error
    assert err.code == int(ErrorCode.SYS_SELF_COLLISION)
    # Cause string should embed the named pair, not raw int indices.
    assert "ssg48_body_simplified.stl vs L4_0" in err.cause

    monkeypatch.setattr(PAROL6_ROBOT, "collision", real)


def test_guard_disabled_when_checker_is_none(monkeypatch):
    monkeypatch.setattr(PAROL6_ROBOT, "collision", None)
    positions = np.zeros((5, 6))
    # No exception, returns None (no-op).
    assert guard_joint_path(positions) is None


def test_no_spurious_self_overlap_at_home_or_joint_limits():
    """Audit the bundled simplified collision STLs against the SRDF.

    Asserts that the checker reports no colliding pairs at home, at each
    joint's lower/upper limit (single-axis), at every (low, high) corner
    of the joint-limit hypercube, and across a handful of seeded random
    configs. If this fails, the assertion message identifies the named
    pair — add it to parol6/urdf_model/srdf/PAROL6.srdf and re-run.
    """
    import itertools

    lo = PAROL6_ROBOT._joint_limits_radian[:, 0]
    hi = PAROL6_ROBOT._joint_limits_radian[:, 1]
    checker = PAROL6_ROBOT.collision

    configs: list[tuple[str, np.ndarray]] = [("home", np.zeros(6))]
    for j in range(6):
        q_lo = np.zeros(6)
        q_lo[j] = lo[j]
        q_hi = np.zeros(6)
        q_hi[j] = hi[j]
        configs.append((f"J{j}=low", q_lo))
        configs.append((f"J{j}=high", q_hi))

    for bits in itertools.product((0, 1), repeat=6):
        q = np.where(np.array(bits, dtype=bool), hi, lo)
        configs.append((f"corner_{''.join(map(str, bits))}", q))

    rng = np.random.default_rng(0xC011)
    for k in range(20):
        configs.append((f"rand_{k}", rng.uniform(lo, hi)))

    failures: list[str] = []
    for label, q in configs:
        pairs = checker.colliding_pairs(np.ascontiguousarray(q, dtype=np.float64))
        if pairs:
            failures.append(f"{label}: {pairs}")

    assert not failures, (
        "Spurious self-collision in the bundled simplified STLs. Add these "
        "pairs to parol6/urdf_model/srdf/PAROL6.srdf and re-run:\n"
        + "\n".join(failures[:10])
        + (f"\n... ({len(failures) - 10} more)" if len(failures) > 10 else "")
    )


def test_collision_check_speed_diagnostic(capsys):
    """Time in_collision and colliding_pairs across a sampled workspace.

    Diagnostic only — does not assert a threshold. Prints percentiles so
    the JogLCommand mid-motion check can be evaluated against the 100 Hz
    tick budget (10 ms). Decision criterion: if `in_collision` p99 is
    well under 1000 us, the JogLCommand check is viable; otherwise drop
    it and rely on the trajectory-build pre-flight guards.

    Run via:  pytest tests/unit/test_collision_integration.py::test_collision_check_speed_diagnostic -v -s
    """
    import time

    rng = np.random.default_rng(0xC0FFEE)
    lo = PAROL6_ROBOT._joint_limits_radian[:, 0]
    hi = PAROL6_ROBOT._joint_limits_radian[:, 1]
    qs = [np.zeros(6)] + [
        np.ascontiguousarray(rng.uniform(lo, hi), dtype=np.float64) for _ in range(99)
    ]
    c = PAROL6_ROBOT.collision

    # warm-up so first-call cache effects don't dominate
    for q in qs[:10]:
        c.in_collision(q)
        c.colliding_pairs(q)

    t_bool: list[int] = []
    for q in qs:
        t0 = time.perf_counter_ns()
        c.in_collision(q)
        t_bool.append(time.perf_counter_ns() - t0)

    t_pairs: list[int] = []
    for q in qs:
        t0 = time.perf_counter_ns()
        c.colliding_pairs(q)
        t_pairs.append(time.perf_counter_ns() - t0)

    def pct(a: list[int], p: float) -> float:
        return float(np.percentile(a, p)) / 1000.0  # ns -> us

    with capsys.disabled():
        print(
            f"\nin_collision     us:"
            f" min={pct(t_bool, 0):.1f}"
            f" med={pct(t_bool, 50):.1f}"
            f" p95={pct(t_bool, 95):.1f}"
            f" p99={pct(t_bool, 99):.1f}"
            f" max={pct(t_bool, 100):.1f}"
        )
        print(
            f"colliding_pairs  us:"
            f" min={pct(t_pairs, 0):.1f}"
            f" med={pct(t_pairs, 50):.1f}"
            f" p95={pct(t_pairs, 95):.1f}"
            f" p99={pct(t_pairs, 99):.1f}"
            f" max={pct(t_pairs, 100):.1f}"
        )
        print("servo tick budget: 10000 us (100 Hz)")
