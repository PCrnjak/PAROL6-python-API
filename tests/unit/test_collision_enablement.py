"""Directional collision enablement gates + geometry sync in the IK worker."""

import multiprocessing
import time
from dataclasses import dataclass

import numpy as np

from parol6.server.ik_worker import (
    _AXIS_DIRS,
    _ENABLE_NEAR_M,
    SyncShapes,
    SyncTool,
    _compute_cart_enable,
    _drain_sync,
    gate_joint_enable_collision,
)


class _FakeChecker:
    def __init__(self, distance, collides):
        self._distance = distance
        self._collides = collides
        self.queries = 0

    def min_distance(self, q):
        return self._distance

    def in_collision(self, q):
        self.queries += 1
        return self._collides(q)


def test_gate_skips_per_direction_checks_when_far():
    joint_en = np.ones(12, dtype=np.uint8)
    checker = _FakeChecker(_ENABLE_NEAR_M + 0.1, lambda q: True)
    gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en.tolist() == [1] * 12
    assert checker.queries == 0  # proximity-gated: no in_collision when far


def test_gate_greys_only_the_colliding_direction():
    joint_en = np.ones(12, dtype=np.uint8)
    # Near collision; stepping joint index 1 (J2) positive trips it.
    checker = _FakeChecker(_ENABLE_NEAR_M - 0.01, lambda q: q[1] > 0.01)
    gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en[2] == 0  # J2+ greyed
    assert joint_en[3] == 1  # J2- still allowed
    assert joint_en[0] == 1 and joint_en[1] == 1  # J1 untouched


@dataclass
class _FakeIK:
    success: bool
    q: np.ndarray


def _cart_enable(checker, solve_ik):
    out = np.ones(12, dtype=np.uint8)
    _compute_cart_enable(
        np.eye(4),
        True,
        np.zeros(6),
        None,
        solve_ik,
        _AXIS_DIRS,
        np.zeros((12, 4, 4), dtype=np.float64),
        out,
        checker=checker,
    )
    return out


def test_cart_gate_greys_colliding_directions_when_near():
    # IK succeeds everywhere; the solved config for X+ (axis 0) collides.
    calls = []

    def solve_ik(robot, target, q_seed, quiet_logging=True):
        calls.append(target.copy())
        return _FakeIK(True, target[:3, 3].repeat(2))

    checker = _FakeChecker(_ENABLE_NEAR_M - 0.01, lambda q: q[0] > 0)
    out = _cart_enable(checker, solve_ik)
    assert out[0] == 0  # X+ greyed (solved config collides)
    assert out[1] == 1  # X- fine
    assert len(calls) == 12


def test_cart_gate_skips_collision_checks_when_far():
    def solve_ik(robot, target, q_seed, quiet_logging=True):
        return _FakeIK(True, np.zeros(6))

    checker = _FakeChecker(_ENABLE_NEAR_M + 0.1, lambda q: True)
    out = _cart_enable(checker, solve_ik)
    assert out.tolist() == [1] * 12
    assert checker.queries == 0


def test_drain_sync_applies_tool_and_shapes():
    applied = []

    class _FakeRobotModule:
        @staticmethod
        def apply_tool(tool_name, variant_key=""):
            applied.append(("tool", tool_name, variant_key))

        @staticmethod
        def apply_shapes(shapes):
            applied.append(("shapes", tuple(shapes)))

    q = multiprocessing.Queue()
    q.put(SyncTool(tool_name="ssg48", variant_key="v1"))
    q.put(SyncShapes(shapes=("wire1", "wire2")))
    deadline = time.monotonic() + 5.0
    while len(applied) < 2 and time.monotonic() < deadline:
        _drain_sync(q, _FakeRobotModule)  # feeder-thread latency: poll until seen
        time.sleep(0.005)
    assert ("tool", "ssg48", "v1") in applied
    assert ("shapes", ("wire1", "wire2")) in applied
    assert _drain_sync(q, _FakeRobotModule) is False  # empty → no-op
    q.close()
    q.cancel_join_thread()
