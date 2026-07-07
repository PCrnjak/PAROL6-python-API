"""Directional collision enablement gates + geometry sync in the IK worker."""

import multiprocessing
import time
from dataclasses import dataclass

import numpy as np

from parol6.commands._collision_guard import collision_blocked
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
    """``distance``/``pairs`` may be values or callables(q) for escape-semantics
    tests. ``pairs`` feeds the new-contact half of the escape rule; the default
    (no pairs anywhere) leaves the distance-trend half in charge."""

    def __init__(self, distance, collides, pairs=()):
        self._distance = distance
        self._collides = collides
        self._pairs = pairs
        self.queries = 0

    def min_distance(self, q):
        return self._distance(q) if callable(self._distance) else self._distance

    def in_collision(self, q):
        self.queries += 1
        return self._collides(q)

    def colliding_pairs(self, q):
        return list(self._pairs(q)) if callable(self._pairs) else list(self._pairs)


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


def test_gate_clamps_probe_to_joint_limits():
    """Geometry just past a mechanical stop must not grey a direction the jog
    (whose own lookahead clamps to qlim) would actually permit."""
    joint_en = np.ones(12, dtype=np.uint8)
    # Near collision; anything beyond J1's upper limit collides.
    limit = np.radians(1.0)
    checker = _FakeChecker(_ENABLE_NEAR_M - 0.01, lambda q: q[0] > limit)
    qlim = (np.full(6, -np.pi), np.full(6, np.pi).copy())
    qlim[1][0] = limit  # J1 upper limit right at the collision boundary
    gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6), qlim=qlim)
    assert joint_en[0] == 1  # J1+ clamped to the limit -> no phantom grey
    # Without the clamp the unreachable probe greys the button.
    joint_en[:] = 1
    gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en[0] == 0


def test_gate_keeps_escaping_directions_when_already_inside():
    """Arm inside a keep-out: only deeper directions grey — never all of them."""
    joint_en = np.ones(12, dtype=np.uint8)
    # Penetrating (in_collision True at q); J1+ escapes (distance grows with
    # q[0]), J1- digs deeper. Other joints don't change the distance.
    checker = _FakeChecker(lambda q: -0.01 + q[0], lambda q: True)
    gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en[0] == 1  # J1+ escapes -> enabled
    assert joint_en[1] == 0  # J1- deeper -> greyed
    assert joint_en[2] == 1 and joint_en[3] == 1  # no-change dirs stay enabled


def test_gate_greys_direction_entering_new_pair_while_inside():
    """Inside keep-out A, a step that leaves the global min improving but
    contacts a NEW pair must grey — parity with the jog guard's pair diff."""
    joint_en = np.ones(12, dtype=np.uint8)
    checker = _FakeChecker(
        lambda q: -0.01 + q[0],  # A dominates; J1+ improves, J1- deepens
        lambda q: True,
        pairs=lambda q: [("L6", "shape:A")]
        + ([("L4", "shape:B")] if q[1] > 0.001 else []),
    )
    gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en[0] == 1  # J1+ pure escape -> enabled
    assert joint_en[2] == 0  # J2+ min unchanged but contacts B -> greyed
    assert joint_en[3] == 1  # J2- no new contact, min unchanged -> enabled


def test_collision_blocked_escape_semantics():
    """Shared jog/guard decision: approach blocks, escape is allowed."""
    # Approaching: current clear, target colliding -> blocked.
    approaching = _FakeChecker(0.02, lambda q: bool(q[0] > 0.5))
    assert collision_blocked(approaching, np.zeros(6), np.full(6, 1.0)) is True
    assert collision_blocked(approaching, np.zeros(6), np.full(6, 0.1)) is False
    # Already inside: escape (distance grows) allowed, deeper blocked.
    inside = _FakeChecker(lambda q: -0.01 + q[0], lambda q: True)
    assert collision_blocked(inside, np.zeros(6), np.full(6, 0.1)) is False
    assert collision_blocked(inside, np.zeros(6), np.full(6, -0.1)) is True


def test_collision_blocked_new_pair_blocks_even_when_global_min_improves():
    """Escaping keep-out A while contacting a NEW pair B must block: the pair
    diff catches what the global min-distance trend (still dominated by the
    deeper A) cannot — the exact rule guard_joint_path already applies."""
    inside = _FakeChecker(
        lambda q: -0.05 + q[0],  # A dominates the global min; +x improves it
        lambda q: True,
        pairs=lambda q: [("L6", "shape:A")]
        + ([("L5", "shape:B")] if q[0] > 0.05 else []),
    )
    # Pure escape (no new contact) stays allowed…
    assert collision_blocked(inside, np.zeros(6), np.full(6, 0.01)) is False
    # …but the same improving trend with a new pair at the target blocks.
    assert collision_blocked(inside, np.zeros(6), np.full(6, 0.1)) is True


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


def test_cart_gate_keeps_escaping_directions_when_already_inside():
    def solve_ik(robot, target, q_seed, quiet_logging=True):
        return _FakeIK(True, target[:3, 3].repeat(2))

    # Penetrating everywhere; solved configs with +x motion escape (distance
    # grows with q[0]), -x digs deeper.
    checker = _FakeChecker(lambda q: -0.01 + q[0], lambda q: True)
    out = _cart_enable(checker, solve_ik)
    assert out[0] == 1  # X+ escapes -> enabled
    assert out[1] == 0  # X- deeper -> greyed


def test_cart_gate_greys_direction_entering_new_pair_while_inside():
    """Inside keep-out A, a Cartesian direction whose solved config contacts a
    NEW pair greys even though the global min-distance holds steady."""

    def solve_ik(robot, target, q_seed, quiet_logging=True):
        return _FakeIK(True, target[:3, 3].repeat(2))  # q = [x, x, y, y, z, z]

    checker = _FakeChecker(
        lambda q: -0.01 + q[0],  # A dominates; only x motion changes it
        lambda q: True,
        pairs=lambda q: [("L6", "shape:A")] + ([("L4", "shape:B")] if q[2] > 0 else []),
    )
    out = _cart_enable(checker, solve_ik)
    assert out[0] == 1  # X+ pure escape -> enabled
    assert out[2] == 0  # Y+ min unchanged but contacts B -> greyed
    assert out[3] == 1  # Y- no new contact -> enabled


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


def test_drain_sync_survives_a_failing_apply():
    """A raising sync is logged + skipped — the worker must not die."""
    applied = []

    class _RaisingModule:
        @staticmethod
        def apply_tool(tool_name, variant_key=""):
            raise RuntimeError("mesh missing")

        @staticmethod
        def apply_shapes(shapes):
            applied.append(tuple(shapes))

    q = multiprocessing.Queue()
    q.put(SyncTool(tool_name="broken"))
    q.put(SyncShapes(shapes=("w",)))
    deadline = time.monotonic() + 5.0
    while not applied and time.monotonic() < deadline:
        _drain_sync(q, _RaisingModule)  # must not raise
        time.sleep(0.005)
    assert applied == [("w",)]
    q.close()
    q.cancel_join_thread()
