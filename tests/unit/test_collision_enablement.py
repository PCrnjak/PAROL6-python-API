"""The directional self-collision enablement gate used by the IK worker."""

import numpy as np

from parol6.server.ik_worker import _ENABLE_NEAR_M, _gate_joint_enable_collision


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
    _gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en.tolist() == [1] * 12
    assert checker.queries == 0  # proximity-gated: no in_collision when far


def test_gate_greys_only_the_colliding_direction():
    joint_en = np.ones(12, dtype=np.uint8)
    # Near collision; stepping joint index 1 (J2) positive trips it.
    checker = _FakeChecker(_ENABLE_NEAR_M - 0.01, lambda q: q[1] > 0.01)
    _gate_joint_enable_collision(checker, np.zeros(6), joint_en, np.zeros(6))
    assert joint_en[2] == 0  # J2+ greyed
    assert joint_en[3] == 1  # J2- still allowed
    assert joint_en[0] == 1 and joint_en[1] == 1  # J1 untouched
