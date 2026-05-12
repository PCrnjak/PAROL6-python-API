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
    # The bundled SRDF disables 5 more pairs (base<->L4/L5/L6 + L1<->L5/L6).
    assert PAROL6_ROBOT.collision.num_collision_pairs == 10


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
            return [(0, 5)]

    monkeypatch.setattr(PAROL6_ROBOT, "collision", FakeChecker())

    positions = np.zeros((5, 6))
    with pytest.raises(MotionError) as exc_info:
        guard_joint_path(positions)
    assert exc_info.value.robot_error.code == int(ErrorCode.SYS_SELF_COLLISION)

    monkeypatch.setattr(PAROL6_ROBOT, "collision", real)


def test_guard_disabled_when_checker_is_none(monkeypatch):
    monkeypatch.setattr(PAROL6_ROBOT, "collision", None)
    positions = np.zeros((5, 6))
    # No exception, returns None (no-op).
    assert guard_joint_path(positions) is None
