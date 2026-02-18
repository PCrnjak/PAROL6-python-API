"""Test _max_vel_ratio_jit velocity ratio computation.

When IK crosses the wrist singularity (J5≈0), J4/J6 can jump ~90° in one tick.
_max_vel_ratio_jit returns the ratio of the worst-case joint delta to its
per-tick hardware velocity limit. A ratio >1.0 triggers CSE speed reduction.
"""

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.servo_commands import _max_vel_ratio_jit
from parol6.config import INTERVAL_S, LIMITS

_HARD_VEL = np.array(LIMITS.joint.hard.velocity, dtype=np.float64)
_MAX_STEP_RAD = _HARD_VEL * INTERVAL_S


def _home_rad() -> np.ndarray:
    return np.deg2rad(
        np.ascontiguousarray(PAROL6_ROBOT.joint.standby_deg, dtype=np.float64)
    )


class TestMaxVelRatioJit:
    def test_large_jump_exceeds_limit(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[3] += np.deg2rad(90.0)

        assert _max_vel_ratio_jit(target, home) > 1.0

    def test_small_move_within_limit(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[0] += _MAX_STEP_RAD[0] * 0.5

        assert _max_vel_ratio_jit(target, home) < 1.0

    def test_exactly_at_limit(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[2] += _MAX_STEP_RAD[2] * 0.999

        assert _max_vel_ratio_jit(target, home) < 1.0

    def test_just_over_limit(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[2] += _MAX_STEP_RAD[2] * 1.001

        ratio = _max_vel_ratio_jit(target, home)
        assert ratio > 1.0
        assert abs(ratio - 1.001) < 0.01

    def test_multi_joint_returns_worst_case(self) -> None:
        """Ratio reflects the worst-case joint, not the sum."""
        home = _home_rad()
        target = home.copy()
        target[0] += _MAX_STEP_RAD[0] * 0.5  # ratio 0.5
        target[3] += np.deg2rad(90.0)  # ratio >> 1
        target[5] -= np.deg2rad(45.0)  # ratio >> 1

        ratio = _max_vel_ratio_jit(target, home)
        assert ratio > 1.0
        # Should be driven by whichever of J4/J6 has the higher ratio
        expected_j4 = np.deg2rad(90.0) / _MAX_STEP_RAD[3]
        expected_j6 = np.deg2rad(45.0) / _MAX_STEP_RAD[5]
        assert abs(ratio - max(expected_j4, expected_j6)) < 0.1

    def test_identical_positions_zero_ratio(self) -> None:
        home = _home_rad()
        assert _max_vel_ratio_jit(home, home) == 0.0
