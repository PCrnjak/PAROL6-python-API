"""Test _vel_scale_and_convert_jit wrist flip velocity limiting.

When IK crosses the wrist singularity (J5≈0), J4/J6 can jump ~90° in one tick.
_vel_scale_and_convert_jit uniformly scales all joint velocities so the
worst-case joint is at its jog limit.
"""

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.servo_commands import _vel_scale_and_convert_jit
from parol6.config import INTERVAL_S, LIMITS, rad_to_steps, steps_to_rad

_JOG_VEL = np.array(LIMITS.joint.jog.velocity, dtype=np.float64)
_MAX_STEP_RAD = _JOG_VEL * INTERVAL_S


def _home_rad() -> np.ndarray:
    return np.deg2rad(
        np.ascontiguousarray(PAROL6_ROBOT.joint.standby_deg, dtype=np.float64)
    )


def _call(current: np.ndarray, target: np.ndarray):
    """Call _vel_scale_and_convert_jit, return (flipping, out_steps, flip_target_q)."""
    scratch = np.zeros(6, dtype=np.float64)
    out_steps = np.zeros(6, dtype=np.int32)
    flip_target_q = np.zeros(6, dtype=np.float64)
    flipping = _vel_scale_and_convert_jit(
        target,
        current,
        scratch,
        out_steps,
        flip_target_q,
    )
    return flipping, out_steps, flip_target_q


def _steps_to_rad(steps: np.ndarray) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_rad(steps, out)
    return out


class TestVelScaleAndConvertJit:
    def test_large_jump_triggers_flip(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[3] += np.deg2rad(90.0)

        flipping, _, flip_target_q = _call(home, target)

        assert flipping is True
        np.testing.assert_array_equal(flip_target_q, target)

    def test_small_move_passes_through(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[0] += _MAX_STEP_RAD[0] * 0.5

        flipping, out_steps, _ = _call(home, target)

        assert flipping is False

        expected_steps = np.zeros(6, dtype=np.int32)
        rad_to_steps(target, expected_steps)
        np.testing.assert_array_equal(out_steps, expected_steps)

    def test_clamped_step_within_jog_limits(self) -> None:
        home = _home_rad()
        target = home.copy()
        target[3] += np.deg2rad(90.0)
        target[5] -= np.deg2rad(90.0)

        flipping, out_steps, _ = _call(home, target)
        assert flipping is True

        home_steps = np.zeros(6, dtype=np.int32)
        rad_to_steps(home, home_steps)

        out_rad = _steps_to_rad(out_steps)
        home_rad_back = _steps_to_rad(home_steps)

        step_rad = np.abs(out_rad - home_rad_back)
        # Allow 1 motor step of tolerance for integer rounding
        rps = PAROL6_ROBOT.radian_per_step_constant
        step_tol = rps / np.abs(PAROL6_ROBOT.joint.ratio)
        assert np.all(step_rad <= _MAX_STEP_RAD + step_tol), (
            f"Step (deg): {np.rad2deg(step_rad)}, "
            f"Limit (deg): {np.rad2deg(_MAX_STEP_RAD)}"
        )

    def test_uniform_scaling(self) -> None:
        """All joints scaled by the same factor, preserving velocity direction."""
        home = _home_rad()
        target = home.copy()
        target[0] += np.deg2rad(10.0)
        target[3] += np.deg2rad(90.0)
        target[5] -= np.deg2rad(45.0)

        flipping, out_steps, _ = _call(home, target)
        assert flipping is True

        # Recover clamped radians from steps
        home_steps = np.zeros(6, dtype=np.int32)
        rad_to_steps(home, home_steps)

        clamped_rad = _steps_to_rad(out_steps)
        home_rad_back = _steps_to_rad(home_steps)

        original_delta = target - home
        clamped_delta = clamped_rad - home_rad_back
        moving = np.abs(original_delta) > 1e-6

        ratios = clamped_delta[moving] / original_delta[moving]
        # Ratios should be approximately equal (steps rounding adds noise)
        np.testing.assert_allclose(ratios, ratios[0], atol=0.01)

    def test_multi_tick_convergence(self) -> None:
        """Repeated calls converge to the target."""
        home = _home_rad()
        target = home.copy()
        target[3] += np.deg2rad(90.0)
        target[5] -= np.deg2rad(90.0)

        current = home.copy()
        for tick in range(10000):
            flipping, out_steps, _ = _call(current, target)
            # Use the steps output as next input (matching production flow)
            steps_to_rad(out_steps, current)
            if not flipping:
                break

        # Should be close to target (within steps quantization)
        rps = PAROL6_ROBOT.radian_per_step_constant
        step_tol = rps / np.abs(PAROL6_ROBOT.joint.ratio)
        np.testing.assert_allclose(current, target, atol=np.max(step_tol))
