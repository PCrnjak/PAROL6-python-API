"""Unit tests for electric gripper ramp simulation in mock transport."""

import math

import numpy as np
import pytest

from parol6.server.transports.mock_serial_transport import _simulate_gripper_ramp_jit
from parol6.tools import ElectricGripperConfig, get_registry


# SSG-48 constants (from firmware + physical measurement)
SSG48_TICK_RANGE = 10_432.0  # 24mm / (pi * 12mm) * 16384
SSG48_MIN_SPEED = 40.0
SSG48_MAX_SPEED = 80_000.0

# MSG constants
MSG_GEAR_PD_MM = 16.67
MSG_ENCODER_CPR = 16_384
MSG_MIN_SPEED = 500.0
MSG_MAX_SPEED = 60_000.0


class TestGripperRampJit:
    """Test _simulate_gripper_ramp_jit directly with known inputs."""

    def _make_state(self, target=128.0, speed_byte=255.0, active=True, pos_f=0.0):
        ramp = np.array(
            [target, speed_byte, 1.0 if active else 0.0], dtype=np.float64
        )
        data_in = np.zeros(6, dtype=np.int32)
        return ramp, data_in, pos_f

    def test_inactive_ramp_is_noop(self):
        ramp, data_in, pos_f = self._make_state(active=False, pos_f=50.0)
        result = _simulate_gripper_ramp_jit(
            ramp, data_in, pos_f, 0.01, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
        )
        assert result == 50.0
        assert data_in[1] == 0  # unchanged

    def test_ramp_is_gradual_not_instant(self):
        """After one tick at moderate speed, position should not yet reach target."""
        ramp, data_in, pos_f = self._make_state(target=200.0, speed_byte=128.0, pos_f=0.0)
        result = _simulate_gripper_ramp_jit(
            ramp, data_in, pos_f, 0.01, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
        )
        # Should have moved but not arrived
        assert result > 0.0
        assert result < 200.0
        assert data_in[1] == int(result + 0.5)

    def test_ramp_converges_to_target(self):
        """Running enough ticks should reach the target."""
        ramp, data_in, pos_f = self._make_state(target=200.0, speed_byte=255.0, pos_f=0.0)
        dt = 0.01  # 100 Hz
        for _ in range(200):  # 2 seconds — well beyond max travel time
            pos_f = _simulate_gripper_ramp_jit(
                ramp, data_in, pos_f, dt, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
            )
        assert pos_f == pytest.approx(200.0)
        assert ramp[2] < 0.5  # deactivated
        assert data_in[1] == 200

    def test_higher_speed_arrives_faster(self):
        """Higher speed byte should reach target in fewer ticks."""
        dt = 0.01

        # Slow speed (byte=25)
        ramp_slow, data_slow, pos_slow = self._make_state(target=200.0, speed_byte=25.0)
        ticks_slow = 0
        for _ in range(500):
            pos_slow = _simulate_gripper_ramp_jit(
                ramp_slow, data_slow, pos_slow, dt, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
            )
            ticks_slow += 1
            if ramp_slow[2] < 0.5:
                break

        # Fast speed (byte=255)
        ramp_fast, data_fast, pos_fast = self._make_state(target=200.0, speed_byte=255.0)
        ticks_fast = 0
        for _ in range(500):
            pos_fast = _simulate_gripper_ramp_jit(
                ramp_fast, data_fast, pos_fast, dt, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
            )
            ticks_fast += 1
            if ramp_fast[2] < 0.5:
                break

        assert ticks_fast < ticks_slow

    def test_ramp_moves_in_both_directions(self):
        """Ramp should work for both increasing and decreasing positions."""
        dt = 0.01
        # Start at 200, target 50 (decreasing)
        ramp, data_in, pos_f = self._make_state(target=50.0, speed_byte=200.0, pos_f=200.0)
        result = _simulate_gripper_ramp_jit(
            ramp, data_in, pos_f, dt, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
        )
        assert result < 200.0  # moved toward target

    def test_ssg48_full_travel_time(self):
        """SSG-48 full travel at max speed should take ~0.13s (13 ticks at 100Hz)."""
        dt = 0.01
        ramp, data_in, pos_f = self._make_state(target=255.0, speed_byte=255.0, pos_f=0.0)
        ticks = 0
        for _ in range(100):
            pos_f = _simulate_gripper_ramp_jit(
                ramp, data_in, pos_f, dt, SSG48_TICK_RANGE, SSG48_MIN_SPEED, SSG48_MAX_SPEED
            )
            ticks += 1
            if ramp[2] < 0.5:
                break
        # Should complete in roughly 13 ticks (0.13s) — allow some tolerance
        assert 10 <= ticks <= 20


class TestElectricGripperConfigSpecs:
    """Verify physical specs on tool configs match expected values."""

    def test_ssg48_config(self):
        cfg = get_registry()["SSG-48"]
        assert isinstance(cfg, ElectricGripperConfig)
        assert cfg.encoder_cpr == 16_384
        assert cfg.gear_pd_mm == 12.0
        assert cfg.firmware_speed_range_tps == (40, 80_000)

    def test_msg_config(self):
        cfg = get_registry()["MSG"]
        assert isinstance(cfg, ElectricGripperConfig)
        assert cfg.encoder_cpr == 16_384
        assert cfg.gear_pd_mm == pytest.approx(16.67, abs=0.01)
        assert cfg.firmware_speed_range_tps == (500, 60_000)

    def test_ssg48_tick_range_derivation(self):
        """Tick range derived from gear PD + travel_m should match expected ~10,432."""
        cfg = get_registry()["SSG-48"]
        assert isinstance(cfg, ElectricGripperConfig)
        from waldoctl.tools import LinearMotion

        motion = next(m for m in cfg.motions if isinstance(m, LinearMotion))
        travel_mm = motion.travel_m * 1000.0
        tick_range = (travel_mm / (math.pi * cfg.gear_pd_mm)) * cfg.encoder_cpr
        assert tick_range == pytest.approx(10_432, rel=0.01)

    def test_msg_100mm_tick_range_derivation(self):
        """MSG 100mm tick range should be ~8,353."""
        cfg = get_registry()["MSG"]
        assert isinstance(cfg, ElectricGripperConfig)
        from waldoctl.tools import LinearMotion

        motion = next(m for m in cfg.motions if isinstance(m, LinearMotion))
        travel_mm = motion.travel_m * 1000.0
        tick_range = (travel_mm / (math.pi * cfg.gear_pd_mm)) * cfg.encoder_cpr
        assert tick_range == pytest.approx(8_353, rel=0.02)
