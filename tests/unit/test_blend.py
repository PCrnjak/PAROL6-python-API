"""Unit tests for joint-space and Cartesian N-command blending."""

import numpy as np
import pytest

from parol6.motion.geometry import (
    _blend_joint_path_into,
    _linear_joint_segment_into,
    build_composite_joint_path,
)


class TestLinearJointSegment:
    """Tests for _linear_joint_segment_into helper."""

    def test_full_segment(self):
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        end = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = np.empty((11, 6), dtype=np.float64)
        _linear_joint_segment_into(start, end, out)
        assert out.shape[0] == 11
        assert np.allclose(out[0], start)
        assert np.allclose(out[-1], end)
        assert np.allclose(out[5], (start + end) / 2)

    def test_partial_segment(self):
        start = np.zeros(6)
        end = np.ones(6)
        out = np.empty((11, 6), dtype=np.float64)
        _linear_joint_segment_into(start, end, out, s_start=0.25, s_end=0.75)
        assert out.shape[0] == 11
        assert np.allclose(out[0], np.full(6, 0.25))
        assert np.allclose(out[-1], np.full(6, 0.75))


class TestBlendJointPath:
    """Tests for _blend_joint_path_into Bezier blend zone."""

    def test_endpoints_match(self):
        entry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        waypoint = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        exit_ = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])

        out = np.empty((21, 6), dtype=np.float64)
        _blend_joint_path_into(entry, waypoint, exit_, out)
        assert out.shape[0] == 21
        assert np.allclose(out[0], entry)
        assert np.allclose(out[-1], exit_)

    def test_c1_continuity_at_entry(self):
        """Tangent at t=0 should point from entry toward waypoint."""
        entry = np.zeros(6)
        waypoint = np.ones(6)
        exit_ = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])

        out = np.empty((101, 6), dtype=np.float64)
        _blend_joint_path_into(entry, waypoint, exit_, out)

        # Numerical derivative at t=0
        dt = 1.0 / 100
        tangent_start = (out[1] - out[0]) / dt

        # Expected tangent: d/dt[(1-t)^2 E + 2t(1-t)W + t^2 X] at t=0
        # = 2(W - E)
        expected_tangent = 2.0 * (waypoint - entry)
        assert np.allclose(tangent_start, expected_tangent, atol=0.1)

    def test_c1_continuity_at_exit(self):
        """Tangent at t=1 should point from waypoint toward exit."""
        entry = np.zeros(6)
        waypoint = np.ones(6)
        exit_ = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])

        out = np.empty((101, 6), dtype=np.float64)
        _blend_joint_path_into(entry, waypoint, exit_, out)

        dt = 1.0 / 100
        tangent_end = (out[-1] - out[-2]) / dt

        # Expected: 2(X - W)
        expected_tangent = 2.0 * (exit_ - waypoint)
        assert np.allclose(tangent_end, expected_tangent, atol=0.1)


class TestBuildCompositeJointPath:
    """Tests for build_composite_joint_path."""

    def test_two_waypoints_no_blend(self):
        """Two waypoints should produce a straight interpolation."""
        w0 = np.zeros(6)
        w1 = np.ones(6)
        result = build_composite_joint_path([w0, w1], [], samples_per_segment=11)
        assert result.shape == (11, 6)
        assert np.allclose(result[0], w0)
        assert np.allclose(result[-1], w1)

    def test_three_waypoints_with_blend(self):
        """Three waypoints with a blend zone at the middle one."""
        w0 = np.zeros(6)
        w1 = np.ones(6)
        w2 = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])

        result = build_composite_joint_path(
            [w0, w1, w2],
            [(0.3, 0.3)],
            samples_per_segment=20,
        )

        assert result.ndim == 2
        assert result.shape[1] == 6
        # Path should start at w0 and end at w2
        assert np.allclose(result[0], w0)
        assert np.allclose(result[-1], w2)

        # Path should NOT pass exactly through w1 (it's blended)
        dists_to_w1 = np.linalg.norm(result - w1, axis=1)
        assert dists_to_w1.min() > 0.01, (
            "Path should round the corner, not pass through w1"
        )

    def test_four_waypoints_two_blend_zones(self):
        """Four waypoints with two blend zones."""
        w0 = np.zeros(6)
        w1 = np.ones(6)
        w2 = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
        w3 = np.full(6, 3.0)

        result = build_composite_joint_path(
            [w0, w1, w2, w3],
            [(0.2, 0.2), (0.2, 0.2)],
            samples_per_segment=20,
        )

        assert result.ndim == 2
        assert result.shape[1] == 6
        assert np.allclose(result[0], w0)
        assert np.allclose(result[-1], w3)

    def test_zero_blend_fracs(self):
        """Zero blend fractions should produce sharp corners (linear segments only)."""
        w0 = np.zeros(6)
        w1 = np.ones(6)
        w2 = np.full(6, 2.0)

        result = build_composite_joint_path(
            [w0, w1, w2],
            [(0.0, 0.0)],
            samples_per_segment=11,
        )

        # With zero blend, path should pass through w1
        dists_to_w1 = np.linalg.norm(result - w1, axis=1)
        assert dists_to_w1.min() < 1e-10, "Zero blend should pass through waypoint"

    def test_large_blend_fracs_clamped(self):
        """Blend fractions > 0.5 should be clamped."""
        w0 = np.zeros(6)
        w1 = np.ones(6)
        w2 = np.full(6, 2.0)

        # Should not raise even with extreme fractions
        result = build_composite_joint_path(
            [w0, w1, w2],
            [(0.9, 0.9)],
            samples_per_segment=20,
        )
        assert result.ndim == 2
        assert np.allclose(result[0], w0)
        assert np.allclose(result[-1], w2)

    def test_wrong_blend_fracs_count_raises(self):
        """Mismatched blend_fracs count should raise ValueError."""
        w0 = np.zeros(6)
        w1 = np.ones(6)
        w2 = np.full(6, 2.0)

        with pytest.raises(ValueError, match="Expected 1 blend_fracs"):
            build_composite_joint_path([w0, w1, w2], [])

    def test_single_waypoint_raises(self):
        with pytest.raises(ValueError, match="Need at least 2"):
            build_composite_joint_path([np.zeros(6)], [])

    def test_path_continuity(self):
        """Adjacent samples should be close (no jumps)."""
        w0 = np.zeros(6)
        w1 = np.ones(6) * 0.5
        w2 = np.ones(6)

        result = build_composite_joint_path(
            [w0, w1, w2],
            [(0.3, 0.3)],
            samples_per_segment=50,
        )

        # Max jump between consecutive samples should be small
        diffs = np.diff(result, axis=0)
        max_jump = np.max(np.abs(diffs))
        assert max_jump < 0.05, f"Max jump {max_jump} too large — path is discontinuous"

    def test_no_double_skip_at_junction(self):
        """Gap at blend-to-linear junction should be <= 1 grid step."""
        w0 = np.zeros(6)
        w1 = np.ones(6) * 0.5
        w2 = np.ones(6)

        result = build_composite_joint_path(
            [w0, w1, w2],
            [(0.3, 0.3)],
            samples_per_segment=50,
        )

        # Compute per-step distances
        diffs = np.linalg.norm(np.diff(result, axis=0), axis=1)
        # The maximum step should not be more than 2x the median step
        median_step = np.median(diffs)
        assert diffs.max() < 3.0 * median_step, (
            f"Max step {diffs.max():.6f} is >3x median {median_step:.6f} — "
            "likely double-skip at junction"
        )

    def test_adaptive_blend_samples(self):
        """Blend sample count should scale with blend fraction."""
        w0 = np.zeros(6)
        w1 = np.ones(6) * 0.5
        w2 = np.ones(6)

        small_blend = build_composite_joint_path(
            [w0, w1, w2],
            [(0.05, 0.05)],
            samples_per_segment=50,
        )
        large_blend = build_composite_joint_path(
            [w0, w1, w2],
            [(0.4, 0.4)],
            samples_per_segment=50,
        )

        # Larger blend fraction should produce more samples
        assert large_blend.shape[0] > small_blend.shape[0], (
            f"Large blend ({large_blend.shape[0]}) should have more samples "
            f"than small blend ({small_blend.shape[0]})"
        )


class TestMaxBlendLookahead:
    """Test the config constant exists."""

    def test_config_exists(self):
        from parol6.config import MAX_BLEND_LOOKAHEAD

        assert isinstance(MAX_BLEND_LOOKAHEAD, int)
        assert MAX_BLEND_LOOKAHEAD >= 1
