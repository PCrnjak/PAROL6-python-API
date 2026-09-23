"""Unit tests for joint-space and Cartesian N-command blending."""

import numpy as np
import pytest

from parol6.motion.geometry import (
    ArcSegment,
    LineSegment,
    _blend_joint_path_into,
    _linear_joint_segment_into,
    build_blended_path,
    build_composite_cartesian_path,
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


def _se3(xyz_m):
    m = np.eye(4)
    m[:3, 3] = xyz_m
    return m


class TestBlendedCartesianPath:
    """The corner zone between segments: what it is between two lines, and
    that an arc joins it like a line does."""

    def test_a_line_line_corner_is_the_quadratic_through_the_corner(self):
        """A chain of straight moves rounds exactly as it always has: the
        cubic zone is the degree-raised quadratic through the corner."""
        a, corner, b = (
            _se3([0.0, 0.0, 0.0]),
            _se3([0.1, 0.0, 0.0]),
            _se3([0.1, 0.1, 0.0]),
        )
        r_mm = 20.0
        path = build_composite_cartesian_path(
            [a, corner, b], [r_mm], samples_per_segment=40
        )
        pc, pa, pb = corner[:3, 3], a[:3, 3], b[:3, 3]
        entry = pc + (pa - pc) / np.linalg.norm(pa - pc) * r_mm / 1000.0
        exit_ = pc + (pb - pc) / np.linalg.norm(pb - pc) * r_mm / 1000.0
        # Dense enough that the distance to the nearest sample is the
        # distance to the curve, well under the tolerance.
        t = np.linspace(0.0, 1.0, 200_001)[:, None]
        quadratic = (1 - t) ** 2 * entry + 2 * (1 - t) * t * pc + t**2 * exit_
        in_zone = 0
        for pose in path:
            q = pose[:3, 3]
            if np.linalg.norm(q - pc) > r_mm / 1000.0 + 1e-12:
                continue
            in_zone += 1
            miss = np.min(np.linalg.norm(quadratic - q, axis=1))
            assert miss < 1e-6, f"a zone sample left the quadratic by {miss:e} m"
        assert in_zone > 10
        # The corner is cut, by less than the zone's radius.
        closest = min(np.linalg.norm(pose[:3, 3] - pc) for pose in path)
        assert 0.001 < closest < r_mm / 1000.0

    def test_an_arc_rounds_into_the_line_after_it(self):
        """line → arc → line with zones at both junctions: the direction of
        travel never jumps, each corner is cut inside its zone, and the
        arc is still its circle where no zone reaches."""
        a = _se3([0.1, 0.0, 0.0])
        via = _se3([0.16, 0.0, -0.06])
        b = _se3([0.22, 0.0, 0.0])
        c = _se3([0.32, 0.0, 0.0])
        start = _se3([0.0, 0.0, 0.0])
        segments = [LineSegment(start, a), ArcSegment(a, via, b), LineSegment(b, c)]
        path = build_blended_path(segments, [20.0, 20.0], samples_per_segment=60)
        pts = np.array([pose[:3, 3] for pose in path])
        steps = np.diff(pts, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        keep = lengths > 1e-9
        dirs = steps[keep] / lengths[keep][:, None]
        cos = np.clip(np.einsum("ij,ij->i", dirs[:-1], dirs[1:]), -1.0, 1.0)
        worst_turn = float(np.degrees(np.arccos(cos)).max())
        assert worst_turn < 12.0, f"the path turned {worst_turn:.1f}° in one step"
        for corner in (a, b):
            closest = float(np.min(np.linalg.norm(pts - corner[:3, 3], axis=1)))
            assert 0.001 < closest <= 0.020 + 1e-9, (
                f"corner cut by {closest * 1000:.2f} mm"
            )
        center = np.array([0.16, 0.0, 0.0])
        low = pts[pts[:, 2] < -0.03]
        assert len(low) > 10
        radial = np.abs(np.linalg.norm(low - center, axis=1) - 0.06)
        assert radial.max() < 1e-6, f"the arc left its circle by {radial.max():e} m"
        assert np.allclose(pts[-1], c[:3, 3])
        assert np.allclose(pts[0], start[:3, 3])
