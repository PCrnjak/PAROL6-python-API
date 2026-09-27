"""
Geometry generation for smooth motion paths.

This module provides pure geometry generators for arcs and splines,
plus blend zone computation for fly-by motion between consecutive segments.

All generators are stateless - they produce Cartesian path geometry without
depending on controller state or executing any motion.
"""

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pinokin import batch_se3_interp, se3_interp, so3_rpy
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from pinokin import Robot

logger = logging.getLogger(__name__)


#: Rotation weight in the multi-segment path metric sqrt(t² + (w·θ)²) [m/rad].
PATH_ROT_WEIGHT_M_PER_RAD: float = 0.15
#: Sampling pitch of a spline or process move on that metric [mm] (par6's
#: ``path_step_m``).
PATH_STEP_MM: float = 2.0
#: Most poses a spline or process move is sampled into (par6's
#: ``CART_PATH_MAX_STEPS``): bounds the IK and timing work of one path.
PATH_MAX_POINTS: int = 3000

#: An arc's end within this of its start asks for a full circle [mm]: the
#: arm settles a hair off the pose it was sent to, so the start the arc is
#: planned from is never exactly the end a script wrote back.
FULL_CIRCLE_MM: float = 1.0
#: A via within this of the line through an arc's start and end names no
#: circle [mm]. The arm settles within a motor step of a commanded pose, a
#: few hundredths of a millimetre at the tool, so a start read back from
#: it stands that far off a line a script drew through it.
ARC_COLLINEAR_MM: float = 0.1

_PATH_ROT_WEIGHT_MM_PER_RAD: float = PATH_ROT_WEIGHT_M_PER_RAD * 1000.0
# Legs shorter than this on the combined metric repeat the waypoint before [mm].
_REPEAT_MM: float = 1e-6


def _rotation_angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Angle between the rotations of two SE3 poses [rad].

    The atan2 of the relative rotation's sine and cosine: exact to rounding
    at zero, where the arccos of a rounded cosine reads a repeated pose as
    turned by up to 3e-8 rad."""
    r = a[:3, :3].T @ b[:3, :3]
    sine = 0.5 * math.sqrt(
        (r[2, 1] - r[1, 2]) ** 2 + (r[0, 2] - r[2, 0]) ** 2 + (r[1, 0] - r[0, 1]) ** 2
    )
    cosine = (r[0, 0] + r[1, 1] + r[2, 2] - 1.0) / 2.0
    return math.atan2(sine, cosine)


def pose_distance_m(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Distance between two SE3 poses on the combined metric
    sqrt(translation² + (w·rotation)²) [m]: a reorientation in place still
    covers distance."""
    translation = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    return math.hypot(translation, PATH_ROT_WEIGHT_M_PER_RAD * _rotation_angle(a, b))


def _intervals(length_mm: float, angle_rad: float) -> int:
    """Sample intervals a piece of path wants at ``PATH_STEP_MM`` on the
    combined metric, at least one."""
    metric = math.hypot(length_mm, _PATH_ROT_WEIGHT_MM_PER_RAD * angle_rad)
    return max(1, math.ceil(metric / PATH_STEP_MM))


def _fit_budget(counts: list[int]) -> None:
    """Scale per-piece interval counts down so the whole path stays within
    ``PATH_MAX_POINTS`` poses, every piece keeping at least one interval."""
    total = sum(counts)
    budget = max(PATH_MAX_POINTS, len(counts) + 1)
    if total < budget:
        return
    factor = (budget - 1) / total
    for i, c in enumerate(counts):
        counts[i] = max(1, round(c * factor))


def joint_path_to_tcp_poses(
    joint_positions: NDArray[np.float64],
    robot: "Robot | None" = None,
) -> NDArray[np.float64]:
    """Convert joint-space path to TCP poses using forward kinematics.

    This is useful for visualizing the actual TCP trajectory that results
    from joint-space interpolation (which traces an arc, not a straight line).

    Args:
        joint_positions: (N, 6) array of joint angles in radians
        robot: pinokin Robot model (uses PAROL6_ROBOT.robot if None)

    Returns:
        (N, 6) array of [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] poses
    """
    if robot is None:
        import parol6.PAROL6_ROBOT as PAROL6_ROBOT

        robot = PAROL6_ROBOT.robot

    # Batch FK in C++ (single call, no Python loop overhead)
    transforms = robot.batch_fk(joint_positions)

    n_points = len(joint_positions)
    tcp_poses = np.empty((n_points, 6), dtype=np.float64)
    rpy_buf = np.empty(3, dtype=np.float64)

    for i, T in enumerate(transforms):
        tcp_poses[i, :3] = T[:3, 3] * 1000.0  # m -> mm
        so3_rpy(T[:3, :3], rpy_buf)
        np.degrees(rpy_buf, out=tcp_poses[i, 3:])

    return tcp_poses


def compute_circle_from_3_points(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Compute the circumscribed circle through 3 non-collinear 3D points (mm).

    An end within ``FULL_CIRCLE_MM`` of the start is a full circle through
    the via opposite the start.

    Args:
        p1, p2, p3: 3D points (shape (3,)) in mm

    Returns:
        (center, radius, normal):
            center: Circle center point (3,)
            radius: Circle radius
            normal: Unit normal of the plane containing the circle (3,)

    Raises:
        ValueError: If the via lies within ``ARC_COLLINEAR_MM`` of the line
            through the start and the end (no unique circle), or all three
            points coincide.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    a = p2 - p1
    b = p3 - p1
    b_len = float(np.linalg.norm(b))

    if b_len < FULL_CIRCLE_MM:
        a_len = float(np.linalg.norm(a))
        if a_len < 1e-12:
            raise ValueError("All three points are coincident.")
        center = (p1 + p2) / 2.0
        radius = a_len / 2.0
        d = a / a_len
        ref = (
            np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        )
        normal = np.cross(d, ref)
        normal /= np.linalg.norm(normal)
        return center, radius, normal

    normal = np.asarray(np.cross(a, b), dtype=np.float64)
    normal_len = float(np.linalg.norm(normal))
    # |a × b| / |b| is how far the via stands off the start-end line.
    if normal_len < ARC_COLLINEAR_MM * b_len:
        raise ValueError("Points are collinear; no unique circle exists.")
    np.divide(normal, normal_len, out=normal)

    # Circumcenter via perpendicular bisector intersection in the plane.
    # C = p1 + s*a + t*b, where s and t satisfy:
    #   (C - p1)·a = |a|²/2   and   (C - p1)·b = |b|²/2
    # Expanding: s*(a·a) + t*(b·a) = (a·a)/2
    #            s*(a·b) + t*(b·b) = (b·b)/2
    aa = float(np.dot(a, a))
    bb = float(np.dot(b, b))
    ab = float(np.dot(a, b))
    det = aa * bb - ab * ab

    s = (bb * aa - ab * bb) / (2.0 * det)
    t = (aa * bb - ab * aa) / (2.0 * det)
    center = p1 + s * a + t * b
    radius = float(np.linalg.norm(center - p1))

    return center, radius, normal


class LineSegment:
    """A straight cartesian segment: position lerp, orientation geodesic.

    The two are interpolated apart, not as one screw motion: a screw that
    turns the tool bows its position off the line between the ends."""

    __slots__ = ("start", "end", "_length_m", "_angle_rad")

    def __init__(self, start: NDArray[np.float64], end: NDArray[np.float64]) -> None:
        self.start = start
        self.end = end
        self._length_m = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
        self._angle_rad = _rotation_angle(start, end)

    def length_mm(self) -> float:
        return self._length_m * 1000.0

    def angle_rad(self) -> float:
        return self._angle_rad

    def sample_into(
        self, out: NDArray[np.float64], s_start: float, s_end: float, skip: int
    ) -> None:
        """Poses at evenly spaced ``t`` from ``s_start`` to ``s_end``, the
        first ``skip`` of them left out, written into ``out``."""
        n_total = out.shape[0] + skip
        t_values = np.linspace(s_start, s_end, n_total)[skip:]
        # The screw's rotation is the geodesic; its translation is replaced.
        batch_se3_interp(self.start, self.end, t_values, out)
        p0 = self.start[:3, 3]
        out[:, :3, 3] = p0 + np.outer(t_values, self.end[:3, 3] - p0)

    def sample(self, t: float, out: NDArray[np.float64]) -> None:
        se3_interp(self.start, self.end, t, out)
        p0 = self.start[:3, 3]
        out[:3, 3] = p0 + t * (self.end[:3, 3] - p0)

    def tangent(self, t: float) -> NDArray[np.float64]:
        d = self.end[:3, 3] - self.start[:3, 3]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else np.zeros(3)


class ArcSegment:
    """A circular arc as a segment: position sweeps the circle through the
    via point from start to end, orientation is the geodesic between the
    two end poses."""

    __slots__ = (
        "start",
        "end",
        "_center_m",
        "_r1_m",
        "_normal",
        "_sweep",
        "_angle_rad",
    )

    def __init__(
        self,
        start: NDArray[np.float64],
        via: NDArray[np.float64],
        end: NDArray[np.float64],
    ) -> None:
        self.start = start
        self.end = end
        start_mm = start[:3, 3] * 1000.0
        end_mm = end[:3, 3] * 1000.0
        center_mm, _radius, normal = compute_circle_from_3_points(
            start_mm, via[:3, 3] * 1000.0, end_mm
        )
        self._center_m = center_mm / 1000.0
        self._normal = normal
        r1 = start[:3, 3] - self._center_m
        r2 = end[:3, 3] - self._center_m
        n1, n2 = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
        if n1 < 1e-9 or n2 < 1e-9:
            raise ValueError("the arc has no radius")
        self._r1_m = r1
        u1, u2 = r1 / n1, r2 / n2
        sweep = float(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
        if float(np.linalg.norm(end_mm - start_mm)) < FULL_CIRCLE_MM:
            sweep = 2.0 * np.pi
        elif float(np.dot(np.cross(u1, u2), normal)) < 0.0:
            sweep = 2.0 * np.pi - sweep
        self._sweep = sweep
        self._angle_rad = _rotation_angle(start, end)

    def length_mm(self) -> float:
        return float(np.linalg.norm(self._r1_m)) * self._sweep * 1000.0

    def angle_rad(self) -> float:
        return self._angle_rad

    def _position(self, t: float) -> NDArray[np.float64]:
        rotation = Rotation.from_rotvec(self._normal * (t * self._sweep))
        return self._center_m + rotation.apply(self._r1_m)

    def sample(self, t: float, out: NDArray[np.float64]) -> None:
        se3_interp(self.start, self.end, t, out)
        out[:3, 3] = self._position(t)

    def sample_into(
        self, out: NDArray[np.float64], s_start: float, s_end: float, skip: int
    ) -> None:
        n_total = out.shape[0] + skip
        t_values = np.linspace(s_start, s_end, n_total)[skip:]
        batch_se3_interp(self.start, self.end, t_values, out)
        rotations = Rotation.from_rotvec(np.outer(t_values * self._sweep, self._normal))
        out[:, :3, 3] = self._center_m + rotations.apply(self._r1_m)

    def tangent(self, t: float) -> NDArray[np.float64]:
        r = self._position(t) - self._center_m
        d = np.cross(self._normal, r)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else np.zeros(3)


def _cubic_blend_into(
    entry_pose: NDArray[np.float64],
    exit_pose: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    out: NDArray[np.float64],
    skip: int = 0,
) -> None:
    """Write a cubic Bézier blend zone into a pre-allocated buffer.

    Position follows the cubic through the control points
    (entry, p1, p2, exit); orientation is the geodesic from entry to exit.
    """
    E = entry_pose[:3, 3]
    X = exit_pose[:3, 3]

    n_total = out.shape[0] + skip
    t = np.linspace(0.0, 1.0, n_total)[skip:]

    batch_se3_interp(entry_pose, exit_pose, t, out)

    omt = 1.0 - t
    out[:, :3, 3] = (
        np.outer(omt * omt * omt, E)
        + np.outer(3.0 * omt * omt * t, p1)
        + np.outer(3.0 * omt * t * t, p2)
        + np.outer(t * t * t, X)
    )


def build_composite_cartesian_path(
    waypoints: list[NDArray[np.float64]],
    blend_radii: list[float],
    samples_per_segment: int | None = None,
) -> NDArray[np.float64]:
    """A polyline through SE3 waypoints with its interior corners rounded:
    :func:`build_blended_path` over straight segments.

    Args:
        waypoints: SE3 poses (4x4) defining the path corners, at least 2.
        blend_radii: Blend radius (mm) for each intermediate waypoint,
            ``len(waypoints) - 2`` of them; ``0`` means stop at the waypoint.
        samples_per_segment: As :func:`build_blended_path` takes it.
    """
    n = len(waypoints)
    if n < 2:
        raise ValueError("Need at least 2 waypoints")
    segments = [LineSegment(waypoints[i], waypoints[i + 1]) for i in range(n - 1)]
    return build_blended_path(segments, blend_radii, samples_per_segment)


def build_blended_path(
    segments: list[LineSegment | ArcSegment],
    blend_radii: list[float],
    samples_per_segment: int | None = None,
) -> NDArray[np.float64]:
    """Build a composite cartesian path from straight and circular segments
    whose junctions are rounded by blend zones.

    Each zone trims both adjoining segments by its radius, measured along
    the segment (arc length on an arc), and joins the two trim points with
    a cubic Bézier whose handles lie along the segments' directions of
    travel there, two thirds of the trim long: the zone is tangent to the
    incoming segment where it starts and to the outgoing one where it
    ends. Between two lines the cubic is exactly the degree-raised
    quadratic through the corner point; an arc's zone follows its
    curvature into and out of the corner. The ABB zone rule applies: a radius never eats more
    than half of either adjoining segment, and two zones sharing a segment
    are scaled down together until they fit.

    Args:
        segments: The path's segments in order, at least one.
        blend_radii: Blend radius (mm) for each junction, ``len(segments) - 1``
            of them; ``0`` means stop at the junction.
        samples_per_segment: Poses per run of a segment, a blend zone
            taking its share of them. ``None`` samples every run and zone
            by its own length at ``PATH_STEP_MM`` on the combined metric,
            ``PATH_MAX_POINTS`` poses at most, so a long straight run is
            sampled as finely as a short one.

    Returns:
        (M, 4, 4) ndarray of SE3 poses forming the complete path.
    """
    n_seg = len(segments)
    if n_seg < 1:
        raise ValueError("Need at least 1 segment")
    if len(blend_radii) != n_seg - 1:
        raise ValueError(f"Expected {n_seg - 1} blend radii, got {len(blend_radii)}")

    seg_lengths = [seg.length_mm() for seg in segments]

    # Clamp blend radii (zone overlap prevention)
    clamped = list(blend_radii)
    for i in range(len(clamped)):
        half_before = seg_lengths[i] / 2.0
        half_after = seg_lengths[i + 1] / 2.0
        clamped[i] = min(clamped[i], half_before, half_after)

    # Adjacent blends: if clamped[i] + clamped[i+1] > seg_lengths[i+1], scale both
    for i in range(len(clamped) - 1):
        seg_len = seg_lengths[i + 1]
        total = clamped[i] + clamped[i + 1]
        if total > seg_len and total > 0:
            scale = seg_len / total
            clamped[i] *= scale
            clamped[i + 1] *= scale

    # Pre-compute per-segment trim fractions
    seg_exit_frac = [0.0] * n_seg
    seg_entry_frac = [0.0] * n_seg
    for i in range(len(clamped)):
        if clamped[i] > 0:
            if seg_lengths[i] > 0:
                seg_exit_frac[i] = clamped[i] / seg_lengths[i]
            if seg_lengths[i + 1] > 0:
                seg_entry_frac[i + 1] = clamped[i] / seg_lengths[i + 1]

    # Workspace buffers for blend zone endpoints (hoisted out of loop)
    entry_buf = np.zeros((4, 4), dtype=np.float64)
    exit_buf = np.zeros((4, 4), dtype=np.float64)

    # Size every piece first, so a budget spreads over the whole path: the
    # runs of each segment between its zones, and the zones.
    pieces: list[tuple[int, bool, float, float]] = []
    counts: list[int] = []
    for seg_idx, seg in enumerate(segments):
        s_start = seg_entry_frac[seg_idx]
        s_end = 1.0 - seg_exit_frac[seg_idx]
        if s_end > s_start + 1e-9:
            pieces.append((seg_idx, False, s_start, s_end))
            span = s_end - s_start
            counts.append(
                samples_per_segment - 1
                if samples_per_segment is not None
                else _intervals(span * seg_lengths[seg_idx], span * seg.angle_rad())
            )
        if seg_idx < len(clamped) and clamped[seg_idx] > 0:
            t_in = 1.0 - seg_exit_frac[seg_idx]
            t_out = seg_entry_frac[seg_idx + 1]
            pieces.append((seg_idx, True, t_in, t_out))
            if samples_per_segment is not None:
                avg_seg_len = (seg_lengths[seg_idx] + seg_lengths[seg_idx + 1]) / 2.0
                frac = clamped[seg_idx] / avg_seg_len if avg_seg_len > 1e-6 else 0.0
                counts.append(_blend_sample_count(frac, samples_per_segment) - 1)
            else:
                # The zone's control polygon is about 2r long; its curve is
                # shorter.
                seg.sample(t_in, entry_buf)
                segments[seg_idx + 1].sample(t_out, exit_buf)
                counts.append(
                    _intervals(
                        2.0 * clamped[seg_idx], _rotation_angle(entry_buf, exit_buf)
                    )
                )
    if samples_per_segment is None:
        _fit_budget(counts)

    out = np.empty((1 + sum(counts), 4, 4), dtype=np.float64)
    row = 0
    for (seg_idx, zone, t_a, t_b), intervals in zip(pieces, counts, strict=True):
        # Pieces share their junction pose; each after the first skips it.
        skip = 1 if row > 0 else 0
        n_write = intervals + 1 - skip
        if not zone:
            segments[seg_idx].sample_into(out[row : row + n_write], t_a, t_b, skip)
        else:
            seg, nxt = segments[seg_idx], segments[seg_idx + 1]
            seg.sample(t_a, entry_buf)
            nxt.sample(t_b, exit_buf)
            handle_m = 2.0 / 3.0 * clamped[seg_idx] / 1000.0
            p1 = entry_buf[:3, 3] + handle_m * seg.tangent(t_a)
            p2 = exit_buf[:3, 3] - handle_m * nxt.tangent(t_b)
            _cubic_blend_into(
                entry_buf, exit_buf, p1, p2, out[row : row + n_write], skip=skip
            )
        row += n_write

    return out[:row]


def build_spline_path(
    waypoints: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Poses along a cubic spline through SE3 ``waypoints``, the first being
    where the path starts; every waypoint is one of the poses.

    Position is a natural cubic spline per axis over chord-length knots:
    uniform knots overshoot between unevenly spaced waypoints, and a
    natural end cannot swing wide of the first and last segments as
    not-a-knot can. Orientation slerps between the waypoints' rotations.

    Both run on one schedule, the distance along the path on the combined
    metric sqrt(t² + (w·θ)²), sampled at ``PATH_STEP_MM``: a waypoint that
    only turns the tool takes its turn there, the tool standing at it,
    rather than all at once between two poses. Position keeps its
    chord-length knots and holds still while the tool turns in place; a
    cubic over the combined metric would swing the tool wide of a
    waypoint it only turns at.

    Returns:
        (M, 4, 4) SE3 poses; one pose when every waypoint is the first.
    """
    points = np.array([w[:3, 3] for w in waypoints], dtype=np.float64) * 1000.0
    rotations = Rotation.from_matrix(np.array([w[:3, :3] for w in waypoints]))
    if len(points) > 1:
        chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
        turn = (rotations[:-1].inv() * rotations[1:]).magnitude()
        # A waypoint that repeats the one before it is nothing to pass through.
        keep = np.concatenate(
            ([True], np.hypot(chord, _PATH_ROT_WEIGHT_MM_PER_RAD * turn) > _REPEAT_MM)
        )
        points = points[keep]
        rotations = rotations[keep]
    if len(points) < 2:
        return np.asarray(waypoints[0], dtype=np.float64)[np.newaxis].copy()

    chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
    turn = (rotations[:-1].inv() * rotations[1:]).magnitude()
    along_knots = np.concatenate(([0.0], np.cumsum(chord)))
    knots = np.concatenate(
        ([0.0], np.cumsum(np.hypot(chord, _PATH_ROT_WEIGHT_MM_PER_RAD * turn)))
    )

    # The cubic runs through the distinct positions only: a waypoint that
    # turns the tool in place shares its position's knot.
    moves = np.concatenate(([True], chord > _REPEAT_MM))
    position: CubicSpline | None = None
    if int(moves.sum()) > 1:
        position = CubicSpline(
            along_knots[moves], points[moves], bc_type="natural", axis=0
        )

    counts = [_intervals(float(chord[i]), float(turn[i])) for i in range(len(chord))]
    _fit_budget(counts)
    u = np.concatenate(
        [knots[:1]]
        + [
            np.linspace(knots[i], knots[i + 1], counts[i] + 1)[1:]
            for i in range(len(counts))
        ]
    )

    out = np.zeros((len(u), 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = Slerp(knots, rotations)(u).as_matrix()
    if position is None:
        out[:, :3, 3] = points[0] / 1000.0
    else:
        out[:, :3, 3] = position(np.interp(u, knots, along_knots)) / 1000.0
    return out


def cartesian_path_knots(cart_poses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized cumulative tool distance along an SE3 pose chain, on the
    metric sqrt(translation² + (w·rotation)²): the path parameter a timing
    solver should key the poses to, so that a constant ``ds/dt`` is a
    constant tool speed. Repeated poses share a knot value; the caller
    drops them."""
    n = len(cart_poses)
    knots = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        knots[i] = knots[i - 1] + pose_distance_m(cart_poses[i - 1], cart_poses[i])
    total = float(knots[-1])
    if total > 1e-12:
        knots /= total
    return knots


def _blend_sample_count(frac: float, samples_per_segment: int) -> int:
    """Compute adaptive blend zone sample count from blend fraction.

    The blend zone replaces *frac* of each of two adjacent segments,
    so its effective arc length is ~2*frac segments.  The 2x multiplier
    keeps the sample density roughly uniform with the linear segments.
    """
    return max(5, int(2.0 * frac * samples_per_segment + 0.5))


# ---------------------------------------------------------------------------
# Joint-space composite path with blend zones
# ---------------------------------------------------------------------------


def build_composite_joint_path(
    waypoints: list[NDArray[np.float64]],
    blend_fracs: list[tuple[float, float]],
    samples_per_segment: int = 50,
) -> NDArray[np.float64]:
    """Build a composite joint-space path with Bezier blend zones.

    Mirrors :func:`build_composite_cartesian_path` but operates entirely in
    joint space.  Blend zone sizes are expressed as fractions of each adjacent
    segment (pre-computed by the caller from FK-based mm->fraction conversion).

    Args:
        waypoints: Joint-angle arrays (radians), length N >= 2.
        blend_fracs: For each of the N-2 intermediate waypoints, a
            ``(frac_before, frac_after)`` tuple giving the fraction of the
            incoming / outgoing segment consumed by the blend zone.
            Values are clamped internally to prevent overlap.
        samples_per_segment: Linear interpolation samples per segment.

    Returns:
        (M, ndof) ndarray of joint positions along the composite path.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n = len(waypoints)
    ndof = len(waypoints[0])
    if n < 2:
        raise ValueError("Need at least 2 waypoints")
    if len(blend_fracs) != max(0, n - 2):
        raise ValueError(
            f"Expected {max(0, n - 2)} blend_fracs, got {len(blend_fracs)}"
        )

    # Trivial 2-waypoint path — no blending needed
    if n == 2:
        out = np.empty((samples_per_segment, ndof), dtype=np.float64)
        _linear_joint_segment_into(
            waypoints[0],
            waypoints[1],
            out,
            0.0,
            1.0,
        )
        return out

    # Clamp fractions to [0, 0.5]
    exit_frac = [min(max(f[0], 0.0), 0.5) for f in blend_fracs]
    entry_frac = [min(max(f[1], 0.0), 0.5) for f in blend_fracs]

    # Build per-segment trim arrays
    seg_start_trim = [0.0] * (n - 1)
    seg_end_trim = [0.0] * (n - 1)
    for i in range(len(blend_fracs)):
        wp_idx = i + 1
        seg_end_trim[wp_idx - 1] = exit_frac[i]
        seg_start_trim[wp_idx] = entry_frac[i]

    # Clamp overlapping trims on any segment
    for s in range(n - 1):
        total = seg_start_trim[s] + seg_end_trim[s]
        if total > 1.0:
            scale = 1.0 / total
            seg_start_trim[s] *= scale
            seg_end_trim[s] *= scale

    # Interleaved precompute: count linear segments and blend zones in order
    total_rows = 0
    for seg_idx in range(n - 1):
        s_start = seg_start_trim[seg_idx]
        s_end = 1.0 - seg_end_trim[seg_idx]
        if s_end > s_start + 1e-9:
            rows = samples_per_segment
            if total_rows > 0 and seg_idx > 0:
                rows -= 1
            total_rows += rows
        blend_idx = seg_idx
        if blend_idx < len(blend_fracs) and (
            exit_frac[blend_idx] > 0 or entry_frac[blend_idx] > 0
        ):
            avg_frac = (exit_frac[blend_idx] + entry_frac[blend_idx]) / 2.0
            bs = _blend_sample_count(avg_frac, samples_per_segment)
            rows = bs
            if total_rows > 0:
                rows -= 1
            total_rows += rows

    out = np.empty((total_rows, ndof), dtype=np.float64)
    row = 0

    for seg_idx in range(n - 1):
        start = waypoints[seg_idx]
        end = waypoints[seg_idx + 1]
        s_start = seg_start_trim[seg_idx]
        s_end = 1.0 - seg_end_trim[seg_idx]

        if s_end > s_start + 1e-9:
            skip = 1 if (row > 0 and seg_idx > 0) else 0
            n_write = samples_per_segment - skip
            _linear_joint_segment_into(
                start,
                end,
                out[row : row + n_write],
                s_start,
                s_end,
                skip=skip,
            )
            row += n_write

        blend_idx = seg_idx
        if blend_idx < len(blend_fracs) and (
            exit_frac[blend_idx] > 0 or entry_frac[blend_idx] > 0
        ):
            entry_q = start + (1.0 - seg_end_trim[seg_idx]) * (end - start)
            corner_q = end
            next_end = waypoints[seg_idx + 2]
            exit_q = end + seg_start_trim[seg_idx + 1] * (next_end - end)

            avg_frac = (exit_frac[blend_idx] + entry_frac[blend_idx]) / 2.0
            bs = _blend_sample_count(avg_frac, samples_per_segment)
            skip = 1 if row > 0 else 0
            n_write = bs - skip
            _blend_joint_path_into(
                entry_q,
                corner_q,
                exit_q,
                out[row : row + n_write],
                skip=skip,
            )
            row += n_write

    return out[:row]


def _linear_joint_segment_into(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    out: NDArray[np.float64],
    s_start: float = 0.0,
    s_end: float = 1.0,
    skip: int = 0,
) -> None:
    """Write linearly interpolated joint positions into pre-allocated buffer.

    Args:
        start: Start joint configuration.
        end: End joint configuration.
        out: Output array, shape (n_samples, ndof). Written in-place.
        s_start: Start interpolation fraction (0-1).
        s_end: End interpolation fraction (0-1).
        skip: Number of initial samples to skip (for junction dedup).
    """
    n_total = out.shape[0] + skip
    delta = end - start
    t = np.linspace(s_start, s_end, n_total)[skip:]
    np.outer(t, delta, out)
    out += start


def _blend_joint_path_into(
    entry_q: NDArray[np.float64],
    waypoint_q: NDArray[np.float64],
    exit_q: NDArray[np.float64],
    out: NDArray[np.float64],
    skip: int = 0,
) -> None:
    """Write quadratic Bezier blend zone into pre-allocated buffer.

    Per-joint: ``q(t) = (1-t)^2 E + 2t(1-t) W + t^2 X``

    Tangent at t=0 matches incoming segment, tangent at t=1 matches outgoing
    segment, giving C1 (velocity) continuity at blend boundaries.

    Args:
        entry_q: Joint angles at blend entry.
        waypoint_q: Joint angles at the corner being rounded.
        exit_q: Joint angles at blend exit.
        out: Output array, shape (n_samples, ndof). Written in-place.
        skip: Number of initial samples to skip.
    """
    n_total = out.shape[0] + skip
    t = np.linspace(0.0, 1.0, n_total)[skip:]
    omt = 1.0 - t
    out[:] = (
        np.outer(omt * omt, entry_q)
        + np.outer(2.0 * omt * t, waypoint_q)
        + np.outer(t * t, exit_q)
    )
