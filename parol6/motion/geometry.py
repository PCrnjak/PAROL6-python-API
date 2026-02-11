"""
Geometry generation for smooth motion paths.

This module provides pure geometry generators for arcs and splines,
plus blend zone computation for fly-by motion between consecutive segments.

All generators are stateless - they produce Cartesian path geometry without
depending on controller state or executing any motion.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pinokin import batch_se3_interp, se3_interp
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from pinokin import Robot

from parol6.config import CONTROL_RATE_HZ, PATH_SAMPLES

logger = logging.getLogger(__name__)

# Default control rate for geometry sampling
DEFAULT_CONTROL_RATE = CONTROL_RATE_HZ


class _ShapeGenerator:
    """Base class for geometry generation."""

    def __init__(self, control_rate: float | None = None):
        self.control_rate = (
            control_rate if control_rate is not None else DEFAULT_CONTROL_RATE
        )


class CircularMotion(_ShapeGenerator):
    """Generate arc trajectories in 3D space.

    Returns (N, 6) arrays of [x, y, z, rx, ry, rz] poses.
    Position units match input units (typically mm).
    Orientation is in degrees.
    """

    def generate_arc(
        self,
        start_pose: NDArray,
        end_pose: NDArray,
        center: NDArray,
        normal: NDArray | None = None,
        clockwise: bool = False,
        n_samples: int = PATH_SAMPLES,
    ) -> np.ndarray:
        """Generate a 3D circular arc trajectory (uniformly sampled geometry).

        Args:
            start_pose: Start pose [x, y, z, rx, ry, rz]
            end_pose: End pose [x, y, z, rx, ry, rz]
            center: Arc center point [x, y, z]
            normal: Normal vector defining arc plane (auto-computed if None)
            clockwise: If True, arc goes clockwise when viewed from normal
            n_samples: Number of sample points along the arc

        Returns:
            (N, 6) array of poses along the arc
        """
        start_pos = start_pose[:3]
        end_pos = end_pose[:3]

        r1 = start_pos - center
        r2 = end_pos - center

        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:
                normal = np.array([0, 0, 1])
        normal_unit = normal / np.linalg.norm(normal)

        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)

        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, normal_unit) < 0:
            arc_angle = 2 * np.pi - arc_angle
        if clockwise:
            arc_angle = -arc_angle

        num_points = max(2, n_samples)

        # Vectorized arc generation using scipy Rotation
        t_values = np.linspace(0, 1, num_points) if num_points > 1 else np.array([1.0])
        angles = t_values * arc_angle

        # Batch rotation using rotvec (axis-angle)
        rotvecs = np.outer(angles, normal_unit)  # (num_points, 3)
        rotations = Rotation.from_rotvec(rotvecs)
        positions = center + rotations.apply(r1)  # (num_points, 3)

        # Batch orientation interpolation (slerp)
        r_start = Rotation.from_euler("xyz", start_pose[3:], degrees=True)
        r_end = Rotation.from_euler("xyz", end_pose[3:], degrees=True)
        key_rots = Rotation.from_quat(np.stack([r_start.as_quat(), r_end.as_quat()]))
        slerp = Slerp(np.array([0.0, 1.0]), key_rots)
        orientations = slerp(t_values).as_euler("xyz", degrees=True)  # (num_points, 3)

        # Combine positions and orientations
        trajectory = np.concatenate([positions, orientations], axis=1)

        return trajectory


class SplineMotion(_ShapeGenerator):
    """Generate smooth spline trajectories through waypoints.

    Uses cubic spline interpolation for position and SLERP for orientation.
    """

    def generate_spline(
        self,
        waypoints: NDArray,
        timestamps: NDArray | None = None,
        duration: float | None = None,
        velocity_start: NDArray | None = None,
        velocity_end: NDArray | None = None,
    ) -> np.ndarray:
        """Generate spline trajectory (uniformly sampled geometry).

        Args:
            waypoints: (N, 6) array of [x, y, z, rx, ry, rz] waypoints
            timestamps: Optional timestamps for each waypoint
            duration: Total duration (overrides timestamps scaling)
            velocity_start: Start velocity for position [vx, vy, vz]
            velocity_end: End velocity for position [vx, vy, vz]

        Returns:
            (N, 6) array of poses along the spline
        """
        waypoints_arr = np.asarray(waypoints, dtype=float)
        num_waypoints = len(waypoints_arr)

        if num_waypoints < 2:
            return waypoints_arr

        if timestamps is None:
            total_dist = 0.0
            for i in range(1, num_waypoints):
                dist = np.linalg.norm(waypoints_arr[i, :3] - waypoints_arr[i - 1, :3])
                total_dist += float(dist)

            if duration is not None:
                total_time = duration
            else:
                total_time = max(0.1, total_dist / 50.0)

            timestamps_arr = np.linspace(0, total_time, num_waypoints)
        else:
            timestamps_arr = np.asarray(timestamps, dtype=float)
            if duration is not None:
                scale = duration / timestamps_arr[-1] if timestamps_arr[-1] > 0 else 1.0
                timestamps_arr = timestamps_arr * scale

        if len(timestamps_arr) != len(waypoints_arr):
            raise ValueError(
                f"Timestamps length ({len(timestamps_arr)}) must match "
                f"waypoints length ({len(waypoints_arr)})"
            )

        pos_splines = []
        for i in range(3):
            bc: Any
            if velocity_start is not None and velocity_end is not None:
                bc = ((1, float(velocity_start[i])), (1, float(velocity_end[i])))
            else:
                bc = "not-a-knot"
            spline = CubicSpline(timestamps_arr, waypoints_arr[:, i], bc_type=bc)
            pos_splines.append(spline)

        # Batch convert euler angles to rotations (vectorized)
        euler_angles = waypoints_arr[:, 3:]
        key_rots = Rotation.from_euler("xyz", euler_angles, degrees=True)
        slerp = Slerp(timestamps_arr, key_rots)

        total_time = float(timestamps_arr[-1])
        num_points = max(2, int(total_time * self.control_rate))
        t_eval = np.linspace(0, total_time, num_points)

        trajectory = np.empty((num_points, 6), dtype=np.float64)
        for i, spline in enumerate(pos_splines):
            trajectory[:, i] = spline(t_eval)
        trajectory[:, 3:] = slerp(t_eval).as_euler("xyz", degrees=True)

        return trajectory


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

    for i, T in enumerate(transforms):
        tcp_poses[i, :3] = T[:3, 3] * 1000.0  # m -> mm
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        tcp_poses[i, 3:] = rpy

    return tcp_poses


def compute_circle_from_3_points(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Compute the circumscribed circle through 3 non-collinear 3D points.

    Args:
        p1, p2, p3: 3D points (shape (3,))

    Returns:
        (center, radius, normal):
            center: Circle center point (3,)
            radius: Circle radius
            normal: Unit normal of the plane containing the circle (3,)

    Raises:
        ValueError: If the 3 points are collinear (no unique circle).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    # Vectors from p1 to p2 and p3
    a = p2 - p1
    b = p3 - p1

    # Normal to the plane
    normal = np.asarray(np.cross(a, b), dtype=np.float64)
    normal_len = float(np.linalg.norm(normal))
    if normal_len < 1e-12:
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
    if abs(det) < 1e-20:
        raise ValueError("Degenerate configuration; cannot compute circle center.")

    s = (bb * aa - ab * bb) / (2.0 * det)
    t = (aa * bb - ab * aa) / (2.0 * det)
    center = p1 + s * a + t * b
    radius = float(np.linalg.norm(center - p1))

    return center, radius, normal


def blend_path_into(
    entry_pose: NDArray[np.float64],
    waypoint_pose: NDArray[np.float64],
    exit_pose: NDArray[np.float64],
    out: NDArray[np.float64],
    skip: int = 0,
) -> None:
    """Write quadratic Bezier blend zone into pre-allocated buffer.

    The blend zone smoothly rounds a corner between two linear Cartesian
    segments. It is tangent to the incoming segment at t=0 and to the outgoing
    segment at t=1.

    Position follows a quadratic Bezier curve::

        P(t) = (1-t)^2*E + 2t(1-t)*W + t^2*X

    Orientation is geodesic (SLERP) from entry to exit.

    Args:
        entry_pose: SE3 pose at blend zone entry (4x4)
        waypoint_pose: SE3 pose at the corner being rounded (4x4)
        exit_pose: SE3 pose at blend zone exit (4x4)
        out: Output array, shape (n_samples, 4, 4). Written in-place.
        skip: Number of initial samples to skip (for junction dedup).
    """
    E = entry_pose[:3, 3]
    W = waypoint_pose[:3, 3]
    X = exit_pose[:3, 3]

    n_total = out.shape[0] + skip
    t = np.linspace(0.0, 1.0, n_total)[skip:]

    # Batch SLERP for orientation (entry -> exit)
    batch_se3_interp(entry_pose, exit_pose, t, out)

    # Override translation with quadratic Bezier position
    omt = 1.0 - t
    out[:, :3, 3] = (
        np.outer(omt * omt, E) + np.outer(2.0 * omt * t, W) + np.outer(t * t, X)
    )


def build_composite_cartesian_path(
    waypoints: list[NDArray[np.float64]],
    blend_radii: list[float],
    samples_per_segment: int = PATH_SAMPLES,
) -> NDArray[np.float64]:
    """Build a composite Cartesian path with blend zones at intermediate waypoints.

    Concatenates linear Cartesian segments connected by quadratic Bezier blend
    zones. Implements ABB-style zone overlap clamping: if two adjacent blend
    zones would overlap, both radii are proportionally reduced so they don't
    exceed half the segment length.

    Args:
        waypoints: List of SE3 poses (4x4) defining the path corners.
            Must have at least 2 waypoints.
        blend_radii: Blend radius (mm) for each intermediate waypoint.
            Length must equal ``len(waypoints) - 2`` (no blend at start/end).
            ``r=0`` means stop at the waypoint (no blending).
        samples_per_segment: Number of linear interpolation samples per segment

    Returns:
        (M, 4, 4) ndarray of SE3 poses forming the complete path.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n = len(waypoints)
    if n < 2:
        raise ValueError("Need at least 2 waypoints")
    if len(blend_radii) != max(0, n - 2):
        raise ValueError(
            f"Expected {max(0, n - 2)} blend radii, got {len(blend_radii)}"
        )

    # No blending for 2-waypoint path
    if n == 2:
        out = np.empty((samples_per_segment, 4, 4), dtype=np.float64)
        _linear_se3_segment_into(waypoints[0], waypoints[1], out)
        return out

    # Compute segment lengths (mm)
    seg_lengths: list[float] = [0.0] * (n - 1)
    for i in range(n - 1):
        seg_lengths[i] = (
            float(np.linalg.norm(waypoints[i + 1][:3, 3] - waypoints[i][:3, 3]))
            * 1000.0
        )

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
    seg_exit_frac = [0.0] * (n - 1)
    seg_entry_frac = [0.0] * (n - 1)
    for i in range(len(clamped)):
        if clamped[i] > 0:
            if seg_lengths[i] > 0:
                seg_exit_frac[i] = clamped[i] / seg_lengths[i]
            if seg_lengths[i + 1] > 0:
                seg_entry_frac[i + 1] = clamped[i] / seg_lengths[i + 1]

    # Interleaved precompute: count linear segments and blend zones in order
    total_rows = 0
    for seg_idx in range(n - 1):
        s_start = seg_entry_frac[seg_idx]
        s_end = 1.0 - seg_exit_frac[seg_idx]
        if s_end > s_start + 1e-9:
            rows = samples_per_segment
            if total_rows > 0 and seg_idx > 0:
                rows -= 1
            total_rows += rows
        if seg_idx < len(clamped) and clamped[seg_idx] > 0:
            avg_seg_len = (seg_lengths[seg_idx] + seg_lengths[seg_idx + 1]) / 2.0
            frac = clamped[seg_idx] / avg_seg_len if avg_seg_len > 1e-6 else 0.0
            bs = _blend_sample_count(frac, samples_per_segment)
            rows = bs
            if total_rows > 0:
                rows -= 1
            total_rows += rows

    out = np.empty((total_rows, 4, 4), dtype=np.float64)
    row = 0

    # Workspace buffers for blend zone endpoints (hoisted out of loop)
    entry_buf = np.zeros((4, 4), dtype=np.float64)
    exit_buf = np.zeros((4, 4), dtype=np.float64)

    for seg_idx in range(n - 1):
        start = waypoints[seg_idx]
        end = waypoints[seg_idx + 1]

        s_start = seg_entry_frac[seg_idx]
        s_end = 1.0 - seg_exit_frac[seg_idx]

        # Linear segment
        if s_end > s_start + 1e-9:
            skip = 1 if (row > 0 and seg_idx > 0) else 0
            n_write = samples_per_segment - skip
            _linear_se3_segment_into(
                start,
                end,
                out[row : row + n_write],
                s_start,
                s_end,
                skip=skip,
            )
            row += n_write

        # Blend zone at end of this segment
        if seg_idx < len(clamped) and clamped[seg_idx] > 0:
            se3_interp(start, end, 1.0 - seg_exit_frac[seg_idx], entry_buf)
            corner = end
            next_end = waypoints[seg_idx + 2]
            se3_interp(end, next_end, seg_entry_frac[seg_idx + 1], exit_buf)

            avg_seg_len = (seg_lengths[seg_idx] + seg_lengths[seg_idx + 1]) / 2.0
            frac = clamped[seg_idx] / avg_seg_len if avg_seg_len > 1e-6 else 0.0
            bs = _blend_sample_count(frac, samples_per_segment)
            skip = 1 if row > 0 else 0
            n_write = bs - skip
            blend_path_into(
                entry_buf,
                corner,
                exit_buf,
                out[row : row + n_write],
                skip=skip,
            )
            row += n_write

    return out[:row]


def _linear_se3_segment_into(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    out: NDArray[np.float64],
    s_start: float = 0.0,
    s_end: float = 1.0,
    skip: int = 0,
) -> None:
    """Write linearly interpolated SE3 poses into pre-allocated buffer.

    Args:
        start: Start SE3 pose (4x4)
        end: End SE3 pose (4x4)
        out: Output array, shape (n_samples, 4, 4). Written in-place.
        s_start: Start interpolation fraction (0-1)
        s_end: End interpolation fraction (0-1)
        skip: Number of initial samples to skip (for junction dedup).
    """
    n_total = out.shape[0] + skip
    s_values = np.linspace(s_start, s_end, n_total)[skip:]
    batch_se3_interp(start, end, s_values, out)


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
