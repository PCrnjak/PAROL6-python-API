"""Planned cartesian paths, sampled from the controller's status stream
while the simulator drives them: a process move rounds its corner and
holds one tool speed, a spline never reverses along unevenly spaced
waypoints, a TRF move runs along the tool axis, a relative WRF rotation
turns about the TCP, and a ``move_l`` with a blend radius rounds into the
``move_c`` after it and out into the ``move_l`` after that."""

import math
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.integration

#: Fraction of the planned-move linear ceiling (0.2 m/s) the moves run at.
SPEED = 0.25
CRUISE_MM_S = 0.2 * 1000.0 * SPEED


def _rotz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle of the rotation taking ``a`` to ``b``."""
    tr = float(np.trace(a.T @ b))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _point_to_segment_mm(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class _TcpSampler:
    """Samples the TCP transform and speed from ``status()``/``tcp_speed()``
    on a background thread; ``positions`` drops the repeats the status
    cache serves between its updates."""

    def __init__(self, client):
        self._client = client
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.frames: list[np.ndarray] = []
        self.speeds: list[float] = []

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._done.is_set():
            status = self._client.status()
            speed = self._client.tcp_speed()
            if status is not None:
                self.frames.append(
                    np.asarray(status.pose, dtype=np.float64).reshape(4, 4)
                )
                self.speeds.append(float(speed) if speed is not None else 0.0)
            time.sleep(0.02)

    def positions(self) -> np.ndarray:
        pts = [f[:3, 3] for f in self.frames]
        kept = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - kept[-1]) > 1e-6:
                kept.append(p)
        return np.asarray(kept)

    def positions_and_speeds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = [f[:3, 3] for f in self.frames]
        kept_p, kept_v = [pts[0]], [self.speeds[0]]
        for p, v in zip(pts[1:], self.speeds[1:], strict=True):
            if np.linalg.norm(p - kept_p[-1]) > 1e-6:
                kept_p.append(p)
                kept_v.append(v)
        return np.asarray(kept_p), np.asarray(kept_v)


def _start(client) -> tuple[list[float], np.ndarray]:
    """The current wire pose and its transform (mm)."""
    pose = client.pose()
    status = client.status()
    assert pose is not None and status is not None
    return list(pose), np.asarray(status.pose, dtype=np.float64).reshape(4, 4)


def _offset(pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    return [pose[0] + dx, pose[1] + dy, pose[2] + dz, pose[3], pose[4], pose[5]]


def _max_turn_deg(pts: np.ndarray, min_step_mm: float) -> float:
    steps = [d for d in np.diff(pts, axis=0) if np.linalg.norm(d) > min_step_mm]
    worst = 0.0
    for a, b in zip(steps[:-1], steps[1:], strict=True):
        cosang = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    return worst


@pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR", "TRAPEZOID"])
def test_move_p_rounds_its_corner_and_holds_one_tool_speed(
    client, server_proc, profile
):
    """An L-shaped process move cuts its corner by a quarter of the shorter
    leg, never stops in it, and cruises at one tool speed, under whichever
    profile times it."""
    assert client.select_profile(profile) > 0
    pose, start = _start(client)
    s = start[:3, 3]
    corner = _offset(pose, 50.0, 0.0, 0.0)
    end = _offset(pose, 50.0, 50.0, 0.0)
    corner_xyz, end_xyz = np.array(corner[:3]), np.array(end[:3])
    radius = 0.25 * 50.0

    with _TcpSampler(client) as sampler:
        assert client.move_p([corner, end], speed=SPEED, timeout=20.0) >= 0
        assert client.wait_motion(timeout=20.0)
    pts, speeds = sampler.positions_and_speeds()
    assert len(pts) > 10

    assert np.linalg.norm(pts[-1] - end_xyz) < 0.5
    miss = float(np.min(np.linalg.norm(pts - corner_xyz, axis=1)))
    print(f"\nmove_p corner miss {miss:.2f} mm (radius {radius:.1f})")
    assert 1.0 < miss <= radius + 0.5, "the corner is rounded, within its radius"

    on_legs = [
        min(
            _point_to_segment_mm(p, s, corner_xyz),
            _point_to_segment_mm(p, corner_xyz, end_xyz),
        )
        for p in pts
        if np.linalg.norm(p - corner_xyz) > radius + 0.5
    ]
    assert max(on_legs) < 0.5, "outside the corner zone the path is the polyline"

    away = (np.linalg.norm(pts - s, axis=1) > 12.0) & (
        np.linalg.norm(pts - end_xyz, axis=1) > 12.0
    )
    cruise = speeds[away]
    print(
        f"{profile} cruise {cruise.min():.1f}..{cruise.max():.1f} mm/s "
        f"of {CRUISE_MM_S:.0f}"
    )
    # Percentiles, not extremes: a control-loop stall on a loaded runner
    # shows as a sample or two of lower measured speed, where a path that
    # varies its speed does so over a stretch of it.
    slow, fast = np.percentile(cruise, [10, 90])
    assert slow > 0.8 * fast, "one tool speed through the corner"
    assert cruise.max() < 1.1 * CRUISE_MM_S


def test_move_s_never_reverses_along_unevenly_spaced_waypoints(client, server_proc):
    """A spline through collinear, unevenly spaced waypoints runs the line
    monotonically: no dip behind the start, no overshoot past the end."""
    assert client.select_profile("TOPPRA") > 0
    pose, start = _start(client)
    s = start[:3, 3]
    waypoints = [_offset(pose, 0.0, d, 0.0) for d in (10.0, 45.0, 60.0)]
    end_xyz = np.array(waypoints[-1][:3])
    axis = (end_xyz - s) / np.linalg.norm(end_xyz - s)

    with _TcpSampler(client) as sampler:
        assert client.move_s(waypoints, speed=SPEED, timeout=20.0) >= 0
        assert client.wait_motion(timeout=20.0)
    pts = sampler.positions()
    assert len(pts) > 10

    along = (pts - s) @ axis
    lateral = np.linalg.norm((pts - s) - np.outer(along, axis), axis=1)
    print(
        f"\nmove_s along {along.min():.2f}..{along.max():.2f} mm, lateral {lateral.max():.2f} mm"
    )
    assert along.min() > -0.3, "the spline never dips behind its start"
    assert along.max() < 60.0 + 0.3, "the spline never overshoots its end"
    assert np.all(np.diff(along) > -0.3), "the tool never turns back along the line"
    assert lateral.max() < 0.5
    for wp in waypoints:
        assert np.min(np.linalg.norm(pts - np.array(wp[:3]), axis=1)) < 1.0
    assert np.linalg.norm(pts[-1] - end_xyz) < 0.5


def test_a_trf_move_l_runs_along_the_tool_axis(client, server_proc):
    """A TRF pose is an offset in the tool frame at the start of the move."""
    pose, start = _start(client)
    s = start[:3, 3]
    expected = s + start[:3, :3] @ np.array([0.0, 0.0, -20.0])

    with _TcpSampler(client) as sampler:
        assert (
            client.move_l(
                [0.0, 0.0, -20.0, 0.0, 0.0, 0.0], frame="TRF", speed=SPEED, timeout=20.0
            )
            >= 0
        )
        assert client.wait_motion(timeout=20.0)
    pts = sampler.positions()
    assert len(pts) > 3

    _, final = _start(client)
    print(f"\nTRF landed {final[:3, 3]} for {expected}")
    assert np.linalg.norm(final[:3, 3] - expected) < 0.5
    assert _rotation_angle_deg(final[:3, :3], start[:3, :3]) < 0.5
    assert max(_point_to_segment_mm(p, s, expected) for p in pts) < 0.3


def test_a_relative_wrf_rotation_turns_about_the_tcp(client, server_proc):
    """``rel`` in WRF applies the rotation about the TCP: the tool turns
    in place, and its position never leaves where it stood."""
    pose, start = _start(client)
    s = start[:3, 3]
    expected_r = _rotz(15.0) @ start[:3, :3]

    with _TcpSampler(client) as sampler:
        assert (
            client.move_l(
                [0.0, 0.0, 0.0, 0.0, 0.0, 15.0],
                frame="WRF",
                rel=True,
                speed=SPEED,
                timeout=20.0,
            )
            >= 0
        )
        assert client.wait_motion(timeout=20.0)

    _, final = _start(client)
    drift = max(float(np.linalg.norm(f[:3, 3] - s)) for f in sampler.frames)
    print(
        f"\nWRF rel rotation: position drift {drift:.3f} mm, orientation error {_rotation_angle_deg(final[:3, :3], expected_r):.3f} deg"
    )
    assert drift < 0.5, "a pure rotation keeps the TCP where it is"
    assert _rotation_angle_deg(final[:3, :3], expected_r) < 0.5
    assert _rotation_angle_deg(final[:3, :3], start[:3, :3]) > 14.0


def test_a_move_l_with_a_radius_rounds_into_the_move_c_after_it(client, server_proc):
    """``move_l(r)`` → ``move_c(r)`` → ``move_l``: one continuous path whose
    corners are cut within ``r`` of their junctions, and whose arc lies on
    its circle outside the blend zones."""
    assert client.select_profile("TOPPRA") > 0
    pose, start = _start(client)
    s = start[:3, 3]
    r = 12.0
    radius = 30.0
    a = _offset(pose, 40.0, 0.0, 0.0)
    a_xyz = np.array(a[:3])
    centre = a_xyz + np.array([radius, 0.0, 0.0])
    via = _offset(
        pose, 40.0 + radius * (1.0 - math.sqrt(0.5)), radius * math.sqrt(0.5), 0.0
    )
    b = _offset(pose, 40.0 + radius, radius, 0.0)
    b_xyz = np.array(b[:3])
    end = _offset(pose, 40.0 + radius, radius + 30.0, 0.0)
    end_xyz = np.array(end[:3])

    with _TcpSampler(client) as sampler:
        assert client.move_l(a, speed=SPEED, r=r, wait=False) >= 0
        assert client.move_c(via, b, speed=SPEED, r=r, wait=False) >= 0
        assert client.move_l(end, speed=SPEED, timeout=30.0) >= 0
        assert client.wait_motion(timeout=30.0)
    pts = sampler.positions()
    assert len(pts) > 20
    assert np.linalg.norm(pts[-1] - end_xyz) < 0.5

    for junction in (a_xyz, b_xyz):
        miss = float(np.min(np.linalg.norm(pts - junction, axis=1)))
        print(f"\ncorner miss {miss:.2f} mm (r {r})")
        assert 1.0 < miss <= r + 0.5

    # The arc's quadrant, with a degree kept clear of the lines before
    # and after it (they lie exactly on its bounding rays, and sampling
    # noise puts them on either side).
    rel = pts - centre
    angle = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    on_arc = (
        (np.linalg.norm(pts - a_xyz, axis=1) > r + 1.0)
        & (np.linalg.norm(pts - b_xyz, axis=1) > r + 1.0)
        & (angle > 91.0)
        & (angle < 179.0)
    )
    assert on_arc.sum() >= 3
    radial = np.abs(np.linalg.norm(rel[on_arc][:, :2], axis=1) - radius)
    print(f"arc radial error {radial.max():.2f} mm over {on_arc.sum()} samples")
    assert radial.max() < 0.5

    body = (np.linalg.norm(pts - s, axis=1) > 8.0) & (
        np.linalg.norm(pts - end_xyz, axis=1) > 8.0
    )
    steps = np.linalg.norm(np.diff(pts[body], axis=0), axis=1)
    assert steps.min() > 0.3, "the chain never comes to rest between its moves"
    assert _max_turn_deg(pts, 0.5) < 30.0, "the path turns gradually, never at a corner"


def _off_geodesic_deg(r: np.ndarray, keys: list[np.ndarray]) -> float:
    """How far ``r`` lies from the piecewise geodesic through ``keys``."""
    from scipy.spatial.transform import Rotation, Slerp

    t = np.linspace(0.0, 1.0, 401)
    worst = math.inf
    for a, b in zip(keys[:-1], keys[1:], strict=True):
        arc = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a, b])))(t)
        rel = arc.inv() * Rotation.from_matrix(r)
        worst = min(worst, float(np.degrees(rel.magnitude()).min()))
    return worst


def test_move_s_turns_the_tool_along_the_geodesic_between_waypoints(
    client, server_proc
):
    """A spline's orientation turns from each waypoint's rotation to the
    next along the shortest arc between them, the rotations read as the
    wire names them (intrinsic XYZ)."""
    from pinokin import se3_from_rpy

    assert client.select_profile("TOPPRA") > 0
    # Clear of the wrist singularity at standby, where any reorientation
    # needs a wrist turn first.
    assert client.teleport([90.0, -80.0, 190.0, 0.0, 30.0, 180.0]) == 1
    pose, start = _start(client)
    waypoints = [
        [pose[0] + 20.0, pose[1], pose[2], pose[3] + 25.0, pose[4] + 20.0, pose[5]],
        [
            pose[0] + 40.0,
            pose[1] + 15.0,
            pose[2],
            pose[3] + 10.0,
            pose[4] + 35.0,
            pose[5] + 30.0,
        ],
    ]
    keys = [start[:3, :3]]
    for wp in waypoints:
        se3 = np.zeros((4, 4))
        rx, ry, rz = np.radians(wp[3:])
        se3_from_rpy(0.0, 0.0, 0.0, rx, ry, rz, se3)
        keys.append(se3[:3, :3].copy())

    with _TcpSampler(client) as sampler:
        assert client.move_s(waypoints, speed=SPEED, timeout=20.0) >= 0
        assert client.wait_motion(timeout=20.0)
    assert len(sampler.frames) > 10
    worst = max(_off_geodesic_deg(f[:3, :3], keys) for f in sampler.frames)
    print(f"\nmove_s orientation off the geodesic by up to {worst:.3f} deg")
    assert worst < 0.5
    assert _rotation_angle_deg(sampler.frames[-1][:3, :3], keys[-1]) < 0.5


def test_a_wrist_turn_is_collision_checked_all_the_way_round(client, server_proc):
    """From standby a tool-frame reorientation first turns J4 a quarter turn
    out of the wrist singularity. A keep-out the wrist sweeps through only
    partway round that turn — clear of where it starts, where it ends and
    of the path after it — refuses the move when it is planned: the arm
    never stirs, and a dry run previews the same refusal."""
    from waldoctl import Sphere

    from parol6 import MotionError
    from parol6.client.dry_run_client import DryRunRobotClient

    before = client.angles()
    assert before is not None
    keep_out = Sphere(
        name="wrist-arc", radius=0.01, pose=(-0.0581, 0.2168, 0.2869, 0.0, 0.0, 0.0)
    )
    try:
        assert client.set_shapes([keep_out]) == 1
        with pytest.raises(MotionError, match="wrist-arc"):
            client.move_l(
                [0.0, 0.0, 0.0, -15.0, 0.0, 0.0], frame="TRF", speed=SPEED, timeout=20.0
            )
    finally:
        assert client.set_shapes([]) == 1
    after = client.angles()
    assert after is not None
    assert np.allclose(after, before, atol=0.05), "the refused move moved the arm"

    preview = DryRunRobotClient(initial_joints_deg=before)
    assert preview.set_shapes([keep_out]) == 1
    index = preview.move_l([0.0, 0.0, 0.0, -15.0, 0.0, 0.0], frame="TRF", speed=SPEED)
    refusal = preview.plan().blocks[index].error
    assert refusal is not None and "wrist-arc" in str(refusal), (
        "the preview ran the turn the arm refuses"
    )
