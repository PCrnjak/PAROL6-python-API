"""Planned cartesian paths, sampled from the controller's status stream
while the simulator drives them: a process move rounds its corner and
holds one tool speed, a spline never reverses along unevenly spaced
waypoints, a TRF move runs along the tool axis, a relative WRF rotation
turns about the TCP, and a ``move_l`` with a blend radius rounds into the
``move_c`` after it and out into the ``move_l`` after that. At the edges:
a full circle under every profile, a tool-frame turn a hair off the wrist
singularity, a line the wrist can follow only by flipping, a waypoint that
only turns the tool, a long straight process run past the base, a leg that
turns the tool, and a spline timed shorter than any arm could run it."""

import math
import threading

import numpy as np
import pytest

pytestmark = pytest.mark.integration

#: Fraction of the planned-move linear ceiling (0.2 m/s) the moves run at.
SPEED = 0.25
CRUISE_MM_S = 0.2 * 1000.0 * SPEED


def _rotz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle of the rotation taking ``a`` to ``b``."""
    tr = float(np.trace(a.T @ b))
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _point_to_segment_mm(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _polyline_miss_mm(p: np.ndarray, pts: np.ndarray) -> float:
    """How far the polyline through ``pts`` passes from ``p``."""
    if len(pts) < 2:
        return float(np.linalg.norm(pts[0] - p))
    return min(
        _point_to_segment_mm(p, a, b) for a, b in zip(pts[:-1], pts[1:], strict=True)
    )


def _wire_rotation(pose: list[float]) -> np.ndarray:
    """The rotation a wire pose names (intrinsic XYZ, degrees)."""
    from pinokin import se3_from_rpy

    se3 = np.zeros((4, 4))
    rx, ry, rz = np.radians(pose[3:])
    se3_from_rpy(0.0, 0.0, 0.0, rx, ry, rz, se3)
    return se3[:3, :3].copy()


class _TcpSampler:
    """Records the TCP transform of every published status frame on a
    background thread; ``positions`` drops repeats."""

    def __init__(self, client):
        self._client = client
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.frames: list[np.ndarray] = []

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join(timeout=2.0)

    def _record(self, status) -> bool:
        self.frames.append(np.array(status.pose, dtype=np.float64).reshape(4, 4))
        return self._done.is_set()

    def _run(self):
        # Woken by each frame's arrival: a polling sleep can wake late enough
        # on a loaded machine to skip whole corners of a fast path.
        while not self._done.is_set():
            self._client.wait_status(self._record, timeout=1.0)

    def positions(self) -> np.ndarray:
        pts = [f[:3, 3] for f in self.frames]
        kept = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - kept[-1]) > 1e-6:
                kept.append(p)
        return np.asarray(kept)


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
    profile times it. The geometry is read off the arm; the speed off the
    plan the arm plays row by row, which a status sample rate cannot
    scatter."""
    from parol6.client.dry_run_client import DryRunRobotClient

    assert client.select_profile(profile) > 0
    pose, start = _start(client)
    s = start[:3, 3]
    corner = _offset(pose, 50.0, 0.0, 0.0)
    end = _offset(pose, 50.0, 50.0, 0.0)
    corner_xyz, end_xyz = np.array(corner[:3]), np.array(end[:3])
    radius = 0.25 * 50.0

    preview = DryRunRobotClient(initial_joints_deg=client.angles())
    assert preview.select_profile(profile) == 1
    index = preview.move_p([corner, end], speed=SPEED)
    record = preview.plan()
    block = record.blocks[index]
    assert block.error is None, block.error
    planned = (
        np.asarray(record.tcp[block.start_row : block.start_row + block.rows, :3])
        * 1000.0
    )
    speeds = np.linalg.norm(np.diff(planned, axis=0), axis=1) / record.row_dt_s

    with _TcpSampler(client) as sampler:
        assert client.move_p([corner, end], speed=SPEED, timeout=20.0) >= 0
        assert client.wait_motion(timeout=20.0)
    pts = sampler.positions()
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

    away = (np.linalg.norm(planned[1:] - s, axis=1) > 12.0) & (
        np.linalg.norm(planned[1:] - end_xyz, axis=1) > 12.0
    )
    cruise = speeds[away]
    print(
        f"{profile} cruise {cruise.min():.1f}..{cruise.max():.1f} mm/s "
        f"of {CRUISE_MM_S:.0f}"
    )
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
    # Between two status samples the tool covers a few millimetres: a
    # waypoint is passed when the sampled polyline runs within 1 mm of it.
    for wp in waypoints:
        w = np.array(wp[:3])
        assert (
            min(
                _point_to_segment_mm(w, a, b)
                for a, b in zip(pts[:-1], pts[1:], strict=True)
            )
            < 1.0
        )
    assert np.linalg.norm(pts[-1] - end_xyz) < 0.5


@pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR", "QUINTIC", "TRAPEZOID"])
def test_a_move_c_that_ends_where_it_starts_runs_the_whole_circle(
    client, server_proc, profile
):
    """An end equal to the start asks for a full circle, through the via
    point opposite the start, under whichever profile times it. The arm
    settles a hair off the pose it was sent to, so the start the arc is
    planned from is never exactly the end the script wrote; an end read
    back from the arm is exactly that start. Either way the circle is
    whole, and takes about as long as its length at the speed asked."""
    from parol6.client.dry_run_client import DryRunRobotClient

    radius = 30.0
    centre = np.array([0.0, 340.0, 210.0])

    def on_circle(angle_deg: float) -> list[float]:
        a = math.radians(angle_deg)
        return [radius * math.cos(a), 340.0, 210.0 + radius * math.sin(a), 90, 0, 90]

    assert client.select_profile(profile) > 0
    start, via = on_circle(0.0), on_circle(180.0)
    assert client.move_j(pose=start, speed=0.5, wait=True, timeout=20.0) >= 0

    for read_back in (False, True):
        end = client.pose() if read_back else start
        here = client.angles()
        assert end is not None and here is not None
        which = "read-back" if read_back else "written"

        preview = DryRunRobotClient(initial_joints_deg=here)
        assert preview.select_profile(profile) == 1
        index = preview.move_c(via=via, end=end, speed=SPEED)
        record = preview.plan()
        planned = record.blocks[index].rows * record.row_dt_s
        assert record.blocks[index].error is None, record.blocks[index].error

        with _TcpSampler(client) as sampler:
            assert (
                client.move_c(via=via, end=end, speed=SPEED, wait=True, timeout=20.0)
                >= 0
            )
            assert client.wait_motion(timeout=20.0)
        pts = sampler.positions()
        assert len(pts) > 10, f"{which} end: the arm did not run the circle"

        # Measured against the circle, not the chords between samples: a slow
        # CI loop spaces the samples out, and a chord cuts inside the arc.
        off = pts - centre
        drift = np.abs(np.hypot(off[:, 0], off[:, 2]) - radius).max()
        assert drift < 1.0, f"{which} end: the arm left the circle by {drift:.1f} mm"
        assert np.abs(off[:, 1]).max() < 1.0
        # A whole turn about the centre passes the via point opposite the start.
        angle = np.degrees(np.unwrap(np.arctan2(off[:, 2], off[:, 0])))
        turn = abs(angle[-1] - angle[0])
        assert abs(turn - 360.0) < 2.0, (
            f"{which} end: the arm turned {turn:.0f}° about the centre"
        )
        assert np.linalg.norm(pts[-1] - np.asarray(start[:3])) < 0.5
        circle_s = 2.0 * math.pi * radius / CRUISE_MM_S
        assert planned < 2.0 * circle_s, (
            f"{which} end: a {circle_s:.1f} s circle planned as {planned:.1f} s"
        )


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
    # A preview's shapes are the process's robot model's: cleared, or every
    # later preview in this process plans around the keep-out too.
    try:
        assert preview.set_shapes([keep_out]) == 1
        index = preview.move_l(
            [0.0, 0.0, 0.0, -15.0, 0.0, 0.0], frame="TRF", speed=SPEED
        )
        refusal = preview.plan().blocks[index].error
    finally:
        assert preview.set_shapes([]) == 1
    assert refusal is not None and "wrist-arc" in str(refusal), (
        "the preview ran the turn the arm refuses"
    )


def test_a_tool_frame_turn_a_hair_off_the_wrist_singularity_runs_with_a_long_tool(
    client, server_proc
):
    """With the SSG-48 fitted the TCP stands 14 cm out from the wrist. A
    hair off the singularity (J5 at 0.3°) a tool-frame turn about the
    tool's x axis leaves it through a turn of the wrist, which swings that
    long tool a fraction of a millimetre on the way: the move still plans,
    runs and turns the tool in place, and a dry run previews it."""
    from parol6.client.dry_run_client import DryRunRobotClient

    near_singular = [90.0, -90.0, 180.0, 0.0, 0.3, 180.0]
    turn = [0.0, 0.0, 0.0, 30.0, 0.0, 0.0]
    fitted = client.select_tool("SSG-48")
    assert fitted >= 0 and client.wait_command(fitted, timeout=10.0)
    try:
        assert client.teleport(near_singular) == 1
        _, start = _start(client)
        assert client.move_l(turn, frame="TRF", speed=SPEED, timeout=20.0) >= 0
        _, final = _start(client)
    finally:
        bare = client.select_tool("NONE")
        assert bare >= 0 and client.wait_command(bare, timeout=10.0)
    assert np.linalg.norm(final[:3, 3] - start[:3, 3]) < 0.5
    assert _rotation_angle_deg(final[:3, :3], start[:3, :3] @ _rotx(30.0)) < 0.5

    preview = DryRunRobotClient(initial_joints_deg=near_singular)
    # The preview fits the tool to the process's robot model: taken off
    # again, or every later preview in this process plans with it.
    try:
        assert preview.select_tool("SSG-48") >= 0
        index = preview.move_l(turn, frame="TRF", speed=SPEED)
        error = preview.plan().blocks[index].error
    finally:
        preview.select_tool("NONE")
    assert error is None, error


def test_a_line_the_wrist_follows_only_by_flipping_is_refused(client, server_proc):
    """From J5 = +20° to the pose with J5 = -20° and J4 a turn of 20° round,
    the straight line passes beside the wrist singularity: following it
    runs J4 into its stop, and the only way on is to flip the whole wrist
    halfway along. That move is refused when it is planned, the arm never
    stirs, and a dry run previews the same refusal."""
    from parol6 import MotionError
    from parol6.client.dry_run_client import DryRunRobotClient
    from parol6.utils.error_codes import ErrorCode

    assert client.teleport([90.0, -90.0, 180.0, 20.0, -20.0, 180.0]) == 1
    target = client.pose()
    assert target is not None
    start = [90.0, -90.0, 180.0, 0.0, 20.0, 180.0]
    assert client.teleport(start) == 1
    before = client.angles()
    assert before is not None

    with pytest.raises(MotionError) as refused:
        client.move_l(target, speed=SPEED, timeout=20.0)
    assert refused.value.code == ErrorCode.IK_PARTIAL_PATH, refused.value
    after = client.angles()
    assert after is not None
    assert np.allclose(after, before, atol=0.05), "the refused move moved the arm"

    preview = DryRunRobotClient(initial_joints_deg=start)
    index = preview.move_l(target, speed=SPEED)
    error = preview.plan().blocks[index].error
    assert error is not None and error.code == ErrorCode.IK_PARTIAL_PATH, error


@pytest.mark.parametrize(
    "turns",
    [
        pytest.param([(0.0, 30.0), (20.0, 30.0)], id="first"),
        pytest.param([(20.0, 0.0), (20.0, 30.0), (40.0, 30.0)], id="inside"),
    ],
)
def test_move_s_turns_the_tool_at_a_waypoint_that_only_turns_it(
    client, server_proc, turns
):
    """A waypoint at the position of the one before it, with the tool
    turned about x, is a waypoint like any other, whether it opens the
    list or sits inside it: the spline runs through it with the tool
    turned there, rather than refusing the move or turning the tool on
    the way to the next waypoint."""
    assert client.select_profile("TOPPRA") > 0
    assert client.teleport([90.0, -80.0, 190.0, 0.0, 30.0, 180.0]) == 1
    pose, start = _start(client)
    waypoints = [
        [pose[0] + dx, pose[1], pose[2], pose[3] + drx, pose[4], pose[5]]
        for dx, drx in turns
    ]
    keys = [_wire_rotation(wp) for wp in waypoints]

    with _TcpSampler(client) as sampler:
        assert client.move_s(waypoints, speed=SPEED, timeout=20.0) >= 0
        assert client.wait_motion(timeout=20.0)
    frames = sampler.frames
    assert len(frames) > 10

    # Within 3 mm and 3° of each waypoint: a status sample lands a few
    # milliseconds either side of the instant the tool passes it.
    for wp, key in zip(waypoints, keys, strict=True):
        miss = min(
            max(
                float(np.linalg.norm(f[:3, 3] - wp[:3])) / 3.0,
                _rotation_angle_deg(f[:3, :3], key) / 3.0,
            )
            for f in frames
        )
        assert miss < 1.0, f"the tool never stood at {np.round(wp, 1).tolist()}"
    worst = max(_off_geodesic_deg(f[:3, :3], [start[:3, :3], *keys]) for f in frames)
    assert worst < 0.5, f"the tool left the turn between the waypoints by {worst:.1f}°"
    lateral = max(float(np.linalg.norm(f[1:3, 3] - start[1:3, 3])) for f in frames)
    assert lateral < 0.5
    assert np.linalg.norm(frames[-1][:3, 3] - waypoints[-1][:3]) < 0.5
    assert _rotation_angle_deg(frames[-1][:3, :3], keys[-1]) < 0.5


def test_move_p_runs_a_long_straight_run_past_the_base(client, server_proc):
    """A process move along a 300 mm line that passes 40 mm from the J1
    axis swings J1 through 150° as the line goes by; the move runs, and
    stays on the line, as a move_l along the same line does."""
    assert client.teleport([-74.98, -97.76, 189.08, -1.51, 73.55, 105.45]) == 1
    pose, start = _start(client)
    s = start[:3, 3]
    end = _offset(pose, 0.0, 300.0, 0.0)
    end_xyz = np.array(end[:3])

    with _TcpSampler(client) as sampler:
        assert client.move_p([pose, end], speed=0.5, timeout=30.0) >= 0
        assert client.wait_motion(timeout=30.0)
    pts = sampler.positions()
    assert len(pts) > 10

    off = max(_point_to_segment_mm(p, s, end_xyz) for p in pts)
    assert off < 0.5, f"the tool left the line by {off:.1f} mm"
    assert np.linalg.norm(pts[-1] - end_xyz) < 0.5


@pytest.mark.parametrize("chain", ["move_p", "move_l"])
def test_a_leg_that_turns_the_tool_still_runs_straight(client, server_proc, chain):
    """A straight leg is a straight line whatever the tool does along it:
    a leg that turns the tool a quarter turn about its own axis runs the
    line between its ends, as a process move and as a blended move_l, and
    rounds the corner into the next leg no more sharply than the same legs
    do with the tool held still."""
    from parol6.client.dry_run_client import DryRunRobotClient

    s = [-50.0, 250.0, 200.0, 90.0, 0.0, 90.0]
    corner = [50.0, 250.0, 200.0, 90.0, 0.0, 180.0]
    end = [50.0, 250.0, 300.0, 90.0, 0.0, 180.0]
    # A process move rounds its corner by a quarter of the shorter leg.
    zone = 0.25 * 100.0 if chain == "move_p" else 5.0

    def run(robot, corner: list[float], end: list[float], **wait) -> int:
        if chain == "move_p":
            return robot.move_p([corner, end], speed=SPEED, **wait)
        assert robot.move_l(corner, speed=SPEED, r=zone, wait=False) >= 0
        return robot.move_l(end, speed=SPEED, **wait)

    assert client.select_profile("TOPPRA") > 0
    assert client.move_j(pose=s, speed=0.5, timeout=20.0) >= 0
    q0 = client.angles()
    assert q0 is not None

    with _TcpSampler(client) as sampler:
        assert run(client, corner, end, timeout=30.0) >= 0
        assert client.wait_motion(timeout=30.0)
    pts = sampler.positions()
    assert len(pts) > 20

    a, c, e = (np.array(p[:3]) for p in (s, corner, end))
    off = max(
        min(_point_to_segment_mm(p, a, c), _point_to_segment_mm(p, c, e))
        for p in pts
        if np.linalg.norm(p - c) > zone + 0.5
    )
    assert off < 0.5, f"{chain}: the turning leg bowed {off:.1f} mm off its line"
    assert np.linalg.norm(pts[-1] - e) < 0.5

    def sharpest_turn(turned: bool) -> float:
        """The largest change of direction between two planned rows."""
        preview = DryRunRobotClient(initial_joints_deg=q0)
        assert preview.select_profile("TOPPRA") == 1
        if turned:
            run(preview, corner, end)
        else:
            run(preview, [*corner[:3], *s[3:]], [*end[:3], *s[3:]])
        record = preview.plan()
        assert all(block.error is None for block in record.blocks)
        rows = np.asarray(record.tcp[:, :3], dtype=np.float64) * 1000.0
        return _max_turn_deg(rows, 0.05)

    turned, held = sharpest_turn(True), sharpest_turn(False)
    assert turned < held + 5.0, (
        f"{chain}: the path kinks {turned:.0f}° between two rows turning the "
        f"tool, {held:.0f}° holding it"
    )


def test_move_s_timed_too_short_still_runs_through_its_waypoints(client, server_proc):
    """A duration no arm could keep is stretched to one it can, never met by
    cutting the path: a spline up, across and back down, timed to 20 ms,
    still passes every waypoint."""
    assert client.select_profile("TOPPRA") > 0
    tool = [180.0, -80.0, 180.0]
    waypoints = [
        [250.0, 0.0, 150.0, *tool],
        [250.0, 0.0, 230.0, *tool],
        [290.0, 0.0, 230.0, *tool],
        [290.0, 0.0, 150.0, *tool],
    ]
    assert client.move_j(pose=waypoints[0], speed=0.5, timeout=20.0) >= 0

    with _TcpSampler(client) as sampler:
        assert client.move_s(waypoints, duration=0.02, timeout=20.0) >= 0
        assert client.wait_motion(timeout=20.0)
    pts = sampler.positions()

    for wp in waypoints:
        miss = _polyline_miss_mm(np.array(wp[:3]), pts)
        assert miss < 2.0, f"the spline passed {miss:.0f} mm from {wp[:3]}"
