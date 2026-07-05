"""Unit tests for PAROL6's pinokin collision integration.

Covers:
- Singleton checker initialization in PAROL6_ROBOT
- SRDF disabled-pair count
- Public Robot.in_collision / colliding_pairs / min_distance / check_trajectory
- guard_joint_path raising TrajectoryPlanningError on a colliding sample
"""

from __future__ import annotations

import numpy as np
import pytest

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
import parol6.config  # noqa: F401 - imports trigger collision-checker init
from parol6 import Robot
from parol6.commands._collision_guard import guard_joint_path
from parol6.commands.base import ExecutionStatusCode
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import TrajectoryPlanningError


def test_singleton_checker_initialized():
    assert PAROL6_ROBOT.collision is not None
    assert PAROL6_ROBOT.collision.num_geometry_objects > 0
    assert PAROL6_ROBOT.collision.num_collision_pairs > 0


def test_srdf_disabled_pairs_reduce_pair_count():
    # Without SRDF: 7 link geometries -> 21 pairs minus 6 parent/child adjacent = 15.
    # The bundled SRDF disables 7 more pairs (base<->L4/L5/L6, L1<->L4/L5/L6, L2<->L4).
    assert PAROL6_ROBOT.collision.num_collision_pairs == 8


def test_home_is_clear():
    q = np.zeros(PAROL6_ROBOT.robot.nq)
    assert PAROL6_ROBOT.collision.in_collision(q) is False


def test_robot_collision_methods_clear_at_home():
    """Public Robot collision methods report clear at/near home on one instance.
    (Robot defines no __del__; an unstarted instance holds no subprocess, so the
    old per-test try/finally del was a no-op.)"""
    r = Robot()
    q = np.zeros(6)
    assert r.in_collision(q) is False
    d = r.min_distance(q)
    assert d > 0.0 and d != float("inf")
    # Tiny perturbation around home — definitely clear.
    q_path = np.linspace(np.zeros(6), 0.01 * np.ones(6), 5)
    assert r.check_trajectory(q_path) == -1


def test_guard_joint_path_clear_returns_none():
    positions = np.zeros((10, 6))
    # No exception means no collision detected.
    guard_joint_path(positions)


def test_guard_joint_path_raises_with_display_vocabulary():
    """Drive the REAL checker into a real keep-out: the guard must raise and
    report pairs in the display vocabulary (URDF link names + shape:<name>),
    never checker-internal identifiers like ``L4_0``."""
    from waldoctl import Box

    # At q=zeros the wrist sits at (0.16, 0, 0.324) — this box envelops it;
    # with the base rotated 90° the arm swings to +Y and is clear.
    PAROL6_ROBOT.apply_shapes(
        [Box(name="block", x=0.25, y=0.25, z=0.25, pose=(0.16, 0.0, 0.324, 0, 0, 0))]
    )
    try:
        q_clear = np.zeros(6)
        q_clear[0] = np.pi / 2
        assert PAROL6_ROBOT.collision.in_collision(q_clear) is False
        assert PAROL6_ROBOT.collision.in_collision(np.zeros(6)) is True

        positions = np.linspace(q_clear, np.zeros(6), 12)
        with pytest.raises(TrajectoryPlanningError) as exc_info:
            guard_joint_path(positions)
        err = exc_info.value.robot_error
        assert err.code == int(ErrorCode.SYS_SELF_COLLISION)
        assert "shape:block" in err.cause
        assert "_0" not in err.cause  # no Pinocchio geometry suffixes leak
        pairs = exc_info.value.colliding_pairs
        assert all("shape:block" in p or not p[0].endswith("_0") for p in pairs)
    finally:
        PAROL6_ROBOT.apply_shapes([])


def test_guard_disabled_when_checker_is_none(monkeypatch):
    monkeypatch.setattr(PAROL6_ROBOT, "collision", None)
    positions = np.zeros((5, 6))
    # No exception, returns None (no-op).
    assert guard_joint_path(positions) is None


def test_no_spurious_self_overlap_at_home_or_joint_limits():
    """Audit the bundled simplified collision STLs against the SRDF.

    Asserts that the checker reports no colliding pairs at home, at each
    joint's lower/upper limit (single-axis), at every (low, high) corner
    of the joint-limit hypercube, and across a handful of seeded random
    configs. If this fails, the assertion message identifies the named
    pair — add it to parol6/urdf_model/srdf/PAROL6.srdf and re-run.
    """
    import itertools

    lo = PAROL6_ROBOT._joint_limits_radian[:, 0]
    hi = PAROL6_ROBOT._joint_limits_radian[:, 1]
    checker = PAROL6_ROBOT.collision

    configs: list[tuple[str, np.ndarray]] = [("home", np.zeros(6))]
    for j in range(6):
        q_lo = np.zeros(6)
        q_lo[j] = lo[j]
        q_hi = np.zeros(6)
        q_hi[j] = hi[j]
        configs.append((f"J{j}=low", q_lo))
        configs.append((f"J{j}=high", q_hi))

    for bits in itertools.product((0, 1), repeat=6):
        q = np.where(np.array(bits, dtype=bool), hi, lo)
        configs.append((f"corner_{''.join(map(str, bits))}", q))

    rng = np.random.default_rng(0xC011)
    for k in range(20):
        configs.append((f"rand_{k}", rng.uniform(lo, hi)))

    failures: list[str] = []
    for label, q in configs:
        pairs = checker.colliding_pairs(np.ascontiguousarray(q, dtype=np.float64))
        if pairs:
            failures.append(f"{label}: {pairs}")

    assert not failures, (
        "Spurious self-collision in the bundled simplified STLs. Add these "
        "pairs to parol6/urdf_model/srdf/PAROL6.srdf and re-run:\n"
        + "\n".join(failures[:10])
        + (f"\n... ({len(failures) - 10} more)" if len(failures) > 10 else "")
    )


def test_collision_check_speed_diagnostic(capsys):
    """Time in_collision and colliding_pairs across a sampled workspace.

    Diagnostic only — does not assert a threshold. The per-tick mid-motion
    collision check shipped (JogL release-decel gate, JogJ stop); this prints
    percentiles to confirm its cost stays well within the 100 Hz tick budget
    (10 ms).

    Run via:  pytest tests/unit/test_collision_integration.py::test_collision_check_speed_diagnostic -v -s
    """
    import time

    rng = np.random.default_rng(0xC0FFEE)
    lo = PAROL6_ROBOT._joint_limits_radian[:, 0]
    hi = PAROL6_ROBOT._joint_limits_radian[:, 1]
    qs = [np.zeros(6)] + [
        np.ascontiguousarray(rng.uniform(lo, hi), dtype=np.float64) for _ in range(99)
    ]
    c = PAROL6_ROBOT.collision

    # warm-up so first-call cache effects don't dominate
    for q in qs[:10]:
        c.in_collision(q)
        c.colliding_pairs(q)

    t_bool: list[int] = []
    for q in qs:
        t0 = time.perf_counter_ns()
        c.in_collision(q)
        t_bool.append(time.perf_counter_ns() - t0)

    t_pairs: list[int] = []
    for q in qs:
        t0 = time.perf_counter_ns()
        c.colliding_pairs(q)
        t_pairs.append(time.perf_counter_ns() - t0)

    def pct(a: list[int], p: float) -> float:
        return float(np.percentile(a, p)) / 1000.0  # ns -> us

    with capsys.disabled():
        print(
            f"\nin_collision     us:"
            f" min={pct(t_bool, 0):.1f}"
            f" med={pct(t_bool, 50):.1f}"
            f" p95={pct(t_bool, 95):.1f}"
            f" p99={pct(t_bool, 99):.1f}"
            f" max={pct(t_bool, 100):.1f}"
        )
        print(
            f"colliding_pairs  us:"
            f" min={pct(t_pairs, 0):.1f}"
            f" med={pct(t_pairs, 50):.1f}"
            f" p95={pct(t_pairs, 95):.1f}"
            f" p99={pct(t_pairs, 99):.1f}"
            f" max={pct(t_pairs, 100):.1f}"
        )
        print("servo tick budget: 10000 us (100 Hz)")


def _box(name: str, **overrides):
    from waldoctl import Box

    kwargs = dict(x=0.1, y=0.1, z=0.1, pose=(1.0, 1.0, 1.0, 0, 0, 0))
    kwargs.update(overrides)
    return Box(name=name, **kwargs)


def test_apply_shapes_rejects_bad_sets_without_mutation():
    """Set-level rejection fails fast — never a half-applied collision world.

    The OLD world must survive a rejected call: the remove-then-add loop only
    runs after validation, so an error reply can't silently disarm barriers.
    (Per-shape value validation lives in Shape construction — the decode gate
    below.)
    """
    bad_sets = [
        ("Duplicate shape name", [_box("b"), _box("b")]),
        # A visual-only marker sharing a keep-out's name shadows it in the
        # frontend's highlight mapping — rejected too.
        ("Duplicate shape name", [_box("b"), _box("b", collision=False)]),
    ]
    try:
        PAROL6_ROBOT.apply_shapes([_box("a")])
        for match, shapes in bad_sets:
            with pytest.raises(ValueError, match=match):
                PAROL6_ROBOT.apply_shapes(shapes)
            assert PAROL6_ROBOT._active_shape_names == ["shape:a"]
    finally:
        PAROL6_ROBOT.apply_shapes([])


def test_decode_gate_rejects_malformed_wire():
    """The server's codec boundary (shape_from_wire at SET_SHAPES intake)
    rejects malformed and degenerate wire data — cases chosen from the
    requirement (NaN/inf/negative/zero/count), not from what the code handles."""
    from waldoctl import shape_from_wire

    good = ("box", [0.1, 0.1, 0.1], [0, 0, 0, 0, 0, 0], True, None, "b")
    assert shape_from_wire(*good).name == "b"
    bad_wires = [
        ("box", [0.1], [0, 0, 0, 0, 0, 0], True, None, "b"),  # wrong count
        ("box", [float("nan"), 0.1, 0.1], [0, 0, 0, 0, 0, 0], True, None, "b"),
        ("box", [float("inf"), 0.1, 0.1], [0, 0, 0, 0, 0, 0], True, None, "b"),
        ("box", [-0.1, 0.1, 0.1], [0, 0, 0, 0, 0, 0], True, None, "b"),
        ("sphere", [0.0], [0, 0, 0, 0, 0, 0], True, None, "b"),  # zero dim
        ("box", [0.1] * 3, [0, 0, float("nan"), 0, 0, 0], True, None, "b"),
        ("box", [0.1] * 3, [0, 0, 0, 0, 0], True, None, "b"),  # short pose
        ("box", [0.1] * 3, [0, 0, 0, 0, 0, 0], True, -0.01, "b"),  # neg margin
        ("plane", [0, 0, 0, 0.5], [0, 0, 0, 0, 0, 0], True, None, "b"),  # 0 normal
    ]
    for wire in bad_wires:
        with pytest.raises(ValueError):
            shape_from_wire(*wire)


def test_per_shape_margin_blocks_at_standoff():
    """A margined keep-out trips ``in_collision`` at its standoff distance,
    not at contact. Real checker, real geometry: the sphere hovers ~8 cm off
    the wrist — clear without a margin, colliding with a 15 cm one — and the
    reported pair names the shape, proving the margin applied to ITS pairs."""
    from waldoctl import Sphere

    q = np.zeros(6)
    checker = PAROL6_ROBOT.collision
    above_wrist = (0.16, 0.0, 0.424, 0, 0, 0)
    try:
        PAROL6_ROBOT.apply_shapes([Sphere(name="s", radius=0.02, pose=above_wrist)])
        assert checker.in_collision(q) is False

        PAROL6_ROBOT.apply_shapes(
            [Sphere(name="s", radius=0.02, pose=above_wrist, margin=0.15)]
        )
        assert checker.in_collision(q) is True  # margin closes the gap
        pairs = PAROL6_ROBOT.display_pairs(checker.colliding_pairs(q))
        assert any("shape:s" in p for p in pairs)
    finally:
        PAROL6_ROBOT.apply_shapes([])
    assert checker.in_collision(q) is False


def test_installation_layer_survives_program_clear():
    """Installation shapes (robot config) are enforced alongside the program
    layer and are untouched by ``apply_shapes`` — including a full clear."""
    from waldoctl import Box

    q = np.zeros(6)
    checker = PAROL6_ROBOT.collision
    try:
        PAROL6_ROBOT.apply_installation_shapes(
            [Box(name="wall", x=0.25, y=0.25, z=0.25, pose=(0.16, 0.0, 0.324, 0, 0, 0))]
        )
        assert checker.in_collision(q) is True
        assert [s.name for s in PAROL6_ROBOT.installation_shapes()] == ["wall"]

        PAROL6_ROBOT.apply_shapes([])  # program clear must not disarm install
        assert checker.in_collision(q) is True
        assert PAROL6_ROBOT.display_pairs(checker.colliding_pairs(q))[0][1] == (
            "install:wall"
        )
    finally:
        PAROL6_ROBOT.apply_installation_shapes([])
    assert checker.in_collision(q) is False


def test_pose_to_matrix_is_extrinsic_xyz_rpy():
    """The waldoctl Shape.pose contract is extrinsic-XYZ RPY (R = Rz·Ry·Rx).
    Tripwire against swapping in ``pinokin.se3_from_rpy`` (Rx·Ry·Rz), which
    would silently mis-orient every multi-axis-tilted shape versus its render."""
    rx, ry, rz = 0.3, -0.5, 1.1
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T = PAROL6_ROBOT._pose_to_matrix([0.1, 0.2, 0.3, rx, ry, rz])
    assert np.allclose(T[:3, :3], Rz @ Ry @ Rx)
    assert not np.allclose(T[:3, :3], Rx @ Ry @ Rz)  # the tempting wrong order
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])


def test_jogl_release_decel_streams_while_escaping():
    """Releasing a Cartesian jog while ESCAPING from inside a keep-out must
    keep streaming the deceleration (escape-aware gate) — the pre-fix bare
    ``in_collision`` skipped every decel send, freezing the target at the
    release point. Real command, real state, real checker, real IK."""
    import time as _time

    from waldoctl import Box

    from parol6.commands.cartesian_commands import JogLCommand
    from parol6.protocol.wire import JogLCmd
    from parol6.server.state import ControllerState

    from parol6.config import deg_to_steps

    state = ControllerState()
    # IK-friendly physical home (wrist at ~(0.237, 0, 0.334)); q=zeros is an
    # IK danger zone where the jog never streams.
    q_home_deg = np.array([0.0, -90.0, 180.0, 0.0, 0.0, 180.0])
    deg_to_steps(q_home_deg, state.Position_in)
    try:
        # Box centred 5 cm below the wrist: +Z is unambiguously the escape.
        PAROL6_ROBOT.apply_shapes(
            [
                Box(
                    name="cage",
                    x=0.15,
                    y=0.15,
                    z=0.15,
                    pose=(0.237, 0.0, 0.284, 0, 0, 0),
                )
            ]
        )
        assert PAROL6_ROBOT.collision.in_collision(np.radians(q_home_deg)) is True

        cmd = JogLCommand(JogLCmd(velocities=[0.0, 0.0, 1.0, 0, 0, 0], duration=5.0))
        cmd.setup(state)
        for _ in range(30):  # held phase: build real velocity (CSE dt=0.01/tick)
            cmd.execute_step(state)
        assert state.Command_out != 0  # held phase streamed (escape allowed)
        cmd._t_end = _time.perf_counter() - 1.0  # deterministic release

        code = cmd.execute_step(state)
        pos0 = state.Position_out.copy()
        moved = False
        for _ in range(500):
            if code != ExecutionStatusCode.EXECUTING:
                break
            code = cmd.execute_step(state)
            if not np.array_equal(state.Position_out, pos0):
                moved = True
        assert code == ExecutionStatusCode.COMPLETED
        assert moved, "decel sends were skipped — target frozen at release point"
    finally:
        PAROL6_ROBOT.apply_shapes([])


def test_dry_run_script_set_shapes_applies_and_replays():
    """A script's ``set_shapes`` through the REAL dry-run dispatch: applies the
    world (raw waldoctl Shapes end-to-end — the pre-fix path crashed with
    ``TypeError: object of type 'method' has no len()``), reads back via
    ``shapes()``, and a subsequent world change replaces it."""
    from waldoctl import Box

    from parol6.client.dry_run_client import DryRunRobotClient

    try:
        c = DryRunRobotClient(initial_joints_deg=[0, -90, 180, 0, 0, 180])
        c.set_shapes(
            [Box(name="bar", x=0.1, y=0.1, z=0.1, pose=(0.9, 0.9, 0.9, 0, 0, 0))]
        )
        assert PAROL6_ROBOT._active_shape_names == ["shape:bar"]
        world = c.shapes()
        assert [s.name for s in world.program] == ["bar"]

        c.set_shapes(
            [Box(name="bar2", x=0.1, y=0.1, z=0.1, pose=(0.9, 0.9, 0.9, 0, 0, 0))]
        )
        assert PAROL6_ROBOT._active_shape_names == ["shape:bar2"]
    finally:
        PAROL6_ROBOT.apply_shapes([])
