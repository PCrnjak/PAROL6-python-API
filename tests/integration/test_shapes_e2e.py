"""SET_SHAPES / SHAPES end-to-end: real client ↔ real (fake-serial) server.

The ack contract under test is the waldoctl convention: 1 = confirmed applied,
0 = unconfirmed (timeout), raise = rejected. The pre-fix client treated
SET_SHAPES as fire-and-forget and returned 1 unconditionally — every assert
here except the plain success one fails against that behavior.

The invalidation tests cover the MoveIt-style contract: a world change
re-guards the streaming trajectory's remaining waypoints and every queued
trajectory at activation — committed motion never sails into a keep-out
declared after it was planned.
"""

import time

import numpy as np
import pytest

from parol6 import MotionError, RobotClient

from tests.conftest import free_udp_port
from waldoctl import Box, Physical

pytestmark = pytest.mark.integration

# The HOME command parks the arm at J1=90; invalidation moves sweep J1 downward.
HOME_J1 = 90.0


def test_preview_attachment_context_survives_only_explicit_reconciliation():
    from parol6.client.dry_run_client import DryRunRobotClient
    from waldoctl import Sphere

    preview = DryRunRobotClient()
    try:
        world = preview.shapes()
        part = Sphere(name="part", radius=0.01).attach(
            flange_pose=(0, 0, 0.25, 0, 0, 0),
            epoch=world.attachment_epoch,
        )
        assert preview.set_shapes([part]) == 1
        preview.estop()
        assert not preview.shapes().attachments_valid
        preview.reset()
        with pytest.raises(ValueError, match="attachment context"):
            preview.move_j(preview.angles(), duration=1)
        with pytest.raises(ValueError, match="attachment context"):
            preview.set_shapes([part])
        part = part.attach(
            flange_pose=part.pose, epoch=preview.shapes().attachment_epoch
        )
        assert preview.set_shapes([part]) == 1
        assert preview.shapes().attachments_valid
        assert preview.set_shapes([part.detach(world_pose=(1, 1, 1, 0, 0, 0))]) == 1
        assert preview.shapes().program[0].attachment is None
    finally:
        preview.set_shapes([])


def test_attached_part_blocks_motion_except_for_declared_contacts(client: RobotClient):
    from dataclasses import replace

    import parol6.PAROL6_ROBOT as model
    from waldoctl import Sphere

    start = client.angles()
    assert start is not None
    target = list(start)
    target[0] -= 40
    flange = model.robot.fkine(np.radians(target))
    local = (0.0, 0.0, 0.25, 0.0, 0.0, 0.0)
    at = flange[:3, 3] + flange[:3, 2] * local[2]
    fixture = Sphere(name="fixture", radius=0.025, pose=(*at, 0.0, 0.0, 0.0))
    fence = replace(fixture, name="fence")
    world = client.shapes()
    assert world is not None
    part = Sphere(name="part", radius=0.025).attach(
        flange_pose=local,
        epoch=world.attachment_epoch,
        allowed_contacts=("shape:fixture",),
    )
    try:
        assert client.set_shapes([fixture, fence, part]) == 1
        applied = client.shapes()
        assert applied is not None and applied.program[-1] == part
        with pytest.raises(MotionError, match="shape:fence"):
            index = client.move_j(target, duration=1.0, wait=False)
            client.wait_command(index, timeout=10.0)
        assert abs(client.angles()[0] - start[0]) < 1.0
        assert client.set_shapes([fixture, part]) == 1
        index = client.move_j(target, duration=1.0, wait=False)
        assert client.wait_command(index, timeout=10.0)
        assert abs(client.angles()[0] - target[0]) < 1.0

        with pytest.raises(MotionError, match="unknown contact"):
            client.set_shapes(
                [
                    fixture,
                    part.attach(
                        flange_pose=local,
                        epoch=world.attachment_epoch,
                        allowed_contacts=("shape:typo",),
                    ),
                ]
            )
        assert client.shapes().program[-1] == part

        assert client.estop() == 1
        _wait_until(
            lambda: not client.shapes().attachments_valid,
            3.0,
            "attachment remained valid after stop",
        )
        assert client.reset() == 1
        with pytest.raises(MotionError, match="attachment context"):
            client.move_j(start, duration=1.0, wait=False)
        fresh = client.shapes()
        assert fresh is not None and fresh.attachment_epoch != world.attachment_epoch
        reconciled = part.attach(
            flange_pose=local,
            epoch=fresh.attachment_epoch,
            allowed_contacts=("shape:fixture",),
        )
        assert client.set_shapes([fixture, reconciled]) == 1
        assert client.shapes().attachments_valid
        released = reconciled.detach(world_pose=(*at, 0.0, 0.0, 0.0))
        assert client.set_shapes([fixture, released]) == 1
        index = client.move_j(start, duration=1.0, wait=False)
        assert client.wait_command(index, timeout=10.0)
    finally:
        client.stop()
        client.set_shapes([])


def _wrist_box(target_deg: list[float], name: str) -> Box:
    """A keep-out enveloping the wrist position of ``target_deg``."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    p = PAROL6_ROBOT.robot.fkine(np.radians(target_deg))[:3, 3]
    return Box(
        name=name,
        x=0.25,
        y=0.25,
        z=0.25,
        pose=(float(p[0]), float(p[1]), float(p[2]), 0, 0, 0),
    )


def _wait_until(pred, timeout: float, msg: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    pytest.fail(msg)


def _j1(client: RobotClient) -> float:
    angles = client.angles()
    assert angles is not None
    return angles[0]


def test_set_shapes_ack_readback_rejection_and_timeout(
    server_proc, client: RobotClient, ports
):
    box = Box(name="table", x=0.6, y=0.4, z=0.02, pose=(0.9, 0.9, -0.01, 0, 0, 0))
    try:
        # Confirmed apply → 1, and readback reports the applied program layer.
        assert client.set_shapes([box]) == 1
        world = client.shapes()
        assert world is not None
        assert tuple(s.name for s in world.program) == ("table",)
        assert world.program[0] == box  # full round-trip, not just the name

        # Server rejection (duplicate names) → ERROR reply → raises; the
        # previously-applied world must survive the rejected call.
        with pytest.raises(MotionError, match="Duplicate"):
            client.set_shapes([box, Box(name="table", x=0.1, y=0.1, z=0.1)])
        world = client.shapes()
        assert world is not None
        assert tuple(s.name for s in world.program) == ("table",)

        # No contact simulation here: a shape declaring physics is refused
        # by name, never silently flattened to its geometry.
        with pytest.raises(MotionError, match="physics"):
            client.set_shapes(
                [
                    box,
                    Box(name="brick", x=0.1, y=0.1, z=0.1, physics=Physical(mass=0.2)),
                ]
            )
        world = client.shapes()
        assert world is not None
        assert tuple(s.name for s in world.program) == ("table",)

        # Unreachable controller → unconfirmed (0), never a fake success.
        # A port the kernel just handed out and nothing bound: arithmetic on
        # the live port runs past 65535 whenever the ephemeral range hands
        # out a high one, which is a connect() overflow, not a dead server.
        dead = RobotClient(host=ports.server_ip, port=free_udp_port(), timeout=0.3)
        assert dead.set_shapes([box]) == 0
        assert dead.shapes() is None
    finally:
        assert client.set_shapes([]) == 1
        world = client.shapes()
        assert world is not None and world.program == ()


def test_set_shapes_mid_flight_halts_streaming_move(client: RobotClient, server_proc):
    """A keep-out declared over the *remaining* path of a streaming move halts
    it with the collision error instead of letting committed motion sail into
    the new keep-out (pre-fix: the segment player never re-checked)."""
    target = [0.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    try:
        idx = client.move_j(target, duration=4.0, wait=False)
        assert idx >= 0
        _wait_until(
            lambda: _j1(client) < HOME_J1 - 5.0, 10.0, "move never started streaming"
        )

        assert client.set_shapes([_wrist_box(target, "blocker")]) == 1
        _wait_until(
            lambda: client.error() is not None,
            3.0,
            "world change never halted the move",
        )
        err = client.error()
        assert err is not None and "shape:blocker" in err.cause

        j1_stop = _j1(client)
        assert j1_stop > 30.0, f"arm reached the keep-out region (J1={j1_stop:.1f})"
        time.sleep(0.3)
        assert abs(_j1(client) - j1_stop) < 0.5, "arm kept moving after the halt"
    finally:
        client.reset_state()
        assert client.set_shapes([]) == 1


def test_set_shapes_mid_flight_rejects_queued_move_at_activation(
    client: RobotClient, server_proc
):
    """A queued move planned against the old world is re-guarded when it
    activates: the first (clear) move completes, the second (now blocked)
    never streams."""
    t1 = [60.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    t2 = [0.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    try:
        i1 = client.move_j(t1, duration=1.5, wait=False)
        i2 = client.move_j(t2, duration=2.0, wait=False)
        assert i1 >= 0 and i2 >= 0
        _wait_until(
            lambda: _j1(client) < HOME_J1 - 2.0, 10.0, "first move never started"
        )

        assert client.set_shapes([_wrist_box(t2, "late-wall")]) == 1

        _wait_until(
            lambda: client.error() is not None,
            10.0,
            "queued move was never invalidated",
        )
        err = client.error()
        assert err is not None and "shape:late-wall" in err.cause
        # The clear first move finished; the blocked second never streamed.
        _wait_until(
            lambda: abs(_j1(client) - 60.0) < 2.0,
            10.0,
            f"arm not at the first target (J1={_j1(client):.1f})",
        )
        time.sleep(0.3)
        assert abs(_j1(client) - 60.0) < 2.0, "second move streamed despite the wall"
    finally:
        client.reset_state()
        assert client.set_shapes([]) == 1


def test_set_shapes_mid_flight_off_path_does_not_disturb_motion(
    client: RobotClient, server_proc
):
    """The re-guard must not manufacture failures: a mid-flight world change
    that stays clear of the path leaves the move to complete normally."""
    target = [30.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    try:
        idx = client.move_j(target, duration=2.5, wait=False)
        assert idx >= 0
        _wait_until(
            lambda: _j1(client) < HOME_J1 - 5.0, 10.0, "move never started streaming"
        )

        far = Box(name="far", x=0.1, y=0.1, z=0.1, pose=(0.9, 0.9, 0.9, 0, 0, 0))
        assert client.set_shapes([far]) == 1

        assert client.wait_command(idx, timeout=10.0), "move did not complete"
        assert client.error() is None
        assert abs(_j1(client) - 30.0) < 1.0
    finally:
        assert client.set_shapes([]) == 1
