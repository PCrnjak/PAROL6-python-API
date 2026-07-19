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
from waldoctl import Box

pytestmark = pytest.mark.integration

# The HOME command parks the arm at J1=90; invalidation moves sweep J1 downward.
HOME_J1 = 90.0


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

        # Unreachable controller → unconfirmed (0), never a fake success.
        dead = RobotClient(
            host=ports.server_ip, port=ports.server_port + 91, timeout=0.3
        )
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
