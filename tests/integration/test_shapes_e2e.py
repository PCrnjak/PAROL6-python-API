"""SET_SHAPES / SHAPES end-to-end: real client ↔ real (fake-serial) server.

The ack contract under test is the waldoctl convention: 1 = confirmed applied,
0 = unconfirmed (timeout), raise = rejected. The pre-fix client treated
SET_SHAPES as fire-and-forget and returned 1 unconditionally — every assert
here except the plain success one fails against that behavior.
"""

import pytest

from parol6 import MotionError, RobotClient
from waldoctl import Box

pytestmark = pytest.mark.integration


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
