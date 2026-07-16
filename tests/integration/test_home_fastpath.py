"""home() on an already-referenced robot is a planned return move.

The firmware switch-seek establishes references and runs only while the
robot is unhomed; once referenced, HOME plans a normal collision-checked
joint move back to the standby pose instead of re-running the sequence.
Proven by a keep-out enveloping the standby pose: the planned return is
refused, where the unchecked firmware sequence would drive through it.
"""

import numpy as np
import pytest

from parol6 import MotionError, RobotClient
from waldoctl import Box

pytestmark = pytest.mark.integration


def test_home_returns_via_planned_move_when_referenced(
    client: RobotClient, server_proc
):
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    standby = list(PAROL6_ROBOT.joint.standby_deg)
    away = [45.0, -60.0, 150.0, 0.0, 30.0, 90.0]
    assert client.move_j(away, duration=2.0, wait=True) >= 0

    p = PAROL6_ROBOT.robot.fkine(np.radians(standby))[:3, 3]
    box = Box(
        name="home-blocker",
        x=0.25,
        y=0.25,
        z=0.25,
        pose=(float(p[0]), float(p[1]), float(p[2]), 0, 0, 0),
    )
    try:
        assert client.set_shapes([box]) == 1
        with pytest.raises(MotionError, match="home-blocker"):
            client.home(wait=True, timeout=30.0)
    finally:
        assert client.set_shapes([]) == 1

    # Cleared: the fast-path return completes and lands on the standby pose.
    assert client.home(wait=True, timeout=30.0) >= 0
    angles = client.angles()
    assert angles is not None
    assert np.allclose(angles, standby, atol=0.5)

    # Degenerate re-home (already at standby) completes as a no-op move.
    assert client.home(wait=True, timeout=10.0) >= 0
