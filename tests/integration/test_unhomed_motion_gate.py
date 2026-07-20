"""Planned motion is refused until the robot is homed.

Before homing, reported joint positions are unreferenced (the boot state is
all-zeros steps — outside J2/J3's limits and physically impossible), so
planning or collision-checking a trajectory from them produces garbage:
found live as a phantom "[54] Self-collision predicted" on a first move
from the boot pose. Planned moves must instead be refused with an
actionable "not homed" error; jogging stays allowed (an unhomed arm may
need to be nudged clear of something before it can home), and ``home``
itself is the way out.
"""

import pytest

from parol6 import MotionError, RobotClient
from parol6.utils.error_codes import ErrorCode

pytestmark = pytest.mark.integration


def test_planned_motion_refused_until_homed(client: RobotClient, server_proc):
    """move_j from the unhomed boot state raises MOTN_NOT_HOMED (not a
    garbage collision prediction); after homing the same move is accepted.
    Jog remains available while unhomed."""
    target = [90.0, -90.0, 180.0, 0.0, 0.0, 170.0]

    # The autouse fixture homes; reset back to the unhomed boot state
    # (Homed_in and Position_in zeroed — exactly how a controller starts).
    client.reset_state()

    # The STATUS stream reports the unhomed state (WC feeds it to dry runs).
    assert client.wait_status(lambda s: not s.homed, timeout=2.0)

    with pytest.raises(MotionError, match="not homed") as exc_info:
        client.move_j(target, duration=1.5, wait=True)
    assert exc_info.value.robot_error.code == int(ErrorCode.MOTN_NOT_HOMED)

    # Jogging an unhomed robot stays allowed — no planning involved.
    assert client.jog_j(0, 0.2, 0.1) >= 0

    # Homing establishes references; the identical move now proceeds.
    assert client.home(wait=True, timeout=30.0) >= 0
    assert client.wait_status(lambda s: s.homed, timeout=2.0)
    assert client.move_j(target, duration=1.5, wait=True) >= 0
