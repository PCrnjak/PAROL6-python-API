"""A guard-rejected command must not fail later unrelated waits.

Pre-fix, ``state.error`` from a rejected move kept broadcasting (streaming
accepts never cleared it) and ``wait_command`` honored any standing error at
or below its index — including status frames that predate its own command's
acceptance. The next ``wait=True`` command then raised a minutes-old
rejection while the robot executed it fine (found live: MCP ``motion.home``
"failing" with a stale self-collision error as the arm visibly homed).
"""

import time

import numpy as np
import pytest

from parol6 import MotionError, RobotClient
from waldoctl import Box

pytestmark = pytest.mark.integration


def _wait_until(pred, timeout: float, msg: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    pytest.fail(msg)


def _reject_move(client: RobotClient) -> None:
    """Drive a real guard rejection: a keep-out enveloping the target wrist."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    target = [0.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    p = PAROL6_ROBOT.robot.fkine(np.radians(target))[:3, 3]
    box = Box(
        name="blocker",
        x=0.25,
        y=0.25,
        z=0.25,
        pose=(float(p[0]), float(p[1]), float(p[2]), 0, 0, 0),
    )
    assert client.set_shapes([box]) == 1
    with pytest.raises(MotionError, match="blocker"):
        client.move_j(target, duration=1.5, wait=True)


def test_rejected_move_does_not_poison_next_wait(client: RobotClient, server_proc):
    """home(wait=True) right after a rejection must wait on its own outcome,
    not raise the previous command's error off a pre-acceptance status frame."""
    try:
        _reject_move(client)
        assert client.home(wait=True, timeout=30.0) >= 0
    finally:
        assert client.set_shapes([]) == 1


def test_streaming_accept_clears_stale_error(client: RobotClient, server_proc):
    """Jog acceptance clears the standing error like every other command path,
    so a rejection can't keep broadcasting across minutes of later activity."""
    try:
        _reject_move(client)
        assert client.error() is not None
        assert client.jog_j(0, 0.2, 0.2) >= 0
        _wait_until(
            lambda: client.error() is None,
            2.0,
            "jog accept never cleared the stale error",
        )
    finally:
        assert client.set_shapes([]) == 1
