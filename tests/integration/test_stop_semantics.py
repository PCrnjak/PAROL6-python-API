"""STOP cancels in-flight motion without latching; ESTOP latches until RESET.

Pre-fix (as HALT), the segment player kept playing the active trajectory —
each tick rewrote Command_out and fresh speeds, clobbering the stop before
transmission — so a "stopped" robot drove on to its target, and queued
motion survived to play out later. And the only stop primitive latched the
controller disabled, so a plain "just stop" left the robot rejecting every
subsequent command.
"""

import time

import numpy as np
import pytest

from parol6 import MotionError, RobotClient

pytestmark = pytest.mark.integration


def _wait_until_moving(client: RobotClient, start: list[float]) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        a = client.angles()
        if a is not None and not np.allclose(a, start, atol=0.2):
            return
        time.sleep(0.02)
    pytest.fail("move never started")


def _assert_frozen(client: RobotClient, target: list[float]) -> list[float]:
    # Fixed observation window: "stays frozen" has no condition to poll for.
    time.sleep(0.2)
    frozen = client.angles()
    assert frozen is not None
    time.sleep(0.5)
    after = client.angles()
    assert after is not None
    assert np.allclose(after, frozen, atol=0.05), (
        f"robot kept moving after stop: {frozen} -> {after}"
    )
    assert not np.allclose(after, target, atol=0.5), (
        "trajectory played to completion despite stop"
    )
    return after


def test_stop_cancels_motion_and_stays_enabled(client: RobotClient, server_proc):
    away = [45.0, -60.0, 150.0, 0.0, 30.0, 90.0]
    queued = [90.0, -45.0, 120.0, 10.0, 20.0, 90.0]
    start = client.angles()
    assert start is not None

    assert client.move_j(away, duration=4.0, wait=False) >= 0
    assert client.move_j(queued, duration=2.0, wait=False) >= 0
    _wait_until_moving(client, start)

    assert client.stop() == 1
    after = _assert_frozen(client, away)

    # No latch: the very next command is accepted, and the canceled/queued
    # motion never resurfaces.
    assert client.home(wait=True, timeout=30.0) >= 0
    final = client.angles()
    assert final is not None
    assert not np.allclose(final, after, atol=0.05)


def test_estop_latches_until_reset(client: RobotClient, server_proc):
    away = [45.0, -60.0, 150.0, 0.0, 30.0, 90.0]
    start = client.angles()
    assert start is not None

    assert client.move_j(away, duration=4.0, wait=False) >= 0
    _wait_until_moving(client, start)

    assert client.estop() == 1
    after = _assert_frozen(client, away)

    with pytest.raises(MotionError, match="disabled"):
        client.home()

    assert client.reset() == 1
    # Reset clears the latch but never resurrects canceled motion.
    time.sleep(0.5)
    resumed = client.angles()
    assert resumed is not None
    assert np.allclose(resumed, after, atol=0.05), (
        "canceled motion resurfaced after reset"
    )
    assert client.home(wait=True, timeout=30.0) >= 0


def test_stop_discards_plans_still_in_the_planner(client: RobotClient, server_proc):
    """Commands the planner has not finished planning when Stop arrives must
    not play afterwards: a plan finished after the cancel is not a queue."""
    start = client.angles()
    assert start is not None
    pose = client.pose()
    assert pose is not None
    away = list(pose)
    away[0] += 40.0
    # Cartesian plans take the planner long enough that Stop lands while
    # most of these are still in its inbox, behind which CancelAll queues.
    for i in range(40):
        target = away if i % 2 == 0 else pose
        assert client.move_l(target, duration=10.0, wait=False) >= 0
    assert client.stop() == 1
    time.sleep(0.3)
    frozen = client.angles()
    assert frozen is not None
    time.sleep(1.5)
    after = client.angles()
    assert after is not None
    assert np.allclose(after, frozen, atol=0.05), (
        f"a plan finished after Stop played: {frozen} -> {after}"
    )
    assert client.queue() == []
    assert client.home(wait=True, timeout=30.0) >= 0


def test_stop_discards_a_queued_tcp_transform_from_the_planner_too(
    client: RobotClient, server_proc
):
    """The planner applies SET_TCP_TRANSFORM when it plans, while the command
    is still queued; a Stop that cancels the queue must take that transform
    back from the planner, or every later plan is solved against a TCP the
    controller never applied."""
    assert client.home(wait=True, timeout=30.0) >= 0
    index = client.set_tcp_transform(0, 0, 0, 0, 0, 0)
    assert index >= 0 and client.wait_command(index, timeout=5.0)
    start = client.angles()
    pose = client.pose()
    assert start is not None and pose is not None
    assert client.delay(3.0) >= 0
    assert client.set_tcp_transform(0, 0, 40, 0, 90, 0) >= 0
    assert client.stop() == 1
    assert client.tcp_transform() == pytest.approx([0] * 6)
    # A move to the pose the arm is already at is planned against the
    # controller's TCP and therefore goes nowhere.
    index = client.move_l(pose, duration=1.0, wait=False)
    assert index >= 0 and client.wait_command(index, timeout=10.0)
    after = client.angles()
    assert after is not None
    assert np.allclose(after, start, atol=0.5), (
        f"the planner kept the cancelled TCP: {start} -> {after}"
    )
