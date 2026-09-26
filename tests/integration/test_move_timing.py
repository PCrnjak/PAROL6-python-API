"""How a planned move is timed, through the client and the simulated
controller: with no timing given it runs at half speed, a positive duration
sets its length whatever speed says, and a timing value out of range is
refused before anything is sent."""

import math

import pytest

from parol6 import RobotClient
from parol6.client.dry_run_client import DryRunRobotClient

pytestmark = pytest.mark.integration


def _queued_seconds(client: RobotClient, moves: int) -> float:
    """Planned seconds the paused queue holds once it holds *moves* moves."""
    seconds: list[float] = []

    def holds(status) -> bool:
        if status.queued_segments != moves:
            return False
        seconds.append(status.queued_duration)
        return True

    assert client.wait_status(holds, timeout=5.0), f"the queue never held {moves}"
    return seconds[-1]


def test_an_untimed_move_runs_at_half_speed_and_a_duration_overrides_speed(
    client: RobotClient,
):
    start = client.angles()
    assert start is not None
    there = list(start)
    there[0] -= 20.0
    try:
        # Paused, so each move is planned and held rather than run.
        assert client.pause() == 1
        assert client.move_j(there, wait=False) >= 0
        untimed = _queued_seconds(client, 1)
        assert client.move_j(start, speed=0.5, wait=False) >= 0
        at_half = _queued_seconds(client, 2) - untimed
        assert client.move_j(there, duration=4.0, speed=0.1, wait=False) >= 0
        timed = _queued_seconds(client, 3) - untimed - at_half
    finally:
        assert client.stop() == 1
    assert untimed == pytest.approx(at_half, abs=0.02), (
        f"a move given no timing planned {untimed:.3f}s, the same move at "
        f"speed 0.5 {at_half:.3f}s"
    )
    assert timed == pytest.approx(4.0, abs=0.02), (
        f"a 4 s move planned {timed:.3f}s: its speed overrode its duration"
    )


def test_a_timing_value_out_of_range_is_refused_before_anything_is_sent(
    client: RobotClient,
):
    """``speed`` and ``accel`` are fractions in (0, 1] and ``duration`` is a
    finite number of seconds >= 0: the live client and the dry run refuse
    anything else with ValueError, not a controller rejection."""
    start = client.angles()
    pose = client.pose()
    assert start is not None and pose is not None
    there = list(start)
    there[0] -= 5.0
    preview = DryRunRobotClient(initial_joints_deg=start)
    for rbt in (client, preview):
        for bad in (0.0, -0.5, math.nan, math.inf, 1.5):
            with pytest.raises(ValueError):
                rbt.move_j(there, speed=bad)
            with pytest.raises(ValueError):
                rbt.move_l(pose, speed=0.5, accel=bad)
            with pytest.raises(ValueError):
                rbt.move_c(pose, pose, speed=bad)
            with pytest.raises(ValueError):
                rbt.move_p([pose, pose], duration=2.0, accel=bad)
        for bad in (-1.0, math.nan, math.inf):
            with pytest.raises(ValueError):
                rbt.move_j(there, duration=bad)
            with pytest.raises(ValueError):
                rbt.move_s([pose, pose], duration=bad)
    assert preview.program_length == 0, "the dry run recorded a refused move"
