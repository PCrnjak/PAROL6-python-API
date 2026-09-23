"""Command indices stay monotonic across RESET.

Pre-fix, ``state.reset()`` recycled ``next_command_index`` to 0 while status
frames generated before the reset — cached client-side and in flight on the
multicast group — still carried the old ``completed_index`` high-water mark.
Any ``wait_command`` on a recycled index was satisfied instantly by such a
frame, reporting success for a command that hadn't run (the autouse
fixture's post-reset ``home(wait=...)`` survived only by frame timing).
Indices now keep counting across reset, so no stale frame can alias a
post-reset command.
"""

import pytest

from parol6 import MotionError, RobotClient
from parol6.utils.error_codes import ErrorCode
from waldoctl import StatusBuffer

pytestmark = pytest.mark.integration


def test_wait_command_ignores_pre_reset_frames(client: RobotClient, server_proc):
    seen: dict[str, int] = {}

    def _capture(s: StatusBuffer) -> bool:
        seen["completed"] = s.completed_index
        return s.completed_index >= 0

    # Ensure the client's cached frame carries the fixture home's completion
    # — the stale high-water mark that used to alias recycled indices.
    assert client.wait_status(_capture, timeout=5.0)

    client.reset_state()
    idx = client.delay(1.0)
    assert idx > seen["completed"], (
        f"index {idx} recycled across reset (pre-reset completed_index was "
        f"{seen['completed']}) — stale frames can alias it"
    )

    # The delay is still running: only a stale pre-reset frame could satisfy
    # this wait early.
    assert not client.wait_command(idx, timeout=0.2), (
        "wait_command satisfied by a stale pre-reset status frame"
    )
    assert client.wait_command(idx, timeout=5.0), "delay never completed"


def test_reset_state_keeps_the_protective_stop_outputs_and_homed(
    client: RobotClient, server_proc
):
    """reset_state restores the program-level state — profile, speed, pause,
    queues, tool, shapes — and nothing physical: it does not clear a
    protective stop (only reset() does), un-home the arm, or change the
    digital outputs."""
    start = client.angles()
    assert start is not None
    away = list(start)
    away[0] += 10.0

    output = client.write_io(0, 1)
    assert output >= 0 and client.wait_command(output, timeout=5.0)
    assert client.select_profile("RUCKIG") == 1
    assert client.set_execution_speed(0.5) == 1
    try:
        assert client.estop() == 1
        assert client.wait_status(lambda s: not s.enabled, timeout=2.0)
        assert client.reset_state() == 1

        with pytest.raises(MotionError) as refused:
            client.move_j(away, speed=0.5)
        assert refused.value.robot_error.code == ErrorCode.SYS_CONTROLLER_DISABLED
        assert client.wait_status(lambda s: s.homed and not s.enabled, timeout=2.0)
        io = client.io()
        assert io is not None and io[2] == 1, f"reset_state changed the outputs: {io}"
        assert client.profile() == "TOPPRA"
        assert client.execution_speed().target_scale == 1.0

        assert client.reset() == 1
        assert client.wait_status(lambda s: s.enabled and s.homed, timeout=2.0)
        moved = client.move_j(away, speed=0.5)
        assert moved >= 0 and client.wait_command(moved, timeout=10.0)
    finally:
        client.reset()
        client.write_io(0, 0)
