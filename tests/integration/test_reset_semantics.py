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

from parol6 import RobotClient
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
