"""What QUEUE reports, against the simulated controller.

The readback is what an operator and the frontend's playback bar read to know
what is still owed: commands the planner has accepted but not started, the one
executing now, and nothing at all once a Stop has cleared the queue. Every
assertion here goes through the client and the real controller, because the
pieces it is made of -- the planner's pending list, the blend consumption, the
executing-index exclusion -- are maintained in three different places.
"""

import numpy as np
import pytest

from parol6 import RobotClient


def _wait(condition, message: str, timeout: float = 10.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    pytest.fail(message)


def test_the_queue_lists_what_is_owed_and_a_stop_clears_it(client: RobotClient):
    start = client.angles()
    assert start is not None
    first, second = list(start), list(start)
    first[0] += 6
    second[0] += 12
    try:
        # Paused, so everything accepted stays owed and nothing moves.
        assert client.pause() == 1
        held = client.move_j(first, duration=1, wait=False)
        queued = client.move_j(second, duration=1, wait=False)
        assert held >= 0 and queued > held
        _wait(
            lambda: len(client.queue() or []) >= 2,
            "the paused queue never listed both accepted commands",
        )
        listed = client.queue()
        assert listed and all(name for name in listed), listed
        assert any("MoveJ" in name for name in listed)
        assert np.allclose(client.angles(), start, atol=0.05)

        # Resuming drains it: what the queue reports is what is still owed.
        assert client.resume() == 1
        assert client.wait_command(queued, timeout=20)
        _wait(lambda: client.queue() == [], "the drained queue still reports work")
        assert np.allclose(client.angles(), second, atol=0.2)

        # A blended chain is consumed as one motion, and the indices it
        # swallowed leave the queue with it rather than lingering as owed work.
        assert client.pause() == 1
        corner = list(second)
        corner[0] -= 6
        blended = client.move_j(corner, duration=1, r=15, wait=False)
        tail = client.move_j(start, duration=1, wait=False)
        _wait(
            lambda: len(client.queue() or []) >= 2,
            "the paused queue never listed the blend chain",
        )
        assert client.resume() == 1
        assert client.wait_command(tail, timeout=20)
        _wait(
            lambda: client.queue() == [],
            "the blend's consumed indices stayed in the queue",
        )
        # The blended command completed with the chain that swallowed it.
        assert blended >= 0 and client.wait_command(blended, timeout=5)

        # With one command executing, the queue reports what is owed after it:
        # the executing index is reported in its own field and listing it again
        # would double-count the motion the arm is already making.
        client.move_j(first, duration=3, wait=False)
        trailing = client.move_j(second, duration=1, wait=False)
        _wait(
            lambda: abs((client.angles() or start)[0] - start[0]) > 0.5,
            "the first move never started",
        )
        listed = client.queue()
        assert listed is not None and len(listed) == 1, (
            f"the executing command is listed as owed work as well: {listed}"
        )
        assert client.wait_command(trailing, timeout=25)

        # Stop clears what was owed, and the readback says so immediately.
        assert client.pause() == 1
        client.move_j(first, duration=2, wait=False)
        client.move_j(second, duration=2, wait=False)
        _wait(
            lambda: len(client.queue() or []) >= 2,
            "the paused queue never listed the commands a Stop must clear",
        )
        assert client.stop() == 1
        _wait(lambda: client.queue() == [], "Stop left work in the queue")
        assert not client.execution_speed().paused, (
            "Stop drops the pause with the queue it was holding"
        )
    finally:
        client.stop()
        client.resume()
        client.set_execution_speed(1)
