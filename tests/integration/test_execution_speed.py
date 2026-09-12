"""Queued execution controls through the client and simulated controller."""

import math
import time

import numpy as np
import pytest

from parol6 import RobotClient


def test_execution_pause_speed_dwell_and_standalone_deadlines(client: RobotClient):
    start = client.angles()
    assert start is not None
    target = list(start)
    target[0] += 8
    try:
        for invalid in [0, -1, 0.09, 1.01, 2, True, math.nan, math.inf, -math.inf]:
            with pytest.raises(ValueError):
                client.set_execution_speed(invalid)
        assert client.pause() == 1
        index = client.move_j(target, duration=2, wait=False)
        before = time.monotonic()
        assert not client.wait_command(index, timeout=0.3)
        assert time.monotonic() - before < 1
        assert client.set_execution_speed(0.5) == 1
        state = client.execution_speed()
        assert state.paused and state.resume_scale == 0.5
        assert not client.wait_command(index, timeout=0.3)
        assert np.allclose(client.angles(), start, atol=0.05)

        assert client.resume() == 1
        assert client.wait_status(lambda s: s.angles[0] > start[0] + 1, timeout=10)
        assert client.pause() == 1
        deadline = time.monotonic() + 5
        while not client.execution_speed().paused:
            assert time.monotonic() < deadline, "pause never reached a hold"
            time.sleep(0.02)
        assert not client.wait_command(index, timeout=0.3)
        held = client.angles()
        assert client.set_execution_speed(0.6) == 1
        assert client.execution_speed().paused
        assert not client.wait_command(index, timeout=0.3)
        assert np.allclose(client.angles(), held, atol=0.05)
        assert client.resume() == 1
        assert client.wait_command(index, timeout=10)
        assert np.allclose(client.angles(), target, atol=0.1)

        index = client.delay(1)
        assert client.wait_status(lambda s: s.executing_index == index, timeout=3)
        assert client.pause() == 1
        assert not client.wait_command(index, timeout=1.3)
        assert client.ping() is not None
        assert client.resume() == 1
        assert client.wait_command(index, timeout=3)

        assert client.pause() == 1
        with pytest.raises(TimeoutError):
            client.move_j(start, duration=1, wait=True, timeout=0.2)
        # Stop discards the queue the pause was holding, and the pause with
        # it: the next queued command runs without a resume.
        assert client.stop() == 1
        assert client.queue() == []
        assert not client.execution_speed().paused
        assert client.move_j(start, duration=1, wait=True, timeout=5) >= 0
        assert np.allclose(client.angles(), start, atol=0.1)
        assert client.pause() == 1
        assert client.reset_state() == 1
        assert not client.execution_speed().paused
        assert client.resume() == 1
    finally:
        client.stop()
        client.resume()
        client.set_execution_speed(1)
