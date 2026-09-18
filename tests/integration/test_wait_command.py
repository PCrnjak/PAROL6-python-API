"""wait_command is paced by the status stream and still completes without it."""

import pytest

from parol6 import RobotClient, config as cfg
from tests.conftest import free_udp_port

pytestmark = pytest.mark.integration


def test_wait_command_completes_without_status_frames(ports, server_proc, monkeypatch):
    monkeypatch.setattr(cfg, "MCAST_PORT", free_udp_port())
    with RobotClient(host=ports.server_ip, port=ports.server_port, timeout=5.0) as deaf:
        assert not deaf.wait_status(lambda s: True, timeout=0.5), "frames still arrive"
        start = deaf.angles()
        assert start is not None
        target = list(start)
        target[0] += 5
        assert deaf.move_j(target, duration=0.5, wait=True, timeout=5.0) >= 0
        assert deaf.move_j(start, duration=0.5, wait=True, timeout=5.0) >= 0
