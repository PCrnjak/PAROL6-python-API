"""What a client reads while the arm moves: ``wait_motion()`` returns only
once everything queued before it has run and the arm has come to rest, and
the TCP speed reads zero once the serial frames it is measured from stop."""

import itertools
import socket
import time

import numpy as np
import pytest

from parol6 import RobotClient
from parol6.config import HOME_ANGLES_DEG, INTERVAL_S
from parol6.protocol.wire import (
    JogJCmd,
    ResponseMsg,
    TcpSpeedCmd,
    TcpSpeedResultStruct,
    decode_message,
    encode_command,
)
from tests.integration.controller_loop import address, push, ready, tick

pytestmark = pytest.mark.integration


def test_wait_motion_returns_once_the_queue_ahead_of_it_has_run(
    client: RobotClient, server_proc
):
    """A move queued behind a delay has not started while the arm stands
    still through the delay: the wait covers the move, not the stillness.
    A queue that a stop discarded ends the wait instead of holding it."""
    start = client.angles()
    assert start is not None
    there = [start[0] - 15.0, *start[1:]]

    assert client.delay(2.0) >= 0
    assert client.move_j(there, duration=1.0, wait=False) >= 0
    assert client.wait_motion(timeout=10.0)
    here = client.angles()
    assert here is not None
    assert np.allclose(here, there, atol=0.5), (
        f"wait_motion returned with the arm at {here}, before the move queued "
        f"behind the delay reached {there}"
    )

    assert client.delay(30.0) >= 0
    assert client.stop() == 1
    assert client.wait_motion(timeout=5.0), "a discarded delay held the wait"


def _tcp_speed(controller, state, sock: socket.socket, req_id: int) -> float:
    """The controller's answer to a TCP_SPEED query, ticking until it comes
    (for up to two seconds: loopback can deliver it some ticks late)."""
    sock.sendto(encode_command(TcpSpeedCmd(), req_id), address(controller))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        tick(controller, state)
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            continue
        reply = decode_message(data)
        if isinstance(reply, ResponseMsg) and reply.req_id == req_id:
            assert isinstance(reply.result, TcpSpeedResultStruct), reply
            return reply.result.speed
    pytest.fail("no reply to the TCP speed query")


def test_tcp_speed_reads_zero_once_the_serial_frames_stop(controller, monkeypatch):
    """A serial dropout mid-move leaves nothing to measure the TCP by: its
    speed reads zero, not the speed it had when the frames stopped."""
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True, at_deg=[float(v) for v in HOME_ANGLES_DEG])
    req_ids = itertools.count(1)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        push(
            controller,
            sock,
            JogJCmd(speeds=[-0.5, 0.0, 0.0, 0.0, 0.0, 0.0], duration=5.0),
        )
        moving = 0.0
        for _ in range(50):
            moving = _tcp_speed(controller, state, sock, next(req_ids))
            if moving > 1.0:
                break
        assert moving > 1.0, f"the jog never moved the TCP ({moving:.2f} mm/s)"

        monkeypatch.setattr(
            controller._transport_mgr, "get_latest_frame", lambda: (None, 0, 0.0)
        )
        deadline = time.monotonic() + 1.0
        speed = _tcp_speed(controller, state, sock, next(req_ids))
        while speed != 0.0 and time.monotonic() < deadline:
            time.sleep(INTERVAL_S)
            speed = _tcp_speed(controller, state, sock, next(req_ids))
        assert speed == 0.0, (
            f"a second into a serial dropout the TCP still read {speed:.1f} mm/s, "
            f"the speed it had when the frames stopped ({moving:.1f} mm/s)"
        )
