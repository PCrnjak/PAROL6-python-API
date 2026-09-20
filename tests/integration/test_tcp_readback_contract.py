"""The TCP readbacks must not answer for a controller that did not reply.

``[0, 0, 0]`` is a legitimate offset -- a tool deliberately cleared -- so a
caller handed one as a not-answered sentinel cannot tell "the offset is zero"
from "there is no controller", and a host that adopts the readback quietly
erases the offset the user just set. The same holds for ``tcp_transform`` and
an identity transform. waldoctl's ``RobotClient`` states both in as many words;
par6 carries the guarantee (par6#80) and this is the parol6 side of it.
"""

import socket
from math import isfinite

import pytest

from parol6 import RobotClient


def _unserved_udp_port() -> int:
    """A port nothing is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _apply_offset(client, x: float, y: float, z: float) -> None:
    """Set the TCP offset and wait for it: SET_TCP_OFFSET lands in queue order."""
    index = client.set_tcp_offset(x, y, z)
    assert index >= 0, f"set_tcp_offset was not accepted (got {index})"
    assert client.wait_command(index, timeout=10.0), "set_tcp_offset did not complete"


@pytest.mark.integration
def test_tcp_readbacks_separate_a_real_zero_from_a_silent_controller(client):
    """A deliberate zero reads back as a value; an unanswered readback raises."""
    _apply_offset(client, 0.0, 0.0, -25.0)
    assert client.tcp_offset() == pytest.approx([0.0, 0.0, -25.0])

    # Clearing back to zero answers a value, and the readback moved to prove it.
    _apply_offset(client, 0.0, 0.0, 0.0)
    assert client.tcp_offset() == pytest.approx([0.0, 0.0, 0.0])

    transform = client.tcp_transform()
    assert len(transform) == 6
    assert all(isfinite(v) for v in transform)

    with RobotClient(
        host="127.0.0.1", port=_unserved_udp_port(), timeout=0.2, retries=1
    ) as mute:
        with pytest.raises(TimeoutError):
            mute.tcp_offset()
        with pytest.raises(TimeoutError):
            mute.tcp_transform()
