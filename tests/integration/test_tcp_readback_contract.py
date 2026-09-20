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


@pytest.mark.integration
def test_tcp_readbacks_separate_a_real_zero_from_a_silent_controller(client):
    """A deliberate zero reads back as a value; an unanswered readback raises."""
    assert client.set_tcp_offset(0.0, 0.0, 0.0) == 1

    assert [float(v) for v in client.tcp_offset()] == [0.0, 0.0, 0.0]

    transform = [float(v) for v in client.tcp_transform()]
    assert len(transform) == 6
    assert all(isfinite(v) for v in transform)

    with RobotClient(
        host="127.0.0.1", port=_unserved_udp_port(), timeout=0.2, retries=1
    ) as mute:
        with pytest.raises(TimeoutError):
            mute.tcp_offset()
        with pytest.raises(TimeoutError):
            mute.tcp_transform()
