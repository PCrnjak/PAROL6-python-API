"""A teleport is a system command on the simulator: the controller answers
it once the pose is applied, the pose is exact so an unreferenced arm reads
referenced afterwards, whatever was driving the arm stops there, and what
cannot be applied is refused before anything moves."""

import socket

import numpy as np
import pytest

from parol6.config import steps_to_deg
from parol6.protocol.wire import ErrorMsg, JogJCmd, OkMsg, TeleportCmd
from parol6.utils.error_catalog import RobotError
from parol6.utils.error_codes import ErrorCode
from tests.integration.controller_loop import push, ready, send, tick, tick_until

pytestmark = pytest.mark.integration


def _angles_deg(state) -> np.ndarray:
    out = np.zeros(6, dtype=np.float64)
    steps_to_deg(state.Position_in, out)
    return out


def test_a_teleport_is_acked_once_applied_and_references_the_arm(controller):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=False)
    target = [10.0, -80.0, 170.0, 5.0, -10.0, 175.0]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        reply = send(controller, state, sock, TeleportCmd(angles=target), 1)
        assert isinstance(reply, OkMsg), reply
        tick_until(
            controller,
            state,
            lambda: bool(state.Homed_in[:6].all()),
            "the teleported arm never read referenced",
        )
        assert np.allclose(_angles_deg(state), target, atol=0.05)

        # A jog is driving the arm; a teleport ends it where it lands.
        push(
            controller,
            sock,
            JogJCmd(speeds=[0.0, 0.0, 0.0, 0.0, 0.0, -0.8], duration=5.0),
        )
        tick_until(
            controller,
            state,
            lambda: _angles_deg(state)[5] < 174.0,
            "the jog never moved the wrist",
        )
        reply = send(controller, state, sock, TeleportCmd(angles=target), 2)
        assert isinstance(reply, OkMsg), reply
        for _ in range(30):
            tick(controller, state)
        assert np.allclose(_angles_deg(state), target, atol=0.05), (
            "the jog carried on after the teleport"
        )

        # Tool positions for a tool that is not fitted are refused, and the
        # arm stays where it is.
        reply = send(
            controller,
            state,
            sock,
            TeleportCmd(
                angles=[0.0, -90.0, 180.0, 0.0, 0.0, 180.0], tool_positions=[0.5]
            ),
            3,
        )
        assert isinstance(reply, ErrorMsg), reply
        assert RobotError.from_wire(reply.message).code == int(
            ErrorCode.COMM_VALIDATION_ERROR
        )
        for _ in range(5):
            tick(controller, state)
        assert np.allclose(_angles_deg(state), target, atol=0.05)

    # What the hard limits and [0, 1] exclude never reaches the wire.
    with pytest.raises(ValueError):
        TeleportCmd(angles=[10.0, -80.0, 170.0, 5.0, -10.0, 1000.0])
    with pytest.raises(ValueError):
        TeleportCmd(angles=[float("nan"), -80.0, 170.0, 5.0, -10.0, 175.0])
    with pytest.raises(ValueError):
        TeleportCmd(angles=target, tool_positions=[1.5])
