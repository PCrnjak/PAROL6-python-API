"""Planned motion is refused until the robot is homed.

Before homing, reported joint positions are unreferenced (the boot state is
all-zeros steps — outside J2/J3's limits and physically impossible), so
planning or collision-checking a trajectory from them produces garbage:
found live as a phantom "[54] Self-collision predicted" on a first move
from the boot pose. Planned moves must instead be refused with an
actionable "not homed" error; jogging stays allowed (an unhomed arm may
need to be nudged clear of something before it can home), and ``home``
itself is the way out.
"""

import socket

import pytest

from parol6 import RobotClient
from parol6.protocol.wire import (
    HomeCmd,
    JogJCmd,
    MoveJCmd,
    OkMsg,
    encode_command,
)
from parol6.utils.error_codes import ErrorCode
from tests.integration.controller_loop import address, ready, send, tick_for, tick_until

pytestmark = pytest.mark.integration


def test_planned_motion_refused_until_homed(controller):
    """move_j from the unhomed boot state is refused with MOTN_NOT_HOMED (not
    a garbage collision prediction); after homing the same move is accepted.
    Jog remains available while unhomed."""
    state = controller.state_manager.get_state()
    # The boot state: nothing referenced, all-zero steps — exactly how a
    # controller starts. Nothing a client sends can un-home a simulator.
    ready(controller, state, homed=False)
    controller._planner.start()
    target = [90.0, -90.0, 180.0, 0.0, 0.0, 170.0]

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        queued = send(controller, state, sock, MoveJCmd(angles=target, duration=1.5), 1)
        assert isinstance(queued, OkMsg), queued
        tick_for(
            controller,
            state,
            lambda: state.error is not None,
            "the unhomed move was never refused",
            seconds=60.0,
        )
        assert state.error is not None
        assert state.error.code == int(ErrorCode.MOTN_NOT_HOMED), state.error

        # Jogging an unhomed robot stays allowed — no planning involved. A
        # jog datagram is not acknowledged; the arm moving is the answer.
        sock.sendto(
            encode_command(
                JogJCmd(speeds=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.0), 2
            ),
            address(controller),
        )
        tick_until(
            controller,
            state,
            lambda: state.Position_in[0] != 0,
            "the unhomed jog never moved the arm",
            ticks=300,
        )

        # Homing establishes references; the identical move now proceeds.
        homing = send(controller, state, sock, HomeCmd(), 3)
        assert isinstance(homing, OkMsg) and homing.index is not None, homing
        homed = homing.index
        tick_for(
            controller,
            state,
            lambda: all(state.Homed_in[:6]) and state.command_completed(homed),
            "homing never referenced the robot",
            seconds=60.0,
        )
        accepted = send(
            controller, state, sock, MoveJCmd(angles=target, duration=1.5), 4
        )
        assert isinstance(accepted, OkMsg), accepted
        assert accepted.index is not None
        moved = accepted.index
        tick_for(
            controller,
            state,
            lambda: state.command_completed(moved) or state.error is not None,
            "the move after homing never completed",
            seconds=60.0,
        )
        assert state.error is None, state.error


def test_home_calibrate_rereferences_homed_robot(client: RobotClient, server_proc):
    """home(calibrate=True) runs the real referencing sequence even when the
    robot is already homed — the firmware drops the homed bits while it seeks
    the end stops, which a substituted planned return move never does — and
    leaves the robot referenced at standby."""
    idx = client.home(calibrate=True)
    assert idx >= 0
    assert client.wait_status(lambda s: not s.homed, timeout=5.0)
    # Progress is published while the firmware seeks the end stops...
    assert client.wait_status(
        lambda s: (
            bool(s.homing.get("active"))
            and any(state.name == "SEEKING" for state, _ in s.homing["joints"])
        ),
        timeout=5.0,
    )
    assert client.wait_command(idx, timeout=30.0)
    assert client.wait_status(lambda s: s.homed, timeout=2.0)
    # ...and the view is empty again once referencing completes.
    assert client.wait_status(lambda s: not s.homing, timeout=2.0)
