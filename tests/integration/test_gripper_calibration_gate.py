"""A jaw move needs a calibration the gripper may only get from the action
queued ahead of it, so the gate is judged when the move's turn comes:
``calibrate(); move()`` runs in order, and a move on a gripper that was
never calibrated fails — as a completion, since the ack came before its
turn — instead of driving uncalibrated jaws."""

import socket

import pytest

from parol6.protocol.wire import OkMsg, ToolActionCmd
from parol6.utils.error_codes import ErrorCode
from tests.integration.controller_loop import ready, send, tick_until

pytestmark = pytest.mark.integration


def test_a_jaw_move_waits_for_the_calibrate_ahead_of_it(controller):
    state = controller.state_manager.get_state()
    ready(controller, state, homed=True)
    state.set_tool("SSG-48")
    move = ToolActionCmd(tool_key="SSG-48", action="move", params=[0.5, 0.5, 600])

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        never = send(controller, state, sock, move, 1)
        assert isinstance(never, OkMsg) and never.index is not None, never
        tick_until(
            controller,
            state,
            lambda: state.command_failure(never.index) is not None,
            "the move on a never-calibrated gripper was not failed",
        )
        failure = state.command_failure(never.index)
        assert failure is not None
        assert failure.code == int(ErrorCode.COMM_VALIDATION_ERROR)
        assert "not calibrated" in failure.cause
        assert state.gripper_hw.feedback_position == 0, "the jaws must not have moved"

        calibrate = send(
            controller,
            state,
            sock,
            ToolActionCmd(tool_key="SSG-48", action="calibrate", params=[]),
            2,
        )
        moved = send(controller, state, sock, move, 3)
        assert isinstance(calibrate, OkMsg) and calibrate.index is not None
        assert isinstance(moved, OkMsg) and moved.index is not None
        tick_until(
            controller,
            state,
            lambda: state.command_completed(moved.index),
            "the move queued behind the calibrate never completed",
            ticks=2000,
        )
        assert state.command_completed(calibrate.index)
        assert state.command_failure(moved.index) is None
        assert abs(state.gripper_hw.feedback_position / 255.0 - 0.5) < 0.05
