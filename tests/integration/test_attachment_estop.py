"""A hardware E-stop with a part attached: the E-stop owns the error while it
is pressed, and the stale-attachment error latches on the tick it is released.

Driven through the control loop's phases in loop order against the fake
serial, whose E-stop input the test presses and releases.
"""

import socket

import pytest

from parol6.protocol.wire import SetShapesCmd, ShapeWire, encode_command
from parol6.server.transports.mock_serial_transport import MockSerialTransport
from parol6.utils.error_codes import ErrorCode
from tests.integration.controller_loop import tick, tick_until
from waldoctl import Sphere

pytestmark = pytest.mark.integration


def test_estop_owns_the_error_until_release_then_the_attachment_latches(controller):
    state = controller.state_manager.get_state()
    robot = controller._transport_mgr.transport
    assert isinstance(robot, MockSerialTransport)
    state.Homed_in[:] = 1
    robot.sync_from_controller_state(state)
    tick_until(
        controller,
        state,
        lambda: state.enabled and all(state.Homed_in[:6]),
        "the fake serial never reported a referenced, enabled robot",
    )

    part = Sphere(name="part", radius=0.01).attach(
        flange_pose=(0.0, 0.0, 0.25, 0.0, 0.0, 0.0), epoch=state.attachment_epoch
    )
    assert controller.udp_transport is not None
    address = ("127.0.0.1", controller.udp_transport.socket.getsockname()[1])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(
            encode_command(SetShapesCmd(shapes=[ShapeWire(*part.to_wire())])), address
        )
        tick_until(
            controller,
            state,
            lambda: state.has_attachments,
            "the part was never attached",
        )

    robot.press_estop(True)
    tick_until(
        controller,
        state,
        lambda: controller.estop_active,
        "the E-stop press was never seen",
    )
    for _ in range(5):
        tick(controller, state)
        assert state.error is not None
        assert state.error.code == ErrorCode.SYS_ESTOP_ACTIVE, state.error
    assert not state.attachments_valid

    robot.press_estop(False)
    tick_until(
        controller,
        state,
        lambda: state.error is not None
        and state.error.code == ErrorCode.COMM_VALIDATION_ERROR,
        "the stale attachment never surfaced after the E-stop released",
    )
    assert state.error is not None and "attachment context" in state.error.cause
    assert not controller.estop_active and state.enabled
