"""Regression test: a SystemCommand's Command_out must survive to the write phase.

RESET sets ``Command_out = ENABLE`` during poll_cmd so the firmware can clear
a latched ``disabled``. ``_execute_commands()`` runs later the same tick, and
its "nothing active" fallback used to overwrite Command_out back to IDLE
before ``_write_to_firmware()`` ran — so the ENABLE never reached the wire and
the robot stayed firmware-disabled after every RESET.
"""

import socket
import time

import pytest

from parol6.protocol.wire import CommandCode, EstopCmd, ResetCmd, encode_command
from parol6.server.controller import Controller, ControllerConfig


@pytest.fixture
def controller():
    ctl = Controller(ControllerConfig(udp_host="127.0.0.1", udp_port=0))
    try:
        yield ctl
    finally:
        if ctl.udp_transport is not None:
            ctl.udp_transport.close_socket()
        if ctl._status_broadcaster is not None:
            ctl._status_broadcaster.close()
        ctl._transport_mgr.disconnect()
        ctl.state_manager.reset_state()


def test_reset_enable_reaches_firmware_write(controller):
    """ESTOP then RESET over real UDP: the tick that dispatches RESET must
    hand ENABLE to the transport write, not the executor fallback's IDLE."""
    state = controller.state_manager.get_state()
    assert controller.udp_transport is not None
    port = controller.udp_transport.socket.getsockname()[1]

    written: list[int] = []
    real_write = controller._transport_mgr.write_frame

    def tap(position_out, speed_out, command_out, *args):
        written.append(command_out)
        return real_write(position_out, speed_out, command_out, *args)

    controller._transport_mgr.write_frame = tap

    def tick() -> None:
        # The phases of _main_control_loop this bug lives in, in loop order.
        # _handle_estop is omitted: it gates on serial-read state, and the
        # protective-stop scenario has estop_active False, so execute runs.
        controller._poll_commands(state)
        controller._execute_commands(state)
        controller._write_to_firmware(state)

    def tick_until(condition, message: str) -> None:
        for _ in range(200):
            tick()
            if condition():
                return
            time.sleep(0.005)
        pytest.fail(message)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(encode_command(EstopCmd()), ("127.0.0.1", port))
        tick_until(lambda: not state.enabled, "ESTOP was never dispatched")

        sock.sendto(encode_command(ResetCmd()), ("127.0.0.1", port))
        tick_until(lambda: state.enabled, "RESET was never dispatched")
    finally:
        sock.close()

    assert CommandCode.ENABLE.value in written, (
        "RESET's ENABLE never reached the firmware write phase — "
        f"written command codes: {sorted(set(written))}"
    )


def test_streaming_packet_keeps_following_configuration_and_stop(controller):
    from parol6.config import MAX_POLL_COUNT
    from parol6.protocol.wire import PingCmd, SelectProfileCmd, TeleportCmd

    state = controller.state_manager.get_state()
    assert controller.udp_transport is not None
    address = ("127.0.0.1", controller.udp_transport.socket.getsockname()[1])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(
            encode_command(TeleportCmd(angles=[90, -90, 180, 0, 0, 180])), address
        )
        for _ in range(MAX_POLL_COUNT - 1):
            sender.sendto(encode_command(PingCmd()), address)
        sender.sendto(encode_command(SelectProfileCmd(profile="QUINTIC")), address)
        sender.sendto(encode_command(EstopCmd()), address)
        for _ in range(2):
            controller._poll_commands(state)
            controller._execute_commands(state)

    assert state.motion_profile == "QUINTIC", (
        "streaming must not discard pending configuration"
    )
    assert not state.enabled, "a queued stop must survive a streaming batch boundary"


def test_a_streaming_flood_is_consumed_in_the_tick_it_arrived_in(controller):
    """A client streaming faster than one tick's batch leaves a backlog in the
    socket. It is stale the moment the tick runs, so the tick has to consume
    it: otherwise the arm follows superseded targets for backlog/batch ticks
    and anything queued behind them — here a stop — waits just as long."""
    from parol6.config import MAX_POLL_COUNT
    from parol6.protocol.wire import JogJCmd

    state = controller.state_manager.get_state()
    assert controller.udp_transport is not None
    address = ("127.0.0.1", controller.udp_transport.socket.getsockname()[1])
    flood = MAX_POLL_COUNT * 4
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for _ in range(flood):
            sender.sendto(
                encode_command(
                    JogJCmd(speeds=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], duration=0.2)
                ),
                address,
            )
        sender.sendto(encode_command(EstopCmd()), address)
        controller._poll_commands(state)
        controller._execute_commands(state)

    assert not state.enabled, (
        f"one tick left {flood} streamed targets unread, so the stop behind "
        f"them was not seen either"
    )
