"""Drive an in-process Controller (the ``controller`` fixture) through the
loop's phases in loop order against the fake serial, and talk to it over
its real UDP socket."""

import socket
import time
from types import SimpleNamespace

import numpy as np
import pytest

import parol6.commands.base as command_base
from parol6.config import INTERVAL_S, deg_to_steps
from parol6.protocol.wire import (
    ErrorMsg,
    OkMsg,
    PingCmd,
    decode_message,
    encode_command,
)
from parol6.server.controller import Controller


def tick(controller: Controller, state) -> None:
    controller._read_from_firmware(state)
    controller._check_attachments(state)
    controller._poll_commands(state)
    controller._handle_estop(state)
    controller._check_attachments(state)
    if not controller.estop_active:
        controller._execute_commands(state)
    controller._write_to_firmware(state)
    controller._transport_mgr.tick_simulation(state.current_tool, tool_teleport_pos=-1)


def tick_until(controller: Controller, state, condition, message: str, ticks=50):
    for _ in range(ticks):
        tick(controller, state)
        if condition():
            return
    pytest.fail(message)


def tick_for(controller: Controller, state, condition, message: str, seconds: float):
    """Tick at the control rate until *condition* holds, for up to *seconds*
    of wall time (a cold planner JITs its motion pipeline)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        tick(controller, state)
        if condition():
            return
        time.sleep(INTERVAL_S)
    pytest.fail(message)


class VirtualClock:
    """The clock the commands' timers read (a jog's duration, a stream's
    grace), advanced one control interval per :meth:`tick`. The timers then
    count ticks, as they do on a loop that holds its rate, whatever the
    machine running the test does to its sleeps."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 0.0
        monkeypatch.setattr(
            command_base, "time", SimpleNamespace(perf_counter=lambda: self.now)
        )

    def tick(self, controller: Controller, state) -> None:
        tick(controller, state)
        self.now += INTERVAL_S


def drain(controller: Controller, state, sock: socket.socket, req_id: int) -> None:
    """Tick until the controller answers a ping sent now. Datagrams from one
    socket to one address are read in the order they were sent, so every
    one sent before the ping has been read by then, however late it was
    delivered."""
    sock.sendto(encode_command(PingCmd(), req_id), address(controller))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        tick(controller, state)
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            continue
        if getattr(decode_message(data), "req_id", None) == req_id:
            return
    pytest.fail("the controller never answered the ping")


def address(controller: Controller) -> tuple[str, int]:
    assert controller.udp_transport is not None
    return ("127.0.0.1", controller.udp_transport.socket.getsockname()[1])


def push(controller: Controller, sock: socket.socket, cmd, req_id: int = 0) -> None:
    """Send a fire-and-forget datagram (jog, servo): nothing answers it."""
    sock.sendto(encode_command(cmd, req_id), address(controller))


def send(controller: Controller, state, sock: socket.socket, cmd, req_id: int):
    """Send *cmd* to the controller, tick until it answers, and return the
    decoded reply."""
    sock.sendto(encode_command(cmd, req_id), address(controller))
    for _ in range(50):
        tick(controller, state)
        try:
            data, _ = sock.recvfrom(4096)
        except BlockingIOError:
            continue
        reply = decode_message(data)
        if isinstance(reply, (OkMsg, ErrorMsg)) and reply.req_id == req_id:
            return reply
    pytest.fail(f"no reply to {type(cmd).__name__}")


def ready(
    controller: Controller, state, *, homed: bool, at_deg: list[float] | None = None
) -> None:
    """Bring the fake serial up enabled, referenced or in the boot state,
    at ``at_deg`` when given."""
    from parol6.server.transports.mock_serial_transport import MockSerialTransport

    robot = controller._transport_mgr.transport
    assert isinstance(robot, MockSerialTransport)
    state.Homed_in[:] = 1 if homed else 0
    if at_deg is not None:
        deg_to_steps(np.asarray(at_deg, dtype=np.float64), state.Position_in)
    elif not homed:
        state.Position_in[:] = 0
    robot.sync_from_controller_state(state)
    tick_until(
        controller,
        state,
        lambda: state.enabled and (all(state.Homed_in[:6]) == homed),
        "the fake serial never reported an enabled robot",
    )
    # The frame read on that tick predates the sync; the next tick
    # produces one from the synced state and the one after reads it.
    tick(controller, state)
    tick(controller, state)
