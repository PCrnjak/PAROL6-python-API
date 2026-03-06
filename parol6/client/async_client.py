"""
Async UDP client for PAROL6 robot control.
"""

import asyncio
import contextlib
import logging
import random
import socket
import struct
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Literal, cast, overload

import msgspec
import numpy as np
from waldoctl import RobotClient as _RobotClientABC, ToolStatus
from waldoctl.tools import ToolSpec

from .. import config as cfg
from ..ack_policy import QUERY_CMD_TYPES, SYSTEM_CMD_TYPES, AckPolicy
from ..utils.error_catalog import RobotError
from ..utils.errors import MotionError
from ..protocol.wire import (
    STRUCT_TO_CMDTYPE,
    AnglesResultStruct,
    decode_status_bin_into,
    CheckpointCmd,
    CurrentActionResultStruct,
    DelayCmd,
    EnablementResultStruct,
    ErrorResultStruct,
    ErrorMsg,
    IOResultStruct,
    GetAnglesCmd,
    GetCurrentActionCmd,
    GetEnablementCmd,
    GetErrorCmd,
    GetIOCmd,
    GetLoopStatsCmd,
    GetPoseCmd,
    GetProfileCmd,
    GetQueueCmd,
    GetSpeedsCmd,
    GetStatusCmd,
    GetTcpSpeedCmd,
    GetToolCmd,
    GetToolStatusCmd,
    HaltCmd,
    HomeCmd,
    JogJCmd,
    JogLCmd,
    LoopStatsResultStruct,
    MoveCCmd,
    MoveJCmd,
    MoveJPoseCmd,
    MoveLCmd,
    MovePCmd,
    MoveSCmd,
    OkMsg,
    PingCmd,
    PingResultStruct,
    PoseResultStruct,
    ProfileResultStruct,
    QueueResultStruct,
    ResetCmd,
    ResetLoopStatsCmd,
    ResumeCmd,
    Response,
    ResponseMsg,
    ServoJCmd,
    ServoJPoseCmd,
    ServoLCmd,
    SetIOCmd,
    SetPortCmd,
    SetProfileCmd,
    SetToolCmd,
    SimulatorCmd,
    SpeedsResultStruct,
    StatusBuffer,
    StatusResultStruct,
    TcpSpeedResultStruct,
    ToolActionCmd,
    ToolResultStruct,
    ToolStatusResultStruct,
    decode_message,
    encode_command,
    encode_command_into,
)
from ..protocol.types import Axis, Frame
from waldoctl import PingResult
from pinokin import so3_rpy

logger = logging.getLogger(__name__)

_AXIS_MAP: dict[str, int] = {"X": 0, "Y": 1, "Z": 2, "RX": 3, "RY": 4, "RZ": 5}


class _UDPClientProtocol(asyncio.DatagramProtocol):
    def __init__(self, rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]]):
        self.rx_queue = rx_queue
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            self.rx_queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            pass  # Drop packet when queue is full (expected under load)

    def error_received(self, exc: Exception) -> None:
        pass

    def connection_lost(self, exc: Exception | None) -> None:
        pass


def _create_multicast_socket(group: str, port: int, iface_ip: str) -> socket.socket:
    """Create and configure a multicast socket with loopback-first semantics."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple listeners on same port
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

    # Bind to port
    try:
        sock.bind(("", port))
    except OSError:
        sock.bind((iface_ip, port))

    # Helper to detect primary NIC IP
    def _detect_primary_ip() -> str:
        tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            tmp.connect(("1.1.1.1", 80))
            return tmp.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            with contextlib.suppress(Exception):
                tmp.close()

    # Join multicast group with fallbacks
    try:
        mreq = struct.pack("=4s4s", socket.inet_aton(group), socket.inet_aton(iface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except Exception:
        try:
            primary_ip = _detect_primary_ip()
            mreq = struct.pack(
                "=4s4s", socket.inet_aton(group), socket.inet_aton(primary_ip)
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            mreq_any = struct.pack("=4sl", socket.inet_aton(group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq_any)

    sock.setblocking(False)
    return sock


def _create_unicast_socket(port: int, host: str) -> socket.socket:
    """Create and configure a plain UDP socket for unicast reception."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((host, port))
    except OSError:
        sock.bind(("", port))
    sock.setblocking(False)
    return sock


if TYPE_CHECKING:
    from typing import Protocol

    class _StatusNotifier(Protocol):
        _shared_status: StatusBuffer
        _status_generation: int
        _status_event: asyncio.Event
        _closed: bool


class _StatusProtocol(asyncio.DatagramProtocol):
    """Protocol handler for status datagrams - decodes directly into shared buffer."""

    def __init__(self, client: "_StatusNotifier"):
        self._client = client
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._client._closed:
            return
        # Zero-allocation decode directly into shared buffer
        if decode_status_bin_into(data, self._client._shared_status):
            self._client._status_generation += 1
            # Event.set() is synchronous and wakes all waiters
            self._client._status_event.set()

    def error_received(self, exc: Exception) -> None:
        pass

    def connection_lost(self, exc: Exception | None) -> None:
        pass


class AsyncRobotClient(_RobotClientABC):
    """
    Async UDP client for the PAROL6 headless controller.

    Motion/control commands: fire-and-forget via UDP
    Query commands: request/response with timeout and simple retry
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 1.0,
        retries: int = 1,
    ) -> None:
        # Endpoint configuration (host/port immutable after endpoint creation)
        self._host = host
        self._port = port
        self.timeout = timeout
        self.retries = retries

        # Pre-allocated buffers for get_pose_rpy
        self._R_buf = np.zeros((3, 3), dtype=np.float64)
        self._rpy_buf = np.zeros(3, dtype=np.float64)

        # Pre-allocated TX buffer for fire-and-forget command encoding
        self._tx_buf = bytearray(256)

        # Persistent asyncio datagram endpoint
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _UDPClientProtocol | None = None
        self._rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]] = asyncio.Queue(
            maxsize=256
        )
        self._ep_lock = asyncio.Lock()

        # Serialize request/response
        self._req_lock = asyncio.Lock()

        # ACK policy (category-based, no stream_mode dependency)
        self._ack_policy = AckPolicy()

        # Shared status state (single buffer, event-based notification)
        self._status_transport: asyncio.DatagramTransport | None = None
        self._status_sock: socket.socket | None = None
        self._shared_status: StatusBuffer = StatusBuffer()
        self._status_generation: int = 0
        self._status_event: asyncio.Event = asyncio.Event()

        # Last command index returned by server for queued commands
        self._last_command_index: int | None = None

        # Active tool key (set by set_tool)
        self._active_tool_key: str | None = None

        # Bound tool specs (populated by Robot.create_async_client)
        self._bound_tools: dict[str, ToolSpec] = {}

        # Lifecycle flag
        self._closed: bool = False

    # --------------- Tool access ---------------

    @property
    def tool(self) -> ToolSpec:
        """Active bound tool. Raises if no tool has been set."""
        key = (self._active_tool_key or "").upper()
        if not key:
            raise RuntimeError("No tool set. Call set_tool() first.")
        return self._bound_tools[key]

    # --------------- Endpoint configuration properties ---------------

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        if self._transport is not None:
            raise RuntimeError(
                "AsyncRobotClient.host is read-only after endpoint creation"
            )
        self._host = value

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        if self._transport is not None:
            raise RuntimeError(
                "AsyncRobotClient.port is read-only after endpoint creation"
            )
        self._port = value

    # --------------- Internal helpers ---------------

    async def _ensure_endpoint(self) -> None:
        """Lazily create a persistent asyncio UDP datagram endpoint."""
        if self._closed:
            raise RuntimeError("AsyncRobotClient is closed")
        if self._transport is not None:
            return
        async with self._ep_lock:
            if self._closed:
                raise RuntimeError("AsyncRobotClient is closed")
            if self._transport is not None:
                return
            loop = asyncio.get_running_loop()
            self._rx_queue = asyncio.Queue(maxsize=256)
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _UDPClientProtocol(self._rx_queue),
                remote_addr=(self.host, self.port),
            )
            self._transport = transport
            self._protocol = protocol
            logger.info(
                f"AsyncRobotClient UDP endpoint: remote={self.host}:{self.port}, timeout={self.timeout}, retries={self.retries}"
            )

            # Start multicast listener
            await self._start_multicast_listener()

    async def _start_multicast_listener(self) -> None:
        """Start listening for multicast/unicast status broadcasts.

        Creates a UDP socket and protocol that decodes status datagrams directly
        into the shared buffer, notifying consumers via condition variable.
        """
        if self._status_transport is not None:
            return

        logger.info(
            f"Status subscriber config: transport={cfg.STATUS_TRANSPORT} group={cfg.MCAST_GROUP} port={cfg.MCAST_PORT} iface={cfg.MCAST_IF}"
        )
        # Quick readiness check (no blind wait): bounded by client's own timeout
        try:
            await self.wait_ready(
                timeout=min(1.0, float(self.timeout or 0.3)), interval=0.5
            )
        except Exception:
            pass

        # Create the socket based on configured transport
        if cfg.STATUS_TRANSPORT == "UNICAST":
            self._status_sock = _create_unicast_socket(
                cfg.MCAST_PORT, cfg.STATUS_UNICAST_HOST
            )
        else:
            try:
                self._status_sock = _create_multicast_socket(
                    cfg.MCAST_GROUP, cfg.MCAST_PORT, cfg.MCAST_IF
                )
            except OSError:
                logging.warning("Multicast socket failed, falling back to unicast")
                self._status_sock = _create_unicast_socket(
                    cfg.MCAST_PORT, cfg.STATUS_UNICAST_HOST
                )

        # Create the datagram endpoint with the status protocol
        loop = asyncio.get_running_loop()
        self._status_transport, _ = await loop.create_datagram_endpoint(
            lambda: _StatusProtocol(self),  # type: ignore[arg-type]
            sock=self._status_sock,
        )

    # --------------- Lifecycle / context management ---------------

    async def close(self) -> None:
        """Release UDP transport and background tasks.

        Safe to call multiple times.
        """
        if self._closed:
            return
        logging.debug("Closing Client...")
        self._closed = True

        # Wake all status_stream consumers
        self._status_event.set()

        # Close status transport - yield first to let pending I/O complete
        if self._status_transport is not None:
            with contextlib.suppress(Exception):
                await asyncio.sleep(0)
                self._status_transport.close()
            self._status_transport = None
        if self._status_sock is not None:
            with contextlib.suppress(Exception):
                self._status_sock.close()
            self._status_sock = None

        # Close UDP command transport
        if self._transport is not None:
            with contextlib.suppress(Exception):
                await asyncio.sleep(0)
                self._transport.close()
            self._transport = None
            self._protocol = None

        # Best-effort drain for RX queue to free memory
        with contextlib.suppress(Exception):
            while not self._rx_queue.empty():
                self._rx_queue.get_nowait()

    async def __aenter__(self) -> "AsyncRobotClient":
        if self._closed:
            raise RuntimeError("AsyncRobotClient is closed")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def status_stream(self) -> AsyncIterator[StatusBuffer]:
        """Async generator that yields status updates from multicast broadcasts.

        Usage:
            async for status in client.status_stream():
                print(f"Angles: {status.angles}")

        This generator terminates automatically when :meth:`close` is
        called on the client, so callers do not need to manually cancel
        their consumer tasks.

        Each yielded StatusBuffer is a copy - safe to store or process async.
        For zero-copy hot paths, use :meth:`status_stream_shared` instead.

        Slow consumers automatically skip to the latest state (desired for real-time).
        """
        async for status in self.status_stream_shared():
            yield status.copy()

    async def status_stream_shared(self) -> AsyncIterator[StatusBuffer]:
        """Zero-copy async generator that yields the shared status buffer.

        Usage:
            async for status in client.status_stream_shared():
                # Process immediately - don't store references
                print(f"Angles: {status.angles}")

        WARNING: The same buffer instance is yielded on every iteration.
        Do not store references to the yielded object - data will be
        overwritten on the next iteration. For safe storage, use
        :meth:`status_stream` or call status.copy().

        This generator terminates automatically when :meth:`close` is
        called on the client.

        Slow consumers automatically skip to the latest state (desired for real-time).
        """
        await self._ensure_endpoint()
        last_gen = 0

        while not self._closed:
            # Clear before waiting - only affects future waits, not current waiters
            self._status_event.clear()

            # Check if we already have new data (arrived between yield and clear)
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                yield self._shared_status
                continue

            # Wait for next update
            await self._status_event.wait()

            if self._closed:
                break
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                yield self._shared_status

    async def _send(self, cmd: msgspec.Struct) -> int:
        """
        Send a binary command based on AckPolicy.

        Returns:
            int (command index ≥ 0) for ACK'd queued commands, -1 on failure,
            bool for system/fire-and-forget/query commands.
        """
        await self._ensure_endpoint()
        assert self._transport is not None

        # Get command type from struct's tag
        cmd_type = STRUCT_TO_CMDTYPE.get(type(cmd))
        if cmd_type is None:
            return False

        # System commands: need stable bytes across await
        if cmd_type in SYSTEM_CMD_TYPES:
            try:
                await self._request_ok_raw(encode_command(cmd), self.timeout)
                return True
            except TimeoutError:
                return False

        # Motion and other non-query commands
        if cmd_type not in QUERY_CMD_TYPES:
            if self._ack_policy.requires_ack(cmd_type):
                try:
                    ok = await self._request_ok_raw(encode_command(cmd), self.timeout)
                    self._last_command_index = ok.index
                    return ok.index if ok.index is not None else 0
                except TimeoutError:
                    return -1
            # Fire-and-forget: reuse pre-allocated buffer (sendto copies)
            encode_command_into(cmd, self._tx_buf)
            self._transport.sendto(self._tx_buf)
            return True

        # Queries via _send: fire-and-forget
        encode_command_into(cmd, self._tx_buf)
        self._transport.sendto(self._tx_buf)
        return True

    async def _request(self, cmd: msgspec.Struct) -> Response | None:
        """Send a query command and wait for a typed response.

        Drains the receive queue until a ResponseMsg is found or timeout.
        Non-ResponseMsg datagrams (e.g. status broadcasts) are discarded.

        Args:
            cmd: Typed command struct

        Returns:
            Typed Response struct, or None on timeout.

        Raises:
            MotionError: If the server responds with an error.
        """
        await self._ensure_endpoint()
        assert self._transport is not None
        data = encode_command(cmd)
        for attempt in range(self.retries + 1):
            try:
                async with self._req_lock:
                    self._transport.sendto(data)
                    end_time = time.monotonic() + self.timeout
                    while time.monotonic() < end_time:
                        try:
                            resp_data, _ = await asyncio.wait_for(
                                self._rx_queue.get(),
                                timeout=max(0.0, end_time - time.monotonic()),
                            )
                            try:
                                parsed = decode_message(resp_data)
                                if isinstance(parsed, ResponseMsg):
                                    return parsed.result
                                if isinstance(parsed, ErrorMsg):
                                    raise MotionError(
                                        RobotError.from_wire(parsed.message)
                                    )
                            except MotionError:
                                raise
                            except Exception:
                                pass  # Ignore non-matching datagrams
                        except (asyncio.TimeoutError, TimeoutError):
                            break
            except MotionError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                pass
            except Exception:
                break
            if attempt < self.retries:
                backoff = min(0.5, 0.05 * (2**attempt)) + random.uniform(0, 0.05)
                await asyncio.sleep(backoff)
        return None

    async def _request_ok_raw(self, data: bytes, timeout: float) -> OkMsg:
        """
        Send pre-encoded binary command and wait for 'OK' or 'ERROR' reply.

        Args:
            data: Pre-encoded msgpack bytes
            timeout: Timeout in seconds.

        Returns OkMsg on OK; raises RuntimeError on ERROR, TimeoutError on timeout.
        """
        await self._ensure_endpoint()
        assert self._transport is not None

        end_time = time.monotonic() + timeout
        async with self._req_lock:
            self._transport.sendto(data)
            while time.monotonic() < end_time:
                try:
                    resp_data, _addr = await asyncio.wait_for(
                        self._rx_queue.get(),
                        timeout=max(0.0, end_time - time.monotonic()),
                    )
                    try:
                        match decode_message(resp_data):
                            case OkMsg() as ok:
                                return ok
                            case ErrorMsg(message):
                                raise MotionError(RobotError.from_wire(message))
                    except msgspec.ValidationError:
                        pass  # Ignore non-matching datagrams
                except (asyncio.TimeoutError, TimeoutError):
                    break
        raise TimeoutError("Timeout waiting for OK")

    # --------------- Motion / Control ---------------

    async def home(self, wait: bool = False, timeout: float = 10.0) -> int:
        """Home the robot to its home position.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Motion

        Example:
            rbt.home()

        Args:
            wait: If True, block until motion completes
            timeout: Maximum time to wait in seconds (only used when wait=True)
        """
        index = await self._send(HomeCmd())
        assert isinstance(index, int)
        if wait and index >= 0:
            await self.wait_command_complete(index, timeout=timeout)
        return index

    async def resume(self) -> int:
        """Re-enable the robot controller, allowing motion commands.

        Category: Control

        Example:
            rbt.resume()

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(ResumeCmd())

    async def halt(self) -> int:
        """Halt the robot — stop all motion and disable.

        Category: Control

        Example:
            rbt.halt()

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(HaltCmd())

    async def simulator_on(self) -> int:
        """Enable simulator mode (no physical robot hardware required).

        The controller will use simulated robot dynamics instead of
        communicating with real hardware over serial.

        Category: Control

        Example:
            rbt.simulator_on()

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(SimulatorCmd(on=True))

    async def simulator_off(self) -> int:
        """Disable simulator mode, switching to real hardware communication.

        Category: Control

        Example:
            rbt.simulator_off()

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(SimulatorCmd(on=False))

    async def set_serial_port(self, port_str: str) -> int:
        """Set the serial port for robot hardware communication.

        Category: Configuration

        Example:
            rbt.set_serial_port("/dev/ttyUSB0")

        Args:
            port_str: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3').

        Returns:
            True if the command was acknowledged successfully.

        Raises:
            ValueError: If port_str is empty.
        """
        if not port_str:
            raise ValueError("No port provided")
        return await self._send(SetPortCmd(port_str=port_str))

    async def reset(self) -> int:
        """Reset controller state to initial values.

        Instantly resets positions to home, clears queues, resets tool/errors.
        Preserves serial connection. Useful for fast test isolation.

        Category: Control

        Example:
            rbt.reset()
        """
        return await self._send(ResetCmd())

    # --------------- Status / Queries ---------------
    async def ping(self) -> PingResult | None:
        """Return parsed ping result with hardware_connected status.

        Category: Query

        Example:
            rbt.ping()
        """
        resp = await self._request(PingCmd())
        if not isinstance(resp, PingResultStruct):
            return None
        return PingResult(hardware_connected=bool(resp.hardware_connected))

    async def get_angles(self) -> list[float] | None:
        """Get current joint angles in degrees [J1, J2, J3, J4, J5, J6].

        Category: Query

        Example:
            angles = rbt.get_angles()
        """
        resp = await self._request(GetAnglesCmd())
        return resp.angles if isinstance(resp, AnglesResultStruct) else None

    async def get_io(self) -> list[int] | None:
        """Get digital I/O status [in1, in2, out1, out2, estop].

        Category: Query

        Example:
            io = rbt.get_io()
        """
        resp = await self._request(GetIOCmd())
        return resp.io if isinstance(resp, IOResultStruct) else None

    async def get_speeds(self) -> list[float] | None:
        """Get current joint speeds in steps/sec [J1, J2, J3, J4, J5, J6].

        Category: Query

        Example:
            speeds = rbt.get_speeds()
        """
        resp = await self._request(GetSpeedsCmd())
        return resp.speeds if isinstance(resp, SpeedsResultStruct) else None

    async def get_pose(
        self, frame: Literal["WRF", "TRF"] = "WRF"
    ) -> list[float] | None:
        """Get 16-element transformation matrix (flattened) with translation in mm.

        Category: Query

        Example:
            pose = rbt.get_pose()
        """
        resp = await self._request(GetPoseCmd(frame=frame))
        return resp.pose if isinstance(resp, PoseResultStruct) else None

    async def get_status(self) -> StatusResultStruct | None:
        """Get aggregate status (pose, angles, speeds, io, tool_status).

        Category: Query

        Example:
            status = rbt.get_status()
        """
        resp = await self._request(GetStatusCmd())
        return resp if isinstance(resp, StatusResultStruct) else None

    async def get_loop_stats(self) -> LoopStatsResultStruct | None:
        """Fetch control-loop runtime metrics.

        Category: Query

        Example:
            stats = rbt.get_loop_stats()
        """
        resp = await self._request(GetLoopStatsCmd())
        return resp if isinstance(resp, LoopStatsResultStruct) else None

    async def reset_loop_stats(self) -> int:
        """Reset control-loop min/max metrics and overrun count.

        Category: Query

        Example:
            rbt.reset_loop_stats()
        """
        return await self._send(ResetLoopStatsCmd())

    async def set_tool(self, tool_name: str) -> int:
        """Set the current end-effector tool configuration.

        Category: Configuration

        Example:
            rbt.set_tool("NONE")

        Args:
            tool_name: Name of the tool ('NONE', 'PNEUMATIC', 'SSG-48', 'MSG', 'VACUUM')

        Returns:
            True if successful
        """
        self._active_tool_key = tool_name.upper()
        return await self._send(SetToolCmd(tool_name=self._active_tool_key))

    async def set_profile(self, profile: str) -> int:
        """Set the motion profile for all moves.

        Category: Configuration

        Example:
            rbt.set_profile("TOPPRA")

        Args:
            profile: Motion profile type ('TOPPRA', 'RUCKIG', 'QUINTIC', 'TRAPEZOID', 'LINEAR')
                Note: RUCKIG is point-to-point only; Cartesian moves will use TOPPRA.

        Returns:
            True if successful
        """
        return await self._send(SetProfileCmd(profile=profile.upper()))

    async def get_profile(self) -> str | None:
        """Get the current motion profile.

        Category: Query

        Example:
            profile = rbt.get_profile()
        """
        resp = await self._request(GetProfileCmd())
        return resp.profile.upper() if isinstance(resp, ProfileResultStruct) else None

    async def get_tool(self) -> ToolResultStruct | None:
        """Get the current tool and available tools.

        Category: Query

        Example:
            tool = rbt.get_tool()
        """
        resp = await self._request(GetToolCmd())
        return resp if isinstance(resp, ToolResultStruct) else None

    async def get_current_action(self) -> CurrentActionResultStruct | None:
        """Get the current executing action (current, state, next, params).

        Category: Query

        Example:
            action = rbt.get_current_action()
        """
        resp = await self._request(GetCurrentActionCmd())
        return resp if isinstance(resp, CurrentActionResultStruct) else None

    async def get_queue(self) -> QueueResultStruct | None:
        """Get queue status with progress tracking.

        Category: Query

        Example:
            queue = rbt.get_queue()
        """
        resp = await self._request(GetQueueCmd())
        return resp if isinstance(resp, QueueResultStruct) else None

    async def get_tool_status(self) -> ToolStatus | None:
        """Get current tool status (key, state, engaged, positions, channels, etc.).

        Category: Query

        Example:
            ts = rbt.get_tool_status()
        """
        resp = await self._request(GetToolStatusCmd())
        if not isinstance(resp, ToolStatusResultStruct):
            return None
        return ToolStatus(
            key=resp.tool_key,
            state=resp.state,
            engaged=resp.engaged,
            part_detected=resp.part_detected,
            fault_code=resp.fault_code,
            positions=tuple(resp.positions),
            channels=tuple(resp.channels),
        )

    async def get_enablement(self) -> EnablementResultStruct | None:
        """Get joint and Cartesian enablement flags.

        Category: Query

        Example:
            en = rbt.get_enablement()
        """
        resp = await self._request(GetEnablementCmd())
        return resp if isinstance(resp, EnablementResultStruct) else None

    async def get_error(self) -> RobotError | None:
        """Get the current error state, or None if no error.

        Category: Query

        Example:
            err = rbt.get_error()
        """
        resp = await self._request(GetErrorCmd())
        if not isinstance(resp, ErrorResultStruct) or resp.error is None:
            return None
        return RobotError.from_wire(resp.error)

    async def get_tcp_speed(self) -> float | None:
        """Get current TCP linear speed in mm/s.

        Category: Query

        Example:
            speed = rbt.get_tcp_speed()
        """
        resp = await self._request(GetTcpSpeedCmd())
        return resp.speed if isinstance(resp, TcpSpeedResultStruct) else None

    # --------------- Helper methods ---------------

    async def get_pose_rpy(self) -> list[float] | None:
        """Get robot pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] using RPY order='xyz'.

        Category: Query

        Example:
            rpy = rbt.get_pose_rpy()
        """
        pose_matrix = await self.get_pose()
        if not pose_matrix or len(pose_matrix) != 16:
            return None

        try:
            x, y, z = pose_matrix[3], pose_matrix[7], pose_matrix[11]
            R = self._R_buf
            R[0, 0] = pose_matrix[0]
            R[0, 1] = pose_matrix[1]
            R[0, 2] = pose_matrix[2]
            R[1, 0] = pose_matrix[4]
            R[1, 1] = pose_matrix[5]
            R[1, 2] = pose_matrix[6]
            R[2, 0] = pose_matrix[8]
            R[2, 1] = pose_matrix[9]
            R[2, 2] = pose_matrix[10]
            so3_rpy(R, self._rpy_buf)
            rpy_deg = np.degrees(self._rpy_buf)
            return [x, y, z, rpy_deg[0], rpy_deg[1], rpy_deg[2]]
        except (ValueError, IndexError):
            return None

    async def get_pose_xyz(self) -> list[float] | None:
        """Get robot position as [x, y, z] in mm.

        Category: Query

        Example:
            xyz = rbt.get_pose_xyz()
        """
        pose_rpy = await self.get_pose_rpy()
        return pose_rpy[:3] if pose_rpy else None

    async def is_estop_pressed(self) -> bool:
        """Check if E-stop is pressed. Returns True if pressed.

        Category: Query

        Example:
            pressed = rbt.is_estop_pressed()
        """
        io_status = await self.get_io()
        if io_status and len(io_status) >= 5:
            return io_status[4] == 0  # E-stop at index 4, 0 means pressed
        return False

    async def is_robot_stopped(self, threshold_speed: float = 2.0) -> bool:
        """Check if robot has stopped moving.

        Category: Query

        Prefer ``wait_command_complete()`` for waiting on specific commands.
        This method polls raw joint speeds and is mainly useful for
        diagnostics or manual stopping logic.

        Example:
            stopped = rbt.is_robot_stopped()

        Args:
            threshold_speed: Speed threshold in steps/sec

        Returns:
            True if all joints below threshold
        """
        speeds = await self.get_speeds()
        if not speeds:
            return False
        return max(abs(s) for s in speeds) < threshold_speed

    async def wait_motion_complete(
        self,
        timeout: float = 10.0,
        settle_window: float = 0.25,
        speed_threshold: float = 0.01,
        angle_threshold: float = 0.5,
        motion_start_timeout: float = 1.0,
    ) -> bool:
        """Wait for robot to stop moving using multicast status broadcasts.

        This method first waits for motion to START (speeds above threshold),
        then waits for motion to COMPLETE (speeds below threshold for settle_window).
        This avoids a race condition where the method returns immediately if
        called before motion has begun.

        Category: Synchronization

        Example:
            rbt.wait_motion_complete()

        Args:
            timeout: Maximum time to wait in seconds
            settle_window: How long robot must be stable to be considered stopped
            speed_threshold: Max joint speed to be considered stopped (rad/s)
            angle_threshold: Max angle change to be considered stopped (degrees)
            motion_start_timeout: Max time to wait for motion to start (seconds)

        Returns:
            True if robot stopped, False if timeout
        """
        await self._ensure_endpoint()

        last_angles: np.ndarray | None = None
        settle_start: float | None = None
        motion_started = False
        start_time = time.monotonic()

        try:
            async with asyncio.timeout(timeout):
                async for status in self.status_stream_shared():
                    speeds = status.speeds
                    angles = status.angles

                    max_speed = float(np.abs(speeds).max())

                    max_angle_change = 0.0
                    if last_angles is not None:
                        max_angle_change = float(np.abs(angles - last_angles).max())
                        last_angles[:] = angles
                    else:
                        last_angles = angles.copy()

                    now = time.monotonic()

                    # Phase 1: Wait for motion to start
                    if not motion_started:
                        if (
                            max_speed >= speed_threshold
                            or max_angle_change >= angle_threshold
                        ):
                            motion_started = True
                            settle_start = None
                        elif now - start_time > motion_start_timeout:
                            motion_started = True

                    # Phase 2: Wait for motion to complete
                    if motion_started:
                        if (
                            max_speed < speed_threshold
                            and max_angle_change < angle_threshold
                        ):
                            if settle_start is None:
                                settle_start = now
                            elif now - settle_start > settle_window:
                                return True
                        else:
                            settle_start = None
        except TimeoutError:
            return False

        return False

    # --------------- Additional waits and utilities ---------------

    async def wait_ready(self, timeout: float = 5.0, interval: float = 0.05) -> bool:
        """Poll ping() until server responds or timeout.

        Args:
            timeout: Maximum time to wait for server to respond
            interval: Polling interval between ping attempts

        Returns:
            True if server responded to PING, False on timeout
        """
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            result = await self.ping()
            if result:
                return True
            await asyncio.sleep(interval)
        return False

    async def wait_for_status(
        self, predicate: Callable[[StatusBuffer], bool], timeout: float = 5.0
    ) -> bool:
        """Wait until a multicast status satisfies predicate(status) within timeout."""
        await self._ensure_endpoint()
        last_gen = 0
        end_time = time.monotonic() + timeout

        while time.monotonic() < end_time and not self._closed:
            self._status_event.clear()

            # Check if we already have new data
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                try:
                    if predicate(self._shared_status):
                        return True
                except Exception:
                    logger.debug("Status predicate raised", exc_info=True)
                continue

            # Wait for next update with timeout
            remaining = max(0.0, end_time - time.monotonic())
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(
                    self._status_event.wait(),
                    timeout=min(remaining, 0.5),
                )
            except asyncio.TimeoutError:
                continue

            if self._closed:
                return False
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                try:
                    if predicate(self._shared_status):
                        return True
                except Exception:
                    logger.debug("Status predicate raised", exc_info=True)
        return False

    async def wait_command_complete(
        self, command_index: int, timeout: float = 10.0
    ) -> bool:
        """Wait until a specific command index has been completed.

        Uses status broadcasts to monitor the server's completed_command_index.
        Raises MotionError if the pipeline reports a planning/execution failure
        at or before the awaited command index.

        Args:
            command_index: The command index to wait for (returned by motion commands).
            timeout: Maximum time to wait in seconds.

        Returns:
            True if the command completed within timeout, False otherwise.

        Raises:
            MotionError: If the pipeline errored at or before command_index.
        """

        def _done(s: StatusBuffer) -> bool:
            if s.completed_index >= command_index:
                return True
            if s.error is not None and s.error.command_index <= command_index:
                return True
            return False

        ok = await self.wait_for_status(_done, timeout=timeout)
        if ok:
            s = self._shared_status
            if s.error is not None and s.error.command_index <= command_index:
                raise MotionError(s.error)
        return ok

    # --------------- Move commands (queued, pre-computed trajectory) ---------------

    @overload
    async def moveJ(
        self,
        target: list[float],
        *,
        duration: float = ...,
        speed: float = ...,
        accel: float = ...,
        r: float = ...,
        rel: bool = ...,
        wait: bool = ...,
        timeout: float = ...,
    ) -> int:
        """Joint-space move to target angles."""
        ...

    @overload
    async def moveJ(
        self,
        target: list[float],
        *,
        pose: list[float],
        duration: float = ...,
        speed: float = ...,
        accel: float = ...,
        r: float = ...,
        wait: bool = ...,
        timeout: float = ...,
    ) -> int:
        """Joint-space move to Cartesian target via IK."""
        ...

    async def moveJ(
        self,
        target: list[float],
        *,
        pose: list[float] | None = None,
        duration: float = 0.0,
        speed: float = 0.0,
        accel: float = 1.0,
        r: float = 0.0,
        rel: bool = False,
        wait: bool = False,
        timeout: float = 10.0,
    ) -> int:
        """Joint-space move. Positional arg = joint angles; pose= kwarg = Cartesian target with IK.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Motion

        Example:
            rbt.moveJ([90, -90, 180, 0, 0, 180], speed=0.5)

        Args:
            target: 6 joint angles in degrees (ignored if pose= is set)
            pose: If set, Cartesian target [x,y,z,rx,ry,rz] — dispatches to MOVEJ_POSE
            duration: Motion duration in seconds (mutually exclusive with speed)
            speed: Speed fraction 0-1 (mutually exclusive with duration)
            accel: Acceleration fraction 0-1
            r: Blend radius in mm (0 = stop at target)
            rel: If True, target is relative to current position
            wait: If True, block until motion completes
        """
        if pose is not None:
            index = await self._send(
                MoveJPoseCmd(
                    pose=pose, duration=duration, speed=speed, accel=accel, r=r
                )
            )
        else:
            index = await self._send(
                MoveJCmd(
                    angles=target,
                    duration=duration,
                    speed=speed,
                    accel=accel,
                    r=r,
                    rel=rel,
                )
            )
        if wait and index >= 0:
            await self.wait_command_complete(index, timeout=timeout)
        return index

    async def moveL(
        self,
        pose: list[float],
        *,
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float = 0.0,
        speed: float = 0.0,
        accel: float = 1.0,
        r: float = 0.0,
        rel: bool = False,
        wait: bool = False,
        timeout: float = 10.0,
    ) -> int:
        """Linear Cartesian move to target pose.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Motion

        Example:
            rbt.moveL([0, 263, 242, 90, 0, 90], speed=0.5)

        Args:
            pose: Target [x,y,z,rx,ry,rz] in mm and degrees
            frame: Reference frame ("WRF" or "TRF")
            duration: Motion duration in seconds
            speed: Speed fraction 0-1
            accel: Acceleration fraction 0-1
            r: Blend radius in mm
            rel: If True, pose is relative delta
            wait: If True, block until motion completes
        """
        cmd = MoveLCmd(
            pose=pose,
            frame=frame,
            duration=duration,
            speed=speed,
            accel=accel,
            r=r,
            rel=rel,
        )
        index = await self._send(cmd)
        if wait and index >= 0:
            await self.wait_command_complete(index, timeout=timeout)
        return index

    async def moveC(
        self,
        via: list[float],
        end: list[float],
        *,
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float | None = None,
        speed: float | None = None,
        accel: float = 1.0,
        r: float = 0.0,
        wait: bool = False,
        timeout: float = 10.0,
    ) -> int:
        """Circular arc through current position -> via -> end.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Motion

        Example:
            rbt.moveC(via=[0, 250, 250, 90, 0, 90], end=[0, 263, 242, 90, 0, 90], speed=0.5)

        Args:
            via: Via-point pose [x,y,z,rx,ry,rz]
            end: End-point pose [x,y,z,rx,ry,rz]
            frame: Reference frame
            duration: Motion duration in seconds
            speed: Speed fraction 0-1
            accel: Acceleration fraction 0-1
            r: Blend radius in mm
            wait: If True, block until motion completes
        """
        cmd = MoveCCmd(
            via=via,
            end=end,
            frame=frame,
            duration=duration,
            speed=speed,
            accel=accel,
            r=r,
        )
        index = await self._send(cmd)
        if wait and index >= 0:
            await self.wait_command_complete(index, timeout=timeout)
        return index

    async def moveS(
        self,
        waypoints: list[list[float]],
        *,
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float | None = None,
        speed: float | None = None,
        accel: float = 1.0,
        wait: bool = False,
        timeout: float = 10.0,
    ) -> int:
        """Cubic spline through waypoints.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Smooth Motion

        Example:
            rbt.moveS([[0, 263, 242, 90, 0, 90], [50, 250, 200, 90, 0, 90]], speed=0.5)

        Args:
            waypoints: List of poses [[x,y,z,rx,ry,rz], ...]
            frame: Reference frame
            duration: Motion duration in seconds
            speed: Speed fraction 0-1
            accel: Acceleration fraction 0-1
            wait: If True, block until motion completes
        """
        cmd = MoveSCmd(
            waypoints=waypoints,
            frame=frame,
            duration=duration,
            speed=speed,
            accel=accel,
        )
        index = await self._send(cmd)
        if wait and index >= 0:
            await self.wait_command_complete(index, timeout=timeout)
        return index

    async def moveP(
        self,
        waypoints: list[list[float]],
        *,
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float | None = None,
        speed: float | None = None,
        accel: float = 1.0,
        wait: bool = False,
        timeout: float = 10.0,
    ) -> int:
        """Process move — constant TCP speed with auto-blending at corners.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Smooth Motion

        Example:
            rbt.moveP([[0, 263, 242, 90, 0, 90], [50, 250, 200, 90, 0, 90]], speed=0.5)

        Args:
            waypoints: List of poses [[x,y,z,rx,ry,rz], ...]
            frame: Reference frame
            duration: Motion duration in seconds
            speed: Speed fraction 0-1
            accel: Acceleration fraction 0-1
            wait: If True, block until motion completes
        """
        cmd = MovePCmd(
            waypoints=waypoints,
            frame=frame,
            duration=duration,
            speed=speed,
            accel=accel,
        )
        index = await self._send(cmd)
        if wait and index >= 0:
            await self.wait_command_complete(index, timeout=timeout)
        return index

    async def checkpoint(self, label: str) -> int:
        """Insert a checkpoint marker in the motion queue.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Synchronization

        Example:
            rbt.checkpoint("pick_done")

        Args:
            label: Checkpoint label for progress tracking
        """
        index = await self._send(CheckpointCmd(label=label))
        return index

    async def wait_for_command(self, index: int, timeout: float = 30.0) -> bool:
        """Wait until a queued command (by index) has completed.

        Category: Synchronization

        Example:
            rbt.wait_for_command(index)

        Args:
            index: Command index returned by a move command
            timeout: Maximum wait time in seconds
        """
        return await self.wait_for_status(
            lambda s: s.completed_index >= index,
            timeout=timeout,
        )

    async def wait_for_checkpoint(self, label: str, timeout: float = 30.0) -> bool:
        """Wait until a checkpoint with the given label has been reached.

        Category: Synchronization

        Example:
            rbt.wait_for_checkpoint("pick_done")

        Args:
            label: Checkpoint label to wait for
            timeout: Maximum wait time in seconds
        """
        return await self.wait_for_status(
            lambda s: s.last_checkpoint == label,
            timeout=timeout,
        )

    # --------------- Servo commands (streaming position) ---------------

    @overload
    async def servoJ(
        self,
        target: list[float],
        *,
        speed: float = ...,
        accel: float = ...,
    ) -> int:
        """Stream joint position target."""
        ...

    @overload
    async def servoJ(
        self,
        target: list[float],
        *,
        pose: list[float],
        speed: float = ...,
        accel: float = ...,
    ) -> int:
        """Stream Cartesian position target via IK."""
        ...

    async def servoJ(
        self,
        target: list[float],
        *,
        pose: list[float] | None = None,
        speed: float = 1.0,
        accel: float = 1.0,
    ) -> int:
        """Streaming joint position target. Fire-and-forget.

        Category: Streaming

        Example:
            rbt.servoJ([90, -90, 180, 0, 0, 180])

        Args:
            target: 6 joint angles in degrees (ignored if pose= is set)
            pose: If set, Cartesian target — dispatches to SERVOJ_POSE
            speed: Speed fraction 0-1
            accel: Acceleration fraction 0-1
        """
        if pose is not None:
            return await self._send(ServoJPoseCmd(pose=pose, speed=speed, accel=accel))
        return await self._send(ServoJCmd(target=target, speed=speed, accel=accel))

    async def servoL(
        self,
        pose: list[float],
        *,
        speed: float = 1.0,
        accel: float = 1.0,
    ) -> int:
        """Streaming linear Cartesian position target. Fire-and-forget.

        Category: Streaming

        Example:
            rbt.servoL([0, 263, 242, 90, 0, 90])

        Args:
            pose: Target [x,y,z,rx,ry,rz] in mm and degrees
            speed: Speed fraction 0-1
            accel: Acceleration fraction 0-1
        """
        return await self._send(ServoLCmd(pose=pose, speed=speed, accel=accel))

    # --------------- Jog commands (streaming velocity) ---------------

    @overload
    async def jogJ(
        self,
        joint: int,
        speed: float,
        duration: float = ...,
        *,
        accel: float = ...,
    ) -> int:
        """Jog a single joint."""
        ...

    @overload
    async def jogJ(
        self,
        *,
        joints: list[int],
        speeds: list[float],
        duration: float = ...,
        accel: float = ...,
    ) -> int:
        """Jog multiple joints simultaneously."""
        ...

    async def jogJ(
        self,
        joint: int | None = None,
        speed: float = 0.0,
        duration: float = 0.1,
        *,
        joints: list[int] | None = None,
        speeds: list[float] | None = None,
        accel: float = 1.0,
    ) -> int:
        """Joint velocity jog. Single-joint or multi-joint.

        Single joint:   jogJ(0, 0.5, 1.0)
        Multi joint:    jogJ(joints=[0,1], speeds=[0.5, -0.3], duration=1.0)

        Category: Jog

        Example:
            rbt.jogJ(0, speed=0.5, duration=1.0)

        Args:
            joint: Joint index (0-5) for single-joint jog
            speed: Signed speed fraction for single-joint jog
            duration: Jog duration in seconds
            joints: List of joint indices for multi-joint jog
            speeds: List of signed speed fractions for multi-joint jog
            accel: Acceleration fraction 0-1
        """
        speed_arr = [0.0] * 6
        if joints is not None and speeds is not None:
            for j, s in zip(joints, speeds):
                speed_arr[j] = s
        elif joint is not None:
            speed_arr[joint] = speed
        else:
            raise ValueError("jogJ requires either joint= or joints=/speeds=")
        return await self._send(
            JogJCmd(speeds=speed_arr, duration=duration, accel=accel)
        )

    @overload
    async def jogL(
        self,
        frame: Frame,
        axis: Axis,
        speed: float,
        duration: float = ...,
        *,
        accel: float = ...,
    ) -> int:
        """Jog a single Cartesian axis."""
        ...

    @overload
    async def jogL(
        self,
        frame: Frame,
        *,
        axes: list[Axis],
        speeds_list: list[float],
        duration: float = ...,
        accel: float = ...,
    ) -> int:
        """Jog multiple Cartesian axes simultaneously."""
        ...

    async def jogL(
        self,
        frame: Frame,
        axis: Axis | None = None,
        speed: float = 0.0,
        duration: float = 0.1,
        *,
        axes: list[Axis] | None = None,
        speeds_list: list[float] | None = None,
        accel: float = 1.0,
    ) -> int:
        """Cartesian velocity jog. Single-axis or multi-axis.

        Single axis:  jogL("WRF", "X", 0.5, 1.0)
        Multi axis:   jogL("WRF", axes=["X","Y"], speeds_list=[0.5, -0.3], duration=1.0)

        Category: Jog

        Example:
            rbt.jogL("WRF", "X", speed=0.5, duration=1.0)

        Args:
            frame: Reference frame ("WRF" or "TRF")
            axis: Axis name for single-axis jog
            speed: Signed speed fraction for single-axis jog
            duration: Jog duration in seconds
            axes: List of axis names for multi-axis jog
            speeds_list: List of signed speed fractions for multi-axis jog
            accel: Acceleration fraction 0-1
        """
        vel = [0.0] * 6
        if axes is not None and speeds_list is not None:
            for a, s in zip(axes, speeds_list):
                vel[_AXIS_MAP[a]] = s
        elif axis is not None:
            vel[_AXIS_MAP[axis]] = speed
        else:
            raise ValueError("jogL requires either axis= or axes=/speeds_list=")
        return await self._send(
            JogLCmd(frame=frame, velocities=vel, duration=duration, accel=accel)
        )

    # --------------- IO / Gripper / Utility ---------------

    async def set_io(self, index: int, value: int) -> int:
        """Set digital output by logical index (0 = first output pin).

        The firmware I/O byte layout is ``[in0, in1, out0, out1, estop, ...]``
        so logical output index 0 maps to bit position 2.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: IO

        Example:
            rbt.set_io(0, 1)   # Set first output HIGH
        """
        if index < 0 or index > 1:
            raise ValueError("Output index must be 0 or 1")
        if value not in (0, 1):
            raise ValueError("I/O value must be 0 or 1")
        # Firmware bit layout: [in0, in1, out0, out1, estop, ...]
        firmware_index = index + 2
        result = await self._send(SetIOCmd(port_index=firmware_index, value=value))
        return result

    async def delay(self, seconds: float) -> int:
        """Insert a non-blocking delay in the motion queue.

        Returns the command index (≥ 0) on success, -1 on failure.

        Category: Synchronization

        Example:
            rbt.delay(1.0)
        """
        if seconds <= 0:
            raise ValueError("Delay must be positive")
        result = await self._send(DelayCmd(seconds=seconds))
        return result

    async def tool_action(
        self,
        tool_key: str,
        action: str,
        params: list | None = None,
        *,
        wait: bool = False,
        timeout: float = 10.0,
    ) -> int:
        """Send a generic tool action command.

        Returns the command index (>= 0) on success, -1 on failure.

        Category: I/O

        Example:
            rbt.tool_action("PNEUMATIC", "open")

        Args:
            tool_key: Tool registry key (e.g. "PNEUMATIC", "SSG-48", "MSG")
            action: Action name (e.g. "open", "close", "move", "calibrate")
            params: Positional parameters (meaning defined by tool config)
            wait: If True, block until action completes
            timeout: Maximum time to wait in seconds (only used when wait=True)
        """
        cmd = ToolActionCmd(
            tool_key=tool_key.strip().upper(),
            action=action.strip().lower(),
            params=params or [],
        )
        result = await self._send(cmd)
        if wait and result >= 0:
            await self.wait_command_complete(result, timeout=timeout)
        return result
