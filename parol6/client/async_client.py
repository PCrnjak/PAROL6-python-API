"""
Async UDP client for PAROL6 robot control.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Literal, cast

import numpy as np
from spatialmath import SO3

from .. import config as cfg
from ..ack_policy import QUERY_COMMANDS, SYSTEM_COMMANDS, AckPolicy
from ..client.status_subscriber import subscribe_status
from ..protocol import wire
from ..protocol.types import Axis, Frame, StatusAggregate

logger = logging.getLogger(__name__)


class _UDPClientProtocol(asyncio.DatagramProtocol):
    def __init__(self, rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]]):
        self.rx_queue = rx_queue
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            self.rx_queue.put_nowait((data, addr))
        except Exception:
            pass

    def error_received(self, exc: Exception) -> None:
        pass

    def connection_lost(self, exc: Exception | None) -> None:
        pass


class AsyncRobotClient:
    """
    Async UDP client for the PAROL6 headless controller.

    Motion/control commands: fire-and-forget via UDP
    Query commands: request/response with timeout and simple retry
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 2.0,
        retries: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries

        # Persistent asyncio datagram endpoint
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _UDPClientProtocol | None = None
        self._rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]] = asyncio.Queue(maxsize=256)
        self._ep_lock = asyncio.Lock()

        # Serialize request/response
        self._req_lock = asyncio.Lock()

        # Stream flag for UI convenience
        self._stream_mode = False
        # ACK policy
        self._ack_policy = AckPolicy.from_env(lambda: self._stream_mode)

        # Multicast listener using subscribe_status
        self._multicast_task: asyncio.Task | None = None
        self._status_queue: asyncio.Queue[StatusAggregate] = asyncio.Queue(maxsize=100)

    # --------------- Internal helpers ---------------

    async def _ensure_endpoint(self) -> None:
        """Lazily create a persistent asyncio UDP datagram endpoint."""
        if self._transport is not None:
            return
        async with self._ep_lock:
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
        """Start listening for multicast status broadcasts using subscribe_status."""
        if self._multicast_task is not None and not self._multicast_task.done():
            return

        logger.info(
            f"Status subscriber config: group={cfg.MCAST_GROUP} port={cfg.MCAST_PORT} iface={cfg.MCAST_IF}"
        )
        # Quick readiness check (no blind wait): bounded by client's own timeout
        try:
            await self.wait_for_server_ready(
                timeout=min(1.0, float(self.timeout or 0.3)), interval=0.5
            )
        except Exception:
            pass

        async def _listener():
            """Consume status broadcasts and queue them."""
            try:
                async for status in subscribe_status():
                    # Put in queue, drop old if full
                    if self._status_queue.full():
                        try:
                            self._status_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._status_queue.put_nowait(status)
                    except asyncio.QueueFull:
                        pass
            except Exception:
                # Subscriber ended, could retry but for now just exit
                pass

        # Start listener task
        self._multicast_task = asyncio.create_task(_listener())

    async def status_stream(self) -> AsyncIterator[StatusAggregate]:
        """
        Async generator that yields status updates from multicast broadcasts.

        Usage:
            async for status in client.status_stream():
                print(f"Speeds: {status.get('speeds')}")
        """
        await self._ensure_endpoint()
        while True:
            status = await self._status_queue.get()
            yield status

    async def _send(self, message: str) -> bool:
        """
        Send a command based on AckPolicy:
        - System commands: wait for server OK/ERROR, return True on OK
        - Motion commands: wait iff policy requires ACK; otherwise fire-and-forget (return True on send)
        - Query commands: should use _request path; if invoked here, just fire-and-forget
        """
        await self._ensure_endpoint()
        assert self._transport is not None

        name = (message or "").split("|", 1)[0].strip().upper()

        # System commands: wait for OK/ERROR
        if name in SYSTEM_COMMANDS:
            return await self._request_ok(message, self.timeout)

        # Motion and other non-query commands
        if name not in QUERY_COMMANDS:
            if self._ack_policy.requires_ack(message):
                return await self._request_ok(message, self.timeout)
            # Fire-and-forget
            self._transport.sendto(message.encode("ascii"))
            return True

        # Queries: fire-and-forget here (query methods use _request())
        self._transport.sendto(message.encode("ascii"))
        return True

    async def _request(self, message: str, bufsize: int = 2048) -> str | None:
        """Send a request and wait for a UDP response."""
        await self._ensure_endpoint()
        assert self._transport is not None
        for attempt in range(self.retries + 1):
            try:
                async with self._req_lock:
                    self._transport.sendto(message.encode("ascii"))
                    data, _addr = await asyncio.wait_for(self._rx_queue.get(), timeout=self.timeout)
                    return data.decode("ascii", errors="ignore")
            except TimeoutError:
                if attempt < self.retries:
                    backoff = min(0.5, 0.05 * (2**attempt)) + random.uniform(0, 0.05)
                    await asyncio.sleep(backoff)
                    continue
            except Exception:
                break
        return None

    async def _request_ok(self, message: str, timeout: float) -> bool:
        """
        Send a command and wait for a simple 'OK' or 'ERROR|...' reply.
        Returns True on OK; raises on ERROR or timeout.
        """
        await self._ensure_endpoint()
        assert self._transport is not None

        end_time = time.time() + timeout
        async with self._req_lock:
            self._transport.sendto(message.encode("ascii"))
            while time.time() < end_time:
                try:
                    data, _addr = await asyncio.wait_for(
                        self._rx_queue.get(), timeout=max(0.0, end_time - time.time())
                    )
                    text = data.decode("ascii", errors="ignore").strip()
                    if text == "OK":
                        return True
                    if text.startswith("ERROR|"):
                        raise RuntimeError(text)
                    # Ignore unrelated datagrams
                except TimeoutError:
                    break
                except Exception:
                    break
        raise TimeoutError("Timeout waiting for OK")

    # --------------- Motion / Control ---------------

    async def home(self) -> bool:
        return await self._send("HOME")

    async def enable(self) -> bool:
        return await self._send("ENABLE")

    async def disable(self) -> bool:
        return await self._send("DISABLE")

    async def stop(self) -> bool:
        """Alias for disable() - stops motion and disables controller."""
        return await self.disable()

    async def start(self) -> bool:
        """Alias for enable() - enables controller."""
        return await self.enable()

    async def stream_on(self) -> bool:
        self._stream_mode = True
        return await self._send("STREAM|ON")

    async def stream_off(self) -> bool:
        self._stream_mode = False
        return await self._send("STREAM|OFF")

    async def simulator_on(self) -> bool:
        return await self._send("SIMULATOR|ON")

    async def simulator_off(self) -> bool:
        return await self._send("SIMULATOR|OFF")

    async def set_serial_port(self, port_str: str) -> bool:
        if not port_str:
            raise ValueError("No port provided")
        return await self._send(f"SET_PORT|{port_str}")

    # --------------- Status / Queries ---------------
    async def ping(self) -> str | None:
        """Return raw 'PONG|...' text (e.g., 'PONG|SERIAL=1') or None on timeout."""
        return await self._request("PING", bufsize=256)

    async def get_angles(self) -> list[float] | None:
        resp = await self._request("GET_ANGLES", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "ANGLES")
        return cast(list[float] | None, vals)

    async def get_io(self) -> list[int] | None:
        resp = await self._request("GET_IO", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "IO")
        return cast(list[int] | None, vals)

    async def get_gripper_status(self) -> list[int] | None:
        resp = await self._request("GET_GRIPPER", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "GRIPPER")
        return cast(list[int] | None, vals)

    async def get_speeds(self) -> list[float] | None:
        resp = await self._request("GET_SPEEDS", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "SPEEDS")
        return cast(list[float] | None, vals)

    async def get_pose(self, frame: Literal["WRF", "TRF"] = "WRF") -> list[float] | None:
        """
        Returns 16-element transformation matrix (flattened) with translation in mm.

        Args:
            frame: Reference frame - "WRF" for World Reference Frame (default),
                   "TRF" for Tool Reference Frame

        Expected wire format: "POSE|p0,p1,p2,...,p15"
        """
        command = f"GET_POSE {frame}" if frame != "WRF" else "GET_POSE"
        resp = await self._request(command, bufsize=2048)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "POSE")
        return cast(list[float] | None, vals)

    async def get_gripper(self) -> list[int] | None:
        """Alias for get_gripper_status for compatibility."""
        return await self.get_gripper_status()

    async def get_status(self) -> dict | None:
        """
        Aggregate status.
        Expected format:
          STATUS|POSE=p0,p1,...,p15|ANGLES=a0,...,a5|IO=in1,in2,out1,out2,estop|GRIPPER=id,pos,spd,cur,status,obj
        Returns dict with keys: pose (list[float] len=16 with translation in mm), angles (list[float] len=6),
                                io (list[int] len=5), gripper (list[int] len>=6)
        """
        resp = await self._request("GET_STATUS", bufsize=4096)
        if not resp:
            return None
        return cast(dict | None, wire.decode_status(resp))

    async def get_loop_stats(self) -> dict | None:
        """
        Fetch control-loop runtime metrics.
        Expected wire format: "LOOP_STATS|{json}"
        """
        resp = await self._request("GET_LOOP_STATS", bufsize=1024)
        if not resp or not resp.startswith("LOOP_STATS|"):
            return None
        return cast(dict, json.loads(resp.split("|", 1)[1]))

    async def set_tool(self, tool_name: str) -> bool:
        """
        Set the current end-effector tool configuration.

        Args:
            tool_name: Name of the tool ('NONE', 'PNEUMATIC', 'ELECTRIC')

        Returns:
            True if successful
        """
        return await self._send(f"SET_TOOL|{tool_name.upper()}")

    async def get_tool(self) -> dict | None:
        """
        Get the current tool configuration and available tools.

        Returns:
            Dict with keys: 'tool' (current tool name), 'available' (list of available tools)
            Expected wire format: "TOOL|{json}"
        """
        resp = await self._request("GET_TOOL", bufsize=1024)
        if not resp or not resp.startswith("TOOL|"):
            return None
        return cast(dict, json.loads(resp.split("|", 1)[1]))

    async def get_current_action(self) -> dict | None:
        """
        Get the current executing action/command and its state.

        Returns:
            Dict with keys: 'current' (current action name), 'state' (action state),
                           'next' (next action if any)
            Expected wire format: "ACTION|{json}"
        """
        resp = await self._request("GET_CURRENT_ACTION", bufsize=1024)
        if not resp or not resp.startswith("ACTION|"):
            return None
        return cast(dict, json.loads(resp.split("|", 1)[1]))

    async def get_queue(self) -> dict | None:
        """
        Get the list of queued non-streamable commands.

        Returns:
            Dict with keys: 'non_streamable' (list of queued commands), 'size' (queue size)
            Expected wire format: "QUEUE|{json}"
        """
        resp = await self._request("GET_QUEUE", bufsize=2048)
        if not resp or not resp.startswith("QUEUE|"):
            return None
        return cast(dict, json.loads(resp.split("|", 1)[1]))

    # --------------- Helper methods ---------------

    async def get_pose_rpy(self) -> list[float] | None:
        """
        Get robot pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] using RPY order='xyz'.
        """
        pose_matrix = await self.get_pose()
        if not pose_matrix or len(pose_matrix) != 16:
            return None

        try:
            x, y, z = pose_matrix[3], pose_matrix[7], pose_matrix[11]
            # Rotation matrix rows (row-major layout)
            R = np.array(
                [
                    [pose_matrix[0], pose_matrix[1], pose_matrix[2]],
                    [pose_matrix[4], pose_matrix[5], pose_matrix[6]],
                    [pose_matrix[8], pose_matrix[9], pose_matrix[10]],
                ]
            )
            # Use xyz convention (rx, ry, rz) - Roll-Pitch-Yaw
            rpy_deg = SO3(R).rpy(order="xyz", unit="deg")
            return [x, y, z, rpy_deg[0], rpy_deg[1], rpy_deg[2]]
        except (ValueError, IndexError, ImportError):
            return None

    async def get_pose_xyz(self) -> list[float] | None:
        """Get robot position as [x, y, z] in mm."""
        pose_rpy = await self.get_pose_rpy()
        return pose_rpy[:3] if pose_rpy else None

    async def is_estop_pressed(self) -> bool:
        """Check if E-stop is pressed. Returns True if pressed."""
        io_status = await self.get_io()
        if io_status and len(io_status) >= 5:
            return io_status[4] == 0  # E-stop at index 4, 0 means pressed
        return False

    async def is_robot_stopped(self, threshold_speed: float = 2.0) -> bool:
        """
        Check if robot has stopped moving.

        Args:
            threshold_speed: Speed threshold in steps/sec

        Returns:
            True if all joints below threshold
        """
        speeds = await self.get_speeds()
        if not speeds:
            return False
        return max(abs(s) for s in speeds) < threshold_speed

    async def wait_until_stopped(
        self,
        timeout: float = 90.0,
        settle_window: float = 1.0,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5,
    ) -> bool:
        """
        Wait for robot to stop moving using multicast status broadcasts.

        Args:
            timeout: Maximum time to wait in seconds
            settle_window: How long robot must be stable to be considered stopped
            speed_threshold: Max joint speed to be considered stopped (steps/sec)
            angle_threshold: Max angle change to be considered stopped (degrees)

        Returns:
            True if robot stopped, False if timeout
        """
        await self._ensure_endpoint()

        last_angles = None
        settle_start = None
        timeout_task = asyncio.create_task(asyncio.sleep(timeout))

        try:
            async for status in self.status_stream():
                if timeout_task.done():
                    return False

                # Check speeds from status
                speeds = status.get("speeds")
                if speeds and isinstance(speeds, Iterable):
                    if max(abs(s) for s in speeds) < speed_threshold:
                        if settle_start is None:
                            settle_start = time.time()
                        elif time.time() - settle_start > settle_window:
                            return True
                    else:
                        settle_start = None

                # Also check angles as fallback
                angles = status.get("angles")
                if angles and not speeds:
                    if last_angles is not None:
                        max_change = max(
                            abs(a - b) for a, b in zip(angles, last_angles, strict=False)
                        )
                        if max_change < angle_threshold:
                            if settle_start is None:
                                settle_start = time.time()
                            elif time.time() - settle_start > settle_window:
                                return True
                        else:
                            settle_start = None
                    last_angles = angles
        finally:
            timeout_task.cancel()
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass

        return False

    # --------------- Additional waits and utilities ---------------

    async def wait_for_server_ready(self, timeout: float = 5.0, interval: float = 0.05) -> bool:
        """Poll ping() until server responds or timeout."""
        end_time = time.time() + timeout
        while time.time() < end_time:
            ok = await self.ping()
            if ok:
                return True
            await asyncio.sleep(interval)
        return False

    async def wait_for_status(
        self, predicate: Callable[[StatusAggregate], bool], timeout: float = 5.0
    ) -> bool:
        """Wait until a multicast status satisfies predicate(status) within timeout."""
        await self._ensure_endpoint()
        end_time = time.time() + timeout
        while time.time() < end_time:
            remaining = max(0.0, end_time - time.time())
            try:
                status = await asyncio.wait_for(self._status_queue.get(), timeout=remaining)
            except TimeoutError:
                break
            try:
                if predicate(status):
                    return True
            except Exception:
                # Ignore predicate exceptions from tests
                pass
        return False

    async def send_raw(
        self, message: str, await_reply: bool = False, timeout: float = 2.0
    ) -> bool | str | None:
        """Send a raw UDP message; optionally await a single reply."""
        await self._ensure_endpoint()
        assert self._transport is not None
        try:
            if not await_reply:
                self._transport.sendto(message.encode("ascii"))
                return True
            async with self._req_lock:
                self._transport.sendto(message.encode("ascii"))
                data, _addr = await asyncio.wait_for(self._rx_queue.get(), timeout=timeout)
                return data.decode("ascii", errors="ignore")
        except TimeoutError:
            return None
        except Exception:
            return None

    # --------------- Motion encoders ---------------

    async def move_joints(
        self,
        joint_angles: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,  # accepted but not sent
        profile: str | None = None,  # accepted but not sent
        tracking: str | None = None,  # accepted but not sent
    ) -> bool:
        if duration is None and speed_percentage is None:
            raise RuntimeError("You must provide either a duration or a speed_percentage.")
        message = wire.encode_move_joint(joint_angles, duration, speed_percentage)
        return await self._send(message)

    async def move_pose(
        self,
        pose: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        if duration is None and speed_percentage is None:
            raise RuntimeError("You must provide either a duration or a speed_percentage.")
        message = wire.encode_move_pose(pose, duration, speed_percentage)
        return await self._send(message)

    async def move_cartesian(
        self,
        pose: list[float],
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        if duration is None and speed_percentage is None:
            raise RuntimeError("Error: You must provide either a duration or a speed_percentage.")
        message = wire.encode_move_cartesian(pose, duration, speed_percentage)
        return await self._send(message)

    async def move_cartesian_rel_trf(
        self,
        deltas: list[float],  # [dx, dy, dz, rx, ry, rz]
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        """
        Send a MOVECARTRELTRF (relative straight-line in TRF) command.
        Provide either duration or speed_percentage (1..100).
        """
        if duration is None and speed_percentage is None:
            raise RuntimeError("Error: You must provide either a duration or a speed_percentage.")
        message = wire.encode_move_cartesian_rel_trf(
            deltas, duration, speed_percentage, accel_percentage, profile, tracking
        )
        return await self._send(message)

    async def jog_joint(
        self,
        joint_index: int,
        speed_percentage: int,
        duration: float | None = None,
        distance_deg: float | None = None,
    ) -> bool:
        """
        Send a JOG command for a single joint (0..5 positive, 6..11 negative for reverse).
        duration and distance_deg are optional; at least one should be provided for one-shot jog.
        """
        if duration is None and distance_deg is None:
            raise RuntimeError("Error: You must provide either a duration or a distance_deg.")
        message = wire.encode_jog_joint(joint_index, speed_percentage, duration, distance_deg)
        return await self._send(message)

    async def jog_cartesian(
        self,
        frame: Frame,
        axis: Axis,
        speed_percentage: int,
        duration: float,
    ) -> bool:
        """
        Send a CARTJOG command (frame 'TRF' or 'WRF', axis in {X+/X-/Y+/.../RZ-}).
        """
        message = wire.encode_cart_jog(frame, axis, speed_percentage, duration)
        return await self._send(message)

    async def jog_multiple(
        self,
        joints: list[int],
        speeds: list[float],
        duration: float,
    ) -> bool:
        """
        Send a MULTIJOG command to jog multiple joints simultaneously for 'duration' seconds.
        """
        if len(joints) != len(speeds):
            raise ValueError("Error: The number of joints must match the number of speeds.")
        joints_str = ",".join(str(j) for j in joints)
        speeds_str = ",".join(str(s) for s in speeds)
        message = f"MULTIJOG|{joints_str}|{speeds_str}|{duration}"
        return await self._send(message)

    async def set_io(self, index: int, value: int) -> bool:
        """
        Set digital I/O bit.
        index: 0..7, value: 0 or 1
        """
        if index < 0 or index > 7:
            raise ValueError("I/O index must be 0..7")
        if value not in (0, 1):
            raise ValueError("I/O value must be 0 or 1")
        return await self._send(f"SET_IO|{index}|{value}")

    async def delay(self, seconds: float) -> bool:
        """
        Insert a non-blocking delay in the motion queue.
        """
        if seconds <= 0:
            raise ValueError("Delay must be positive")
        return await self._send(f"DELAY|{seconds}")

    # --------------- IO / Gripper ---------------

    async def control_pneumatic_gripper(self, action: str, port: int) -> bool:
        """
        Control pneumatic gripper via digital outputs.
        action: 'open' or 'close'
        port: 1 or 2
        """
        action = action.lower()
        if action not in ("open", "close"):
            raise ValueError("Invalid pneumatic action")
        if port not in (1, 2):
            raise ValueError("Invalid pneumatic port")
        message = f"PNEUMATICGRIPPER|{action}|{port}"
        return await self._send(message)

    async def control_electric_gripper(
        self,
        action: str,
        position: int | None = 255,
        speed: int | None = 150,
        current: int | None = 500,
    ) -> bool:
        """
        Control electric gripper.
        action: 'move' or 'calibrate'
        position: 0..255
        speed: 0..255
        current: 100..1000 (mA)
        """
        action = action.lower()
        if action not in ("move", "calibrate"):
            raise ValueError("Invalid electric gripper action")
        pos = 0 if position is None else int(position)
        spd = 0 if speed is None else int(speed)
        cur = 100 if current is None else int(current)
        message = f"ELECTRICGRIPPER|{action}|{pos}|{spd}|{cur}"
        return await self._send(message)

    # --------------- GCODE ---------------

    async def execute_gcode(self, gcode_line: str) -> bool:
        """
        Execute a single GCODE line.
        """
        message = wire.encode_gcode(gcode_line)
        return await self._send(message)

    async def execute_gcode_program(self, program_lines: list[str]) -> bool:
        """
        Execute a GCODE program from a list of lines.
        """
        for i, line in enumerate(program_lines):
            if "|" in line:
                raise SyntaxError(f"Line {i + 1} contains invalid '|'")
        message = wire.encode_gcode_program_inline(program_lines)
        return await self._send(message)

    async def load_gcode_file(self, filepath: str) -> bool:
        """
        Load and execute a GCODE program from a file.
        """
        message = f"GCODE_PROGRAM|FILE|{filepath}"
        return await self._send(message)

    async def get_gcode_status(self) -> dict | None:
        """
        Get the current status of the GCODE interpreter.
        """
        resp = await self._request("GET_GCODE_STATUS", bufsize=2048)
        if not resp or not resp.startswith("GCODE_STATUS|"):
            return None

        status_json = resp.split("|", 1)[1]
        return json.loads(status_json)

    async def pause_gcode_program(self) -> bool:
        """Pause the currently running GCODE program."""
        return await self._send("GCODE_PAUSE")

    async def resume_gcode_program(self) -> bool:
        """Resume a paused GCODE program."""
        return await self._send("GCODE_RESUME")

    async def stop_gcode_program(self) -> bool:
        """Stop the currently running GCODE program."""
        return await self._send("GCODE_STOP")

    # --------------- Smooth motion ---------------

    async def smooth_circle(
        self,
        center: list[float],
        radius: float,
        plane: Literal["XY", "XZ", "YZ"] = "XY",
        frame: Literal["WRF", "TRF"] = "WRF",
        center_mode: Literal["ABSOLUTE", "TOOL", "RELATIVE"] = "ABSOLUTE",
        entry_mode: Literal["AUTO", "TANGENT", "DIRECT", "NONE"] = "NONE",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
    ) -> bool:
        """
        Execute a smooth circular motion.

        Args:
            center: [x, y, z] center point in mm
            radius: Circle radius in mm
            plane: Plane of the circle ('XY', 'XZ', or 'YZ')
            frame: Reference frame ('WRF' for World, 'TRF' for Tool)
            center_mode: How to interpret center point
            entry_mode: How to approach circle if not on perimeter
            start_pose: Optional [x, y, z, rx, ry, rz] start pose
            duration: Time to complete the circle in seconds
            speed_percentage: Speed as percentage (1-100)
            clockwise: Direction of motion
            trajectory_type: Type of trajectory
            jerk_limit: Optional jerk limit for s_curve trajectory
        """
        if duration is None and speed_percentage is None:
            raise RuntimeError("Error: You must provide either duration or speed_percentage.")
        center_str = ",".join(map(str, center))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        clockwise_str = "1" if clockwise else "0"
        timing_str = f"DURATION|{duration}" if duration is not None else f"SPEED|{speed_percentage}"
        traj_params = f"|{trajectory_type}"
        if trajectory_type == "s_curve" and jerk_limit is not None:
            traj_params += f"|{jerk_limit}"
        elif trajectory_type != "cubic":
            traj_params += "|DEFAULT"
        mode_params = f"|{center_mode}|{entry_mode}"
        command = (
            f"SMOOTH_CIRCLE|{center_str}|{radius}|{plane}|{frame}|{start_str}|"
            f"{timing_str}|{clockwise_str}{traj_params}{mode_params}"
        )
        return await self._send(command)

    async def smooth_arc_center(
        self,
        end_pose: list[float],
        center: list[float],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
    ) -> bool:
        """
        Execute a smooth arc motion defined by center point.

        Args:
            end_pose: [x, y, z, rx, ry, rz] end pose (mm and degrees)
            center: [x, y, z] arc center point in mm
            frame: Reference frame ('WRF' for World, 'TRF' for Tool)
            start_pose: Optional [x, y, z, rx, ry, rz] start pose
            duration: Time to complete the arc in seconds
            speed_percentage: Speed as percentage (1-100)
            clockwise: Direction of motion
            trajectory_type: Type of trajectory
            jerk_limit: Optional jerk limit for s_curve trajectory
        """
        if duration is None and speed_percentage is None:
            raise RuntimeError("Error: You must provide either a duration or a speed_percentage.")
        end_str = ",".join(map(str, end_pose))
        center_str = ",".join(map(str, center))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        clockwise_str = "1" if clockwise else "0"
        timing_str = f"DURATION|{duration}" if duration is not None else f"SPEED|{speed_percentage}"
        traj_params = f"|{trajectory_type}"
        if trajectory_type == "s_curve" and jerk_limit is not None:
            traj_params += f"|{jerk_limit}"
        elif trajectory_type != "cubic":
            traj_params += "|DEFAULT"
        command = (
            f"SMOOTH_ARC_CENTER|{end_str}|{center_str}|{frame}|{start_str}|"
            f"{timing_str}|{clockwise_str}{traj_params}"
        )
        return await self._send(command)

    async def smooth_arc_param(
        self,
        end_pose: list[float],
        radius: float,
        arc_angle: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        clockwise: bool = False,
    ) -> bool:
        """
        Execute a smooth arc motion defined parametrically (radius and angle).
        """
        if duration is None and speed_percentage is None:
            raise RuntimeError("You must provide either a duration or a speed_percentage.")
        end_str = ",".join(map(str, end_pose))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        timing_str = f"DURATION|{duration}" if duration is not None else f"SPEED|{speed_percentage}"
        parts = [
            "SMOOTH_ARC_PARAM",
            end_str,
            str(radius),
            str(arc_angle),
            frame,
            start_str,
            timing_str,
            trajectory_type,
        ]
        if trajectory_type == "s_curve" and jerk_limit is not None:
            parts.append(str(jerk_limit))
        if clockwise:
            parts.append("CW")
        return await self._send("|".join(parts))

    async def smooth_spline(
        self,
        waypoints: list[list[float]],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
    ) -> bool:
        """
        Execute a smooth spline motion through waypoints.

        Args:
            waypoints: List of [x, y, z, rx, ry, rz] poses (mm and degrees)
            frame: Reference frame ('WRF' for World, 'TRF' for Tool)
            start_pose: Optional [x, y, z, rx, ry, rz] start pose
            duration: Total time for the motion in seconds
            speed_percentage: Speed as percentage (1-100)
            trajectory_type: Type of trajectory
            jerk_limit: Optional jerk limit for s_curve trajectory
            wait_for_ack: Enable command tracking
            timeout: Timeout for acknowledgment
            non_blocking: Return immediately with command ID
        """
        if duration is None and speed_percentage is None:
            raise RuntimeError("Error: You must provide either duration or speed_percentage.")
        num_waypoints = len(waypoints)
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        timing_str = f"DURATION|{duration}" if duration is not None else f"SPEED|{speed_percentage}"
        waypoint_strs: list[str] = []
        for wp in waypoints:
            waypoint_strs.extend(map(str, wp))
        parts = ["SMOOTH_SPLINE", str(num_waypoints), frame, start_str, timing_str, trajectory_type]
        if trajectory_type == "s_curve" and jerk_limit is not None:
            parts.append(str(jerk_limit))
        elif trajectory_type == "s_curve":
            parts.append("DEFAULT")
        parts.extend(waypoint_strs)
        return await self._send("|".join(parts))

    async def smooth_helix(
        self,
        center: list[float],
        radius: float,
        pitch: float,
        height: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        clockwise: bool = False,
    ) -> bool:
        """
        Execute a smooth helical motion.

        Args:
            center: [x, y, z] helix center point in mm
            radius: Helix radius in mm
            pitch: Vertical distance per revolution in mm
            height: Total height of helix in mm
            frame: Reference frame ('WRF' for World, 'TRF' for Tool)
            trajectory_type: Type of trajectory
            jerk_limit: Optional jerk limit for s_curve trajectory
            start_pose: Optional [x, y, z, rx, ry, rz] start pose
            duration: Time to complete the helix in seconds
            speed_percentage: Speed as percentage (1-100)
            clockwise: Direction of motion
        """
        if duration is None and speed_percentage is None:
            raise RuntimeError("Error: You must provide either duration or speed_percentage.")
        center_str = ",".join(map(str, center))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        clockwise_str = "1" if clockwise else "0"
        timing_str = f"DURATION|{duration}" if duration is not None else f"SPEED|{speed_percentage}"
        traj_params = f"|{trajectory_type}"
        if trajectory_type == "s_curve" and jerk_limit is not None:
            traj_params += f"|{jerk_limit}"
        elif trajectory_type != "cubic":
            traj_params += "|DEFAULT"
        command = (
            f"SMOOTH_HELIX|{center_str}|{radius}|{pitch}|{height}|{frame}|{start_str}|"
            f"{timing_str}|{clockwise_str}{traj_params}"
        )
        return await self._send(command)

    async def smooth_blend(
        self,
        segments: list[dict],
        blend_time: float = 0.5,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
    ) -> bool:
        """
        Execute a blended motion through multiple segments.

        Args:
            segments: List of segment dictionaries
            blend_time: Time to blend between segments in seconds
            frame: Reference frame ('WRF' for World, 'TRF' for Tool)
            start_pose: Optional [x, y, z, rx, ry, rz] start pose
            duration: Total time for entire motion
            speed_percentage: Speed as percentage (1-100) for entire motion
            wait_for_ack: Enable command tracking
            timeout: Timeout for acknowledgment
            non_blocking: Return immediately with command ID
        """
        num_segments = len(segments)
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        if duration is None and speed_percentage is None:
            timing_str = "DEFAULT"
        elif duration is not None:
            timing_str = f"DURATION|{duration}"
        else:
            timing_str = f"SPEED|{speed_percentage}"

        segment_strs = []
        for seg in segments:
            seg_type = seg["type"]
            if seg_type == "LINE":
                end_str = ",".join(map(str, seg["end"]))
                seg_str = f"LINE|{end_str}|{seg.get('duration', 2.0)}"
            elif seg_type == "CIRCLE":
                center_str = ",".join(map(str, seg["center"]))
                clockwise_str = "1" if seg.get("clockwise", False) else "0"
                seg_str = f"CIRCLE|{center_str}|{seg['radius']}|{seg['plane']}|{seg.get('duration', 3.0)}|{clockwise_str}"
            elif seg_type == "ARC":
                end_str = ",".join(map(str, seg["end"]))
                center_str = ",".join(map(str, seg["center"]))
                clockwise_str = "1" if seg.get("clockwise", False) else "0"
                seg_str = f"ARC|{end_str}|{center_str}|{seg.get('duration', 2.0)}|{clockwise_str}"
            elif seg_type == "SPLINE":
                waypoints_str = ";".join([",".join(map(str, wp)) for wp in seg["waypoints"]])
                seg_str = (
                    f"SPLINE|{len(seg['waypoints'])}|{waypoints_str}|{seg.get('duration', 3.0)}"
                )
            else:
                continue
            segment_strs.append(seg_str)

        command = (
            f"SMOOTH_BLEND|{num_segments}|{blend_time}|{frame}|{start_str}|{timing_str}|"
            + "||".join(segment_strs)
        )
        return await self._send(command)

    async def smooth_waypoints(
        self,
        waypoints: list[list[float]],
        blend_radii: Literal["AUTO"] | list[float] = "AUTO",
        blend_mode: Literal["parabolic", "circular", "none"] = "parabolic",
        via_modes: list[str] | None = None,
        max_velocity: float = 100.0,
        max_acceleration: float = 500.0,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "quintic",
        duration: float | None = None,
    ) -> bool:
        """
        Execute a waypoint trajectory with blending.
        """
        if not waypoints or any(len(wp) != 6 for wp in waypoints):
            raise ValueError("Waypoints must be a non-empty list of [x,y,z,rx,ry,rz]")
        wp_str = ";".join(",".join(map(str, wp)) for wp in waypoints)
        if blend_radii == "AUTO":
            radii_str = "AUTO"
        else:
            if len(blend_radii) != max(0, len(waypoints) - 2):
                raise ValueError(f"Blend radii count must be {max(0, len(waypoints) - 2)}")
            radii_str = ",".join(map(str, blend_radii))
        if via_modes is None:
            via_modes_list: list[str] = ["via"] * len(waypoints)
        else:
            via_modes_list = list(via_modes)
        if len(via_modes_list) != len(waypoints):
            raise ValueError("via_modes length must match waypoints length")
        if any(vm not in ("via", "stop") for vm in via_modes_list):
            raise ValueError("via_modes entries must be 'via' or 'stop'")
        via_str = ",".join(via_modes_list)
        parts = [
            "SMOOTH_WAYPOINTS",
            wp_str,
            radii_str,
            blend_mode,
            via_str,
            str(max_velocity),
            str(max_acceleration),
            frame,
            trajectory_type,
        ]
        if duration is not None:
            parts.append(str(duration))
        return await self._send("|".join(parts))

    # --------------- Work coordinate helpers ---------------

    async def set_work_coordinate_offset(
        self,
        coordinate_system: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> bool:
        """
        Set work coordinate system offsets (G54-G59).

        Args:
            coordinate_system: Work coordinate system to set ('G54' through 'G59')
            x: X axis offset in mm (None to keep current)
            y: Y axis offset in mm (None to keep current)
            z: Z axis offset in mm (None to keep current)

        Returns:
            Success message, command ID, or dict with status details

        Example:
            # Set G54 origin to current position
            await client.set_work_coordinate_offset('G54', x=0, y=0, z=0)

            # Offset G55 by 100mm in X
            await client.set_work_coordinate_offset('G55', x=100)
        """
        valid_systems = ["G54", "G55", "G56", "G57", "G58", "G59"]
        if coordinate_system not in valid_systems:
            raise RuntimeError(
                f"Invalid coordinate system: {coordinate_system}. Must be one of {valid_systems}"
            )

        coord_num = int(coordinate_system[1:]) - 53  # G54=1, G55=2, etc.
        offset_params = []
        if x is not None:
            offset_params.append(f"X{x}")
        if y is not None:
            offset_params.append(f"Y{y}")
        if z is not None:
            offset_params.append(f"Z{z}")

        # Always select CS first, then apply offset if any
        ok = await self.execute_gcode(coordinate_system)
        if not ok:
            return False
        if offset_params:
            offset_cmd = f"G10 L2 P{coord_num} {' '.join(offset_params)}"
            return await self.execute_gcode(offset_cmd)
        return True

    async def zero_work_coordinates(
        self,
        coordinate_system: str = "G54",
    ) -> bool:
        """
        Set the current position as zero in the specified work coordinate system.

        Args:
            coordinate_system: Work coordinate system to zero ('G54' through 'G59')

        Returns:
            Success message, command ID, or dict with status details

        Example:
            # Set current position as origin in G54
            await client.zero_work_coordinates('G54')
        """
        # This sets the current position as 0,0,0 in the work coordinate system
        return await self.set_work_coordinate_offset(coordinate_system, x=0, y=0, z=0)
