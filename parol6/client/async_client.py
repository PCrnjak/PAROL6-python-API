"""
Async UDP client for PAROL6 robot control.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import List, Dict, Optional, Literal, cast

from ..protocol.types import Frame, Axis
from ..protocol import wire


class _UDPClientProtocol(asyncio.DatagramProtocol):
    def __init__(self, rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]]):
        self.rx_queue = rx_queue
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr) -> None:
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
            self._transport = transport  # type: ignore[assignment]
            self._protocol = protocol  # type: ignore[assignment]

    async def _send(self, message: str) -> str:
        """Fire-and-forget UDP send."""
        await self._ensure_endpoint()
        assert self._transport is not None
        self._transport.sendto(message.encode("utf-8"))
        return f"Sent: {message}"

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
            except asyncio.TimeoutError:
                if attempt < self.retries:
                    backoff = min(0.5, 0.05 * (2 ** attempt)) + random.uniform(0, 0.05)
                    await asyncio.sleep(backoff)
                    continue
            except Exception:
                break
        return None

    # --------------- Motion / Control ---------------

    async def ping(self) -> bool:
        """True if the controller responds with 'PONG'."""
        resp = await self._request("PING", bufsize=256)
        return bool(resp and resp.strip().upper().startswith("PONG"))

    async def home(self) -> str:
        return await self._send("HOME")

    async def stop(self) -> str:
        return await self._send("STOP")

    async def enable(self) -> str:
        return await self._send("ENABLE")

    async def disable(self) -> str:
        return await self._send("DISABLE")

    async def clear_error(self) -> str:
        return await self._send("CLEAR_ERROR")

    async def stream_on(self) -> str:
        self._stream_mode = True
        return await self._send("STREAM|ON")

    async def stream_off(self) -> str:
        self._stream_mode = False
        return await self._send("STREAM|OFF")

    async def set_com_port(self, port_str: str) -> str:
        if not port_str:
            return "No port provided"
        return await self._send(f"SET_PORT|{port_str}")

    # --------------- Status / Queries ---------------

    async def get_angles(self) -> list[float] | None:
        resp = await self._request("GET_ANGLES", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "ANGLES")
        return cast(List[float] | None, vals)

    async def get_io(self) -> list[int] | None:
        resp = await self._request("GET_IO", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "IO")
        return cast(List[int] | None, vals)

    async def get_gripper_status(self) -> list[int] | None:
        resp = await self._request("GET_GRIPPER", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "GRIPPER")
        return cast(List[int] | None, vals)

    async def get_speeds(self) -> list[float] | None:
        resp = await self._request("GET_SPEEDS", bufsize=1024)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "SPEEDS")
        return cast(List[float] | None, vals)

    async def get_pose(self) -> list[float] | None:
        """
        Returns 16-element transformation matrix (flattened) or None on failure.
        Expected wire format: "POSE|p0,p1,p2,...,p15"
        """
        resp = await self._request("GET_POSE", bufsize=2048)
        if not resp:
            return None
        vals = wire.decode_simple(resp, "POSE")
        return cast(List[float] | None, vals)

    async def get_gripper(self) -> list[int] | None:
        """Alias for get_gripper_status for compatibility."""
        return await self.get_gripper_status()

    async def get_status(self) -> dict | None:
        """
        Aggregate status.
        Expected format:
          STATUS|POSE=p0,p1,...,p15|ANGLES=a0,...,a5|IO=in1,in2,out1,out2,estop|GRIPPER=id,pos,spd,cur,status,obj
        Returns dict with keys: pose (list[float] len=16), angles (list[float] len=6),
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
        return cast(Dict, json.loads(resp.split("|", 1)[1]))

    # --------------- Helper methods ---------------

    async def get_pose_rpy(self) -> list[float] | None:
        """
        Get robot pose as [x, y, z, rx, ry, rz] in mm and degrees.
        Converts 4x4 matrix to xyz + RPY Euler angles.
        """
        pose_matrix = await self.get_pose()
        if not pose_matrix or len(pose_matrix) != 16:
            return None
        
        try:
            # Extract translation
            x, y, z = pose_matrix[3], pose_matrix[7], pose_matrix[11]
            
            # Extract rotation matrix elements 
            r11, _, _ = pose_matrix[0], pose_matrix[1], pose_matrix[2]
            r21, r22, r23 = pose_matrix[4], pose_matrix[5], pose_matrix[6] 
            r31, r32, r33 = pose_matrix[8], pose_matrix[9], pose_matrix[10]
            
            # Convert to RPY (XYZ convention) in degrees
            # Handle gimbal lock cases
            sy = math.sqrt(r11*r11 + r21*r21)
            
            if sy > 1e-6:  # Not at gimbal lock
                rx = math.atan2(r32, r33)
                ry = math.atan2(-r31, sy)  
                rz = math.atan2(r21, r11)
            else:  # Gimbal lock case
                rx = math.atan2(-r23, r22)
                ry = math.atan2(-r31, sy)
                rz = 0.0
            
            # Convert to degrees
            rx_deg = math.degrees(rx)
            ry_deg = math.degrees(ry)
            rz_deg = math.degrees(rz)
            
            return [x, y, z, rx_deg, ry_deg, rz_deg]
            
        except (ValueError, IndexError):
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
        poll_interval: float = 0.2,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5
    ) -> bool:
        """
        Wait for robot to stop moving with responsive polling.
        
        Args:
            timeout: Maximum time to wait in seconds
            settle_window: How long robot must be stable to be considered stopped
            poll_interval: How often to check status
            speed_threshold: Max joint speed to be considered stopped (steps/sec)
            angle_threshold: Max angle change to be considered stopped (degrees)  
            show_progress: Print dots to show progress
            
        Returns:
            True if robot stopped, False if timeout
        """
        
        start_time = time.time()
        last_angles = None
        settle_start = None

        while time.time() - start_time < timeout:
            # Try speed-based detection first (preferred)
            speeds = await self.get_speeds()
            if speeds:
                if max(abs(s) for s in speeds) < speed_threshold:
                    if settle_start is None:
                        settle_start = time.time()
                    elif time.time() - settle_start > settle_window:
                        return True
                else:
                    settle_start = None
            else:
                # Fallback to angle-based detection
                angles = await self.get_angles()
                if angles:
                    if last_angles is not None:
                        max_change = max(abs(a - b) for a, b in zip(angles, last_angles))
                        if max_change < angle_threshold:
                            if settle_start is None:
                                settle_start = time.time()
                            elif time.time() - settle_start > settle_window:
                                return True
                        else:
                            settle_start = None
                    last_angles = angles
            await asyncio.sleep(poll_interval)
        return False

    # --------------- Motion encoders ---------------

    async def move_joints(
        self,
        joint_angles: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,  # accepted but not sent
        profile: str | None = None,  # accepted but not sent
        tracking: str | None = None,  # accepted but not sent
    ) -> str:
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
    ) -> str:
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
    ) -> str:
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
    ) -> str:
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
    ) -> str:
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
    ) -> str:
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
    ) -> str:
        """
        Send a MULTIJOG command to jog multiple joints simultaneously for 'duration' seconds.
        """
        if len(joints) != len(speeds):
            raise ValueError("Error: The number of joints must match the number of speeds.")
        joints_str = ",".join(str(j) for j in joints)
        speeds_str = ",".join(str(s) for s in speeds)
        message = f"MULTIJOG|{joints_str}|{speeds_str}|{duration}"
        return await self._send(message)

    # --------------- IO / Gripper ---------------

    async def control_pneumatic_gripper(self, action: str, port: int) -> str:
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
    ) -> str:
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

    async def execute_gcode(self, gcode_line: str) -> str:
        """
        Execute a single GCODE line.
        """
        message = wire.encode_gcode(gcode_line)
        return await self._send(message)

    async def execute_gcode_program(self, program_lines: list[str]) -> str:
        """
        Execute a GCODE program from a list of lines.
        """
        for i, line in enumerate(program_lines):
            if "|" in line:
                raise SyntaxError(f"Line {i+1} contains invalid '|'")
        message = wire.encode_gcode_program_inline(program_lines)
        return await self._send(message)

    async def load_gcode_file(self, filepath: str) -> str:
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
        
        status_json = resp.split('|', 1)[1]
        return json.loads(status_json)

    async def pause_gcode_program(self) -> str:
        """Pause the currently running GCODE program."""
        return await self._send("GCODE_PAUSE")

    async def resume_gcode_program(self) -> str:
        """Resume a paused GCODE program."""
        return await self._send("GCODE_RESUME")

    async def stop_gcode_program(self) -> str:
        """Stop the currently running GCODE program."""
        return await self._send("GCODE_STOP")

    # --------------- Smooth motion ---------------

    async def smooth_circle(
        self,
        center: List[float],
        radius: float,
        plane: Literal["XY", "XZ", "YZ"] = "XY",
        frame: Literal["WRF", "TRF"] = "WRF",
        center_mode: Literal["ABSOLUTE", "TOOL", "RELATIVE"] = "ABSOLUTE",
        entry_mode: Literal["AUTO", "TANGENT", "DIRECT", "NONE"] = "NONE",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> str:
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
        end_pose: List[float],
        center: List[float],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> str:
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
            raise RuntimeError("Error: You must provide either duration or speed_percentage.")
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

    async def smooth_spline(
        self,
        waypoints: List[List[float]],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> str:
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
        center: List[float],
        radius: float,
        pitch: float,
        height: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
    ) -> str:
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
        segments: List[Dict],
        blend_time: float = 0.5,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
    ) -> str:
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
                seg_str = f"SPLINE|{len(seg['waypoints'])}|{waypoints_str}|{seg.get('duration', 3.0)}"
            else:
                continue
            segment_strs.append(seg_str)

        command = (
            f"SMOOTH_BLEND|{num_segments}|{blend_time}|{frame}|{start_str}|{timing_str}|"
            + "||".join(segment_strs)
        )
        return await self._send(command)

    # --------------- Work coordinate helpers ---------------

    async def set_work_coordinate_offset(
        self,
        coordinate_system: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> str:
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
            raise RuntimeError(f"Invalid coordinate system: {coordinate_system}. Must be one of {valid_systems}")

        coord_num = int(coordinate_system[1:]) - 53  # G54=1, G55=2, etc.
        offset_params = []
        if x is not None:
            offset_params.append(f"X{x}")
        if y is not None:
            offset_params.append(f"Y{y}")
        if z is not None:
            offset_params.append(f"Z{z}")

        # Always select CS first, then apply offset if any
        await self.execute_gcode(coordinate_system)
        if offset_params:
            offset_cmd = f"G10 L2 P{coord_num} {' '.join(offset_params)}"
            return await self.execute_gcode(offset_cmd)
        return f"Sent: {coordinate_system}"

    async def zero_work_coordinates(
        self,
        coordinate_system: str = "G54",
    ) -> str:
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
        return await self.set_work_coordinate_offset(
            coordinate_system, 
            x=0, y=0, z=0
        )
