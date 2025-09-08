"""
Async UDP client for PAROL6 robot control.

Provides an async-first interface for all robot operations with optional acknowledgment tracking.
"""

import json
import os
import socket
import time
import math
from typing import Union, List, Optional, Literal, Dict

from ..protocol.types import Frame, Axis, IOStatus, GripperStatus, StatusAggregate, TrackingStatus
from ..utils.tracking import send_robot_command_tracked, send_and_wait


class AsyncRobotClient:
    """
    Async UDP client for the PAROL6 headless controller.

    This client provides async methods for all robot operations:
    - Motion commands (home, stop, move_joints, move_pose, move_cartesian, jog)
    - Status queries (get_angles, get_io, get_gripper_status, get_status)
    - Control commands (enable, disable, clear_error, set_com_port, stream on/off)
    - GCODE operations

    All methods support optional acknowledgment tracking for reliable operation.
    """

    def __init__(
        self, 
        host: str = "127.0.0.1", 
        port: int = 5001, 
        timeout: float = 2.0, 
        retries: int = 1, 
        ack_port: int = 5002
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.ack_port = ack_port

    # --------------- Internal helpers ---------------

    async def _send(self, message: str) -> str:
        """Fire-and-forget UDP send."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode("utf-8"), (self.host, self.port))
        return f"Sent: {message}"

    async def _request(self, message: str, bufsize: int = 2048) -> str | None:
        """Send a request and wait for a UDP response (with retry)."""
        for _ in range(self.retries + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(self.timeout)
                    sock.sendto(message.encode("utf-8"), (self.host, self.port))
                    data, _ = sock.recvfrom(bufsize)
                    return data.decode("utf-8")
            except TimeoutError:
                continue
            except Exception:
                break
        return None

    async def _send_tracked(self, message: str, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Send with optional tracking support."""
        # Check if tracking is explicitly disabled
        disable_tracking = os.getenv("PAROL6_DISABLE_TRACKING", "").lower() in ("1", "true", "yes", "on")
        
        # Only use tracking if wait_for_ack or non_blocking is requested AND tracking is not disabled
        if (wait_for_ack or non_blocking) and not disable_tracking:
            result = send_and_wait(message, timeout, non_blocking)
            return result if result is not None else {"status": "ERROR", "details": "Send failed"}
        elif wait_for_ack and disable_tracking:
            # If ACK was requested but tracking is disabled, return a NO_TRACKING response
            await self._send(message)
            return {"status": "NO_TRACKING", "details": "Tracking disabled by environment", "completed": True, "command_id": None}
        else:
            # Fire-and-forget send without initializing tracker
            return await self._send(message)

    # --------------- Motion / Control ---------------

    async def ping(self) -> bool:
        """True if the controller responds with a 'PONG' message."""
        resp = await self._request("PING", bufsize=256)
        return bool(resp and resp.strip().upper().startswith("PONG"))

    async def home(self, wait_for_ack: bool = False, timeout: float = 30.0, non_blocking: bool = False) -> Union[str, dict]:
        """Send HOME command."""
        return await self._send_tracked("HOME", wait_for_ack, timeout, non_blocking)

    async def stop(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Send STOP command."""
        return await self._send_tracked("STOP", wait_for_ack, timeout, non_blocking)

    async def enable(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Requires server support for ENABLE command."""
        return await self._send_tracked("ENABLE", wait_for_ack, timeout, non_blocking)

    async def disable(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Requires server support for DISABLE command."""
        return await self._send_tracked("DISABLE", wait_for_ack, timeout, non_blocking)

    async def clear_error(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Requires server support for CLEAR_ERROR command."""
        return await self._send_tracked("CLEAR_ERROR", wait_for_ack, timeout, non_blocking)

    async def stream_on(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Enable zero-queue streaming mode on the server."""
        return await self._send_tracked("STREAM|ON", wait_for_ack, timeout, non_blocking)

    async def stream_off(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """Disable zero-queue streaming mode on the server."""
        return await self._send_tracked("STREAM|OFF", wait_for_ack, timeout, non_blocking)

    async def set_com_port(self, port_str: str, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        """
        Best-effort COM port change. Requires server support to take effect immediately.
        """
        if not port_str:
            return "No port provided"
        return await self._send_tracked(f"SET_PORT|{port_str}", wait_for_ack, timeout, non_blocking)

    # --------------- Status / Queries ---------------

    async def get_angles(self) -> list[float] | None:
        """
        Returns list of 6 angles in degrees or None on failure.
        Expected wire format: "ANGLES|j1,j2,j3,j4,j5,j6"
        """
        resp = await self._request("GET_ANGLES", bufsize=1024)
        if not resp:
            return None
        parts = resp.split("|")
        if len(parts) != 2 or parts[0] != "ANGLES":
            return None
        return [float(v) for v in parts[1].split(",")]

    async def get_io(self) -> list[int] | None:
        """
        Returns [IN1, IN2, OUT1, OUT2, ESTOP] or None on failure.
        Expected wire format: "IO|in1,in2,out1,out2,estop"
        """
        resp = await self._request("GET_IO", bufsize=1024)
        if not resp:
            return None
        parts = resp.split("|")
        if len(parts) != 2 or parts[0] != "IO":
            return None
        return [int(v) for v in parts[1].split(",")]

    async def get_gripper_status(self) -> list[int] | None:
        """
        Returns [ID, Position, Speed, Current, StatusByte, ObjectDetected] or None.
        Expected wire format: "GRIPPER|id,pos,spd,cur,status,obj"
        """
        resp = await self._request("GET_GRIPPER", bufsize=1024)
        if not resp:
            return None
        parts = resp.split("|")
        if len(parts) != 2 or parts[0] != "GRIPPER":
            return None
        return [int(v) for v in parts[1].split(",")]

    async def get_speeds(self) -> list[float] | None:
        """
        Returns list of 6 joint speeds in steps/sec or None on failure.
        Expected wire format: "SPEEDS|s1,s2,s3,s4,s5,s6"
        """
        resp = await self._request("GET_SPEEDS", bufsize=1024)
        if not resp:
            return None
        parts = resp.split("|")
        if len(parts) != 2 or parts[0] != "SPEEDS":
            return None
        return [float(v) for v in parts[1].split(",")]

    async def get_pose(self) -> list[float] | None:
        """
        Returns 16-element transformation matrix (flattened) or None on failure.
        Expected wire format: "POSE|p0,p1,p2,...,p15"
        """
        resp = await self._request("GET_POSE", bufsize=2048)
        if not resp:
            return None
        parts = resp.split("|")
        if len(parts) != 2 or parts[0] != "POSE":
            return None
        return [float(v) for v in parts[1].split(",")]

    async def get_gripper(self) -> list[int] | None:
        """Alias for get_gripper_status for compatibility."""
        return await self.get_gripper_status()

    async def get_status(self) -> dict | None:
        """
        Aggregate status if supported by controller.
        Expected format:
          STATUS|POSE=p0,p1,...,p15|ANGLES=a0,...,a5|IO=in1,in2,out1,out2,estop|GRIPPER=id,pos,spd,cur,status,obj
        Returns dict with keys: pose (list[float] len=16), angles (list[float] len=6),
                                io (list[int] len=5), gripper (list[int] len>=6)
        """
        resp = await self._request("GET_STATUS", bufsize=4096)
        if not resp or not resp.startswith("STATUS|"):
            return None
        # Split top-level sections after "STATUS|"
        sections = resp.split("|")[1:]
        result: dict[str, object] = {
            "pose": None,
            "angles": None,
            "io": None,
            "gripper": None,
        }
        for sec in sections:
            if sec.startswith("POSE="):
                vals = [float(x) for x in sec[len("POSE=") :].split(",") if x]
                result["pose"] = vals
            elif sec.startswith("ANGLES="):
                vals = [float(x) for x in sec[len("ANGLES=") :].split(",") if x]
                result["angles"] = vals
            elif sec.startswith("IO="):
                vals = [int(x) for x in sec[len("IO=") :].split(",") if x]
                result["io"] = vals
            elif sec.startswith("GRIPPER="):
                vals = [int(x) for x in sec[len("GRIPPER=") :].split(",") if x]
                result["gripper"] = vals
        return result

    # --------------- Helper methods for compatibility ---------------

    async def get_pose_rpy(self) -> list[float] | None:
        """
        Get robot pose as [x, y, z, rx, ry, rz] in mm and degrees.
        Converts 4x4 matrix to xyz + RPY Euler angles.
        """
        pose_matrix = await self.get_pose()
        if not pose_matrix or len(pose_matrix) != 16:
            return None
        
        try:
            # Extract translation (convert to mm if needed - assume matrix is in mm)
            x, y, z = pose_matrix[3], pose_matrix[7], pose_matrix[11]
            
            # Extract rotation matrix elements 
            r11, r12, r13 = pose_matrix[0], pose_matrix[1], pose_matrix[2]
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
                rz = 0
            
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
        
        max_speed = max(abs(s) for s in speeds)
        return max_speed < threshold_speed

    async def wait_until_stopped(
        self,
        timeout: float = 90.0,
        settle_window: float = 1.0,
        poll_interval: float = 0.2,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5,
        show_progress: bool = False
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
        import asyncio
        
        start_time = time.time()
        last_angles = None
        settle_start = None
        last_progress = 0
        
        if show_progress:
            print("Waiting for robot to stop...", end="", flush=True)
        
        while time.time() - start_time < timeout:
            # Try speed-based detection first (preferred)
            speeds = await self.get_speeds()
            if speeds:
                max_speed = max(abs(s) for s in speeds)
                if max_speed < speed_threshold:
                    if settle_start is None:
                        settle_start = time.time()
                    elif time.time() - settle_start > settle_window:
                        if show_progress:
                            print(" stopped via speeds!")
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
                                if show_progress:
                                    print(" stopped via angle delta!")
                                return True
                        else:
                            settle_start = None
                    last_angles = angles
            
            # Show progress dots every few seconds
            if show_progress and int(time.time() - start_time) > last_progress:
                print(".", end="", flush=True)
                last_progress = int(time.time() - start_time)
            
            await asyncio.sleep(poll_interval)
        
        if show_progress:
            print(" timeout!")
        return False

    # --------------- Extended controls / motion ---------------

    async def move_joints(
        self,
        joint_angles: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,  # kept for API compatibility; not sent
        profile: str | None = None,  # kept for API compatibility; not sent
        tracking: str | None = None,  # kept for API compatibility; not sent
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send minimal MOVEJOINT wire format expected by the server:
          MOVEJOINT|j1|j2|j3|j4|j5|j6|DUR|SPD
        Use "NONE" for omitted duration/speed.
        """
        if duration is None and speed_percentage is None:
            error = "Error: You must provide either a duration or a speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        angles_str = "|".join(str(a) for a in joint_angles)
        dur_str = "NONE" if duration is None else str(duration)
        spd_str = "NONE" if speed_percentage is None else str(speed_percentage)
        message = f"MOVEJOINT|{angles_str}|{dur_str}|{spd_str}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def move_pose(
        self,
        pose: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,  # kept; not sent
        profile: str | None = None,  # kept; not sent
        tracking: str | None = None,  # kept; not sent
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send minimal MOVEPOSE wire format expected by the server:
          MOVEPOSE|x|y|z|rx|ry|rz|DUR|SPD
        Use "NONE" for omitted duration/speed.
        """
        if duration is None and speed_percentage is None:
            error = "Error: You must provide either a duration or a speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        pose_str = "|".join(str(v) for v in pose)
        dur_str = "NONE" if duration is None else str(duration)
        spd_str = "NONE" if speed_percentage is None else str(speed_percentage)
        message = f"MOVEPOSE|{pose_str}|{dur_str}|{spd_str}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def move_cartesian(
        self,
        pose: list[float],
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,  # kept; not sent
        profile: str | None = None,  # kept; not sent
        tracking: str | None = None,  # kept; not sent
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send minimal MOVECART wire format expected by the server:
          MOVECART|x|y|z|rx|ry|rz|DUR|SPD
        Use "NONE" for omitted duration/speed.
        """
        if duration is None and speed_percentage is None:
            error = "Error: You must provide either a duration or a speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        pose_str = "|".join(str(v) for v in pose)
        dur_str = "NONE" if duration is None else str(duration)
        spd_str = "NONE" if speed_percentage is None else str(speed_percentage)
        message = f"MOVECART|{pose_str}|{dur_str}|{spd_str}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def move_cartesian_rel_trf(
        self,
        deltas: list[float],  # [dx, dy, dz, rx, ry, rz] in mm/deg relative to TRF
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send a MOVECARTRELTRF (relative straight-line in TRF) command.
        Provide either duration or speed_percentage (1..100).
        Optional: accel_percentage, trajectory profile, and tracking mode.
        """
        delta_str = "|".join(str(v) for v in deltas)
        dur_str = "NONE" if duration is None else str(duration)
        spd_str = "NONE" if speed_percentage is None else str(speed_percentage)
        acc_str = "NONE" if accel_percentage is None else str(int(accel_percentage))
        prof_str = (profile or "NONE").upper()
        track_str = (tracking or "NONE").upper()
        message = f"MOVECARTRELTRF|{delta_str}|{dur_str}|{spd_str}|{acc_str}|{prof_str}|{track_str}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def jog_joint(
        self,
        joint_index: int,
        speed_percentage: int,
        duration: float | None = None,
        distance_deg: float | None = None,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send a JOG command for a single joint (0..5 positive, 6..11 negative for reverse).
        duration and distance_deg are optional; at least one should be provided for one-shot jog.
        For press-and-hold UI, send short duration repeatedly.
        """
        if duration is None and distance_deg is None:
            error = "Error: You must provide either a duration or a distance_deg."
            return {'status': 'INVALID', 'details': error}
        
        dur_str = "NONE" if duration is None else str(duration)
        dist_str = "NONE" if distance_deg is None else str(distance_deg)
        message = f"JOG|{joint_index}|{speed_percentage}|{dur_str}|{dist_str}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def jog_cartesian(
        self, 
        frame: Frame, 
        axis: Axis, 
        speed_percentage: int, 
        duration: float,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send a CARTJOG command (frame 'TRF' or 'WRF', axis in {X+/X-/Y+/.../RZ-}).
        """
        message = f"CARTJOG|{frame}|{axis}|{speed_percentage}|{duration}"
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def jog_multiple(
        self, 
        joints: list[int], 
        speeds: list[float], 
        duration: float,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Send a MULTIJOG command to jog multiple joints simultaneously for 'duration' seconds.
        """
        if len(joints) != len(speeds):
            error = "Error: The number of joints must match the number of speeds."
            return {'status': 'INVALID', 'details': error}
        
        joints_str = ",".join(str(j) for j in joints)
        speeds_str = ",".join(str(s) for s in speeds)
        message = f"MULTIJOG|{joints_str}|{speeds_str}|{duration}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    # --------------- IO / Gripper ---------------

    async def control_pneumatic_gripper(
        self, 
        action: str, 
        port: int,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Control pneumatic gripper via digital outputs.
        action: 'open' or 'close'
        port: 1 or 2
        """
        action = action.lower()
        if action not in ("open", "close"):
            return {'status': 'INVALID', 'details': 'Invalid pneumatic action'}
        if port not in (1, 2):
            return {'status': 'INVALID', 'details': 'Invalid pneumatic port'}
        
        message = f"PNEUMATICGRIPPER|{action}|{port}"
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def control_electric_gripper(
        self,
        action: str,
        position: int | None = 255,
        speed: int | None = 150,
        current: int | None = 500,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Control electric gripper.
        action: 'move' or 'calibrate'
        position: 0..255
        speed: 0..255
        current: 100..1000 (mA)
        """
        action = action.lower()
        if action not in ("move", "calibrate"):
            return {'status': 'INVALID', 'details': 'Invalid electric gripper action'}
        pos = 0 if position is None else int(position)
        spd = 0 if speed is None else int(speed)
        cur = 100 if current is None else int(current)
        
        message = f"ELECTRICGRIPPER|{action}|{pos}|{spd}|{cur}"
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    # --------------- GCODE operations ---------------

    async def execute_gcode(
        self,
        gcode_line: str,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Execute a single GCODE line.
        """
        message = f"GCODE|{gcode_line}"
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def execute_gcode_program(
        self,
        program_lines: list[str],
        wait_for_ack: bool = False,
        timeout: float = 30.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Execute a GCODE program from a list of lines.
        """
        # Validate program lines don't contain problematic characters
        for i, line in enumerate(program_lines):
            if '|' in line:
                error_msg = f"Line {i+1} contains pipe character '|' which is not allowed"
                return {'status': 'INVALID', 'details': error_msg}
        
        # Join lines with semicolons for inline program
        program_str = ';'.join(program_lines)
        message = f"GCODE_PROGRAM|INLINE|{program_str}"
        
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def load_gcode_file(
        self,
        filepath: str,
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """
        Load and execute a GCODE program from a file.
        """
        message = f"GCODE_PROGRAM|FILE|{filepath}"
        return await self._send_tracked(message, wait_for_ack, timeout, non_blocking)

    async def get_gcode_status(self) -> dict | None:
        """
        Get the current status of the GCODE interpreter.
        """
        resp = await self._request("GET_GCODE_STATUS", bufsize=2048)
        if not resp or not resp.startswith("GCODE_STATUS|"):
            return None
        
        status_json = resp.split('|', 1)[1]
        try:
            return json.loads(status_json)
        except json.JSONDecodeError:
            return None

    async def pause_gcode_program(
        self,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """Pause the currently running GCODE program."""
        return await self._send_tracked("GCODE_PAUSE", wait_for_ack, timeout, non_blocking)

    async def resume_gcode_program(
        self,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """Resume a paused GCODE program."""
        return await self._send_tracked("GCODE_RESUME", wait_for_ack, timeout, non_blocking)

    async def stop_gcode_program(
        self,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
        """Stop the currently running GCODE program."""
        return await self._send_tracked("GCODE_STOP", wait_for_ack, timeout, non_blocking)

    # --------------- Smooth motion commands ---------------

    async def smooth_circle(
        self,
        center: List[float],
        radius: float,
        plane: Literal['XY', 'XZ', 'YZ'] = 'XY',
        frame: Literal['WRF', 'TRF'] = 'WRF',
        center_mode: Literal['ABSOLUTE', 'TOOL', 'RELATIVE'] = 'ABSOLUTE',
        entry_mode: Literal['AUTO', 'TANGENT', 'DIRECT', 'NONE'] = 'NONE',
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
        jerk_limit: Optional[float] = None,
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
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
            wait_for_ack: Enable command tracking
            timeout: Timeout for acknowledgment
            non_blocking: Return immediately with command ID
        """
        if duration is None and speed_percentage is None:
            error = "Error: You must provide either duration or speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        center_str = ",".join(map(str, center))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        clockwise_str = "1" if clockwise else "0"
        
        # Format timing
        if duration is not None:
            timing_str = f"DURATION|{duration}"
        else:
            timing_str = f"SPEED|{speed_percentage}"
        
        # Add trajectory type and new parameters
        traj_params = f"|{trajectory_type}"
        if trajectory_type == 's_curve' and jerk_limit is not None:
            traj_params += f"|{jerk_limit}"
        elif trajectory_type != 'cubic':
            traj_params += "|DEFAULT"  # Use default jerk limit for s_curve
        
        # Add center_mode and entry_mode parameters
        mode_params = f"|{center_mode}|{entry_mode}"
        
        command = f"SMOOTH_CIRCLE|{center_str}|{radius}|{plane}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}{mode_params}"
        
        return await self._send_tracked(command, wait_for_ack, timeout, non_blocking)

    async def smooth_arc_center(
        self,
        end_pose: List[float],
        center: List[float],
        frame: Literal['WRF', 'TRF'] = 'WRF',
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
        jerk_limit: Optional[float] = None,
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
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
            wait_for_ack: Enable command tracking
            timeout: Timeout for acknowledgment
            non_blocking: Return immediately with command ID
        """
        if duration is None and speed_percentage is None:
            error = "Error: You must provide either duration or speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        end_str = ",".join(map(str, end_pose))
        center_str = ",".join(map(str, center))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        clockwise_str = "1" if clockwise else "0"
        
        # Format timing
        if duration is not None:
            timing_str = f"DURATION|{duration}"
        else:
            timing_str = f"SPEED|{speed_percentage}"
        
        # Add trajectory type if not default
        traj_params = f"|{trajectory_type}"
        if trajectory_type == 's_curve' and jerk_limit is not None:
            traj_params += f"|{jerk_limit}"
        elif trajectory_type != 'cubic':
            traj_params += "|DEFAULT"
        
        command = f"SMOOTH_ARC_CENTER|{end_str}|{center_str}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}"
        
        return await self._send_tracked(command, wait_for_ack, timeout, non_blocking)

    async def smooth_spline(
        self,
        waypoints: List[List[float]],
        frame: Literal['WRF', 'TRF'] = 'WRF',
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
        jerk_limit: Optional[float] = None,
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
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
            error = "Error: You must provide either duration or speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        num_waypoints = len(waypoints)
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        
        # Format timing
        if duration is not None:
            timing_str = f"DURATION|{duration}"
        else:
            timing_str = f"SPEED|{speed_percentage}"
        
        # Format waypoints - flatten each waypoint's 6 values
        waypoint_strs = []
        for wp in waypoints:
            waypoint_strs.extend(map(str, wp))
        
        # Build command with trajectory type
        command_parts = [f"SMOOTH_SPLINE", str(num_waypoints), frame, start_str, timing_str]
        
        # Add trajectory type
        command_parts.append(trajectory_type)
        if trajectory_type == 's_curve' and jerk_limit is not None:
            command_parts.append(str(jerk_limit))
        elif trajectory_type == 's_curve':
            command_parts.append("DEFAULT")
        
        # Add waypoints
        command_parts.extend(waypoint_strs)
        command = "|".join(command_parts)
        
        return await self._send_tracked(command, wait_for_ack, timeout, non_blocking)

    async def smooth_helix(
        self,
        center: List[float],
        radius: float,
        pitch: float,
        height: float,
        frame: Literal['WRF', 'TRF'] = 'WRF',
        trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
        jerk_limit: Optional[float] = None,
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
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
            wait_for_ack: Enable command tracking
            timeout: Timeout for acknowledgment
            non_blocking: Return immediately with command ID
        """
        if duration is None and speed_percentage is None:
            error = "Error: You must provide either duration or speed_percentage."
            return {'status': 'INVALID', 'details': error}
        
        center_str = ",".join(map(str, center))
        start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
        clockwise_str = "1" if clockwise else "0"
        
        # Format timing
        if duration is not None:
            timing_str = f"DURATION|{duration}"
        else:
            timing_str = f"SPEED|{speed_percentage}"
        
        # Add trajectory type parameters
        traj_params = f"|{trajectory_type}"
        if trajectory_type == 's_curve' and jerk_limit is not None:
            traj_params += f"|{jerk_limit}"
        elif trajectory_type != 'cubic':
            traj_params += "|DEFAULT"  # Use default jerk limit for s_curve
        
        command = f"SMOOTH_HELIX|{center_str}|{radius}|{pitch}|{height}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}"
        
        return await self._send_tracked(command, wait_for_ack, timeout, non_blocking)

    async def smooth_blend(
        self,
        segments: List[Dict],
        blend_time: float = 0.5,
        frame: Literal['WRF', 'TRF'] = 'WRF',
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        wait_for_ack: bool = False,
        timeout: float = 15.0,
        non_blocking: bool = False
    ) -> Union[str, dict]:
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
        
        # Format timing
        if duration is None and speed_percentage is None:
            # Use individual segment durations
            timing_str = "DEFAULT"
        elif duration is not None:
            timing_str = f"DURATION|{duration}"
        else:
            timing_str = f"SPEED|{speed_percentage}"
        
        # Format segments
        segment_strs = []
        for seg in segments:
            seg_type = seg['type']
            
            if seg_type == 'LINE':
                end_str = ",".join(map(str, seg['end']))
                seg_str = f"LINE|{end_str}|{seg.get('duration', 2.0)}"
                
            elif seg_type == 'CIRCLE':
                center_str = ",".join(map(str, seg['center']))
                clockwise_str = "1" if seg.get('clockwise', False) else "0"
                seg_str = f"CIRCLE|{center_str}|{seg['radius']}|{seg['plane']}|{seg.get('duration', 3.0)}|{clockwise_str}"
                
            elif seg_type == 'ARC':
                end_str = ",".join(map(str, seg['end']))
                center_str = ",".join(map(str, seg['center']))
                clockwise_str = "1" if seg.get('clockwise', False) else "0"
                seg_str = f"ARC|{end_str}|{center_str}|{seg.get('duration', 2.0)}|{clockwise_str}"
                
            elif seg_type == 'SPLINE':
                waypoints_str = ";".join([",".join(map(str, wp)) for wp in seg['waypoints']])
                seg_str = f"SPLINE|{len(seg['waypoints'])}|{waypoints_str}|{seg.get('duration', 3.0)}"
                
            else:
                continue
                
            segment_strs.append(seg_str)
        
        # Build command with || separators between segments
        command = f"SMOOTH_BLEND|{num_segments}|{blend_time}|{frame}|{start_str}|{timing_str}|" + "||".join(segment_strs)
        
        return await self._send_tracked(command, wait_for_ack, timeout, non_blocking)
