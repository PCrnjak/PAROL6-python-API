"""
Query commands that return immediate status information.

All query commands are marked as is_immediate=True to bypass the command queue
and execute immediately when received.
"""

from __future__ import annotations

import json
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, TYPE_CHECKING

from parol6.commands.base import CommandBase, ExecutionStatus, ExecutionStatusCode
from parol6.server.command_registry import register_command
import parol6.PAROL6_ROBOT as PAROL6_ROBOT

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@dataclass
@register_command("GET_POSE")
class GetPoseCommand(CommandBase):
    """Get current robot pose matrix."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_POSE command."""
        if parts[0].upper() == "GET_POSE":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return pose data."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
            current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A
            pose_flat = current_pose_matrix.flatten()
            pose_str = ",".join(map(str, pose_flat))
            response_message = f"POSE|{pose_str}"
            # Use UDPTransport API
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get pose: {e}")
            return ExecutionStatus.failed(f"Failed to get pose: {e}")
        
        self.finish()
        return ExecutionStatus.completed("Pose sent")


@dataclass
@register_command("GET_ANGLES")
class GetAnglesCommand(CommandBase):
    """Get current joint angles in degrees."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_ANGLES command."""
        if parts[0].upper() == "GET_ANGLES":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return angle data."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)]
            angles_deg = np.rad2deg(angles_rad)
            angles_str = ",".join(map(str, angles_deg))
            response_message = f"ANGLES|{angles_str}"
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get angles: {e}")
            return ExecutionStatus.failed(f"Failed to get angles: {e}")
        
        self.finish()
        return ExecutionStatus.completed("Angles sent")


@dataclass
@register_command("GET_IO")
class GetIOCommand(CommandBase):
    """Get current I/O status."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_IO command."""
        if parts[0].upper() == "GET_IO":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return I/O data."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            io_status_str = ",".join(map(str, state.InOut_in[:5]))
            response_message = f"IO|{io_status_str}"
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get I/O: {e}")
            return ExecutionStatus.failed(f"Failed to get I/O: {e}")
        
        self.finish()
        return ExecutionStatus.completed("I/O sent")


@dataclass
@register_command("GET_GRIPPER")
class GetGripperCommand(CommandBase):
    """Get current gripper status."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_GRIPPER command."""
        if parts[0].upper() == "GET_GRIPPER":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return gripper data."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            gripper_status_str = ",".join(map(str, state.Gripper_data_in))
            response_message = f"GRIPPER|{gripper_status_str}"
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get gripper status: {e}")
            return ExecutionStatus.failed(f"Failed to get gripper status: {e}")
        
        self.finish()
        return ExecutionStatus.completed("Gripper sent")


@dataclass
@register_command("GET_SPEEDS")
class GetSpeedsCommand(CommandBase):
    """Get current joint speeds."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_SPEEDS command."""
        if parts[0].upper() == "GET_SPEEDS":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return speed data."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            speeds_str = ",".join(map(str, state.Speed_in))
            response_message = f"SPEEDS|{speeds_str}"
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get speeds: {e}")
            return ExecutionStatus.failed(f"Failed to get speeds: {e}")
        
        self.finish()
        return ExecutionStatus.completed("Speeds sent")


@dataclass
@register_command("GET_STATUS")
class GetStatusCommand(CommandBase):
    """Get aggregated robot status (pose, angles, I/O, gripper)."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_STATUS command."""
        if parts[0].upper() == "GET_STATUS":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return aggregated status."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            # Get pose
            try:
                q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
                current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A
                pose_flat = current_pose_matrix.flatten()
                pose_str = ",".join(map(str, pose_flat))
            except Exception:
                pose_str = ",".join(["0"] * 16)
            
            angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)]
            angles_deg = np.rad2deg(angles_rad)
            angles_str = ",".join(map(str, angles_deg))
            
            io_status_str = ",".join(map(str, state.InOut_in[:5]))
            gripper_status_str = ",".join(map(str, state.Gripper_data_in))
            
            response_message = f"STATUS|POSE={pose_str}|ANGLES={angles_str}|IO={io_status_str}|GRIPPER={gripper_status_str}"
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get status: {e}")
            return ExecutionStatus.failed(f"Failed to get status: {e}")
        
        self.finish()
        return ExecutionStatus.completed("Status sent")


@dataclass
@register_command("GET_GCODE_STATUS")
class GetGcodeStatusCommand(CommandBase):
    """Get GCODE interpreter status."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_GCODE_STATUS command."""
        if parts[0].upper() == "GET_GCODE_STATUS":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
        if gcode_interpreter is not None:
            self.gcode_interpreter = gcode_interpreter
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return GCODE status."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            if self.gcode_interpreter:
                gcode_status = self.gcode_interpreter.get_status()
            else:
                gcode_status = {
                    'is_running': False,
                    'is_paused': False,
                    'current_line': None,
                    'total_lines': 0,
                    'state': {}
                }
            
            status_json = json.dumps(gcode_status)
            response_message = f"GCODE_STATUS|{status_json}"
            self.udp_transport.send_response(response_message, self.addr)
        except Exception as e:
            self.fail(f"Failed to get GCODE status: {e}")
            return ExecutionStatus.failed(f"Failed to get GCODE status: {e}")
        
        self.finish()
        return ExecutionStatus.completed("GCODE status sent")


@dataclass
@register_command("GET_SERVER_STATE")
class GetServerStateCommand(CommandBase):
    """Get server state information."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_SERVER_STATE command."""
        if parts[0].upper() == "GET_SERVER_STATE":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState') -> None:
        """No setup needed for query commands."""
        pass
    
    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs) -> bool:
        """Execute immediately and return server state."""
        udp_transport = kwargs.get('udp_transport')
        addr = kwargs.get('addr')
        state = kwargs.get('state')
        
        if not udp_transport or not addr:
            self.fail("Missing UDP transport or address")
            return True
        
        try:
            # Build state information
            server_state = {
                "listening": {
                    "transport": "udp",
                    "address": f"{state.ip}:{state.port}" if state else "127.0.0.1:5001"
                },
                "serial_connected": bool(state and state.ser and getattr(state.ser, "is_open", False)),
                "homed": any(bool(h) for h in Homed_in) if isinstance(Homed_in, list) else False,
                "queue_depth": len(state.command_queue) if state and hasattr(state, 'command_queue') else 0,
                "active_command": type(state.active_command).__name__ if state and state.active_command else None,
                "stream_mode": bool(state.stream_mode) if state else False,
                "uptime_s": float(time.time() - state.start_time) if state and state.start_time > 0 else 0.0,
            }
            
            payload = f"SERVER_STATE|{json.dumps(server_state)}"
            udp_transport.send(payload, addr)
        except Exception as e:
            self.fail(f"Failed to get server state: {e}")
        
        self.finish()
        return True


@dataclass
@register_command("PING")
class PingCommand(CommandBase):
    """Respond to ping requests."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a PING command."""
        if parts[0].upper() == "PING":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return PONG."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            self.udp_transport.send_response("PONG", self.addr)
        except Exception as e:
            self.fail(f"Failed to send PONG: {e}")
            return ExecutionStatus.failed(f"Failed to send PONG: {e}")
        
        self.finish()
        return ExecutionStatus.completed("PONG")
