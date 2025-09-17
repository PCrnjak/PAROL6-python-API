"""
Query commands that return immediate status information.

All query commands are marked as is_immediate=True to bypass the command queue
and execute immediately when received.
"""

from __future__ import annotations

import json
import time
import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING

from parol6.commands.base import QueryCommand, ExecutionStatus, ExecutionStatusCode
from parol6.server.command_registry import register_command
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.server.status_cache import get_cache
from parol6 import config as cfg

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command("GET_POSE")
class GetPoseCommand(QueryCommand):
    """Get current robot pose matrix."""
    
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


@register_command("GET_ANGLES")
class GetAnglesCommand(QueryCommand):
    """Get current joint angles in degrees."""
    
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


@register_command("GET_IO")
class GetIOCommand(QueryCommand):
    """Get current I/O status."""
    
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


@register_command("GET_GRIPPER")
class GetGripperCommand(QueryCommand):
    """Get current gripper status."""
    
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


@register_command("GET_SPEEDS")
class GetSpeedsCommand(QueryCommand):
    """Get current joint speeds."""
    
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


@register_command("GET_STATUS")
class GetStatusCommand(QueryCommand):
    """Get aggregated robot status (pose, angles, I/O, gripper) from cache."""
    
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
        """Execute immediately and return cached aggregated status (ASCII)."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            payload = get_cache().to_ascii()
            self.udp_transport.send_response(payload, self.addr)
        except Exception as e:
            self.fail(f"Failed to get status: {e}")
            return ExecutionStatus.failed(f"Failed to get status: {e}")
        
        self.finish()
        return ExecutionStatus.completed("Status sent")


@register_command("GET_GCODE_STATUS")
class GetGcodeStatusCommand(QueryCommand):
    """Get GCODE interpreter status."""
    
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




@register_command("GET_LOOP_STATS")
class GetLoopStatsCommand(QueryCommand):
    """Return control-loop metrics (no ACK dependency)."""

    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        if parts[0].upper() == "GET_LOOP_STATS":
            return True, None
        return False, None

    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr

    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        try:
            target_hz = 1.0 / max(cfg.INTERVAL_S, 1e-9)
            ema_hz = (1.0 / state.ema_period_s) if state.ema_period_s > 0.0 else 0.0
            payload = {
                "target_hz": float(target_hz),
                "loop_count": int(state.loop_count),
                "overrun_count": int(state.overrun_count),
                "last_period_s": float(state.last_period_s),
                "ema_period_s": float(state.ema_period_s),
                "ema_hz": float(ema_hz),
            }
            self.udp_transport.send_response(f"LOOP_STATS|{json.dumps(payload)}", self.addr)
        except Exception as e:
            self.fail(f"Failed to get loop stats: {e}")
            return ExecutionStatus.failed(f"Failed to get loop stats: {e}")
        self.finish()
        return ExecutionStatus.completed("Loop stats sent")


@register_command("PING")
class PingCommand(QueryCommand):
    """Respond to ping requests."""
    
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
        """Execute immediately and return PONG with serial connectivity bit."""
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        
        try:
            # Consider serial "connected" if we've observed a fresh serial frame recently
            serial_connected = 1 if get_cache().age_s() <= cfg.STATUS_STALE_S else 0
            self.udp_transport.send_response(f"PONG|SERIAL={serial_connected}", self.addr)
        except Exception as e:
            self.fail(f"Failed to send PONG: {e}")
            return ExecutionStatus.failed(f"Failed to send PONG: {e}")
        
        self.finish()
        return ExecutionStatus.completed("PONG")
