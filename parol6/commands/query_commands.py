"""
Query commands that return immediate status information.
"""

import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING

from parol6.commands.base import QueryCommand, ExecutionStatus
from parol6.server.command_registry import register_command
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.server.status_cache import get_cache
from parol6.server.state import get_fkine_flat_mm
from parol6 import config as cfg
from parol6.tools import list_tools

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command("GET_POSE")
class GetPoseCommand(QueryCommand):
    """Get current robot pose matrix."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_POSE command."""
        if parts[0].upper() == "GET_POSE":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return pose data with translation in mm."""  
        pose_flat = get_fkine_flat_mm()
        pose_str = ",".join(map(str, pose_flat))
        self.reply_ascii("POSE", pose_str)
        
        self.finish()
        return ExecutionStatus.completed("Pose sent")


@register_command("GET_ANGLES")
class GetAnglesCommand(QueryCommand):
    """Get current joint angles in degrees."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_ANGLES command."""
        if parts[0].upper() == "GET_ANGLES":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return angle data."""
        angles_rad = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)
        angles_deg = np.rad2deg(angles_rad)
        angles_str = ",".join(map(str, angles_deg))
        self.reply_ascii("ANGLES", angles_str)
        
        self.finish()
        return ExecutionStatus.completed("Angles sent")


@register_command("GET_IO")
class GetIOCommand(QueryCommand):
    """Get current I/O status."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_IO command."""
        if parts[0].upper() == "GET_IO":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return I/O data."""
        io_status_str = ",".join(map(str, state.InOut_in[:5]))
        self.reply_ascii("IO", io_status_str)
        
        self.finish()
        return ExecutionStatus.completed("I/O sent")


@register_command("GET_GRIPPER")
class GetGripperCommand(QueryCommand):
    """Get current gripper status."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_GRIPPER command."""
        if parts[0].upper() == "GET_GRIPPER":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return gripper data."""
        gripper_status_str = ",".join(map(str, state.Gripper_data_in))
        self.reply_ascii("GRIPPER", gripper_status_str)
        
        self.finish()
        return ExecutionStatus.completed("Gripper sent")


@register_command("GET_SPEEDS")
class GetSpeedsCommand(QueryCommand):
    """Get current joint speeds."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_SPEEDS command."""
        if parts[0].upper() == "GET_SPEEDS":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return speed data."""
        speeds_str = ",".join(map(str, state.Speed_in))
        self.reply_ascii("SPEEDS", speeds_str)
        
        self.finish()
        return ExecutionStatus.completed("Speeds sent")


@register_command("GET_STATUS")
class GetStatusCommand(QueryCommand):
    """Get aggregated robot status (pose, angles, I/O, gripper) from cache."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_STATUS command."""
        if parts[0].upper() == "GET_STATUS":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return cached aggregated status (ASCII)."""
        # Always refresh cache from current state before replying
        cache = get_cache()
        cache.update_from_state(state)
        payload = cache.to_ascii()
        self.reply_text(payload)  # Already has full format
        
        self.finish()
        return ExecutionStatus.completed("Status sent")


@register_command("GET_GCODE_STATUS")
class GetGcodeStatusCommand(QueryCommand):
    """Get GCODE interpreter status."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_GCODE_STATUS command."""
        if parts[0].upper() == "GET_GCODE_STATUS":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return GCODE status."""
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
        
        self.reply_json("GCODE_STATUS", gcode_status)
        
        self.finish()
        return ExecutionStatus.completed("GCODE status sent")




@register_command("GET_LOOP_STATS")
class GetLoopStatsCommand(QueryCommand):
    """Return control-loop metrics (no ACK dependency)."""
    __slots__ = ()

    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        if parts[0].upper() == "GET_LOOP_STATS":
            return True, None
        return False, None

    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
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
        self.reply_json("LOOP_STATS", payload)
        self.finish()
        return ExecutionStatus.completed("Loop stats sent")


@register_command("PING")
class PingCommand(QueryCommand):
    """Respond to ping requests."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a PING command."""
        if parts[0].upper() == "PING":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return PONG with serial connectivity bit."""
        # Consider serial "connected" if we've observed a fresh serial frame recently
        serial_connected = 1 if get_cache().age_s() <= cfg.STATUS_STALE_S else 0
        self.reply_ascii("PONG", f"SERIAL={serial_connected}")
        
        self.finish()
        return ExecutionStatus.completed("PONG")


@register_command("GET_TOOL")
class GetToolCommand(QueryCommand):
    """Get current tool configuration and available tools."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GET_TOOL command."""
        if parts[0].upper() == "GET_TOOL":
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute immediately and return current tool info."""
        
        payload = {
            "tool": state.current_tool,
            "available": list_tools()
        }
        self.reply_json("TOOL", payload)
        
        self.finish()
        return ExecutionStatus.completed("Tool info sent")
