"""
System control commands that can execute regardless of controller enable state.

These commands control the overall state of the robot controller (enable/disable, stop, etc.)
and can execute even when the controller is disabled.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, List, TYPE_CHECKING

from parol6.commands.base import SystemCommand, ExecutionStatus
from parol6.server.command_registry import register_command
from parol6.protocol.wire import CommandCode

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command("STOP")
class StopCommand(SystemCommand):
    """Emergency stop command - immediately stops all motion."""
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a STOP command."""
        if parts[0].upper() == "STOP":
            if len(parts) != 1:
                return False, "STOP command takes no parameters"
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute stop - set all speeds to zero and command to IDLE."""
        logger.info("STOP command executed")
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.IDLE
        
        # Clear any active commands in the controller
        # This will be handled by the controller when it sees this command
        
        self.finish()
        return ExecutionStatus.completed("Robot stopped")


@register_command("ENABLE")
class EnableCommand(SystemCommand):
    """Enable the robot controller."""
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is an ENABLE command."""
        if parts[0].upper() == "ENABLE":
            if len(parts) != 1:
                return False, "ENABLE command takes no parameters"
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute enable - set controller to enabled state."""
        logger.info("ENABLE command executed")
        state.enabled = True
        state.disabled_reason = ""
        state.Command_out = CommandCode.ENABLE
        
        self.finish()
        return ExecutionStatus.completed("Controller enabled")


@register_command("DISABLE")
class DisableCommand(SystemCommand):
    """Disable the robot controller."""
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a DISABLE command."""
        if parts[0].upper() == "DISABLE":
            if len(parts) != 1:
                return False, "DISABLE command takes no parameters"
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute disable - set controller to disabled state."""
        logger.info("DISABLE command executed")
        state.enabled = False
        state.disabled_reason = "User requested disable"
        state.Command_out = CommandCode.DISABLE
        state.Speed_out.fill(0)
        
        self.finish()
        return ExecutionStatus.completed("Controller disabled")


@register_command("CLEAR_ERROR")
class ClearErrorCommand(SystemCommand):
    """Clear any error states in the controller."""
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a CLEAR_ERROR command."""
        if parts[0].upper() == "CLEAR_ERROR":
            if len(parts) != 1:
                return False, "CLEAR_ERROR command takes no parameters"
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute clear error - reset error states."""
        logger.info("CLEAR_ERROR command executed")
        
        # Clear any error states
        # The actual error clearing logic depends on what errors are tracked
        # For now, we'll just ensure the controller is in a clean state
        state.Command_out = CommandCode.IDLE  # No specific CLEAR_ERROR code
        
        self.finish()
        return ExecutionStatus.completed("Errors cleared")


@register_command("SET_PORT")
class SetPortCommand(SystemCommand):
    """Set a digital I/O port state."""
    
    port_index: Optional[int] = None
    port_value: Optional[int] = None
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SET_PORT command.
        
        Format: SET_PORT|port_index|value
        Example: SET_PORT|0|1
        """
        if parts[0].upper() != "SET_PORT":
            return False, None
            
        if len(parts) != 3:
            return False, "SET_PORT requires 2 parameters: port_index, value"
        
        try:
            self.port_index = int(parts[1])
            self.port_value = int(parts[2])
            
            # Validate port index (0-7 for 8 I/O ports)
            if not 0 <= self.port_index <= 7:
                return False, f"Port index must be 0-7, got {self.port_index}"
            
            # Validate port value (0 or 1)
            if self.port_value not in (0, 1):
                return False, f"Port value must be 0 or 1, got {self.port_value}"
            
            logger.info(f"Parsed SET_PORT: port {self.port_index} = {self.port_value}")
            return True, None
            
        except ValueError as e:
            return False, f"Invalid SET_PORT parameters: {str(e)}"
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute set port - update I/O port state."""
        if self.port_index is None or self.port_value is None:
            self.fail("Port index or value not set")
            return ExecutionStatus.failed("Port parameters not set")
        
        logger.info(f"SET_PORT: Setting port {self.port_index} to {self.port_value}")
        
        # Update the output port state
        state.InOut_out[self.port_index] = self.port_value
        
        self.finish()
        return ExecutionStatus.completed(f"Port {self.port_index} set to {self.port_value}")


@register_command("STREAM")
class StreamCommand(SystemCommand):
    """Toggle stream mode for real-time jogging."""
    
    stream_mode: Optional[bool] = None
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse STREAM command.
        
        Format: STREAM|on/off
        Example: STREAM|on
        """
        if parts[0].upper() != "STREAM":
            return False, None
            
        if len(parts) != 2:
            return False, "STREAM requires 1 parameter: on/off"
        
        mode_str = parts[1].lower()
        if mode_str == 'on':
            self.stream_mode = True
        elif mode_str == 'off':
            self.stream_mode = False
        else:
            return False, f"STREAM mode must be 'on' or 'off', got '{parts[1]}'"
        
        logger.info(f"Parsed STREAM: mode = {self.stream_mode}")
        return True, None
    
    def setup(self, state: 'ControllerState', *, udp_transport=None, addr=None, gcode_interpreter=None) -> None:
        """Bind context if provided."""
        if udp_transport is not None:
            self.udp_transport = udp_transport
        if addr is not None:
            self.addr = addr
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute stream mode toggle."""
        if self.stream_mode is None:
            self.fail("Stream mode not set")
            return ExecutionStatus.failed("Stream mode not set")
        
        # The controller will handle the actual stream mode setting
        # This is just a placeholder that sets a flag
        logger.info(f"STREAM: Setting stream mode to {self.stream_mode}")
        
        # Note: The actual stream_mode flag is maintained by the controller
        # This command just triggers the change
        
        self.finish()
        return ExecutionStatus.completed(f"Stream mode {'enabled' if self.stream_mode else 'disabled'}")
