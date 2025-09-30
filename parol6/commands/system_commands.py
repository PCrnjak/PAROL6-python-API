"""
System control commands that can execute regardless of controller enable state.

These commands control the overall state of the robot controller (enable/disable, stop, etc.)
and can execute even when the controller is disabled.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple, Optional, List, TYPE_CHECKING

from parol6.commands.base import SystemCommand, ExecutionStatus, parse_int, parse_bool
from parol6.server.command_registry import register_command
from parol6.protocol.wire import CommandCode
from parol6.config import save_com_port

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command("STOP")
class StopCommand(SystemCommand):
    """Emergency stop command - immediately stops all motion."""
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a STOP command."""
        if parts[0].upper() == "STOP":
            if len(parts) != 1:
                return False, "STOP command takes no parameters"
            return True, None
        return False, None
    
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
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is an ENABLE command."""
        if parts[0].upper() == "ENABLE":
            if len(parts) != 1:
                return False, "ENABLE command takes no parameters"
            return True, None
        return False, None
    
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
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a DISABLE command."""
        if parts[0].upper() == "DISABLE":
            if len(parts) != 1:
                return False, "DISABLE command takes no parameters"
            return True, None
        return False, None
    
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
    __slots__ = ()
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a CLEAR_ERROR command."""
        if parts[0].upper() == "CLEAR_ERROR":
            if len(parts) != 1:
                return False, "CLEAR_ERROR command takes no parameters"
            return True, None
        return False, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute clear error - reset error states."""
        logger.info("CLEAR_ERROR command executed")
        
        # Clear any error states
        # The actual error clearing logic depends on what errors are tracked
        # For now, we'll just ensure the controller is in a clean state
        state.Command_out = CommandCode.IDLE  # No specific CLEAR_ERROR code
        
        self.finish()
        return ExecutionStatus.completed("Errors cleared")


@register_command("SET_IO")
class SetIOCommand(SystemCommand):
    """Set a digital I/O port state."""
    
    __slots__ = ("port_index", "port_value")
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SET_IO command.
        
        Format: SET_IO|port_index|value
        Example: SET_IO|0|1
        """
        if parts[0].upper() != "SET_IO":
            return False, None
            
        if len(parts) != 3:
            return False, "SET_IO requires 2 parameters: port_index, value"
        
        self.port_index = parse_int(parts[1])
        self.port_value = parse_int(parts[2])
        
        if self.port_index is None or self.port_value is None:
            return False, "Port index and value must be integers"
        
        # Validate port index (0-7 for 8 I/O ports)
        if not 0 <= self.port_index <= 7:
            return False, f"Port index must be 0-7, got {self.port_index}"
        
        # Validate port value (0 or 1)
        if self.port_value not in (0, 1):
            return False, f"Port value must be 0 or 1, got {self.port_value}"
        
        logger.info(f"Parsed SET_IO: port {self.port_index} = {self.port_value}")
        return True, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute set port - update I/O port state."""
        if self.port_index is None or self.port_value is None:
            self.fail("Port index or value not set")
            return ExecutionStatus.failed("Port parameters not set")
        
        logger.info(f"SET_IO: Setting port {self.port_index} to {self.port_value}")
        
        # Update the output port state
        state.InOut_out[self.port_index] = self.port_value
        
        self.finish()
        return ExecutionStatus.completed(f"Port {self.port_index} set to {self.port_value}")


@register_command("SET_PORT")
class SetSerialPortCommand(SystemCommand):
    """Set the serial COM port used by the controller."""
    __slots__ = ("port_str",)

    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SET_PORT command.

        Format: SET_PORT|serial_port
        Example: SET_PORT|/dev/ttyACM0
        """
        if parts[0].upper() != "SET_PORT":
            return False, None

        if len(parts) != 2:
            return False, "SET_PORT requires 1 parameter: serial_port"

        port = (parts[1] or "").strip()
        if not port:
            return False, "Serial port cannot be empty"

        self.port_str = port
        logger.info(f"Parsed SET_PORT: serial_port={self.port_str}")
        return True, None

    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Persist the serial port selection; controller may reconnect based on this."""
        if not self.port_str:
            self.fail("No serial port provided")
            return ExecutionStatus.failed("No serial port provided")

        ok = save_com_port(self.port_str)
        if not ok:
            self.fail("Failed to save COM port")
            return ExecutionStatus.failed("Failed to save COM port")

        self.finish()
        # Include details so the controller can reconnect immediately
        return ExecutionStatus.completed("Serial port saved", details={"serial_port": self.port_str})


@register_command("STREAM")
class StreamCommand(SystemCommand):
    """Toggle stream mode for real-time jogging."""
    
    __slots__ = ("stream_mode",)
    
    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse STREAM command.
        
        Format: STREAM|on/off
        Example: STREAM|on
        """
        if parts[0].upper() != "STREAM":
            return False, None
            
        if len(parts) != 2:
            return False, "STREAM requires 1 parameter: on/off"
        
        self.stream_mode = parse_bool(parts[1])
        if parts[1].lower() not in ('on', 'off', '1', '0', 'true', 'false'):
            return False, f"STREAM mode must be 'on' or 'off', got '{parts[1]}'"
        
        logger.info(f"Parsed STREAM: mode = {self.stream_mode}")
        return True, None
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute stream mode toggle."""
        if self.stream_mode is None:
            self.fail("Stream mode not set")
            return ExecutionStatus.failed("Stream mode not set")
        
        # The controller will handle the actual stream mode setting
        # This is just a placeholder that sets a flag
        logger.info(f"STREAM: Setting stream mode to {self.stream_mode}")
        
        state.stream_mode = self.stream_mode
        
        self.finish()
        return ExecutionStatus.completed(f"Stream mode {'enabled' if self.stream_mode else 'disabled'}")


@register_command("SIMULATOR")
class SimulatorCommand(SystemCommand):
    """Toggle simulator (fake serial) mode on/off."""

    __slots__ = ("mode_on",)

    def do_match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Parse SIMULATOR command.

        Format: SIMULATOR|on/off
        Example: SIMULATOR|on
        """
        if parts[0].upper() != "SIMULATOR":
            return False, None

        if len(parts) != 2:
            return False, "SIMULATOR requires 1 parameter: on/off"

        self.mode_on = parse_bool(parts[1])
        if parts[1].lower() not in ('on', 'off', '1', '0', 'true', 'false', 'yes', 'no'):
            return False, "SIMULATOR parameter must be 'on' or 'off'"

        logger.info(f"Parsed SIMULATOR: mode_on={self.mode_on}")
        return True, None

    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Execute simulator toggle by setting env var and returning details to trigger reconfiguration."""
        if self.mode_on is None:
            self.fail("Simulator mode not set")
            return ExecutionStatus.failed("Simulator mode not set")

        # Set environment variable used by transport factory
        os.environ["PAROL6_FAKE_SERIAL"] = "1" if self.mode_on else "0"
        logger.info(f"SIMULATOR command executed: {'ON' if self.mode_on else 'OFF'}")

        self.finish()
        return ExecutionStatus.completed(
            f"Simulator {'ON' if self.mode_on else 'OFF'}",
            details={"simulator_mode": "on" if self.mode_on else "off"},
        )
