"""
GCODE command wrappers for robot control.

These commands integrate the GCODE interpreter with the robot command system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List, TYPE_CHECKING, Any

from parol6.commands.base import CommandBase, ExecutionStatus
from parol6.server.command_registry import register_command
from parol6.gcode import GcodeInterpreter
import parol6.PAROL6_ROBOT as PAROL6_ROBOT

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@dataclass
@register_command("GCODE")
class GcodeCommand(CommandBase):
    """Execute a single GCODE line."""
    
    gcode_line: str = ""
    interpreter: Optional[GcodeInterpreter] = None
    generated_commands: List[str] = None
    current_command_index: int = 0
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GCODE command."""
        if parts[0].upper() == "GCODE" and len(parts) >= 2:
            # Rejoin the GCODE line (it might contain | characters)
            self.gcode_line = '|'.join(parts[1:])
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport: Any = None, addr: Any = None, gcode_interpreter: Any = None) -> None:
        """Set up GCODE interpreter (injected) and parse the line."""
        # Prefer injected, controller-owned interpreter
        self.interpreter = gcode_interpreter or self.interpreter or GcodeInterpreter()
        try:
            # Update interpreter position with current robot position
            current_angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)]
            current_pose_matrix = PAROL6_ROBOT.robot.fkine(current_angles_rad).A
            current_xyz = current_pose_matrix[:3, 3]
            self.interpreter.state.update_position({
                'X': current_xyz[0] * 1000,
                'Y': current_xyz[1] * 1000,
                'Z': current_xyz[2] * 1000
            })
            # Parse and store generated robot commands (strings)
            self.generated_commands = self.interpreter.parse_line(self.gcode_line) or []
        except Exception as e:
            self.fail(f"GCODE parsing error: {str(e)}")
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Return generated commands for the controller to enqueue."""
        # Report back the list so controller can enqueue them
        details = {}
        if self.generated_commands:
            details['enqueue'] = self.generated_commands
        self.finish()
        return ExecutionStatus.completed("GCODE parsed", details=details)


@dataclass
@register_command("GCODE_PROGRAM")
class GcodeProgramCommand(CommandBase):
    """Load and execute a GCODE program."""
    
    program_type: str = ""  # 'FILE' or 'INLINE'
    program_data: str = ""
    interpreter: Optional[GcodeInterpreter] = None
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GCODE_PROGRAM command."""
        if parts[0].upper() == "GCODE_PROGRAM" and len(parts) >= 3:
            self.program_type = parts[1].upper()
            
            if self.program_type == "FILE":
                self.program_data = parts[2]
            elif self.program_type == "INLINE":
                # Join remaining parts and split by semicolon for inline programs
                self.program_data = '|'.join(parts[2:])
            else:
                return False, "Invalid GCODE_PROGRAM type (expected FILE or INLINE)"
            
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport: Any = None, addr: Any = None, gcode_interpreter: Any = None) -> None:
        """Load the GCODE program using the injected controller interpreter."""
        # Prefer injected, controller-owned interpreter
        self.interpreter = gcode_interpreter or self.interpreter or GcodeInterpreter()
        try:
            if self.program_type == "FILE":
                if not self.interpreter.load_file(self.program_data):
                    self.fail(f"Failed to load GCODE file: {self.program_data}")
                    return
            elif self.program_type == "INLINE":
                program_lines = self.program_data.split(';')
                if not self.interpreter.load_program(program_lines):
                    self.fail("Failed to load inline GCODE program")
                    return
            # Start program execution
            self.interpreter.start_program()
        except Exception as e:
            self.fail(f"GCODE program error: {str(e)}")
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Signal that the program was loaded; controller will fetch commands."""
        self.finish()
        return ExecutionStatus.completed("GCODE program loaded")


@dataclass
@register_command("GCODE_STOP")
class GcodeStopCommand(CommandBase):
    """Stop GCODE program execution."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GCODE_STOP command."""
        if parts[0].upper() == "GCODE_STOP":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport: Any = None, addr: Any = None, gcode_interpreter: Any = None) -> None:
        """Bind interpreter if provided."""
        if gcode_interpreter:
            self.gcode_interpreter = gcode_interpreter
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Stop the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.stop_program()
        self.finish()
        return ExecutionStatus.completed("GCODE stopped")


@dataclass
@register_command("GCODE_PAUSE")
class GcodePauseCommand(CommandBase):
    """Pause GCODE program execution."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GCODE_PAUSE command."""
        if parts[0].upper() == "GCODE_PAUSE":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport: Any = None, addr: Any = None, gcode_interpreter: Any = None) -> None:
        if gcode_interpreter:
            self.gcode_interpreter = gcode_interpreter
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Pause the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.pause_program()
        self.finish()
        return ExecutionStatus.completed("GCODE paused")


@dataclass
@register_command("GCODE_RESUME")
class GcodeResumeCommand(CommandBase):
    """Resume GCODE program execution."""
    
    is_immediate: bool = True
    
    def match(self, parts: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if this is a GCODE_RESUME command."""
        if parts[0].upper() == "GCODE_RESUME":
            return True, None
        return False, None
    
    def setup(self, state: 'ControllerState', *, udp_transport: Any = None, addr: Any = None, gcode_interpreter: Any = None) -> None:
        if gcode_interpreter:
            self.gcode_interpreter = gcode_interpreter
    
    def execute_step(self, state: 'ControllerState') -> ExecutionStatus:
        """Resume the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.start_program()  # resumes if already loaded
        self.finish()
        return ExecutionStatus.completed("GCODE resumed")
