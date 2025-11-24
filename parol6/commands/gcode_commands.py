"""
GCODE command wrappers for robot control.

These commands integrate the GCODE interpreter with the robot command system.
"""

from typing import TYPE_CHECKING

from parol6.commands.base import CommandBase, ExecutionStatus
from parol6.gcode import GcodeInterpreter
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_matrix

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command("GCODE")
class GcodeCommand(CommandBase):
    """Execute a single GCODE line."""

    __slots__ = (
        "gcode_line",
        "interpreter",
        "generated_commands",
        "current_command_index",
    )
    gcode_line: str
    interpreter: GcodeInterpreter | None
    generated_commands: list[str]
    current_command_index: int

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Check if this is a GCODE command."""
        if parts[0].upper() == "GCODE" and len(parts) >= 2:
            # Rejoin the GCODE line (it might contain | characters)
            self.gcode_line = "|".join(parts[1:])
            return True, None
        return False, None

    def do_setup(self, state: "ControllerState") -> None:
        """Set up GCODE interpreter and parse the line."""
        # Use injected interpreter or create one
        self.interpreter = (
            self.gcode_interpreter or self.interpreter or GcodeInterpreter()
        )
        assert self.interpreter is not None
        # Update interpreter position with current robot position
        current_pose_matrix = get_fkine_matrix()
        current_xyz = current_pose_matrix[:3, 3]
        self.interpreter.state.update_position(
            {
                "X": current_xyz[0] * 1000,
                "Y": current_xyz[1] * 1000,
                "Z": current_xyz[2] * 1000,
            }
        )
        # Parse and store generated robot commands (strings)
        self.generated_commands = self.interpreter.parse_line(self.gcode_line) or []

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Return generated commands for the controller to enqueue."""
        # Report back the list so controller can enqueue them
        details = {}
        if self.generated_commands:
            details["enqueue"] = self.generated_commands
        self.finish()
        return ExecutionStatus.completed("GCODE parsed", details=details)


@register_command("GCODE_PROGRAM")
class GcodeProgramCommand(CommandBase):
    """Load and execute a GCODE program."""

    __slots__ = ("program_type", "program_data", "interpreter")
    program_type: str
    program_data: str
    interpreter: GcodeInterpreter | None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Check if this is a GCODE_PROGRAM command."""
        if parts[0].upper() == "GCODE_PROGRAM" and len(parts) >= 3:
            self.program_type = parts[1].upper()

            if self.program_type == "FILE":
                self.program_data = parts[2]
            elif self.program_type == "INLINE":
                # Join remaining parts and split by semicolon for inline programs
                self.program_data = "|".join(parts[2:])
            else:
                return False, "Invalid GCODE_PROGRAM type (expected FILE or INLINE)"

            return True, None
        return False, None

    def do_setup(self, state: ControllerState) -> None:
        """Load the GCODE program using the interpreter."""
        # Use injected interpreter or create one
        self.interpreter = (
            self.gcode_interpreter or self.interpreter or GcodeInterpreter()
        )
        assert self.interpreter is not None
        if self.program_type == "FILE":
            if not self.interpreter.load_file(self.program_data):
                raise RuntimeError(f"Failed to load GCODE file: {self.program_data}")
        elif self.program_type == "INLINE":
            program_lines = self.program_data.split(";")
            if not self.interpreter.load_program(program_lines):
                raise RuntimeError("Failed to load inline GCODE program")
        else:
            raise ValueError("Invalid GCODE_PROGRAM type (expected FILE or INLINE)")
        # Start program execution
        self.interpreter.start_program()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Signal that the program was loaded; controller will fetch commands."""
        self.finish()
        return ExecutionStatus.completed("GCODE program loaded")


@register_command("GCODE_STOP")
class GcodeStopCommand(CommandBase):
    """Stop GCODE program execution."""

    __slots__ = ()
    is_immediate: bool = True

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Check if this is a GCODE_STOP command."""
        if parts[0].upper() == "GCODE_STOP":
            return True, None
        return False, None

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Stop the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.stop_program()
        self.finish()
        return ExecutionStatus.completed("GCODE stopped")


@register_command("GCODE_PAUSE")
class GcodePauseCommand(CommandBase):
    """Pause GCODE program execution."""

    __slots__ = ()
    is_immediate: bool = True

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Check if this is a GCODE_PAUSE command."""
        if parts[0].upper() == "GCODE_PAUSE":
            return True, None
        return False, None

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Pause the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.pause_program()
        self.finish()
        return ExecutionStatus.completed("GCODE paused")


@register_command("GCODE_RESUME")
class GcodeResumeCommand(CommandBase):
    """Resume GCODE program execution."""

    __slots__ = ()
    is_immediate: bool = True

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Check if this is a GCODE_RESUME command."""
        if parts[0].upper() == "GCODE_RESUME":
            return True, None
        return False, None

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Resume the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.start_program()  # resumes if already loaded
        self.finish()
        return ExecutionStatus.completed("GCODE resumed")
