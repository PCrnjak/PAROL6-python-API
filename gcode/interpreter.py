"""
Main GCODE Interpreter for PAROL6 Robot

Processes GCODE programs and converts them to robot commands.
Manages state, coordinates, and command execution.
"""

import os
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from .parser import GcodeParser, GcodeToken
from .state import GcodeState
from .coordinates import WorkCoordinateSystem
from .commands import create_command_from_token, GcodeCommand


class GcodeInterpreter:
    """Main GCODE interpreter that processes GCODE into robot commands"""
    
    def __init__(self, state_file: Optional[str] = None, coord_file: Optional[str] = None):
        """
        Initialize GCODE interpreter
        
        Args:
            state_file: Path to state persistence file
            coord_file: Path to coordinate system persistence file
        """
        # Initialize components
        self.parser = GcodeParser()
        
        # Use default paths if not provided
        if state_file is None:
            state_file = os.path.join(os.path.dirname(__file__), 'gcode_state.json')
        if coord_file is None:
            coord_file = os.path.join(os.path.dirname(__file__), 'work_coordinates.json')
        
        self.state = GcodeState(state_file)
        self.coord_system = WorkCoordinateSystem(coord_file)
        
        # Program execution state
        self.program_lines = []
        self.current_line = 0
        self.is_running = False
        self.is_paused = False
        self.single_block = False
        
        # Command queue
        self.command_queue = []
        
        # Error tracking
        self.errors = []
    
    def parse_line(self, gcode_line: str) -> List[str]:
        """
        Parse a single GCODE line and return robot commands
        
        Args:
            gcode_line: Single line of GCODE
            
        Returns:
            List of robot command strings
        """
        robot_commands = []
        
        # Parse the line into tokens
        tokens = self.parser.parse_line(gcode_line)
        
        for token in tokens:
            # Validate token
            is_valid, error_msg = self.parser.validate_token(token)
            if not is_valid:
                self.errors.append(error_msg)
                if self.state.program_running:
                    # Stop on error during program execution
                    self.stop_program()
                continue
            
            # Update state with modal commands
            self.state.update_from_token(token)
            
            # Handle work coordinate changes
            if token.code_type == 'G' and int(token.code_number) in [54, 55, 56, 57, 58, 59]:
                self.coord_system.set_active_system(f'G{int(token.code_number)}')
            
            # Create command object
            command = create_command_from_token(token, self.state, self.coord_system)
            
            if command:
                # Get robot command string
                robot_cmd = command.to_robot_command()
                if robot_cmd:
                    robot_commands.append(robot_cmd)
                
                # Update position after movement commands
                if hasattr(command, 'target_position'):
                    self.state.update_position(command.target_position)
                
                # Handle special commands
                if hasattr(command, 'is_program_end') and command.is_program_end:
                    self.stop_program()
                elif hasattr(command, 'requires_resume') and command.requires_resume:
                    # Check if this is an optional stop (M1)
                    if hasattr(command, 'is_optional') and command.is_optional:
                        # Only pause if optional_stop is enabled
                        if self.state.optional_stop:
                            self.pause_program()
                    else:
                        # M0 always pauses
                        self.pause_program()
        
        return robot_commands
    
    def parse_program(self, program: Union[str, List[str]]) -> List[str]:
        """
        Parse a complete GCODE program
        
        Args:
            program: GCODE program as string or list of lines
            
        Returns:
            List of all robot commands
        """
        if isinstance(program, str):
            lines = program.split('\n')
        else:
            lines = program
        
        all_commands = []
        self.errors = []
        
        for line in lines:
            commands = self.parse_line(line)
            all_commands.extend(commands)
            
            # Stop if errors encountered
            if self.errors and not self.state.program_running:
                break
        
        return all_commands
    
    def load_program(self, program: Union[str, List[str]]) -> bool:
        """
        Load a GCODE program for execution
        
        Args:
            program: GCODE program as string or list of lines
            
        Returns:
            True if program loaded successfully
        """
        if isinstance(program, str):
            self.program_lines = program.split('\n')
        else:
            self.program_lines = program
        
        self.current_line = 0
        self.errors = []
        self.command_queue = []
        
        # Validate program with proper line number tracking
        for line_num, line in enumerate(self.program_lines, 1):
            tokens = self.parser.parse_line(line)
            for token in tokens:
                is_valid, error_msg = self.parser.validate_token(token)
                if not is_valid:
                    self.errors.append(f"Line {line_num}: {error_msg}")
        
        return len(self.errors) == 0
    
    def load_file(self, filepath: str) -> bool:
        """
        Load GCODE program from file
        
        Args:
            filepath: Path to GCODE file
            
        Returns:
            True if file loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                program = f.read()
            return self.load_program(program)
        except Exception as e:
            self.errors.append(f"Error loading file: {e}")
            return False
    
    def start_program(self) -> bool:
        """
        Start or resume program execution
        
        Returns:
            True if program started successfully
        """
        if not self.program_lines:
            self.errors.append("No program loaded")
            return False
        
        self.is_running = True
        self.is_paused = False
        self.state.program_running = True
        return True
    
    def pause_program(self) -> None:
        """Pause program execution"""
        self.is_paused = True
        self.state.program_running = False
        # Note: Command queue is not cleared so position in program is maintained
        # Commands already in queue can be optionally processed or discarded by the caller
    
    def stop_program(self) -> None:
        """Stop program execution and reset"""
        self.is_running = False
        self.is_paused = False
        self.current_line = 0
        self.state.program_running = False
        self.command_queue = []
    
    def set_optional_stop(self, enabled: bool) -> None:
        """
        Enable or disable optional stop (M1)
        
        Args:
            enabled: True to enable optional stop, False to disable
        """
        self.state.optional_stop = enabled
    
    def get_next_command(self) -> Optional[str]:
        """
        Get next robot command from the program
        
        Returns:
            Robot command string or None if no more commands
        """
        # Return queued commands first
        if self.command_queue:
            return self.command_queue.pop(0)
        
        # Check if program is running
        if not self.is_running or self.is_paused:
            return None
        
        # Process lines until we get a command or reach end
        while self.current_line < len(self.program_lines):
            line = self.program_lines[self.current_line]
            self.current_line += 1
            
            # Parse line and get commands
            commands = self.parse_line(line)
            
            if commands:
                # Add to queue
                self.command_queue.extend(commands)
                
                # Return first command
                if self.command_queue:
                    command = self.command_queue.pop(0)
                    
                    # Check for single block mode
                    if self.single_block:
                        self.pause_program()
                    
                    return command
            
            # Check for errors
            if self.errors:
                self.stop_program()
                return None
        
        # End of program
        self.stop_program()
        return None
    
    def set_work_offset(self, coord_system: str, axis: str, value: float) -> bool:
        """
        Set work coordinate offset
        
        Args:
            coord_system: Work coordinate system (G54-G59)
            axis: Axis (X, Y, Z, A, B, C)
            value: Offset value in mm
            
        Returns:
            True if successful
        """
        # Validate coordinate system
        if coord_system not in ['G54', 'G55', 'G56', 'G57', 'G58', 'G59']:
            self.errors.append(f"Invalid coordinate system: {coord_system}")
            return False
        
        # Validate axis
        if axis not in ['X', 'Y', 'Z', 'A', 'B', 'C']:
            self.errors.append(f"Invalid axis: {axis}")
            return False
            
        return self.coord_system.set_offset(coord_system, axis, value)
    
    def set_current_as_zero(self, machine_position: List[float]) -> None:
        """
        Set current position as zero in active work coordinate system
        
        Args:
            machine_position: Current machine position [X, Y, Z, RX, RY, RZ]
        """
        # Convert to dictionary
        pos_dict = {
            'X': machine_position[0],
            'Y': machine_position[1],
            'Z': machine_position[2],
            'A': machine_position[3] if len(machine_position) > 3 else 0,
            'B': machine_position[4] if len(machine_position) > 4 else 0,
            'C': machine_position[5] if len(machine_position) > 5 else 0
        }
        
        self.coord_system.set_current_as_zero(pos_dict)
    
    def get_status(self) -> Dict:
        """
        Get interpreter status
        
        Returns:
            Dictionary with status information
        """
        return {
            'state': self.state.get_status(),
            'coord_system': self.coord_system.active_system,
            'program_running': self.is_running,
            'program_paused': self.is_paused,
            'current_line': self.current_line,
            'total_lines': len(self.program_lines),
            'errors': self.errors[-5:] if self.errors else []  # Last 5 errors
        }
    
    def reset(self) -> None:
        """Reset interpreter to initial state"""
        self.state.reset()
        self.stop_program()
        self.errors = []
    
    def set_feed_override(self, percentage: float) -> None:
        """
        Set feed rate override percentage
        
        Args:
            percentage: Override percentage (0-200)
        """
        self.state.feed_override = np.clip(percentage, 0, 200)
    
    def set_single_block(self, enabled: bool) -> None:
        """
        Enable/disable single block mode
        
        Args:
            enabled: True to enable single block mode
        """
        self.single_block = enabled
        self.state.single_block = enabled