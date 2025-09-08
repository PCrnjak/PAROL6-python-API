"""
GCODE State Management for PAROL6 Robot

Tracks modal states during GCODE execution including:
- Coordinate systems (G54-G59)
- Positioning modes (G90/G91)
- Units (G20/G21)
- Feed rates and spindle speeds
- Active plane (G17/G18/G19)
"""

import json
import os
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class GcodeState:
    """Tracks modal GCODE state during execution"""
    
    # Motion modes
    motion_mode: str = 'G0'  # G0, G1, G2, G3
    positioning_mode: str = 'G90'  # G90 (absolute) or G91 (incremental)
    
    # Coordinate system
    work_coordinate: str = 'G54'  # G54-G59
    work_offsets: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'G54': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
        'G55': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
        'G56': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
        'G57': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
        'G58': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
        'G59': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0}
    })
    
    # Current position (in work coordinates)
    current_position: Dict[str, float] = field(default_factory=lambda: {
        'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0
    })
    
    # Machine position (absolute, no offsets)
    machine_position: Dict[str, float] = field(default_factory=lambda: {
        'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0
    })
    
    # Units and scaling
    units: str = 'G21'  # G20 (inches) or G21 (mm)
    units_scale: float = 1.0  # Multiplier to convert to mm
    
    # Feed and speed
    feed_rate: float = 100.0  # mm/min
    spindle_speed: float = 0.0  # RPM
    
    # Plane selection for arcs
    plane: str = 'G17'  # G17 (XY), G18 (XZ), G19 (YZ)
    
    # Tool
    tool_number: int = 0
    tool_length_offset: float = 0.0
    
    # Program control
    program_running: bool = False
    single_block: bool = False
    optional_stop: bool = False  # M1 optional stop enable
    feed_override: float = 100.0  # Percentage
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize GCODE state, optionally loading from file
        
        Args:
            state_file: Path to JSON file for persistent state
        """
        # Initialize all dataclass fields with their defaults first
        self.motion_mode = 'G0'
        self.positioning_mode = 'G90'
        self.work_coordinate = 'G54'
        self.work_offsets = {
            'G54': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
            'G55': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
            'G56': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
            'G57': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
            'G58': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0},
            'G59': {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0}
        }
        self.current_position = {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0}
        self.machine_position = {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0}
        self.units = 'G21'
        self.units_scale = 1.0
        self.feed_rate = 100.0
        self.spindle_speed = 0.0
        self.plane = 'G17'
        self.tool_number = 0
        self.tool_length_offset = 0.0
        self.program_running = False
        self.single_block = False
        self.optional_stop = False
        self.feed_override = 100.0
        
        # Now handle the state file
        self.state_file = state_file
        if state_file and os.path.exists(state_file):
            self.load_state()
    
    def update_from_token(self, token) -> None:
        """
        Update state based on a GCODE token
        
        Args:
            token: GcodeToken object
        """
        if token.code_type == 'G':
            code = int(token.code_number)
            
            # Motion modes
            if code in [0, 1, 2, 3]:
                self.motion_mode = f'G{code}'
            
            # Plane selection
            elif code in [17, 18, 19]:
                self.plane = f'G{code}'
            
            # Units
            elif code == 20:  # Inches
                self.units = 'G20'
                self.units_scale = 25.4  # Convert inches to mm
            elif code == 21:  # Millimeters
                self.units = 'G21'
                self.units_scale = 1.0
            
            # Work coordinates
            elif code in [54, 55, 56, 57, 58, 59]:
                self.work_coordinate = f'G{code}'
            
            # Positioning mode
            elif code == 90:
                self.positioning_mode = 'G90'
            elif code == 91:
                self.positioning_mode = 'G91'
        
        elif token.code_type == 'F':
            # Feed rate
            self.feed_rate = token.code_number * self.units_scale
        
        elif token.code_type == 'S':
            # Spindle speed
            self.spindle_speed = token.code_number
        
        elif token.code_type == 'T':
            # Tool number
            self.tool_number = int(token.code_number)
    
    def get_work_offset(self) -> Dict[str, float]:
        """Get current work coordinate offset"""
        return self.work_offsets[self.work_coordinate]
    
    def set_work_offset(self, axis: str, value: float) -> None:
        """
        Set work coordinate offset for an axis
        
        Args:
            axis: Axis name (X, Y, Z, A, B, C)
            value: Offset value in mm
        """
        if axis in self.work_offsets[self.work_coordinate]:
            self.work_offsets[self.work_coordinate][axis] = value
            if self.state_file:
                self.save_state()
    
    def work_to_machine(self, work_coords: Dict[str, float]) -> Dict[str, float]:
        """
        Convert work coordinates to machine coordinates
        
        Args:
            work_coords: Dictionary of work coordinates
            
        Returns:
            Dictionary of machine coordinates
        """
        machine_coords = {}
        offset = self.get_work_offset()
        
        for axis in ['X', 'Y', 'Z', 'A', 'B', 'C']:
            if axis in work_coords:
                # Apply work offset and tool offset (for Z)
                machine_coords[axis] = work_coords[axis] + offset.get(axis, 0)
                if axis == 'Z':
                    machine_coords[axis] += self.tool_length_offset
        
        return machine_coords
    
    def machine_to_work(self, machine_coords: Dict[str, float]) -> Dict[str, float]:
        """
        Convert machine coordinates to work coordinates
        
        Args:
            machine_coords: Dictionary of machine coordinates
            
        Returns:
            Dictionary of work coordinates
        """
        work_coords = {}
        offset = self.get_work_offset()
        
        for axis in ['X', 'Y', 'Z', 'A', 'B', 'C']:
            if axis in machine_coords:
                # Remove work offset and tool offset (for Z)
                work_coords[axis] = machine_coords[axis] - offset.get(axis, 0)
                if axis == 'Z':
                    work_coords[axis] -= self.tool_length_offset
        
        return work_coords
    
    def calculate_target_position(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate target position based on positioning mode and parameters
        
        Args:
            parameters: Dictionary of axis values from GCODE
            
        Returns:
            Dictionary of target positions in work coordinates
        """
        target = self.current_position.copy()
        
        for axis in ['X', 'Y', 'Z', 'A', 'B', 'C']:
            if axis in parameters:
                value = parameters[axis] * self.units_scale
                
                if self.positioning_mode == 'G90':  # Absolute
                    target[axis] = value
                else:  # G91 - Incremental
                    target[axis] = self.current_position[axis] + value
        
        return target
    
    def update_position(self, new_position: Dict[str, float]) -> None:
        """
        Update current position
        
        Args:
            new_position: New position in work coordinates
        """
        self.current_position.update(new_position)
        # Update machine position
        machine_pos = self.work_to_machine(new_position)
        self.machine_position.update(machine_pos)
    
    def reset(self) -> None:
        """Reset state to defaults"""
        self.motion_mode = 'G0'
        self.positioning_mode = 'G90'
        self.work_coordinate = 'G54'
        self.units = 'G21'
        self.units_scale = 1.0
        self.feed_rate = 100.0
        self.spindle_speed = 0.0
        self.plane = 'G17'
        self.tool_number = 0
        self.tool_length_offset = 0.0
        self.program_running = False
        self.single_block = False
        self.feed_override = 100.0
        
        # Keep work offsets but reset positions
        self.current_position = {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0}
        self.machine_position = {'X': 0, 'Y': 0, 'Z': 0, 'A': 0, 'B': 0, 'C': 0}
    
    def save_state(self) -> None:
        """Save state to JSON file"""
        if self.state_file:
            # Save complete modal state
            state_dict = {
                'work_offsets': self.work_offsets,
                'units': self.units,
                'work_coordinate': self.work_coordinate,
                'tool_length_offset': self.tool_length_offset,
                'motion_mode': self.motion_mode,
                'positioning_mode': self.positioning_mode,
                'plane': self.plane,
                'feed_rate': self.feed_rate,
                'spindle_speed': self.spindle_speed,
                'current_position': self.current_position,
                'machine_position': self.machine_position
            }
            
            # Create directory if needed (only if path has a directory component)
            dir_name = os.path.dirname(self.state_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
    
    def load_state(self) -> None:
        """Load state from JSON file"""
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state_dict = json.load(f)
                
                # Load complete modal state
                self.work_offsets = state_dict.get('work_offsets', self.work_offsets)
                self.units = state_dict.get('units', self.units)
                self.work_coordinate = state_dict.get('work_coordinate', self.work_coordinate)
                self.tool_length_offset = state_dict.get('tool_length_offset', 0.0)
                self.motion_mode = state_dict.get('motion_mode', self.motion_mode)
                self.positioning_mode = state_dict.get('positioning_mode', self.positioning_mode)
                self.plane = state_dict.get('plane', self.plane)
                self.feed_rate = state_dict.get('feed_rate', self.feed_rate)
                self.spindle_speed = state_dict.get('spindle_speed', self.spindle_speed)
                if 'current_position' in state_dict:
                    self.current_position = state_dict['current_position']
                if 'machine_position' in state_dict:
                    self.machine_position = state_dict['machine_position']
                
                # Update units scale
                self.units_scale = 25.4 if self.units == 'G20' else 1.0
            except Exception as e:
                print(f"Error loading state file: {e}")
    
    def get_status(self) -> Dict:
        """Get current state as dictionary for status reporting"""
        return {
            'motion_mode': self.motion_mode,
            'positioning_mode': self.positioning_mode,
            'work_coordinate': self.work_coordinate,
            'units': self.units,
            'feed_rate': self.feed_rate,
            'spindle_speed': self.spindle_speed,
            'plane': self.plane,
            'tool_number': self.tool_number,
            'current_position': self.current_position.copy(),
            'machine_position': self.machine_position.copy(),
            'program_running': self.program_running,
            'optional_stop': self.optional_stop
        }