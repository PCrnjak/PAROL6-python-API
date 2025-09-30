"""
GCODE Command Mappings for PAROL6 Robot

Maps GCODE commands to robot motion commands.
Implements command objects that interface with the existing robot API.
"""

import numpy as np
from typing import Dict, Optional
from parol6.PAROL6_ROBOT import cart
from .state import GcodeState
from .coordinates import WorkCoordinateSystem
from .utils import ijk_to_center, radius_to_center, validate_arc
from parol6.commands.base import CommandBase

class GcodeCommand(CommandBase):
    """Base class for GCODE commands"""
    
    def __init__(self):
        super().__init__()
    
    def to_robot_command(self) -> str:
        """
        Convert to robot API command string
        
        Returns:
            Command string for robot API
        """
        return ""


class G0Command(GcodeCommand):
    """G0 - Rapid positioning command"""
    
    def __init__(self, target_position: Dict[str, float], state: GcodeState, coord_system: WorkCoordinateSystem):
        """
        Initialize G0 rapid move command
        
        Args:
            target_position: Target position in work coordinates
            state: Current GCODE state
            coord_system: Work coordinate system
        """
        super().__init__()
        self.target_position = target_position
        self.state = state
        self.coord_system = coord_system
        
        # Convert to machine coordinates
        self.machine_position = coord_system.work_to_machine(target_position)
        
        # Convert to robot coordinates [X, Y, Z, RX, RY, RZ]
        self.robot_position = coord_system.gcode_to_robot_coords(self.machine_position)
    
    def to_robot_command(self) -> str:
        """
        Convert to MovePose command for robot API
        
        G0 uses rapid movement (100% speed)
        """
        # Format: MOVEPOSE|X|Y|Z|RX|RY|RZ|duration|speed
        # Where duration="None" for speed-based, speed="None" for duration-based
        x, y, z = self.robot_position[0:3]
        rx, ry, rz = self.robot_position[3:6] if len(self.robot_position) >= 6 else [0, 0, 0]
        
        # G0 uses rapid speed (100%)
        speed_percentage = 100
        duration = "None"  # Speed-based movement
        
        command = f"MOVEPOSE|{x:.3f}|{y:.3f}|{z:.3f}|{rx:.3f}|{ry:.3f}|{rz:.3f}|{duration}|{speed_percentage}"
        return command


class G1Command(GcodeCommand):
    """G1 - Linear interpolation command"""
    
    def __init__(self, target_position: Dict[str, float], state: GcodeState, coord_system: WorkCoordinateSystem):
        """
        Initialize G1 linear move command
        
        Args:
            target_position: Target position in work coordinates
            state: Current GCODE state
            coord_system: Work coordinate system
        """
        super().__init__()
        self.target_position = target_position
        self.state = state
        self.coord_system = coord_system
        
        # Convert to machine coordinates
        self.machine_position = coord_system.work_to_machine(target_position)
        
        # Convert to robot coordinates
        self.robot_position = coord_system.gcode_to_robot_coords(self.machine_position)
        
        # Get feed rate from state (mm/min)
        self.feed_rate = state.feed_rate
    
    def to_robot_command(self) -> str:
        """
        Convert to MoveCart command for robot API
        
        G1 uses linear interpolation with specified feed rate
        """
        # Format: MOVECART|X|Y|Z|RX|RY|RZ|duration|speed
        x, y, z = self.robot_position[0:3]
        rx, ry, rz = self.robot_position[3:6] if len(self.robot_position) >= 6 else [0, 0, 0]
        
        # Convert feed rate (mm/min) to speed percentage
        # Import robot speed limits from configuration
        # Values are in m/s, convert to mm/min
        max_speed_mm_min = cart.vel.linear.max * 1000 * 60  # m/s to mm/min
        min_speed_mm_min = cart.vel.linear.min * 1000 * 60  # m/s to mm/min
        
        # Map feed rate to percentage (0-100)
        speed_percentage = np.interp(
            self.feed_rate,
            [min_speed_mm_min, max_speed_mm_min],
            [0, 100]
        )
        speed_percentage = np.clip(speed_percentage, 0, 100)
        
        duration = "None"  # Speed-based movement
        
        command = f"MOVECART|{x:.3f}|{y:.3f}|{z:.3f}|{rx:.3f}|{ry:.3f}|{rz:.3f}|{duration}|{speed_percentage:.1f}"
        return command


class G2Command(GcodeCommand):
    """G2 - Clockwise circular interpolation command"""
    
    def __init__(self, target_position: Dict[str, float], 
                 arc_params: Dict[str, float],
                 state: GcodeState, 
                 coord_system: WorkCoordinateSystem):
        """
        Initialize G2 clockwise arc command
        
        Args:
            target_position: Target (end) position in work coordinates
            arc_params: Arc parameters (I, J, K offsets or R radius)
            state: Current GCODE state
            coord_system: Work coordinate system
        """
        super().__init__()
        self.target_position = target_position
        self.arc_params = arc_params
        self.state = state
        self.coord_system = coord_system
        
        # Get current position
        self.start_position = state.current_position.copy()
        
        # Determine arc center based on parameters
        if 'R' in arc_params:
            # Radius format
            self.center = radius_to_center(
                self.start_position,
                target_position,
                arc_params['R'],
                clockwise=True,
                plane=state.plane
            )
        else:
            # IJK offset format
            ijk = {}
            for key in ['I', 'J', 'K']:
                if key in arc_params:
                    ijk[key] = arc_params[key]
            self.center = ijk_to_center(
                self.start_position,
                ijk,
                plane=state.plane
            )
        
        # Validate arc
        if not validate_arc(self.start_position, target_position, self.center, state.plane):
            self.is_valid = False
            self.error_message = "Invalid arc: start and end radii don't match"
        
        # Convert positions to machine coordinates
        self.machine_start = coord_system.work_to_machine(self.start_position)
        self.machine_end = coord_system.work_to_machine(target_position)
        self.machine_center = coord_system.work_to_machine(self.center)
        
        # Convert to robot coordinates
        self.robot_start = coord_system.gcode_to_robot_coords(self.machine_start)
        self.robot_end = coord_system.gcode_to_robot_coords(self.machine_end)
        self.robot_center = coord_system.gcode_to_robot_coords(self.machine_center)
        
        # Get feed rate from state
        self.feed_rate = state.feed_rate
    
    def to_robot_command(self) -> str:
        """
        Convert to SMOOTH_ARC_CENTER command for robot API
        
        G2 uses clockwise arc interpolation with specified feed rate
        """
        # Format: SMOOTH_ARC_CENTER|end_x|end_y|end_z|end_rx|end_ry|end_rz|center_x|center_y|center_z|frame|start_x|start_y|start_z|start_rx|start_ry|start_rz|duration|speed|clockwise
        
        # Extract positions
        end_x, end_y, end_z = self.robot_end[0:3]
        end_rx, end_ry, end_rz = self.robot_end[3:6] if len(self.robot_end) >= 6 else [0, 0, 0]
        
        center_x, center_y, center_z = self.robot_center[0:3]
        
        start_x, start_y, start_z = self.robot_start[0:3]
        start_rx, start_ry, start_rz = self.robot_start[3:6] if len(self.robot_start) >= 6 else [0, 0, 0]
        
        # Convert feed rate to speed percentage
        max_speed_mm_min = cart.vel.linear.max * 1000 * 60
        min_speed_mm_min = cart.vel.linear.min * 1000 * 60
        
        speed_percentage = np.interp(
            self.feed_rate,
            [min_speed_mm_min, max_speed_mm_min],
            [0, 100]
        )
        speed_percentage = np.clip(speed_percentage, 0, 100)
        
        # Build command string
        end_str = f"{end_x:.3f}|{end_y:.3f}|{end_z:.3f}|{end_rx:.3f}|{end_ry:.3f}|{end_rz:.3f}"
        center_str = f"{center_x:.3f}|{center_y:.3f}|{center_z:.3f}"
        start_str = f"{start_x:.3f}|{start_y:.3f}|{start_z:.3f}|{start_rx:.3f}|{start_ry:.3f}|{start_rz:.3f}"
        
        # Use speed-based movement
        duration = "None"
        frame = "0"  # World frame
        clockwise = "True"  # G2 is clockwise
        
        command = f"SMOOTH_ARC_CENTER|{end_str}|{center_str}|{frame}|{start_str}|{duration}|{speed_percentage:.1f}|{clockwise}"
        return command


class G3Command(GcodeCommand):
    """G3 - Counter-clockwise circular interpolation command"""
    
    def __init__(self, target_position: Dict[str, float], 
                 arc_params: Dict[str, float],
                 state: GcodeState, 
                 coord_system: WorkCoordinateSystem):
        """
        Initialize G3 counter-clockwise arc command
        
        Args:
            target_position: Target (end) position in work coordinates
            arc_params: Arc parameters (I, J, K offsets or R radius)
            state: Current GCODE state
            coord_system: Work coordinate system
        """
        super().__init__()
        self.target_position = target_position
        self.arc_params = arc_params
        self.state = state
        self.coord_system = coord_system
        
        # Get current position
        self.start_position = state.current_position.copy()
        
        # Determine arc center based on parameters
        if 'R' in arc_params:
            # Radius format
            self.center = radius_to_center(
                self.start_position,
                target_position,
                arc_params['R'],
                clockwise=False,  # G3 is counter-clockwise
                plane=state.plane
            )
        else:
            # IJK offset format
            ijk = {}
            for key in ['I', 'J', 'K']:
                if key in arc_params:
                    ijk[key] = arc_params[key]
            self.center = ijk_to_center(
                self.start_position,
                ijk,
                plane=state.plane
            )
        
        # Validate arc
        if not validate_arc(self.start_position, target_position, self.center, state.plane):
            self.is_valid = False
            self.error_message = "Invalid arc: start and end radii don't match"
        
        # Convert positions to machine coordinates
        self.machine_start = coord_system.work_to_machine(self.start_position)
        self.machine_end = coord_system.work_to_machine(target_position)
        self.machine_center = coord_system.work_to_machine(self.center)
        
        # Convert to robot coordinates
        self.robot_start = coord_system.gcode_to_robot_coords(self.machine_start)
        self.robot_end = coord_system.gcode_to_robot_coords(self.machine_end)
        self.robot_center = coord_system.gcode_to_robot_coords(self.machine_center)
        
        # Get feed rate from state
        self.feed_rate = state.feed_rate
    
    def to_robot_command(self) -> str:
        """
        Convert to SMOOTH_ARC_CENTER command for robot API
        
        G3 uses counter-clockwise arc interpolation with specified feed rate
        """
        # Format: SMOOTH_ARC_CENTER|end_x|end_y|end_z|end_rx|end_ry|end_rz|center_x|center_y|center_z|frame|start_x|start_y|start_z|start_rx|start_ry|start_rz|duration|speed|clockwise
        
        # Extract positions
        end_x, end_y, end_z = self.robot_end[0:3]
        end_rx, end_ry, end_rz = self.robot_end[3:6] if len(self.robot_end) >= 6 else [0, 0, 0]
        
        center_x, center_y, center_z = self.robot_center[0:3]
        
        start_x, start_y, start_z = self.robot_start[0:3]
        start_rx, start_ry, start_rz = self.robot_start[3:6] if len(self.robot_start) >= 6 else [0, 0, 0]
        
        # Convert feed rate to speed percentage
        max_speed_mm_min = cart.vel.linear.max * 1000 * 60
        min_speed_mm_min = cart.vel.linear.min * 1000 * 60
        
        speed_percentage = np.interp(
            self.feed_rate,
            [min_speed_mm_min, max_speed_mm_min],
            [0, 100]
        )
        speed_percentage = np.clip(speed_percentage, 0, 100)
        
        # Build command string
        end_str = f"{end_x:.3f}|{end_y:.3f}|{end_z:.3f}|{end_rx:.3f}|{end_ry:.3f}|{end_rz:.3f}"
        center_str = f"{center_x:.3f}|{center_y:.3f}|{center_z:.3f}"
        start_str = f"{start_x:.3f}|{start_y:.3f}|{start_z:.3f}|{start_rx:.3f}|{start_ry:.3f}|{start_rz:.3f}"
        
        # Use speed-based movement
        duration = "None"
        frame = "0"  # World frame
        clockwise = "False"  # G3 is counter-clockwise
        
        command = f"SMOOTH_ARC_CENTER|{end_str}|{center_str}|{frame}|{start_str}|{duration}|{speed_percentage:.1f}|{clockwise}"
        return command


class G4Command(GcodeCommand):
    """G4 - Dwell command"""
    
    def __init__(self, dwell_time: float):
        """
        Initialize G4 dwell command
        
        Args:
            dwell_time: Dwell time in seconds
        """
        super().__init__()
        # Validate and clamp dwell time
        if dwell_time < 0.0:
            self.dwell_time = 0.0
        elif dwell_time > 300.0:  # Max 5 minutes
            self.dwell_time = 300.0
        else:
            self.dwell_time = dwell_time
    
    def to_robot_command(self) -> str:
        """
        Convert to Delay command for robot API
        """
        # Format: DELAY|seconds
        command = f"DELAY|{self.dwell_time:.3f}"
        return command


class G28Command(GcodeCommand):
    """G28 - Return to home command"""
    
    def __init__(self):
        """Initialize G28 home command"""
        super().__init__()
    
    def to_robot_command(self) -> str:
        """
        Convert to Home command for robot API
        """
        # Format: HOME
        command = "HOME"
        return command


class M3Command(GcodeCommand):
    """M3 - Spindle/Gripper on CW (close gripper)"""
    
    def __init__(self, gripper_port: int = 1):
        """Initialize M3 gripper close command"""
        super().__init__()
        # Validate gripper port
        if gripper_port not in [1, 2]:
            self.is_valid = False
            self.error_message = f"Invalid gripper port {gripper_port}. Must be 1 or 2"
            self.gripper_port = 1  # Default to port 1
        else:
            self.gripper_port = gripper_port
    
    def to_robot_command(self) -> str:
        """
        Convert to Gripper command for robot API
        """
        # Format: PNEUMATICGRIPPER|action|port
        # M3 maps to gripper close
        command = f"PNEUMATICGRIPPER|close|{self.gripper_port}"
        return command


class M5Command(GcodeCommand):
    """M5 - Spindle/Gripper off (open gripper)"""
    
    def __init__(self, gripper_port: int = 1):
        """Initialize M5 gripper open command"""
        super().__init__()
        # Validate gripper port
        if gripper_port not in [1, 2]:
            self.is_valid = False
            self.error_message = f"Invalid gripper port {gripper_port}. Must be 1 or 2"
            self.gripper_port = 1  # Default to port 1
        else:
            self.gripper_port = gripper_port
    
    def to_robot_command(self) -> str:
        """
        Convert to Gripper command for robot API
        """
        # Format: PNEUMATICGRIPPER|action|port
        # M5 maps to gripper open
        command = f"PNEUMATICGRIPPER|open|{self.gripper_port}"
        return command


class M0Command(GcodeCommand):
    """M0 - Program stop"""
    
    def __init__(self):
        """Initialize M0 stop command"""
        super().__init__()
        # This command will need special handling in the interpreter
        self.requires_resume = True
    
    def to_robot_command(self) -> str:
        """
        M0 doesn't directly map to a robot command
        It's handled by the interpreter
        """
        return ""


class M1Command(GcodeCommand):
    """M1 - Optional program stop"""
    
    def __init__(self):
        """Initialize M1 optional stop command"""
        super().__init__()
        # This command will need special handling in the interpreter
        # It only stops if optional_stop is enabled
        self.requires_resume = True
        self.is_optional = True
    
    def to_robot_command(self) -> str:
        """
        M1 doesn't directly map to a robot command
        It's handled by the interpreter based on optional_stop setting
        """
        return ""


class M4Command(GcodeCommand):
    """M4 - Spindle/Gripper on CCW (alternative gripper action)"""
    
    def __init__(self, gripper_port: int = 1):
        """Initialize M4 gripper CCW command"""
        super().__init__()
        # Validate gripper port
        if gripper_port not in [1, 2]:
            self.is_valid = False
            self.error_message = f"Invalid gripper port {gripper_port}. Must be 1 or 2"
            self.gripper_port = 1  # Default to port 1
        else:
            self.gripper_port = gripper_port
    
    def to_robot_command(self) -> str:
        """
        Convert to Gripper command for robot API
        
        Note: M4 typically means counter-clockwise spindle rotation.
        For a gripper, this could map to a different action or be unsupported.
        Currently mapping to gripper toggle or alternative action.
        """
        # For PAROL6, M4 might not have a direct gripper equivalent
        # Could be used for electric gripper with different mode
        # For now, we'll treat it similar to M3 but document the difference
        command = f"PNEUMATICGRIPPER|close|{self.gripper_port}"
        return command


class M30Command(GcodeCommand):
    """M30 - Program end"""
    
    def __init__(self):
        """Initialize M30 end command"""
        super().__init__()
        self.is_program_end = True
    
    def to_robot_command(self) -> str:
        """
        M30 doesn't directly map to a robot command
        It signals the end of the program
        """
        return ""


def create_command_from_token(token, state: GcodeState, coord_system: WorkCoordinateSystem) -> Optional[GcodeCommand]:
    """
    Create a command object from a GCODE token
    
    Args:
        token: GcodeToken object
        state: Current GCODE state
        coord_system: Work coordinate system
        
    Returns:
        GcodeCommand object or None
    """
    if token.code_type == 'G':
        code = int(token.code_number)
        
        if code == 0:  # Rapid positioning
            # Calculate target position
            target = state.calculate_target_position(token.parameters)
            return G0Command(target, state, coord_system)
        
        elif code == 1:  # Linear interpolation
            # Calculate target position
            target = state.calculate_target_position(token.parameters)
            return G1Command(target, state, coord_system)
        
        elif code == 2:  # Clockwise circular interpolation
            # Calculate target position
            target = state.calculate_target_position(token.parameters)
            # Extract arc parameters (I, J, K or R)
            arc_params = {}
            for key in ['I', 'J', 'K', 'R']:
                if key in token.parameters:
                    arc_params[key] = token.parameters[key]
            
            if not arc_params:
                # No arc parameters provided, treat as linear move
                return G1Command(target, state, coord_system)
            
            return G2Command(target, arc_params, state, coord_system)
        
        elif code == 3:  # Counter-clockwise circular interpolation
            # Calculate target position
            target = state.calculate_target_position(token.parameters)
            # Extract arc parameters (I, J, K or R)
            arc_params = {}
            for key in ['I', 'J', 'K', 'R']:
                if key in token.parameters:
                    arc_params[key] = token.parameters[key]
            
            if not arc_params:
                # No arc parameters provided, treat as linear move
                return G1Command(target, state, coord_system)
            
            return G3Command(target, arc_params, state, coord_system)
        
        elif code == 4:  # Dwell
            # Get dwell time from P (milliseconds) or S (seconds)
            if 'P' in token.parameters:
                dwell_time = token.parameters['P'] / 1000.0  # Convert ms to seconds
            elif 'S' in token.parameters:
                dwell_time = token.parameters['S']
            else:
                dwell_time = 0
            return G4Command(dwell_time)
        
        elif code == 28:  # Home
            return G28Command()
        
        # Modal commands that change state but don't generate movement
        elif code in [17, 18, 19, 20, 21, 54, 55, 56, 57, 58, 59, 90, 91]:
            # These are handled by state updates
            return None
    
    elif token.code_type == 'M':
        code = int(token.code_number)
        
        if code == 0:  # Program stop
            return M0Command()
        
        elif code == 1:  # Optional program stop
            return M1Command()
        
        elif code == 3:  # Gripper close
            # Check for P parameter for port number (default 1)
            port = int(token.parameters.get('P', 1))
            return M3Command(gripper_port=port)
        
        elif code == 4:  # Gripper CCW / alternative action
            # Check for P parameter for port number (default 1)
            port = int(token.parameters.get('P', 1))
            return M4Command(gripper_port=port)
        
        elif code == 5:  # Gripper open
            # Check for P parameter for port number (default 1)
            port = int(token.parameters.get('P', 1))
            return M5Command(gripper_port=port)
        
        elif code == 30:  # Program end
            return M30Command()
    
    elif token.code_type in ['F', 'S', 'T', 'COMMENT']:
        # These don't generate commands, just update state
        return None
    
    return None
