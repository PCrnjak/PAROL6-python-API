"""
GCODE Implementation for PAROL6 Robot

This module provides GCODE parsing and execution capabilities for the PAROL6 robot.
It translates standard GCODE commands into robot motion commands.

Main components:
- parser.py: GCODE tokenization and parsing
- state.py: Modal state tracking for GCODE execution
- commands.py: Command mapping from GCODE to robot commands
- coordinates.py: Work coordinate system management
- interpreter.py: Main GCODE interpreter
- utils.py: Utility functions for conversions and calculations
"""

from .parser import GcodeParser, GcodeToken
from .state import GcodeState
from .interpreter import GcodeInterpreter
from .coordinates import WorkCoordinateSystem

__version__ = "0.1.0"
__all__ = ['GcodeParser', 'GcodeToken', 'GcodeState', 'GcodeInterpreter', 'WorkCoordinateSystem']