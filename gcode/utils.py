"""
Utility functions for GCODE processing

Provides conversion functions, calculations, and helpers for GCODE implementation.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


def feed_rate_to_duration(distance: float, feed_rate: float) -> float:
    """
    Convert feed rate to duration for a given distance
    
    Args:
        distance: Distance to travel in mm
        feed_rate: Feed rate in mm/min
        
    Returns:
        Duration in seconds
    """
    if feed_rate <= 0:
        return 0
    
    # Convert mm/min to mm/s
    feed_rate_mm_s = feed_rate / 60.0
    
    # Calculate duration
    duration = distance / feed_rate_mm_s
    
    return duration


def feed_rate_to_speed_percentage(feed_rate: float, 
                                 min_speed: float = 120.0,
                                 max_speed: float = 3600.0) -> float:
    """
    Convert feed rate to speed percentage
    
    Args:
        feed_rate: Feed rate in mm/min
        min_speed: Minimum speed in mm/min (default 120 = 2 mm/s)
        max_speed: Maximum speed in mm/min (default 3600 = 60 mm/s)
        
    Returns:
        Speed percentage (0-100)
    """
    # Clamp feed rate to valid range
    feed_rate = np.clip(feed_rate, min_speed, max_speed)
    
    # Map to percentage
    speed_percentage = np.interp(feed_rate, [min_speed, max_speed], [0, 100])
    
    return speed_percentage


def speed_percentage_to_feed_rate(speed_percentage: float,
                                 min_speed: float = 120.0,
                                 max_speed: float = 3600.0) -> float:
    """
    Convert speed percentage to feed rate
    
    Args:
        speed_percentage: Speed percentage (0-100)
        min_speed: Minimum speed in mm/min
        max_speed: Maximum speed in mm/min
        
    Returns:
        Feed rate in mm/min
    """
    # Clamp percentage
    speed_percentage = np.clip(speed_percentage, 0, 100)
    
    # Map to feed rate
    feed_rate = np.interp(speed_percentage, [0, 100], [min_speed, max_speed])
    
    return feed_rate


def calculate_distance(start: Dict[str, float], end: Dict[str, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        start: Starting position {X, Y, Z, ...}
        end: Ending position {X, Y, Z, ...}
        
    Returns:
        Distance in mm
    """
    distance = 0
    for axis in ['X', 'Y', 'Z']:
        if axis in start and axis in end:
            distance += (end[axis] - start[axis]) ** 2
    
    return math.sqrt(distance)


def ijk_to_center(start: Dict[str, float], ijk: Dict[str, float], 
                 plane: str = 'G17') -> Dict[str, float]:
    """
    Convert IJK offsets to arc center point
    
    Args:
        start: Starting position
        ijk: IJK offset values (relative to start)
        plane: Active plane (G17=XY, G18=XZ, G19=YZ)
        
    Returns:
        Center point coordinates
    """
    center = start.copy()
    
    if plane == 'G17':  # XY plane
        if 'I' in ijk:
            center['X'] = start.get('X', 0) + ijk['I']
        if 'J' in ijk:
            center['Y'] = start.get('Y', 0) + ijk['J']
    elif plane == 'G18':  # XZ plane
        if 'I' in ijk:
            center['X'] = start.get('X', 0) + ijk['I']
        if 'K' in ijk:
            center['Z'] = start.get('Z', 0) + ijk['K']
    elif plane == 'G19':  # YZ plane
        if 'J' in ijk:
            center['Y'] = start.get('Y', 0) + ijk['J']
        if 'K' in ijk:
            center['Z'] = start.get('Z', 0) + ijk['K']
    
    return center


def radius_to_center(start: Dict[str, float], end: Dict[str, float], 
                    radius: float, clockwise: bool = True,
                    plane: str = 'G17') -> Dict[str, float]:
    """
    Calculate arc center from radius
    
    Args:
        start: Starting position
        end: Ending position
        radius: Arc radius (positive for <180°, negative for >180°)
        clockwise: True for G2, False for G3
        plane: Active plane
        
    Returns:
        Center point coordinates
    """
    # Get the two axes involved in the arc based on plane
    if plane == 'G17':  # XY plane
        axis1, axis2 = 'X', 'Y'
    elif plane == 'G18':  # XZ plane
        axis1, axis2 = 'X', 'Z'
    else:  # G19 - YZ plane
        axis1, axis2 = 'Y', 'Z'
    
    # Get start and end coordinates
    x1 = start.get(axis1, 0)
    y1 = start.get(axis2, 0)
    x2 = end.get(axis1, 0)
    y2 = end.get(axis2, 0)
    
    # Calculate midpoint
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    
    # Calculate distance between points
    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx**2 + dy**2)
    
    # Check if arc is possible
    if d > 2 * abs(radius):
        raise ValueError(f"Arc radius {radius} too small for distance {d}")
    
    # Calculate distance from midpoint to center
    if abs(d) < 1e-10:  # Points are the same
        return start.copy()
    
    h = math.sqrt(radius**2 - (d/2)**2)
    
    # Calculate perpendicular direction
    px = -dy / d
    py = dx / d
    
    # Determine direction based on radius sign and clockwise flag
    if radius < 0:
        h = -h
    if not clockwise:
        h = -h
    
    # Calculate center
    center = start.copy()
    center[axis1] = mx + h * px
    center[axis2] = my + h * py
    
    return center


def validate_arc(start: Dict[str, float], end: Dict[str, float],
                center: Dict[str, float], plane: str = 'G17') -> bool:
    """
    Validate arc parameters
    
    Args:
        start: Starting position
        end: Ending position
        center: Arc center
        plane: Active plane
        
    Returns:
        True if arc is valid
    """
    # Get the two axes involved
    if plane == 'G17':
        axis1, axis2 = 'X', 'Y'
    elif plane == 'G18':
        axis1, axis2 = 'X', 'Z'
    else:
        axis1, axis2 = 'Y', 'Z'
    
    # Calculate radii from center to start and end
    r_start = math.sqrt(
        (start.get(axis1, 0) - center.get(axis1, 0))**2 +
        (start.get(axis2, 0) - center.get(axis2, 0))**2
    )
    
    r_end = math.sqrt(
        (end.get(axis1, 0) - center.get(axis1, 0))**2 +
        (end.get(axis2, 0) - center.get(axis2, 0))**2
    )
    
    # Check if radii are approximately equal (within 0.01mm)
    return abs(r_start - r_end) < 0.01


def estimate_motion_time(start: Dict[str, float], end: Dict[str, float],
                        feed_rate: float, is_rapid: bool = False) -> float:
    """
    Estimate time for a motion command
    
    Args:
        start: Starting position
        end: Ending position
        feed_rate: Feed rate in mm/min
        is_rapid: True for G0 rapid moves
        
    Returns:
        Estimated time in seconds
    """
    if is_rapid:
        # Rapid moves use maximum speed
        # Use robot's actual max linear velocity (60 mm/s)
        rapid_speed = 60.0  # mm/s (from PAROL6_ROBOT.Cartesian_linear_velocity_max)
        distance = calculate_distance(start, end)
        return distance / rapid_speed
    else:
        # Regular moves use feed rate
        distance = calculate_distance(start, end)
        return feed_rate_to_duration(distance, feed_rate)


def parse_gcode_file(filepath: str) -> List[str]:
    """
    Parse GCODE file and return list of lines
    
    Args:
        filepath: Path to GCODE file
        
    Returns:
        List of GCODE lines
    """
    lines = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Remove whitespace and convert to uppercase
                line = line.strip()
                if line and not line.startswith('%'):  # Skip empty lines and % markers
                    lines.append(line)
    except Exception as e:
        print(f"Error reading GCODE file: {e}")
    
    return lines


def format_gcode_number(value: float, decimals: int = 3) -> str:
    """
    Format number for GCODE output
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        
    Returns:
        Formatted string without trailing zeros
    """
    formatted = f"{value:.{decimals}f}"
    # Remove trailing zeros and decimal point if not needed
    formatted = formatted.rstrip('0').rstrip('.')
    return formatted


def split_into_segments(start: Dict[str, float], end: Dict[str, float],
                       max_segment_length: float = 10.0) -> List[Dict[str, float]]:
    """
    Split long moves into smaller segments
    
    Args:
        start: Starting position
        end: Ending position
        max_segment_length: Maximum segment length in mm
        
    Returns:
        List of intermediate positions
    """
    distance = calculate_distance(start, end)
    
    if distance <= max_segment_length:
        return [end]
    
    # Calculate number of segments
    num_segments = int(math.ceil(distance / max_segment_length))
    
    # Generate intermediate points
    points = []
    for i in range(1, num_segments + 1):
        t = i / num_segments
        point = {}
        for axis in ['X', 'Y', 'Z', 'A', 'B', 'C']:
            if axis in start and axis in end:
                point[axis] = start[axis] + t * (end[axis] - start[axis])
        points.append(point)
    
    return points


def inches_to_mm(value: float) -> float:
    """Convert inches to millimeters"""
    return value * 25.4


def mm_to_inches(value: float) -> float:
    """Convert millimeters to inches"""
    return value / 25.4