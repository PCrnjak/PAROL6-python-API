"""
Zero-Overhead Robot API with Optional Acknowledgments
======================================================
This version guarantees ZERO resource overhead when tracking is not used.
The tracking system is only initialized when explicitly requested.
"""

import socket
from typing import List, Optional, Literal, Dict, Tuple, Union
import time
import threading
import uuid
import json
import os
from datetime import datetime, timedelta

def _get_env_int(name: str, default: int) -> int:
    """
    Safe environment variable parsing for integers.
    Returns default for unset or empty string values.
    """
    value = os.getenv(name)
    if not value:  # None or empty string
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {name}='{value}' is not a valid integer")

# Global configuration with environment variable overrides  
SERVER_IP = os.getenv("PAROL6_SERVER_IP", "127.0.0.1")
SERVER_PORT = _get_env_int("PAROL6_SERVER_PORT", 5001)
ACK_PORT = _get_env_int("PAROL6_ACK_PORT", 5002)

# Global tracker - starts as None (no resources)
_command_tracker = None
_tracker_lock = threading.Lock()

def reset_tracking():
    """
    Reset and cleanup the command tracker.
    Useful for tests and cleanup scenarios.
    """
    global _command_tracker, _tracker_lock
    
    with _tracker_lock:
        if _command_tracker:
            _command_tracker._cleanup()
            _command_tracker = None

# ============================================================================
# ORIGINAL SEND FUNCTION - ZERO OVERHEAD
# ============================================================================

def send_robot_command(command_string: str):
    """
    Original send function - NO TRACKING, NO OVERHEAD.
    This is what gets called for all backward-compatible operations.
    
    Resource usage:
    - No threads
    - No extra sockets
    - No memory allocation
    - Exactly the same as your original implementation
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command_string.encode('utf-8'), (SERVER_IP, SERVER_PORT))
        return f"Successfully sent command: '{command_string[:50]}...'"
    except Exception as e:
        return f"Error sending command: {e}"

# ============================================================================
# TRACKING SYSTEM - ONLY LOADED WHEN NEEDED
# ============================================================================

class LazyCommandTracker:
    """
    Command tracker with lazy initialization.
    Resources are ONLY allocated when tracking is actually used.
    """
    
    def __init__(self, listen_port=None, history_size=100):
        # Use ACK_PORT constant if not specified
        if listen_port is None:
            listen_port = ACK_PORT
        self.listen_port = listen_port
        self.history_size = history_size
        self.command_history = {}
        self.lock = threading.Lock()
        
        # Lazy initialization flags
        self._initialized = False
        self._thread = None
        self._socket = None
        self._running = False
    
    def _lazy_init(self):
        """
        Initialize resources only when first tracking is requested.
        This is called ONLY when someone uses tracking features.
        """
        if self._initialized:
            return True
            
        try:
            print("[Tracker] First tracking request - initializing resources...")
            
            # Socket initialization
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(('', self.listen_port))
            self._socket.settimeout(0.1)
            
            # Start thread
            self._running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            
            self._initialized = True
            print(f"[Tracker] Initialized on port {self.listen_port}")
            return True
            
        except Exception as e:
            print(f"[Tracker] Failed to initialize: {e}")
            self._cleanup()
            return False
    
    def _cleanup(self):
        """Clean up resources"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._socket:
            self._socket.close()
            self._socket = None
        self._initialized = False
    
    def _listen_loop(self):
        """Listener thread - only runs if tracking is used"""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(2048)
                message = data.decode('utf-8')
                
                parts = message.split('|', 3)
                if parts[0] == 'ACK' and len(parts) >= 3:
                    cmd_id = parts[1]
                    status = parts[2]
                    details = parts[3] if len(parts) > 3 else ""
                    
                    with self.lock:
                        if cmd_id in self.command_history:
                            self.command_history[cmd_id].update({
                                'status': status,
                                'details': details,
                                'ack_time': datetime.now(),
                                'completed': status in ['COMPLETED', 'FAILED', 'INVALID', 'CANCELLED']
                            })
                    
                    # Clean old entries (only if we have many)
                    if len(self.command_history) > self.history_size:
                        self._cleanup_old_entries()
                        
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    pass  # Silently continue
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory growth"""
        with self.lock:
            now = datetime.now()
            expired = [cmd_id for cmd_id, info in self.command_history.items()
                      if now - info['sent_time'] > timedelta(seconds=30)]
            for cmd_id in expired:
                del self.command_history[cmd_id]
    
    def track_command(self, command: str) -> Tuple[str, Optional[str]]:
        """
        Track a command - initializes tracker if needed.
        Returns (modified_command, cmd_id)
        """
        # Initialize on first use
        if not self._initialized:
            if not self._lazy_init():
                # Initialization failed - fall back to non-tracking
                return command, None
        
        # Generate ID and modify command
        cmd_id = str(uuid.uuid4())[:8]
        tracked_command = f"{cmd_id}|{command}"
        
        # Register in history
        with self.lock:
            self.command_history[cmd_id] = {
                'command': command,
                'sent_time': datetime.now(),
                'status': 'SENT',
                'details': '',
                'completed': False
            }
        
        return tracked_command, cmd_id
    
    def get_status(self, cmd_id: str) -> Optional[Dict]:
        """Get status if tracker is initialized"""
        if not self._initialized:
            return None
        with self.lock:
            return self.command_history.get(cmd_id, None)
    
    def wait_for_completion(self, cmd_id: str, timeout: float = 5.0) -> Dict:
        """Wait for completion if tracker is initialized"""
        if not self._initialized:
            return {'status': 'NO_TRACKING', 'details': 'Tracker not initialized', 'completed': True}
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(cmd_id)
            if status and status['completed']:
                return status
            time.sleep(0.01)
        
        return self.get_status(cmd_id) or {
            'status': 'TIMEOUT',
            'details': 'No acknowledgment received',
            'completed': True
        }
    
    def is_active(self) -> bool:
        """Check if tracker is initialized and running"""
        return self._initialized and self._running

# ============================================================================
# LAZY TRACKER ACCESS
# ============================================================================

def _get_tracker_if_needed() -> Optional[LazyCommandTracker]:
    """
    Get tracker ONLY if tracking is requested.
    This ensures zero overhead for non-tracking operations.
    """
    global _command_tracker, _tracker_lock
    
    # Fast path - tracker already exists
    if _command_tracker is not None:
        return _command_tracker
    
    # Slow path - create tracker (only happens once)
    with _tracker_lock:
        if _command_tracker is None:
            _command_tracker = LazyCommandTracker()
        return _command_tracker

# ============================================================================
# ENHANCED SEND WITH OPTIONAL TRACKING
# ============================================================================

def send_robot_command_tracked(command_string: str) -> Tuple[str, Optional[str]]:
    """
    Send with tracking - initializes tracker on first use.
    
    Resource impact:
    - First call: Starts tracker thread
    - Subsequent calls: Minimal overhead (UUID generation)
    """
    tracker = _get_tracker_if_needed()
    if tracker:
        tracked_cmd, cmd_id = tracker.track_command(command_string)
        if cmd_id:
            # Send tracked command
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(tracked_cmd.encode('utf-8'), (SERVER_IP, SERVER_PORT))
                return f"Command sent with tracking (ID: {cmd_id})", cmd_id
            except Exception as e:
                return f"Error: {e}", None
    
    # Fall back to non-tracked
    return send_robot_command(command_string), None

def send_and_wait(
    command_string: str, 
    timeout: float = 2.0, 
    non_blocking: bool = False
    ) -> Union[Dict, str, None]:
    """
    Send and wait for acknowledgment OR return a command_id immediately.
    First use initializes tracker.
    """
    result, cmd_id = send_robot_command_tracked(command_string)
    
    if cmd_id:
        # If non_blocking is True, return the ID right away
        if non_blocking:
            return cmd_id
            
        # Otherwise, proceed with the original blocking logic
        tracker = _get_tracker_if_needed()
        if tracker:
            status_dict = tracker.wait_for_completion(cmd_id, timeout)
            # Add the command_id to the returned dictionary
            status_dict['command_id'] = cmd_id
            return status_dict
    
    # Fallback for tracking failures
    if non_blocking:
        return None
    else:
        return {'status': 'NO_TRACKING', 'details': result, 'completed': True, 'command_id': None}

# ============================================================================
# BACKWARD COMPATIBLE MOVEMENT FUNCTIONS - ZERO OVERHEAD BY DEFAULT
# ============================================================================

def move_robot_joints(
    joint_angles: List[float],
    duration: Optional[float] = None,
    speed_percentage: Optional[int] = None,
    wait_for_ack: bool = False,  # Default: No tracking, no overhead
    timeout: float = 2.0,
    non_blocking: bool = False
):
    """
    Move robot joints.
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    # Validation
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either a duration or a speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    # Build command
    angles_str = "|".join(map(str, joint_angles))
    duration_str = str(duration) if duration is not None else "None"
    speed_str = str(speed_percentage) if speed_percentage is not None else "None"
    command = f"MOVEJOINT|{angles_str}|{duration_str}|{speed_str}"
    
    # Send with or without tracking
    if wait_for_ack:
        # User explicitly requested tracking - initialize if needed
        return send_and_wait(command, timeout, non_blocking)
    else:
        # Default path - NO TRACKING, NO OVERHEAD
        return send_robot_command(command)

def move_robot_pose(
    pose: List[float],
    duration: Optional[float] = None,
    speed_percentage: Optional[int] = None,
    wait_for_ack: bool = False,  # Default: No tracking
    timeout: float = 2.,
    non_blocking: bool = False
):
    """
    Move to pose - zero overhead by default.
    """
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either a duration or a speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    pose_str = "|".join(map(str, pose))
    duration_str = str(duration) if duration is not None else "None"
    speed_str = str(speed_percentage) if speed_percentage is not None else "None"
    command = f"MOVEPOSE|{pose_str}|{duration_str}|{speed_str}"
    
    if wait_for_ack:
        result = send_and_wait(command, timeout, non_blocking)
        return result if result is not None else {'status': 'ERROR', 'details': 'Send failed'}
    else:
        return send_robot_command(command)
    
def jog_robot_joint(
    joint_index: int,
    speed_percentage: int,
    duration: Optional[float] = None,
    distance_deg: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 2.0,
    non_blocking: bool = False
):
    """
    Jogs a single robot joint for a specified time or distance.
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    if duration is None and distance_deg is None:
        error = "Error: You must provide either a duration or a distance_deg."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            error = "Error: Duration must be a valid number."
            return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    duration_str = str(duration) if duration is not None else "None"
    distance_str = str(distance_deg) if distance_deg is not None else "None"
    command = f"JOG|{joint_index}|{speed_percentage}|{duration_str}|{distance_str}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def jog_multiple_joints(
    joints: List[int], 
    speeds: List[float], 
    duration: float,
    wait_for_ack: bool = False,
    timeout: float = 2.0,
    non_blocking: bool = False
) -> Union[str, Dict]:
    """
    Jogs multiple robot joints simultaneously for a specified duration.

    Args:
        joints: List of joint indices (0-5 for positive, 6-11 for negative)
        speeds: List of corresponding speeds (1-100%)
        duration: Duration of the jog in seconds
        wait_for_ack: Enable command tracking (default False)
        timeout: Timeout for acknowledgment
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    if len(joints) != len(speeds):
        error = "Error: The number of joints must match the number of speeds."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    joints_str = ",".join(map(str, joints))
    speeds_str = ",".join(map(str, speeds))
    command = f"MULTIJOG|{joints_str}|{speeds_str}|{duration}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def jog_cartesian(
    frame: Literal['TRF', 'WRF'],
    axis: Literal['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-', 'RX+', 'RX-', 'RY+', 'RY-', 'RZ+', 'RZ-'],
    speed_percentage: int,
    duration: float,
    wait_for_ack: bool = False,
    timeout: float = 2.0,
    non_blocking: bool = False
):
    """
    Jogs the robot's end-effector continuously in Cartesian space.
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            error = "Error: Duration must be a valid number."
            return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    command = f"CARTJOG|{frame}|{axis}|{speed_percentage}|{duration}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def move_robot_cartesian(
    pose: List[float],
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 2.0,
    non_blocking: bool = False
) -> Union[str, Dict]:
    """
    Moves the robot's end-effector to a specific Cartesian pose in a straight line.
    
    Args:
        pose: Target pose as [x, y, z, r, p, y] (mm and degrees)
        duration: Total time for the movement in seconds
        speed_percentage: Movement speed as a percentage (1-100)
        wait_for_ack: Enable command tracking (default False)
        timeout: Timeout for acknowledgment
        
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    # Validate timing arguments
    if (duration is None and speed_percentage is None):
        error = "Error: You must provide either 'duration' or 'speed_percentage'."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    if (duration is not None and speed_percentage is not None):
        error = "Error: Please provide either 'duration' or 'speed_percentage', not both."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    # Prepare command arguments
    duration_arg = 'NONE'
    speed_arg = 'NONE'
    
    if duration is not None:
        try:
            if float(duration) <= 0:
                error = "Error: Duration must be a positive number."
                return {'status': 'INVALID', 'details': error} if wait_for_ack else error
            duration_arg = str(duration)
        except (ValueError, TypeError):
            error = "Error: Duration must be a valid number."
            return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    if speed_percentage is not None:
        try:
            speed_val = float(speed_percentage)
            if not (0 < speed_val <= 100):
                error = "Error: Speed percentage must be between 1 and 100."
                return {'status': 'INVALID', 'details': error} if wait_for_ack else error
            speed_arg = str(speed_val)
        except (ValueError, TypeError):
            error = "Error: Speed percentage must be a valid number."
            return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    # Construct command
    pose_str = "|".join(map(str, pose))
    command = f"MOVECART|{pose_str}|{duration_arg}|{speed_arg}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def control_pneumatic_gripper(
    action: Literal['open', 'close'], 
    port: Literal[1, 2],
    wait_for_ack: bool = False,
    timeout: float = 2.0,
    non_blocking: bool = False
):
    """
    Controls the pneumatic gripper.
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    command = f"PNEUMATICGRIPPER|{action}|{port}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def control_electric_gripper(
    action: Literal['move', 'calibrate'],
    position: Optional[int] = 255,
    speed: Optional[int] = 150,
    current: Optional[int] = 500,
    wait_for_ack: bool = False,
    timeout: float = 2.0,
    non_blocking: bool = False
):
    """
    Controls the electric gripper.
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead, no tracking
    - wait_for_ack=True: Initializes tracker on first use
    """
    action_str = "move" if action == 'move' else 'calibrate'
    command = f"ELECTRICGRIPPER|{action_str}|{position}|{speed}|{current}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)
    
# ============================================================================
# SMOOTH MOTION COMMANDS - WITH START POSITION AND DUAL TIMING SUPPORT
# ============================================================================

def smooth_circle(
    center: List[float],
    radius: float,
    plane: Literal['XY', 'XZ', 'YZ'] = 'XY',
    frame: Literal['WRF', 'TRF'] = 'WRF',
    center_mode: Literal['ABSOLUTE', 'TOOL', 'RELATIVE'] = 'ABSOLUTE',
    entry_mode: Literal['AUTO', 'TANGENT', 'DIRECT', 'NONE'] = 'NONE',
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    clockwise: bool = False,
    trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
    jerk_limit: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 10.0,
    non_blocking: bool = False
):
    """
    Execute a smooth circular motion.
    
    Args:
        center: [x, y, z] center point in mm
        radius: Circle radius in mm
        plane: Plane of the circle ('XY', 'XZ', or 'YZ')
        frame: Reference frame ('WRF' for World, 'TRF' for Tool)
        center_mode: How to interpret center point:
                    'ABSOLUTE' - Use exact coordinates (default)
                    'TOOL' - Center at current tool position
                    'RELATIVE' - Offset from current position
        entry_mode: How to approach circle if not on perimeter:
                   'AUTO' - Generate smooth entry trajectory
                   'TANGENT' - Approach tangentially
                   'DIRECT' - Direct line to nearest point
                   'NONE' - Start immediately (default)
        start_pose: Optional [x, y, z, rx, ry, rz] start pose (mm and degrees).
                   If None, starts from current position.
        duration: Time to complete the circle in seconds
        speed_percentage: Speed as percentage (1-100)
        clockwise: Direction of motion
        trajectory_type: Type of trajectory ('cubic', 'quintic', 's_curve'). Default 'cubic'
        jerk_limit: Optional jerk limit for s_curve trajectory (units/s³)
        wait_for_ack: Enable command tracking (default False)
        timeout: Timeout for acknowledgment
        non_blocking: Return immediately with command ID
    
    Resource usage:
    - wait_for_ack=False (default): ZERO overhead
    - wait_for_ack=True: Initializes tracker on first use
    """
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either duration or speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    center_str = ",".join(map(str, center))
    start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
    clockwise_str = "1" if clockwise else "0"
    
    # Format timing
    if duration is not None:
        timing_str = f"DURATION|{duration}"
    else:
        timing_str = f"SPEED|{speed_percentage}"
    
    # Add trajectory type and new parameters
    traj_params = f"|{trajectory_type}"
    if trajectory_type == 's_curve' and jerk_limit is not None:
        traj_params += f"|{jerk_limit}"
    elif trajectory_type != 'cubic':
        traj_params += "|DEFAULT"  # Use default jerk limit for s_curve
    
    # Add center_mode and entry_mode parameters
    mode_params = f"|{center_mode}|{entry_mode}"
    
    command = f"SMOOTH_CIRCLE|{center_str}|{radius}|{plane}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}{mode_params}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def smooth_arc_center(
    end_pose: List[float],
    center: List[float],
    frame: Literal['WRF', 'TRF'] = 'WRF',
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    clockwise: bool = False,
    trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
    jerk_limit: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 10.0,
    non_blocking: bool = False
):
    """
    Execute a smooth arc motion defined by center point.
    
    Args:
        end_pose: [x, y, z, rx, ry, rz] end pose (mm and degrees)
        center: [x, y, z] arc center point in mm
        frame: Reference frame ('WRF' for World, 'TRF' for Tool)
        start_pose: Optional [x, y, z, rx, ry, rz] start pose.
                   If None, starts from current position.
                   If specified, adds smooth transition from current position.
        duration: Time to complete the arc in seconds
        speed_percentage: Speed as percentage (1-100)
        clockwise: Direction of motion
        trajectory_type: Type of trajectory ('cubic', 'quintic', 's_curve'). Default 'cubic'
        jerk_limit: Optional jerk limit for s_curve trajectory (units/s³)
        wait_for_ack: Enable command tracking
        timeout: Timeout for acknowledgment
        non_blocking: Return immediately with command ID
    """
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either duration or speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    end_str = ",".join(map(str, end_pose))
    center_str = ",".join(map(str, center))
    start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
    clockwise_str = "1" if clockwise else "0"
    
    # Format timing
    if duration is not None:
        timing_str = f"DURATION|{duration}"
    else:
        timing_str = f"SPEED|{speed_percentage}"
    
    # Add trajectory type if not default
    traj_params = f"|{trajectory_type}"
    if trajectory_type == 's_curve' and jerk_limit is not None:
        traj_params += f"|{jerk_limit}"
    elif trajectory_type != 'cubic':
        traj_params += "|DEFAULT"
    
    command = f"SMOOTH_ARC_CENTER|{end_str}|{center_str}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def smooth_arc_parametric(
    end_pose: List[float],
    radius: float,
    arc_angle: float,
    frame: Literal['WRF', 'TRF'] = 'WRF',
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    clockwise: bool = False,
    trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
    jerk_limit: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 10.0,
    non_blocking: bool = False
):
    """
    Execute a smooth arc motion defined by radius and angle.
    
    Args:
        end_pose: [x, y, z, rx, ry, rz] end pose (mm and degrees)
        radius: Arc radius in mm
        arc_angle: Arc angle in degrees
        frame: Reference frame ('WRF' for World, 'TRF' for Tool)
        start_pose: Optional [x, y, z, rx, ry, rz] start pose.
                   If None, starts from current position.
        duration: Time to complete the arc in seconds
        speed_percentage: Speed as percentage (1-100)
        clockwise: Direction of motion
        trajectory_type: Type of trajectory profile ('cubic', 'quintic', or 's_curve')
        jerk_limit: Maximum jerk for s_curve trajectory (mm/s^3)
        wait_for_ack: Enable command tracking
        timeout: Timeout for acknowledgment
        non_blocking: Return immediately with command ID
    """
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either duration or speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    end_str = ",".join(map(str, end_pose))
    start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
    clockwise_str = "1" if clockwise else "0"
    
    # Format timing
    if duration is not None:
        timing_str = f"DURATION|{duration}"
    else:
        timing_str = f"SPEED|{speed_percentage}"
    
    # Add trajectory type if not default
    traj_params = f"|{trajectory_type}"
    if trajectory_type == 's_curve' and jerk_limit is not None:
        traj_params += f"|{jerk_limit}"
    elif trajectory_type != 'cubic':
        traj_params += "|DEFAULT"
    
    command = f"SMOOTH_ARC_PARAM|{end_str}|{radius}|{arc_angle}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def smooth_spline(
    waypoints: List[List[float]],
    frame: Literal['WRF', 'TRF'] = 'WRF',
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
    jerk_limit: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 10.0,
    non_blocking: bool = False
):
    """
    Execute a smooth spline motion through waypoints.
    
    Args:
        waypoints: List of [x, y, z, rx, ry, rz] poses (mm and degrees)
        frame: Reference frame ('WRF' for World, 'TRF' for Tool)
        start_pose: Optional [x, y, z, rx, ry, rz] start pose.
                   If None, starts from current position.
                   If specified and different from first waypoint, adds transition.
        duration: Total time for the motion in seconds
        speed_percentage: Speed as percentage (1-100)
        trajectory_type: Type of trajectory ('cubic', 'quintic', 's_curve'). Default 'cubic'
        jerk_limit: Optional jerk limit for s_curve trajectory (units/s³)
        wait_for_ack: Enable command tracking
        timeout: Timeout for acknowledgment
        non_blocking: Return immediately with command ID
    """
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either duration or speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    num_waypoints = len(waypoints)
    start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
    
    # Format timing
    if duration is not None:
        timing_str = f"DURATION|{duration}"
    else:
        timing_str = f"SPEED|{speed_percentage}"
    
    # Format waypoints - flatten each waypoint's 6 values
    waypoint_strs = []
    for wp in waypoints:
        waypoint_strs.extend(map(str, wp))
    
    # Build command with trajectory type
    command_parts = ["SMOOTH_SPLINE", str(num_waypoints), frame, start_str, timing_str]
    
    # Add trajectory type
    command_parts.append(trajectory_type)
    if trajectory_type == 's_curve' and jerk_limit is not None:
        command_parts.append(str(jerk_limit))
    elif trajectory_type == 's_curve':
        command_parts.append("DEFAULT")
    
    # Add waypoints
    command_parts.extend(waypoint_strs)
    command = "|".join(command_parts)
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def smooth_helix(
    center: List[float],
    radius: float,
    pitch: float,
    height: float,
    frame: Literal['WRF', 'TRF'] = 'WRF',
    trajectory_type: Literal['cubic', 'quintic', 's_curve'] = 'cubic',
    jerk_limit: Optional[float] = None,
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    clockwise: bool = False,
    wait_for_ack: bool = False,
    timeout: float = 10.0,
    non_blocking: bool = False
):
    """
    Execute a smooth helical motion.
    
    Args:
        center: [x, y, z] helix center point in mm
        radius: Helix radius in mm
        pitch: Vertical distance per revolution in mm
        height: Total height of helix in mm
        frame: Reference frame ('WRF' for World, 'TRF' for Tool)
        trajectory_type: Type of trajectory ('cubic', 'quintic', 's_curve'). Default 'cubic'
        jerk_limit: Optional jerk limit for s_curve trajectory (units/s³)
        start_pose: Optional [x, y, z, rx, ry, rz] start pose.
                   If None, starts from current position on helix perimeter.
        duration: Time to complete the helix in seconds
        speed_percentage: Speed as percentage (1-100)
        clockwise: Direction of motion
        wait_for_ack: Enable command tracking
        timeout: Timeout for acknowledgment
        non_blocking: Return immediately with command ID
    """
    if duration is None and speed_percentage is None:
        error = "Error: You must provide either duration or speed_percentage."
        return {'status': 'INVALID', 'details': error} if wait_for_ack else error
    
    center_str = ",".join(map(str, center))
    start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
    clockwise_str = "1" if clockwise else "0"
    
    # Format timing
    if duration is not None:
        timing_str = f"DURATION|{duration}"
    else:
        timing_str = f"SPEED|{speed_percentage}"
    
    # Add trajectory type parameters
    traj_params = f"|{trajectory_type}"
    if trajectory_type == 's_curve' and jerk_limit is not None:
        traj_params += f"|{jerk_limit}"
    elif trajectory_type != 'cubic':
        traj_params += "|DEFAULT"  # Use default jerk limit for s_curve
    
    command = f"SMOOTH_HELIX|{center_str}|{radius}|{pitch}|{height}|{frame}|{start_str}|{timing_str}|{clockwise_str}{traj_params}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def smooth_blend(
    segments: List[Dict],
    blend_time: float = 0.5,
    frame: Literal['WRF', 'TRF'] = 'WRF',
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    wait_for_ack: bool = False,
    timeout: float = 15.0,
    non_blocking: bool = False
):
    """
    Execute a blended motion through multiple segments.
    
    Args:
        segments: List of segment dictionaries, each containing:
            - 'type': 'LINE', 'CIRCLE', 'ARC', or 'SPLINE'
            - Additional parameters based on type
        blend_time: Time to blend between segments in seconds
        frame: Reference frame ('WRF' for World, 'TRF' for Tool)
        start_pose: Optional [x, y, z, rx, ry, rz] start pose for first segment.
                   If None, starts from current position.
        duration: Total time for entire motion (scales all segments proportionally)
        speed_percentage: Speed as percentage (1-100) for entire motion
        wait_for_ack: Enable command tracking
        timeout: Timeout for acknowledgment
        non_blocking: Return immediately with command ID
        
    Example:
        segments = [
            {'type': 'LINE', 'end': [x,y,z,rx,ry,rz], 'duration': 2.0},
            {'type': 'CIRCLE', 'center': [x,y,z], 'radius': 50, 'plane': 'XY', 
             'duration': 3.0, 'clockwise': False},
            {'type': 'ARC', 'end': [x,y,z,rx,ry,rz], 'center': [x,y,z], 
             'duration': 2.0, 'clockwise': True}
        ]
    """
    num_segments = len(segments)
    start_str = ",".join(map(str, start_pose)) if start_pose else "CURRENT"
    
    # Format timing
    if duration is None and speed_percentage is None:
        # Use individual segment durations
        timing_str = "DEFAULT"
    elif duration is not None:
        timing_str = f"DURATION|{duration}"
    else:
        timing_str = f"SPEED|{speed_percentage}"
    
    # Format segments
    segment_strs = []
    for seg in segments:
        seg_type = seg['type']
        
        if seg_type == 'LINE':
            end_str = ",".join(map(str, seg['end']))
            seg_str = f"LINE|{end_str}|{seg.get('duration', 2.0)}"
            
        elif seg_type == 'CIRCLE':
            center_str = ",".join(map(str, seg['center']))
            clockwise_str = "1" if seg.get('clockwise', False) else "0"
            seg_str = f"CIRCLE|{center_str}|{seg['radius']}|{seg['plane']}|{seg.get('duration', 3.0)}|{clockwise_str}"
            
        elif seg_type == 'ARC':
            end_str = ",".join(map(str, seg['end']))
            center_str = ",".join(map(str, seg['center']))
            clockwise_str = "1" if seg.get('clockwise', False) else "0"
            seg_str = f"ARC|{end_str}|{center_str}|{seg.get('duration', 2.0)}|{clockwise_str}"
            
        elif seg_type == 'SPLINE':
            waypoints_str = ";".join([",".join(map(str, wp)) for wp in seg['waypoints']])
            seg_str = f"SPLINE|{len(seg['waypoints'])}|{waypoints_str}|{seg.get('duration', 3.0)}"
            
        else:
            continue
            
        segment_strs.append(seg_str)
    
    # Build command with || separators between segments
    command = f"SMOOTH_BLEND|{num_segments}|{blend_time}|{frame}|{start_str}|{timing_str}|" + "||".join(segment_strs)
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def smooth_waypoints(
    waypoints: List[List[float]],
    blend_radii: Union[str, List[float]] = 'auto',
    blend_mode: Literal['parabolic', 'circular', 'none'] = 'parabolic',
    via_modes: Optional[List[Literal['via', 'stop']]] = None,
    max_velocity: Optional[float] = None,
    max_acceleration: Optional[float] = None,
    trajectory_type: Literal['quintic', 's_curve', 'cubic'] = 'quintic',
    frame: Literal['WRF', 'TRF'] = 'WRF',
    duration: Optional[float] = None,
    wait_for_ack: bool = True,
    timeout: float = 30.0,
    non_blocking: bool = False
):
    """
    Move through waypoints with corner cutting (mstraj-style).
    
    Implements smooth trajectory planning through multiple waypoints with
    automatic corner blending to avoid acceleration spikes at sharp turns.
    
    Args:
        waypoints: List of waypoint poses [x, y, z, rx, ry, rz]
        blend_radii: 'auto' for automatic calculation, or list of radii in mm
        blend_mode: Type of corner blend ('parabolic', 'circular', 'none')
        via_modes: 'via' (pass through) or 'stop' for each waypoint
        max_velocity: Override maximum velocity (mm/s)
        max_acceleration: Override maximum acceleration (mm/s²)
        trajectory_type: Trajectory interpolation type
        frame: Reference frame ('WRF' or 'TRF')
        duration: Optional total duration for trajectory
        wait_for_ack: Enable command acknowledgment
        timeout: Command timeout in seconds
        non_blocking: If True with wait_for_ack, returns immediately with command ID
        
    Returns:
        Command ID string or response dictionary
        
    Example:
        # Simple square path with automatic corner blending
        waypoints = [
            [100, 0, 100, 0, 0, 0],
            [100, 100, 100, 0, 0, 0],
            [0, 100, 100, 0, 0, 0],
            [0, 0, 100, 0, 0, 0]
        ]
        smooth_waypoints(waypoints, blend_radii='auto')
        
        # Manual blend radii with stop at second waypoint
        smooth_waypoints(
            waypoints, 
            blend_radii=[0, 20, 30, 0],  # No blend at start/end
            via_modes=['via', 'stop', 'via', 'via']
        )
    """
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints required")
    
    # Format waypoints
    waypoints_str = ""
    for wp in waypoints:
        if len(wp) != 6:
            raise ValueError("Each waypoint must have 6 values [x,y,z,rx,ry,rz]")
        waypoints_str += f"{wp[0]},{wp[1]},{wp[2]},{wp[3]},{wp[4]},{wp[5]}|"
    waypoints_str = waypoints_str.rstrip('|')
    
    # Format blend radii
    if blend_radii == 'auto':
        blend_str = "auto"
    else:
        if len(blend_radii) != len(waypoints):
            raise ValueError("blend_radii must match number of waypoints")
        blend_str = ",".join(str(r) for r in blend_radii)
    
    # Format via modes
    if via_modes is None:
        via_str = ",".join(['via'] * len(waypoints))
    else:
        if len(via_modes) != len(waypoints):
            raise ValueError("via_modes must match number of waypoints")
        via_str = ",".join(via_modes)
    
    # Format constraints
    vel_str = str(max_velocity) if max_velocity else "default"
    acc_str = str(max_acceleration) if max_acceleration else "default"
    dur_str = str(duration) if duration else "auto"
    
    # Build command
    command = (f"SMOOTH_WAYPOINTS|{len(waypoints)}|{blend_str}|{blend_mode}|"
              f"{via_str}|{vel_str}|{acc_str}|{trajectory_type}|"
              f"{frame}|{dur_str}|{waypoints_str}")
    
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

# ============================================================================
# CONVENIENCE FUNCTIONS FOR SMOOTH MOTION CHAINS
# ============================================================================

def chain_smooth_motions(
    motions: List[Dict],
    ensure_continuity: bool = True,
    frame: Literal['WRF', 'TRF'] = 'WRF',  # ADD THIS
    wait_for_ack: bool = True,
    timeout: float = 30.0
):
    """
    Chain multiple smooth motions together with automatic continuity.
    
    Args:
        motions: List of motion dictionaries, each with 'type' and parameters
        ensure_continuity: If True, automatically sets start_pose of each motion
                          to end of previous motion for perfect continuity
        frame: Reference frame for all motions ('WRF' or 'TRF')  # ADD THIS
        wait_for_ack: Enable command tracking
        timeout: Timeout per motion
        
    Example:
        chain_smooth_motions([
            {'type': 'circle', 'center': [200, 0, 200], 'radius': 50, 'duration': 5},
            {'type': 'arc', 'end_pose': [250, 50, 200, 0, 0, 90], 'center': [225, 25, 200], 'duration': 3},
            {'type': 'helix', 'center': [250, 50, 150], 'radius': 30, 'pitch': 20, 'height': 100, 'duration': 8}
        ], frame='TRF')  # Can now specify frame
    """
    results = []
    last_end_pose = None
    
    for i, motion in enumerate(motions):
        motion_type = motion.get('type', '').lower()
        
        # Add frame to motion parameters
        motion['frame'] = frame
        
        # Add start_pose from previous motion if ensuring continuity
        if ensure_continuity and last_end_pose and i > 0:
            motion['start_pose'] = last_end_pose
        
        # Execute the appropriate motion (add frame parameter to each call)
        if motion_type == 'circle':
            result = smooth_circle(**{k: v for k, v in motion.items() if k != 'type'}, 
                                  wait_for_ack=wait_for_ack, timeout=timeout)
            last_end_pose = None  # Circles return to start
            
        elif motion_type == 'arc' or motion_type == 'arc_center':
            result = smooth_arc_center(**{k: v for k, v in motion.items() if k != 'type'},
                                      wait_for_ack=wait_for_ack, timeout=timeout)
            last_end_pose = motion.get('end_pose')
            
        elif motion_type == 'arc_param' or motion_type == 'arc_parametric':
            result = smooth_arc_parametric(**{k: v for k, v in motion.items() if k != 'type'},
                                          wait_for_ack=wait_for_ack, timeout=timeout)
            last_end_pose = motion.get('end_pose')
            
        elif motion_type == 'spline':
            result = smooth_spline(**{k: v for k, v in motion.items() if k != 'type'},
                                  wait_for_ack=wait_for_ack, timeout=timeout)
            waypoints = motion.get('waypoints', [])
            last_end_pose = waypoints[-1] if waypoints else None
            
        elif motion_type == 'helix':
            result = smooth_helix(**{k: v for k, v in motion.items() if k != 'type'},
                                 wait_for_ack=wait_for_ack, timeout=timeout)
            last_end_pose = None
            
        else:
            result = {'status': 'INVALID', 'details': f'Unknown motion type: {motion_type}'}
        
        results.append(result)
        
        # Track command result validation
        if wait_for_ack and isinstance(result, dict) and result.get('status') == 'FAILED':
            print(f"Motion {i+1} failed: {result.get('details')}")
            break
    
    return results

def execute_trajectory(
    trajectory: List[List[float]],
    timing_mode: Literal['duration', 'speed'] = 'duration',
    timing_value: float = 5.0,
    motion_type: Literal['spline', 'linear'] = 'spline',
    frame: Literal['WRF', 'TRF'] = 'WRF',  # ADD THIS
    wait_for_ack: bool = True,
    timeout: float = 30.0,
):
    """
    High-level function to execute a trajectory using the best method.
    
    Args:
        trajectory: List of poses [x, y, z, rx, ry, rz]
        timing_mode: 'duration' for total time, 'speed' for percentage
        timing_value: Duration in seconds or speed percentage
        motion_type: 'spline' for smooth curves, 'linear' for point-to-point
        frame: Reference frame ('WRF' or 'TRF')  # ADD THIS
        wait_for_ack: Enable command tracking (recommended for trajectories)
        timeout: Timeout for acknowledgment
    """
    if motion_type == 'spline':
        if timing_mode == 'duration':
            return smooth_spline(trajectory, frame=frame, duration=timing_value,  # ADD frame
                               wait_for_ack=wait_for_ack, timeout=timeout)
        else:
            return smooth_spline(trajectory, frame=frame, speed_percentage=timing_value,  # ADD frame
                               wait_for_ack=wait_for_ack, timeout=timeout)
    else:
        # Linear motion - send as individual move commands
        results = []
        for pose in trajectory:
            if timing_mode == 'duration':
                segment_duration = timing_value / len(trajectory)
                # Note: move_robot_cartesian doesn't support TRF, only smooth motions do
                result = move_robot_cartesian(pose, duration=segment_duration,
                                             wait_for_ack=wait_for_ack, timeout=timeout)
            else:
                result = move_robot_cartesian(pose, speed_percentage=timing_value,
                                             wait_for_ack=wait_for_ack, timeout=timeout)
            results.append(result)
            
            # Track command result validation
            if wait_for_ack and result.get('status') == 'FAILED':
                break
        
        return results

# ============================================================================
# BASIC FUNCTIONS
# ============================================================================

def delay_robot(duration: float, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False):
    """Delay - optional tracking"""
    command = f"DELAY|{duration}"
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def home_robot(wait_for_ack: bool = False, timeout: float = 30.0, non_blocking: bool = False):
    """Home robot - optional tracking (longer timeout for homing)"""
    command = "HOME"
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

def stop_robot_movement(wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False):
    """Stop robot - optional tracking"""
    command = "STOP"
    if wait_for_ack:
        return send_and_wait(command, timeout, non_blocking)
    else:
        return send_robot_command(command)

# ============================================================================
# GET FUNCTIONS - ZERO OVERHEAD, IMMEDIATE RESPONSE
# ============================================================================

def get_robot_pose():
    """
    Get the robot's current end-effector pose.
    Returns [x, y, z, roll, pitch, yaw] or None if it fails.
    
    Resource usage: ZERO overhead - simple request/response
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_POSE"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(2048)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'POSE' and len(parts) == 2:
                pose_values = [float(v) for v in parts[1].split(',')]
                if len(pose_values) == 16:
                    # Convert 4x4 matrix to [x,y,z,r,p,y]
                    import numpy as np
                    from spatialmath import SE3
                    
                    pose_matrix = np.array(pose_values).reshape((4, 4))
                    T = SE3(pose_matrix, check=False)
                    xyz_mm = T.t * 1000  # Convert to mm
                    rpy_deg = T.rpy(unit='deg', order='xyz')
                    
                    # Convert numpy float64 to regular Python floats
                    return [float(x) for x in xyz_mm] + [float(r) for r in rpy_deg]
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for pose response")
        return None
    except Exception as e:
        print(f"Error getting robot pose: {e}")
        return None

def get_robot_joint_angles():
    """
    Get the robot's current joint angles in degrees.
    Returns list of 6 angles or None if it fails.
    
    Resource usage: ZERO overhead - simple request/response
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_ANGLES"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(1024)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'ANGLES' and len(parts) == 2:
                angles = [float(v) for v in parts[1].split(',')]
                return angles
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for angles response")
        return None
    except Exception as e:
        print(f"Error getting robot angles: {e}")
        return None

def get_robot_io(verbose = False):
    """
    Get the robot's current digital I/O status.
    Returns [IN1, IN2, OUT1, OUT2, ESTOP] or None if it fails.
    
    Resource usage: ZERO overhead - simple request/response
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_IO"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(1024)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'IO' and len(parts) == 2:
                io_values = [int(v) for v in parts[1].split(',')]

                if verbose:
                    print("--- I/O Status ---")
                    print(f"  IN1:   {io_values[0]} | {'ON' if io_values[0] else 'OFF'}")
                    print(f"  IN2:   {io_values[1]} | {'ON' if io_values[1] else 'OFF'}")
                    print(f"  OUT1:  {io_values[2]} | {'ON' if io_values[2] else 'OFF'}")
                    print(f"  OUT2:  {io_values[3]} | {'ON' if io_values[3] else 'OFF'}")
                    # More intuitive E-stop display
                    if io_values[4] == 0:
                        print(f"  ESTOP: {io_values[4]} | PRESSED (Emergency Stop Active!)")
                    else:
                        print(f"  ESTOP: {io_values[4]} | OK (Normal Operation)")
                    print("--------------------------")

                return io_values
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for I/O response")
        return None
    except Exception as e:
        print(f"Error getting robot I/O: {e}")
        return None

def get_electric_gripper_status(verbose = False):
    """
    Get the electric gripper's current status.
    Returns [ID, Position, Speed, Current, StatusByte, ObjectDetected] or None.
    
    Resource usage: ZERO overhead - simple request/response
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_GRIPPER"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(1024)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'GRIPPER' and len(parts) == 2:
                gripper_values = [int(v) for v in parts[1].split(',')]
                
                # Decode the status byte
                status_byte = gripper_values[4] if len(gripper_values) > 4 else 0
                is_active = (status_byte & 0b00000001) != 0
                is_moving = (status_byte & 0b00000010) != 0
                is_calibrated = (status_byte & 0b10000000) != 0
                
                # Interpret object detection
                object_detection = gripper_values[5] if len(gripper_values) > 5 else 0
                if object_detection == 1:
                    detection_text = "Yes (closing)"
                elif object_detection == 2:
                    detection_text = "Yes (opening)"
                else:
                    detection_text = "No"


                if verbose:
                    # Print formatted status
                    print("--- Electric Gripper Status ---")
                    print(f"  Device ID:         {gripper_values[0]}")
                    print(f"  Current Position:  {gripper_values[1]}")
                    print(f"  Current Speed:     {gripper_values[2]}")
                    print(f"  Current Current:   {gripper_values[3]}")
                    print(f"  Object Detected:   {detection_text}")
                    print(f"  Status Byte:       {bin(status_byte)}")
                    print(f"    - Calibrated:    {is_calibrated}")
                    print(f"    - Active:        {is_active}")
                    print(f"    - Moving:        {is_moving}")
                    print("-------------------------------")
                
                return gripper_values
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for gripper response")
        return None
    except Exception as e:
        print(f"Error getting gripper status: {e}")
        return None

def get_robot_joint_speeds():
    """
    Get the robot's current joint speeds in steps/sec.
    Returns list of 6 speed values or None if it fails.
    
    Resource usage: ZERO overhead - simple request/response
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_SPEEDS"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(1024)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'SPEEDS' and len(parts) == 2:
                speeds = [float(v) for v in parts[1].split(',')]
                return speeds
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for speeds response")
        return None
    except Exception as e:
        print(f"Error getting robot speeds: {e}")
        return None

def get_robot_pose_matrix():
    """
    Get the robot's current pose as a 4x4 transformation matrix.
    Returns 4x4 numpy array or None if it fails.
    
    Resource usage: ZERO overhead - simple request/response
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_POSE"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(2048)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'POSE' and len(parts) == 2:
                pose_values = [float(v) for v in parts[1].split(',')]
                if len(pose_values) == 16:
                    import numpy as np
                    return np.array(pose_values).reshape((4, 4))
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for pose response")
        return None
    except Exception as e:
        print(f"Error getting robot pose matrix: {e}")
        return None

def is_robot_stopped(threshold_speed: float = 2.0) -> bool:
    """
    Check if the robot has stopped moving.
    
    Args:
        threshold_speed: Speed threshold in steps/sec
        
    Returns:
        True if all joints below threshold, False otherwise
        
    Resource usage: ZERO overhead - simple request/response
    """
    speeds = get_robot_joint_speeds()
    if not speeds:
        return False
    
    max_speed = max(abs(s) for s in speeds)
    return max_speed < threshold_speed

def is_estop_pressed() -> bool:
    """
    Check if the E-stop is currently pressed.
    
    Returns:
        True if E-stop is pressed, False otherwise
        
    Resource usage: ZERO overhead - simple request/response
    """
    io_status = get_robot_io()
    if io_status and len(io_status) >= 5:
        return io_status[4] == 0  # E-stop is at index 4, 0 means pressed
    return False

def get_robot_status() -> Dict:
    """
    Get comprehensive robot status in one call.
    
    Returns:
        Dictionary with pose, angles, speeds, IO, gripper status
        
    Resource usage: Multiple requests but still zero overhead
    """
    return {
        'pose': get_robot_pose(),
        'angles': get_robot_joint_angles(),
        'speeds': get_robot_joint_speeds(),
        'io': get_robot_io(),
        'gripper': get_electric_gripper_status(),
        'stopped': is_robot_stopped(),
        'estop': is_estop_pressed()
    }

# ============================================================================
# TRACKING FUNCTIONS - ONLY FOR EXPLICIT USE
# ============================================================================

def check_command_status(command_id: str) -> Optional[Dict]:
    """
    Check status - returns None if tracker not initialized.
    Does NOT initialize tracker (read-only).
    """
    if _command_tracker and _command_tracker.is_active():
        return _command_tracker.get_status(command_id)
    return None

def is_tracking_active() -> bool:
    """
    Check if tracking is active.
    Returns False if never used (zero overhead check).
    """
    return _command_tracker is not None and _command_tracker.is_active()

def get_tracking_stats() -> Dict:
    """
    Get resource usage statistics.
    """
    if _command_tracker and _command_tracker.is_active():
        with _command_tracker.lock:
            return {
                'active': True,
                'commands_tracked': len(_command_tracker.command_history),
                'memory_bytes': len(str(_command_tracker.command_history)),
                'thread_active': _command_tracker._thread.is_alive() if _command_tracker._thread else False
            }
    else:
        return {
            'active': False,
            'commands_tracked': 0,
            'memory_bytes': 0,
            'thread_active': False
        }

# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON OPERATIONS
# ============================================================================

def wait_for_robot_stopped(timeout: float = 10.0, poll_rate: float = 0.1) -> bool:
    """
    Wait for the robot to stop moving.
    
    Args:
        timeout: Maximum time to wait in seconds
        poll_rate: How often to check in seconds
        
    Returns:
        True if robot stopped, False if timeout
    """
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if is_robot_stopped():
            return True
        time.sleep(poll_rate)
    
    return False

def safe_move_with_retry(
    move_func,
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs
):
    """
    Execute a move command with automatic retry on failure.
    
    Args:
        move_func: The movement function to call
        *args: Arguments for the movement function
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        **kwargs: Keyword arguments for the movement function
        
    Returns:
        Result from the movement function or error dict
    """
    import time
    
    # Ensure tracking is enabled for retry logic
    kwargs['wait_for_ack'] = True
    
    for attempt in range(max_retries):
        result = move_func(*args, **kwargs)
        
        if isinstance(result, dict):
            if result.get('status') in ['COMPLETED', 'QUEUED', 'EXECUTING']:
                return result
            elif result.get('status') in ['FAILED', 'TIMEOUT', 'CANCELLED']:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {result.get('details', 'Unknown error')}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"All {max_retries} attempts failed")
                    return result
        else:
            # Non-tracked response, assume success
            return result
    
    return {'status': 'FAILED', 'details': f'Failed after {max_retries} attempts'}

# =============================================================================
# GCODE FUNCTIONALITY
# =============================================================================

def execute_gcode(gcode_line: str, wait_for_ack: bool = False, timeout: float = 5.0):
    """
    Execute a single GCODE line.
    
    Args:
        gcode_line: The GCODE command to execute (e.g., "G0 X100 Y100 Z50")
        wait_for_ack: If True, wait for command acknowledgment
        timeout: Maximum time to wait for acknowledgment in seconds
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    
    Examples:
        # Simple rapid move
        execute_gcode("G0 X100 Y100 Z50")
        
        # Linear move with feed rate
        execute_gcode("G1 X150 Y150 Z30 F1000")
        
        # Set work coordinate system
        execute_gcode("G54")
        
        # Dwell for 2 seconds
        execute_gcode("G4 P2000")
    """
    command = f"GCODE|{gcode_line}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout=timeout)
    else:
        return send_robot_command(command)

def execute_gcode_program(program_lines: list, wait_for_ack: bool = False, timeout: float = 30.0):
    """
    Execute a GCODE program from a list of lines.
    
    Args:
        program_lines: List of GCODE lines to execute
        wait_for_ack: If True, wait for program to be loaded
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    
    Example:
        program = [
            "G21",           # Set units to mm
            "G90",           # Absolute positioning
            "G0 Z50",        # Raise Z
            "G0 X0 Y0",      # Go to origin
            "G1 Z10 F500",   # Lower Z slowly
            "G1 X100 F1000", # Move X
            "G1 Y100",       # Move Y
            "G1 X0",         # Move back X
            "G1 Y0",         # Move back Y
            "G0 Z50",        # Raise Z
            "M30"            # Program end
        ]
        execute_gcode_program(program)
    """
    # Validate program lines don't contain problematic characters
    for i, line in enumerate(program_lines):
        if '|' in line:
            error_msg = f"Line {i+1} contains pipe character '|' which is not allowed"
            if wait_for_ack:
                return {'status': 'INVALID', 'details': error_msg}
            else:
                print(f"Warning: {error_msg}")
                # Remove pipe characters as fallback
                program_lines[i] = line.replace('|', '')
    
    # Join lines with semicolons for inline program
    program_str = ';'.join(program_lines)
    command = f"GCODE_PROGRAM|INLINE|{program_str}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout=timeout)
    else:
        return send_robot_command(command)

def load_gcode_file(filepath: str, wait_for_ack: bool = False, timeout: float = 10.0):
    """
    Load and execute a GCODE program from a file.
    
    Args:
        filepath: Path to the GCODE file
        wait_for_ack: If True, wait for file to be loaded
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    
    Example:
        load_gcode_file("path/to/program.gcode")
    """
    command = f"GCODE_PROGRAM|FILE|{filepath}"
    
    if wait_for_ack:
        return send_and_wait(command, timeout=timeout)
    else:
        return send_robot_command(command)

def get_gcode_status():
    """
    Get the current status of the GCODE interpreter.
    
    Returns:
        Dict containing GCODE interpreter status including:
        - state: Current modal state (G90/G91, G20/G21, etc.)
        - work_coordinate: Active work coordinate system (G54-G59)
        - position: Current position in work coordinates
        - program_running: Whether a program is executing
        - program_line: Current line number being executed
        - errors: Any error messages
    
    Example:
        status = get_gcode_status()
        print(f"Current work coordinate: {status['work_coordinate']}")
        print(f"Program running: {status['program_running']}")
    """
    # Use the same pattern as other GET functions
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
            client_socket.settimeout(2.0)
            
            request_message = "GET_GCODE_STATUS"
            client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
            
            data, _ = client_socket.recvfrom(2048)
            response = data.decode('utf-8')
            
            if response and response.startswith("GCODE_STATUS|"):
                status_json = response.split('|', 1)[1]
                try:
                    return json.loads(status_json)
                except json.JSONDecodeError:
                    print(f"Error parsing GCODE status: {status_json}")
                    return None
            
            return None
            
    except socket.timeout:
        print("Timeout waiting for GCODE status response")
        return None
    except Exception as e:
        print(f"Error getting GCODE status: {e}")
        return None

def pause_gcode_program(wait_for_ack: bool = False, timeout: float = 5.0):
    """
    Pause the currently running GCODE program.
    
    Args:
        wait_for_ack: If True, wait for pause confirmation
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    """
    command = "GCODE_PAUSE"
    
    if wait_for_ack:
        return send_and_wait(command, timeout=timeout)
    else:
        return send_robot_command(command)

def resume_gcode_program(wait_for_ack: bool = False, timeout: float = 5.0):
    """
    Resume a paused GCODE program.
    
    Args:
        wait_for_ack: If True, wait for resume confirmation
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    """
    command = "GCODE_RESUME"
    
    if wait_for_ack:
        return send_and_wait(command, timeout=timeout)
    else:
        return send_robot_command(command)

def stop_gcode_program(wait_for_ack: bool = False, timeout: float = 5.0):
    """
    Stop the currently running GCODE program.
    
    Args:
        wait_for_ack: If True, wait for stop confirmation
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    """
    command = "GCODE_STOP"
    
    if wait_for_ack:
        return send_and_wait(command, timeout=timeout)
    else:
        return send_robot_command(command)

def set_work_coordinate_offset(coordinate_system: str, x: float = None, y: float = None, 
                              z: float = None, wait_for_ack: bool = False, timeout: float = 5.0):
    """
    Set work coordinate system offsets (G54-G59).
    
    Args:
        coordinate_system: Work coordinate system to set ('G54' through 'G59')
        x: X axis offset in mm (None to keep current)
        y: Y axis offset in mm (None to keep current)
        z: Z axis offset in mm (None to keep current)
        wait_for_ack: If True, wait for confirmation
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    
    Example:
        # Set G54 origin to current position
        set_work_coordinate_offset('G54', x=0, y=0, z=0)
        
        # Offset G55 by 100mm in X
        set_work_coordinate_offset('G55', x=100)
    """
    # Validate coordinate system format
    valid_systems = ['G54', 'G55', 'G56', 'G57', 'G58', 'G59']
    if coordinate_system not in valid_systems:
        error_msg = f'Invalid coordinate system: {coordinate_system}. Must be one of {valid_systems}'
        if wait_for_ack:
            return {'status': 'INVALID', 'details': error_msg}
        else:
            print(error_msg)
            return False
    
    # Build GCODE command to set work offsets
    gcode_commands = []
    
    # Select the coordinate system
    gcode_commands.append(coordinate_system)
    
    # Set offsets using G10 L2 P[n] X[x] Y[y] Z[z]
    # P1=G54, P2=G55, etc.
    coord_num = int(coordinate_system[1:]) - 53  # G54=1, G55=2, etc.
    
    offset_params = []
    if x is not None:
        offset_params.append(f"X{x}")
    if y is not None:
        offset_params.append(f"Y{y}")
    if z is not None:
        offset_params.append(f"Z{z}")
    
    if offset_params:
        gcode_commands.append(f"G10 L2 P{coord_num} {' '.join(offset_params)}")
    
    # Execute the commands
    for cmd in gcode_commands:
        result = execute_gcode(cmd, wait_for_ack=wait_for_ack, timeout=timeout)
        if wait_for_ack and isinstance(result, dict) and result.get('status') != 'COMPLETED':
            return result
    
    return True if not wait_for_ack else {'status': 'COMPLETED', 'details': 'Work offsets set'}

def zero_work_coordinates(coordinate_system: str = 'G54', wait_for_ack: bool = False, timeout: float = 5.0):
    """
    Set the current position as zero in the specified work coordinate system.
    
    Args:
        coordinate_system: Work coordinate system to zero ('G54' through 'G59')
        wait_for_ack: If True, wait for confirmation
        timeout: Maximum time to wait for acknowledgment
    
    Returns:
        True if successful, or dict with status details if wait_for_ack is True
    
    Example:
        # Set current position as origin in G54
        zero_work_coordinates('G54')
    """
    # This sets the current position as 0,0,0 in the work coordinate system
    return set_work_coordinate_offset(coordinate_system, x=0, y=0, z=0, 
                                    wait_for_ack=wait_for_ack, timeout=timeout)
