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
import queue
import uuid
from collections import deque
from datetime import datetime, timedelta

# Global configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5001

# Global tracker - starts as None (no resources)
_command_tracker = None
_tracker_lock = threading.Lock()

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
    
    def __init__(self, listen_port=5002, history_size=100):
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
            
            # Create socket
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
    
    def track_command(self, command: str) -> Tuple[str, str]:
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
    
    # Handle cases where a command_id could not be generated
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
        return send_and_wait(command, timeout, non_blocking)
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
) -> str:
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
) -> str:
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
    start_pose: Optional[List[float]] = None,
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None,
    clockwise: bool = False,
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
        start_pose: Optional [x, y, z, rx, ry, rz] start pose (mm and degrees).
                   If None, starts from current position.
        duration: Time to complete the circle in seconds
        speed_percentage: Speed as percentage (1-100)
        clockwise: Direction of motion
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
    
    command = f"SMOOTH_CIRCLE|{center_str}|{radius}|{plane}|{frame}|{start_str}|{timing_str}|{clockwise_str}"
    
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
    
    command = f"SMOOTH_ARC_CENTER|{end_str}|{center_str}|{frame}|{start_str}|{timing_str}|{clockwise_str}"
    
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
    
    command = f"SMOOTH_ARC_PARAM|{end_str}|{radius}|{arc_angle}|{frame}|{start_str}|{timing_str}|{clockwise_str}"
    
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
    
    # Build command
    command_parts = [f"SMOOTH_SPLINE", str(num_waypoints), frame, start_str, timing_str]
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
    
    command = f"SMOOTH_HELIX|{center_str}|{radius}|{pitch}|{height}|{frame}|{start_str}|{timing_str}|{clockwise_str}"
    
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
        
        # Check for failures if tracking
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
            
            # Check for failures if tracking
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