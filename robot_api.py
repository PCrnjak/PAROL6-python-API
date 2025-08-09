import socket
from typing import List, Optional, Literal
import numpy as np
import time
from spatialmath import SE3

def send_robot_command(command_string: str):
    """
    Encodes and sends a command string to the robot controller via UDP.
    This is the core communication function used by all other tools.
    """
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 5001
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command_string.encode('utf-8'), (SERVER_IP, SERVER_PORT))
        success_message = f"Successfully sent command: '{command_string}'"
        print(success_message)
        return success_message
    except Exception as e:
        error_message = f"Error sending command: {e}"
        print(error_message)
        return error_message

def move_robot_joints(
    joint_angles: List[float],
    duration: Optional[float] = None,
    speed_percentage: Optional[int] = None
):
    """Moves the robot's joints to a specific configuration."""
    if duration is None and speed_percentage is None:
        return "Error: You must provide either a duration or a speed_percentage."
    
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            return "Error: Duration must be a valid number."
        
    angles_str = "|".join(map(str, joint_angles))
    duration_str = str(duration) if duration is not None else "None"
    speed_str = str(speed_percentage) if speed_percentage is not None else "None"
    command = f"MOVEJOINT|{angles_str}|{duration_str}|{speed_str}"
    return send_robot_command(command)

def move_robot_pose(
    pose: List[float],
    duration: Optional[float] = None,
    speed_percentage: Optional[int] = None
):
    """Moves the robot's end-effector to a specific Cartesian pose."""
    if duration is None and speed_percentage is None:
        return "Error: You must provide either a duration or a speed_percentage."
    
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            return "Error: Duration must be a valid number."
        
    pose_str = "|".join(map(str, pose))
    duration_str = str(duration) if duration is not None else "None"
    speed_str = str(speed_percentage) if speed_percentage is not None else "None"
    command = f"MOVEPOSE|{pose_str}|{duration_str}|{speed_str}"
    return send_robot_command(command)

def jog_robot_joint(
    joint_index: int,
    speed_percentage: int,
    duration: Optional[float] = None,
    distance_deg: Optional[float] = None
):
    """Jogs a single robot joint for a specified time or distance."""
    if duration is None and distance_deg is None:
        return "Error: You must provide either a duration or a distance_deg."
    
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            return "Error: Duration must be a valid number."
        
    duration_str = str(duration) if duration is not None else "None"
    distance_str = str(distance_deg) if distance_deg is not None else "None"
    command = f"JOG|{joint_index}|{speed_percentage}|{duration_str}|{distance_str}"
    return send_robot_command(command)

def jog_cartesian(
    frame: Literal['TRF', 'WRF'],
    axis: Literal['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-', 'RX+', 'RX-', 'RY+', 'RY-', 'RZ+', 'RZ-'],
    speed_percentage: int,
    duration: float
):
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            return "Error: Duration must be a valid number."
        
    """Jogs the robot's end-effector continuously in Cartesian space."""
    command = f"CARTJOG|{frame}|{axis}|{speed_percentage}|{duration}"
    return send_robot_command(command)

def move_robot_cartesian(
    pose: List[float],
    duration: Optional[float] = None,
    speed_percentage: Optional[float] = None
) -> str:
    """
    Moves the robot's end-effector to a specific Cartesian pose in a straight line.

    You must provide **either** `duration` or `speed_percentage` to define the movement's speed, but not both.

    Args:
        pose (List[float]): The target pose as [x, y, z, r, p, y]. 
                            Positions are in mm, rotations are in degrees.
        duration (Optional[float]): The total time for the movement in seconds.
        speed_percentage (Optional[float]): The movement speed as a percentage (1-100).

    Returns:
        str: A confirmation or error message.
    """
    # --- 1. Validate that one, and only one, timing argument is provided ---
    if (duration is None and speed_percentage is None):
        return "Error: You must provide either 'duration' or 'speed_percentage'."
    
    if (duration is not None and speed_percentage is not None):
        return "Error: Please provide either 'duration' or 'speed_percentage', not both."

    # --- 2. Prepare command arguments, validating the provided values ---
    duration_arg = 'NONE'
    speed_arg = 'NONE'

    if duration is not None:
        try:
            if float(duration) <= 0:
                return "Error: Duration must be a positive number."
            duration_arg = str(duration)
        except (ValueError, TypeError):
            return "Error: Duration must be a valid number."
    
    if speed_percentage is not None:
        try:
            speed_val = float(speed_percentage)
            if not (0 < speed_val <= 100):
                return "Error: Speed percentage must be between 1 and 100."
            speed_arg = str(speed_val)
        except (ValueError, TypeError):
            return "Error: Speed percentage must be a valid number."

    # --- 3. Construct the final command string ---
    pose_str = "|".join(map(str, pose))
    command = f"MOVECART|{pose_str}|{duration_arg}|{speed_arg}"
    
    return send_robot_command(command)

def delay_robot(duration: float):
    """Pauses the robot's command queue execution for a specified time."""
    if duration is not None:
        try:
            duration = float(duration)
        except (ValueError, TypeError):
            return "Error: Duration must be a valid number."
        
    command = f"DELAY|{duration}"
    return send_robot_command(command)

def control_pneumatic_gripper(action: Literal['open', 'close'], port: Literal[1, 2]):
    """Controls the pneumatic gripper."""
    command = f"PNEUMATICGRIPPER|{action}|{port}"
    return send_robot_command(command)

def control_electric_gripper(
    action: Literal['move', 'calibrate'],
    position: Optional[int] = 255,
    speed: Optional[int] = 150,
    current: Optional[int] = 500
):
    """Controls the electric gripper."""
    action_str = "None" if action == 'move' else 'calibrate'
    command = f"ELECTRICGRIPPER|{action_str}|{position}|{speed}|{current}"
    return send_robot_command(command)

def home_robot():
    """Initiates the robot's homing sequence."""
    command = "HOME"
    return send_robot_command(command)

def stop_robot_movement():
    """Immediately stops all robot motion and clears the command queue."""
    command = "STOP"
    return send_robot_command(command)

def get_robot_pose():
    """
    Requests the robot's current end-effector pose via UDP.
    Sends a 'GET_POSE' request and waits for a response.
    Returns a list [x, y, z, roll, pitch, yaw] or None if it fails.
    (x,y,z in mm; r,p,y in degrees)
    """
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 5001
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        client_socket.settimeout(2.0)
        
        request_message = "GET_POSE"
        client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
        
        try:
            data, _ = client_socket.recvfrom(2048)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'POSE' and len(parts) == 2:
                pose_values = [float(v) for v in parts[1].split(',')]
                if len(pose_values) == 16:
                    # We have a valid 4x4 matrix
                    pose_matrix = np.array(pose_values).reshape((4, 4))
                    
                    # Convert matrix to the desired [x,y,z,r,p,y] format
                    T = SE3(pose_matrix, check=False)
                    xyz_mm = T.t * 1000  # Convert from meters to millimeters
                    rpy_deg = T.rpy(unit='deg', order='xyz') # Get roll, pitch, yaw in degrees

                    pose_list = list(xyz_mm) + list(rpy_deg)
                    
                    print(f"Successfully received and parsed robot pose: {pose_list}")
                    return pose_list
                else:
                    print(f"Error: Received pose data has incorrect length: {len(pose_values)}")
                    return None
            else:
                print(f"Error: Received malformed pose data: {response_str}")
                return None

        except socket.timeout:
            print("Error: Timeout waiting for pose response from robot controller.")
            return None
        except Exception as e:
            print(f"An error occurred while getting robot pose: {e}")
            return None

def get_robot_joint_angles():
    """
    Requests the robot's current joint angles in degrees.
    Returns a list of 6 angle values or None if it fails.
    """
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = 5001
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        client_socket.settimeout(2.0)
        
        request_message = "GET_ANGLES"
        client_socket.sendto(request_message.encode('utf-8'), (SERVER_IP, SERVER_PORT))
        
        try:
            data, _ = client_socket.recvfrom(1024)
            response_str = data.decode('utf-8')
            
            parts = response_str.split('|')
            if parts[0] == 'ANGLES' and len(parts) == 2:
                angle_values = [float(v) for v in parts[1].split(',')]
                print(f"Successfully received joint angles: {angle_values}")
                return angle_values
            else:
                print(f"Error: Received malformed angle data: {response_str}")
                return None

        except socket.timeout:
            print("Error: Timeout waiting for angle response from robot controller.")
            return None
        except Exception as e:
            print(f"An error occurred while getting robot angles: {e}")
            return None