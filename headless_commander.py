'''
A full fledged "API" for the PAROL6 robot. To use this, you should pair it with the "robot_api.py" where you can import commands
from said file and use them anywhere within your code. This Python script will handle sending and performing all the commands
to the PAROL6 robot, as well as E-Stop functionality and safety limitations.

To run this program, you must use the "experimental-kinematics" branch of the "PAROL-commander-software" on GitHub
which can be found through this link: https://github.com/PCrnjak/PAROL-commander-software/tree/experimental_kinematics.
You must also save these files into the following folder: "Project Files\PAROL-commander-software\GUI\files".
'''

# * If you press estop robot will stop and you need to enable it by pressing e

from roboticstoolbox import DHRobot, RevoluteDH, ERobot, ELink, ETS, trapezoidal, quintic
import roboticstoolbox as rp
from math import pi, sin, cos
import numpy as np
from oclock import Timer, loop, interactiveloop
import time
import socket
from spatialmath import SE3
import select
import serial
import platform
import os
import re
import logging
import struct
try:
    import keyboard  # type: ignore
except Exception:
    keyboard = None
from typing import Optional, Tuple
from spatialmath.base import trinterp
from collections import namedtuple, deque
try:
    import PAROL6_ROBOT as PAROL6_ROBOT
except ModuleNotFoundError:
    import os, sys
    legacy = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'PAROL-commander-software', 'GUI', 'files')
    if os.path.exists(os.path.join(legacy, 'PAROL6_ROBOT.py')):
        sys.path.append(legacy)
        import PAROL6_ROBOT as PAROL6_ROBOT
    else:
        raise
from smooth_motion import CircularMotion, SplineMotion, MotionBlender

# Set interval
INTERVAL_S = 0.01
prev_time = 0

# Jogging speed scale to slightly increase jog caps while respecting absolute limits
JOG_SPEED_SCALE = 1.2

logging.basicConfig(level = logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
    datefmt='%H:%M:%S'
)
logging.disable(logging.DEBUG)

# =========================
# Runtime flags and globals
# =========================
enabled = True
soft_error = False
disabled_reason = ""

# Ensure serial globals exist on all platforms
ser: Optional[serial.Serial] = None
com_port_str: Optional[str] = None
# Non-blocking serial reconnect throttle
last_reconnect_attempt = 0.0


my_os = platform.system()
if my_os == "Windows":
    # Try to read the COM port from a file
    try:
        with open("com_port.txt", "r") as f:
            com_port_str = f.read().strip()
            ser = serial.Serial(port=com_port_str, baudrate=3000000, timeout=0)
            print(f"Connected to saved COM port: {com_port_str}")
    except (FileNotFoundError, serial.SerialException):
        # If the file doesn't exist or the port is invalid, ask the user
        while True:
            com_port = input("Enter the COM port (e.g., COM9): ")
            try:
                ser = serial.Serial(port=com_port, baudrate=3000000, timeout=0)
                print(f"Successfully connected to {com_port}")
                # Save the successful port to the file
                with open("com_port.txt", "w") as f:
                    f.write(com_port)
                break
            except serial.SerialException:
                print(f"Could not open port {com_port}. Please try again.")

# in big endian machines, first byte of binary representation of the multibyte data-type is stored first. 
int_to_3_bytes = struct.Struct('>I').pack # BIG endian order

# data for output string (data that is being sent to the robot)
#######################################################################################
#######################################################################################
start_bytes =  [0xff,0xff,0xff] 
start_bytes = bytes(start_bytes)

end_bytes =  [0x01,0x02] 
end_bytes = bytes(end_bytes)


# data for input string (Data that is being sent by the robot)
#######################################################################################
#######################################################################################
input_byte = 0 # Here save incoming bytes from serial

start_cond1_byte = bytes([0xff])
start_cond2_byte = bytes([0xff])
start_cond3_byte = bytes([0xff])

end_cond1_byte = bytes([0x01])
end_cond2_byte = bytes([0x02])

start_cond1 = 0 #Flag if start_cond1_byte is received
start_cond2 = 0 #Flag if start_cond2_byte is received
start_cond3 = 0 #Flag if start_cond3_byte is received

good_start = 0 #Flag if we got all 3 start condition bytes
data_len = 0 #Length of the data after -3 start condition bytes and length byte, so -4 bytes

data_buffer = [b'']*255 #Here save all data after data length byte
data_counter = 0 #Data counter for incoming bytes; compared to data length to see if we have correct length
#######################################################################################
#######################################################################################
prev_positions = [0,0,0,0,0,0]
prev_speed = [0,0,0,0,0,0]
robot_pose = [0,0,0,0,0,0] #np.array([0,0,0,0,0,0])
#######################################################################################
#######################################################################################

# --- Wrapper class to make integers mutable when passed to functions ---
class CommandValue:
    def __init__(self, value):
        self.value = value

#######################################################################################
#######################################################################################
Position_out = [1,11,111,1111,11111,10]
Speed_out = [2,21,22,23,24,25]
Command_out = CommandValue(255)
Affected_joint_out = [1,1,1,1,1,1,1,1]
InOut_out = [0,0,0,0,0,0,0,0]
Timeout_out = 0
#Positon,speed,current,command,mode,ID
Gripper_data_out = [1,1,1,1,0,0]
#######################################################################################
#######################################################################################
# Data sent from robot to PC
Position_in = [31,32,33,34,35,36]
Speed_in = [41,42,43,44,45,46]
Homed_in = [0,0,0,0,0,0,0,0]
InOut_in = [1,1,1,1,1,1,1,1]
Temperature_error_in = [1,1,1,1,1,1,1,1]
Position_error_in = [1,1,1,1,1,1,1,1]
Timeout_error = 0
# how much time passed between 2 sent commands (2byte value, last 2 digits are decimal so max value is 655.35ms?)
Timing_data_in = [0]
XTR_data =   0

# --- State variables for program execution ---
Robot_mode = "Dummy"  # Start in an idle state
Program_step = 0      # Which line of the program to run
Command_step = 0      # The current step within a single command
Command_len = 0       # The total steps for the current command
ik_error = 0          # Flag for inverse kinematics errors
error_state = 0       # General error flag
program_running = False # A flag to start and stop the program

# This will be your "program"
command_list = []

#ID,Position,speed,current,status,obj_detection
Gripper_data_in = [1,1,1,1,1,1] 

# Global variable to track previous tolerance for logging changes
_prev_tolerance = None

def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range to handle angle wrapping"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def unwrap_angles(q_solution, q_current):
    """
    Unwrap angles in the solution to be closest to current position.
    This handles the angle wrapping issue where -179° and 181° are close but appear far.
    """
    q_unwrapped = q_solution.copy()
    
    for i in range(len(q_solution)):
        # Calculate the difference
        diff = q_solution[i] - q_current[i]
        
        # If the difference is more than pi, we need to unwrap
        if diff > np.pi:
            # Solution is too far in positive direction, subtract 2*pi
            q_unwrapped[i] = q_solution[i] - 2 * np.pi
        elif diff < -np.pi:
            # Solution is too far in negative direction, add 2*pi
            q_unwrapped[i] = q_solution[i] + 2 * np.pi
    
    return q_unwrapped

IKResult = namedtuple('IKResult', 'success q iterations residual tolerance_used violations')

def calculate_adaptive_tolerance(robot, q, strict_tol=1e-10, loose_tol=1e-7):
    """
    Calculate adaptive tolerance based on proximity to singularities.
    Near singularities: looser tolerance for easier convergence.
    Away from singularities: stricter tolerance for precise solutions.
    
    Parameters
    ----------
    robot : DHRobot
        Robot model
    q : array_like
        Joint configuration
    strict_tol : float
        Strict tolerance away from singularities (default: 1e-10)
    loose_tol : float
        Loose tolerance near singularities (1e-7)
        
    Returns
    -------
    float
        Adaptive tolerance value
    """
    global _prev_tolerance
    
    q_array = np.array(q, dtype=float)
    
    # Calculate manipulability measure (closer to 0 = closer to singularity)
    manip = robot.manipulability(q_array)
    singularity_threshold = 0.001
    
    sing_normalized = np.clip(manip / singularity_threshold, 0.0, 1.0)
    adaptive_tol = loose_tol + (strict_tol - loose_tol) * sing_normalized
    
    # Log tolerance changes (only log significant changes to avoid spam)
    if _prev_tolerance is None or abs(adaptive_tol - _prev_tolerance) / _prev_tolerance > 0.5:
        tol_category = "LOOSE" if adaptive_tol > 1e-7 else "MODERATE" if adaptive_tol > 5e-10 else "STRICT"
        print(f"Adaptive IK tolerance: {adaptive_tol:.2e} ({tol_category}) - Manipulability: {manip:.8f} (threshold: {singularity_threshold})")
        _prev_tolerance = adaptive_tol
    
    return adaptive_tol

def calculate_configuration_dependent_max_reach(q_seed):
    """
    Calculate maximum reach based on joint configuration, particularly joint 5.
    When joint 5 is at 90 degrees, the effective reach is reduced by approximately 0.045.
    
    Parameters
    ----------
    q_seed : array_like
        Current joint configuration in radians
        
    Returns
    -------
    float
        Configuration-dependent maximum reach threshold
    """
    base_max_reach = 0.44  # Base maximum reach from experimentation
    
    j5_angle = q_seed[4] if len(q_seed) > 4 else 0.0
    j5_normalized = normalize_angle(j5_angle)
    angle_90_deg = np.pi / 2
    angle_neg_90_deg = -np.pi / 2
    dist_from_90 = abs(j5_normalized - angle_90_deg)
    dist_from_neg_90 = abs(j5_normalized - angle_neg_90_deg)
    dist_from_90_deg = min(dist_from_90, dist_from_neg_90)
    reduction_range = np.pi / 4  # 45 degrees
    if dist_from_90_deg <= reduction_range:
        # Calculate reduction factor based on proximity to 90°
        proximity_factor = 1.0 - (dist_from_90_deg / reduction_range)
        reach_reduction = 0.045 * proximity_factor
        effective_max_reach = base_max_reach - reach_reduction
        
        return effective_max_reach
    else:
        return base_max_reach

def solve_ik_with_adaptive_tol_subdivision(
        robot: DHRobot,
        target_pose: SE3,
        current_q,
        current_pose: SE3 | None = None,
        max_depth: int = 4,
        ilimit: int = 100,
        jogging: bool = False
):
    """
    Uses adaptive tolerance based on proximity to singularities:
    - Near singularities: looser tolerance for easier convergence
    - Away from singularities: stricter tolerance for precise solutions
    If necessary, recursively subdivide the motion until ikine_LMS converges
    on every segment. Finally check that solution respects joint limits. From experimentation,
    jogging with lower tolerances often produces q_paths that do not differ from current_q,
    essentially freezing the robot.

    Parameters
    ----------
    robot : DHRobot
        Robot model
    target_pose : SE3
        Target pose to reach
    current_q : array_like
        Current joint configuration
    current_pose : SE3, optional
        Current pose (computed if None)
    max_depth : int, optional
        Maximum subdivision depth (default: 8)
    ilimit : int, optional
        Maximum iterations for IK solver (default: 100)

    Returns
    -------
    IKResult
        success  - True/False
        q_path   - (mxn) array of the final joint configuration 
        iterations, residual  - aggregated diagnostics
        tolerance_used - which tolerance was used
        violations - joint limit violations (if any)
    """
    if current_pose is None:
        current_pose = robot.fkine(current_q)

    # ── inner recursive solver───────────────────
    def _solve(Ta: SE3, Tb: SE3, q_seed, depth, tol):
        """Return (path_list, success_flag, iterations, residual)."""
        # Calculate current and target reach
        current_reach = np.linalg.norm(Ta.t)
        target_reach = np.linalg.norm(Tb.t)
        
        # Check if this is an inward movement (recovery)
        is_recovery = target_reach < current_reach
        
        # Calculate configuration-dependent maximum reach based on joint 5 position
        max_reach_threshold = calculate_configuration_dependent_max_reach(q_seed)
        
        # Determine damping based on reach and movement direction
        if is_recovery:
            # Recovery mode - always use low damping
            damping = 0.0000001
        else:
            # Check if we're near configuration-dependent max reach
            # print(f"current_reach:{current_reach:.3f}, max_reach_threshold:{max_reach_threshold:.3f}")
            if not is_recovery and target_reach > max_reach_threshold:
                print(f"Target reach limit exceeded: {target_reach:.3f} > {max_reach_threshold:.3f}")
                return [], False, depth, 0
            else:
                damping = 0.0000001  # Normal low damping
        
        res = robot.ikine_LMS(Tb, q0=q_seed, ilimit=ilimit, tol=tol, wN=damping)
        if res.success:
            q_good = unwrap_angles(res.q, q_seed)      # << unwrap vs *previous*
            return [q_good], True, res.iterations, res.residual

        if depth >= max_depth:
            return [], False, res.iterations, res.residual
        # split the segment and recurse
        Tc = SE3(trinterp(Ta.A, Tb.A, 0.5))            # mid-pose (screw interp)

        left_path,  ok_L, it_L, r_L = _solve(Ta, Tc, q_seed, depth+1, tol)
        if not ok_L:
            return [], False, it_L, r_L

        q_mid = left_path[-1]                          # last solved joint set
        right_path, ok_R, it_R, r_R = _solve(Tc, Tb, q_mid, depth+1, tol)

        return (
            left_path + right_path,
            ok_R,
            it_L + it_R,
            r_R
        )

    # ── kick-off with adaptive tolerance ──────────────────────────────────
    if jogging:
        adaptive_tol = 1e-10
    else:
        adaptive_tol = calculate_adaptive_tolerance(robot, current_q)
    path, ok, its, resid = _solve(current_pose, target_pose, current_q, 0, adaptive_tol)
    # Check if solution respects joint limits
    target_q = path[-1] if len(path) != 0 else None
    solution_valid, violations = PAROL6_ROBOT.check_joint_limits(current_q, target_q)
    if ok and solution_valid:
        return IKResult(True, path[-1], its, resid, adaptive_tol, violations)
    else:
        return IKResult(False, None, its, resid, adaptive_tol, violations)

#Setup IP address and Simulator port
ip = "127.0.0.1" #Loopback address
port = 5001
# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))
print(f'Start listening to {ip}:{port}')

def Unpack_data(data_buffer_list, Position_in,Speed_in,Homed_in,InOut_in,Temperature_error_in,Position_error_in,Timeout_error,Timing_data_in,
         XTR_data,Gripper_data_in):

    Joints = []
    Speed = []

    for i in range(0,18, 3):
        variable = data_buffer_list[i:i+3]
        Joints.append(variable)

    for i in range(18,36, 3):
        variable = data_buffer_list[i:i+3]
        Speed.append(variable)


    for i in range(6):
        var =  b'\x00' + b''.join(Joints[i])
        Position_in[i] = Fuse_3_bytes(var)
        var =  b'\x00' + b''.join(Speed[i])
        Speed_in[i] = Fuse_3_bytes(var)

    Homed = data_buffer_list[36]
    IO_var = data_buffer_list[37]
    temp_error = data_buffer_list[38]
    position_error = data_buffer_list[39]
    timing_data = data_buffer_list[40:42]
    Timeout_error_var = data_buffer_list[42]
    xtr2 = data_buffer_list[43]
    device_ID = data_buffer_list[44]
    Gripper_position = data_buffer_list[45:47]
    Gripper_speed = data_buffer_list[47:49]
    Gripper_current = data_buffer_list[49:51]
    Status = data_buffer_list[51]
    # The original object_detection byte at index 52 is ignored as it is not reliable.
    CRC_byte = data_buffer_list[53]
    endy_byte1 = data_buffer_list[54]
    endy_byte2 = data_buffer_list[55]

    # ... (Code for Homed, IO_var, temp_error, etc. remains the same) ...

    temp = Split_2_bitfield(int.from_bytes(Homed,"big"))
    for i in range(8):
        Homed_in[i] = temp[i]

    temp = Split_2_bitfield(int.from_bytes(IO_var,"big"))
    for i in range(8):
        InOut_in[i] = temp[i]

    temp = Split_2_bitfield(int.from_bytes(temp_error,"big"))
    for i in range(8):
        Temperature_error_in[i] = temp[i]

    temp = Split_2_bitfield(int.from_bytes(position_error,"big"))
    for i in range(8):
        Position_error_in[i] = temp[i]

    var = b'\x00' + b'\x00' + b''.join(timing_data)
    Timing_data_in[0] = Fuse_3_bytes(var)
    Timeout_error = int.from_bytes(Timeout_error_var,"big")
    XTR_data = int.from_bytes(xtr2,"big")

    # --- Gripper Data Unpacking ---
    Gripper_data_in[0] = int.from_bytes(device_ID,"big")

    var =  b'\x00'+ b'\x00' + b''.join(Gripper_position)
    Gripper_data_in[1] = Fuse_2_bytes(var)

    var =  b'\x00'+ b'\x00' + b''.join(Gripper_speed)
    Gripper_data_in[2] = Fuse_2_bytes(var)

    var =  b'\x00'+ b'\x00' + b''.join(Gripper_current)
    Gripper_data_in[3] = Fuse_2_bytes(var)

    # --- Start of Corrected Logic ---
    # This section now mirrors the working logic from GUI_PAROL_latest.py
    
    # 1. Store the raw status byte (from index 51)
    status_byte = int.from_bytes(Status,"big")
    Gripper_data_in[4] = status_byte

    # 2. Split the status byte into a list of 8 individual bits
    status_bits = Split_2_bitfield(status_byte)
    
    # 3. Combine the 3rd and 4th bits (at indices 2 and 3) to get the true object detection status
    # This creates a 2-bit number (0-3) which represents the full state.
    object_detection_status = (status_bits[2] << 1) | status_bits[3]
    Gripper_data_in[5] = object_detection_status
    # --- End of Corrected Logic ---

def Pack_data(Position_out,Speed_out,Command_out,Affected_joint_out,InOut_out,Timeout_out,Gripper_data_out):

    # Len is defined by all bytes EXCEPT start bytes and len
    # Start bytes = 3
    len = 52 #1
    Position = [Position_out[0],Position_out[1],Position_out[2],Position_out[3],Position_out[4],Position_out[5]]  #18
    Speed = [Speed_out[0], Speed_out[1], Speed_out[2], Speed_out[3], Speed_out[4], Speed_out[5],] #18
    Command = Command_out#1
    Affected_joint = Affected_joint_out
    InOut = InOut_out #1
    Timeout = Timeout_out #1
    Gripper_data = Gripper_data_out #9
    CRC_byte = 228 #1
    # End bytes = 2


    test_list = []
    #print(test_list)

    #x = bytes(start_bytes)
    test_list.append((start_bytes))
    
    test_list.append(bytes([len]))


    # Position data
    for i in range(6):
        position_split = Split_2_3_bytes(Position[i])
        test_list.append(position_split[1:4])

    # Speed data
    for i in range(6):
        speed_split = Split_2_3_bytes(Speed[i])
        test_list.append(speed_split[1:4])

    # Command data
    test_list.append(bytes([Command]))

    # Affected joint data
    Affected_list = Fuse_bitfield_2_bytearray(Affected_joint[:])
    test_list.append(Affected_list)

    # Inputs outputs data
    InOut_list = Fuse_bitfield_2_bytearray(InOut[:])
    test_list.append(InOut_list)

    # Timeout data
    test_list.append(bytes([Timeout]))

    # Gripper position
    Gripper_position = Split_2_3_bytes(Gripper_data[0])
    test_list.append(Gripper_position[2:4])

    # Gripper speed
    Gripper_speed = Split_2_3_bytes(Gripper_data[1])
    test_list.append(Gripper_speed[2:4])

    # Gripper current
    Gripper_current = Split_2_3_bytes(Gripper_data[2])
    test_list.append(Gripper_current[2:4])  

    # Gripper command
    test_list.append(bytes([Gripper_data[3]]))
    # Gripper mode
    test_list.append(bytes([Gripper_data[4]]))
    
    # ==========================================================
    # === FIX: Make sure calibrate is a one-shot command      ====
    # ==========================================================
    # If the mode was set to calibrate (1) or clear_error (2), reset it
    # back to normal (0) for the next cycle. This prevents an endless loop.
    if Gripper_data_out[4] == 1 or Gripper_data_out[4] == 2:
        Gripper_data_out[4] = 0
    # ==========================================================
    
    # Gripper ID
    test_list.append(bytes([Gripper_data[5]]))
 
    # CRC byte
    test_list.append(bytes([CRC_byte]))

    # END bytes
    test_list.append((end_bytes))
    
    #print(test_list)
    return test_list

def Get_data(Position_in,Speed_in,Homed_in,InOut_in,Temperature_error_in,Position_error_in,Timeout_error,Timing_data_in,
         XTR_data,Gripper_data_in):
    global input_byte 
    global ser

    global start_cond1_byte 
    global start_cond2_byte 
    global start_cond3_byte 

    global end_cond1_byte 
    global end_cond2_byte 

    global start_cond1 
    global start_cond2 
    global start_cond3 

    global good_start 
    global data_len 

    global data_buffer 
    global data_counter

    # Ensure serial is available before reading
    if ser is None or not ser.is_open:
        return

    while ser.in_waiting > 0:
        input_byte = ser.read(1)

        #UNCOMMENT THIS TO GET ALL DATA FROM THE ROBOT PRINTED
        #print(input_byte) 

        # When data len is received start is good and after that put all data in receive buffer
        # Data len is ALL data after it; that includes input buffer, end bytes and CRC
        if (good_start != 1):
            # All start bytes are good and next byte is data len
            if (start_cond1 == 1 and start_cond2 == 1 and start_cond3 == 1):
                good_start = 1
                data_len = input_byte
                data_len = struct.unpack('B', data_len)[0]
                logging.debug("data len we got from robot packet= ")
                logging.debug(input_byte)
                logging.debug("good start for DATA that we received at PC")
            # Third start byte is good
            if (input_byte == start_cond3_byte and start_cond2 == 1 and start_cond1 == 1):
                start_cond3 = 1
                #print("good cond 3 PC")
            #Third start byte is bad, reset all flags
            elif (start_cond2 == 1 and start_cond1 == 1):
                #print("bad cond 3 PC")
                start_cond1 = 0
                start_cond2 = 0
            # Second start byte is good
            if (input_byte == start_cond2_byte and start_cond1 == 1):
                start_cond2 = 1
                #print("good cond 2 PC ")
            #Second start byte is bad, reset all flags   
            elif (start_cond1 == 1):
                #print("Bad cond 2 PC")
                start_cond1 = 0
            # First start byte is good
            if (input_byte == start_cond1_byte):
                start_cond1 = 1
                #print("good cond 1 PC")
        else:
            # Here data goes after good  start
            data_buffer[data_counter] = input_byte
            if (data_counter == data_len - 1):

                logging.debug("Data len PC")
                logging.debug(data_len)
                logging.debug("End bytes are:")
                logging.debug(data_buffer[data_len -1])
                logging.debug(data_buffer[data_len -2])

                # Here if last 2 bytes are end condition bytes we process the data 
                if (data_buffer[data_len -1] == end_cond2_byte and data_buffer[data_len - 2] == end_cond1_byte):

                    logging.debug("GOOD END CONDITION PC")
                    logging.debug("I UNPACKED RAW DATA RECEIVED FROM THE ROBOT")
                    Unpack_data(data_buffer, Position_in,Speed_in,Homed_in,InOut_in,Temperature_error_in,Position_error_in,Timeout_error,Timing_data_in,
                    XTR_data,Gripper_data_in)
                    logging.debug("DATA UNPACK FINISHED")
                    # ako su dobri izračunaj crc
                    # if crc dobar raspakiraj podatke
                    # ako je dobar paket je dobar i spremi ga u nove variable!
                
                # Print every byte
                #print("podaci u data bufferu su:")
                #for i in range(data_len):
                    #print(data_buffer[i])

                good_start = 0
                start_cond1 = 0
                start_cond3 = 0
                start_cond2 = 0
                data_len = 0
                data_counter = 0
            else:
                data_counter = data_counter + 1

# Split data to 3 bytes 
def Split_2_3_bytes(var_in):
    y = int_to_3_bytes(var_in & 0xFFFFFF) # converts my int value to bytes array
    return y

# Splits byte to bitfield list
def Split_2_bitfield(var_in):
    #return [var_in >> i & 1 for i in range(7,-1,-1)] 
    return [(var_in >> i) & 1 for i in range(7, -1, -1)]

# Fuses 3 bytes to 1 signed int
def Fuse_3_bytes(var_in):
    value = struct.unpack(">I", bytearray(var_in))[0] # converts bytes array to int

    # convert to negative number if it is negative
    if value >= 1<<23:
        value -= 1<<24

    return value

# Fuses 2 bytes to 1 signed int
def Fuse_2_bytes(var_in):
    value = struct.unpack(">I", bytearray(var_in))[0] # converts bytes array to int

    # convert to negative number if it is negative
    if value >= 1<<15:
        value -= 1<<16

    return value

# Fuse bitfield list to byte
def Fuse_bitfield_2_bytearray(var_in):
    number = 0
    for b in var_in:
        number = (2 * number) + b
    return bytes([number])

# Check if there is element 1 in the list. 
# If yes return its index, if no element is 1 return -1
def check_elements(lst):
    for i, element in enumerate(lst):
        if element == 1:
            return i
    return -1  # Return -1 if no element is 1

def quintic_scaling(s: float) -> float:
    """
    Calculates a smooth 0-to-1 scaling factor for progress 's'
    using a quintic polynomial, ensuring smooth start/end accelerations.
    """
    return 6 * (s**5) - 15 * (s**4) + 10 * (s**3)

#########################################################################
# Robot Commands Start Here
#########################################################################

class HomeCommand:
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """
    def __init__(self):
        self.is_valid = True
        self.is_finished = False
        # State machine: START -> WAIT_FOR_UNHOMED -> WAIT_FOR_HOMED -> FINISHED
        self.state = "START"
        # Counter to send the home command for multiple cycles
        self.start_cmd_counter = 10  # Send command 100 for 10 cycles (0.1s)
        # Safety timeout (20 seconds at 0.01s interval)
        self.timeout_counter = 2000
        print("Initializing Home command...")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """
        Manages the homing command and monitors for completion using a state machine.
        """
        if self.is_finished:
            return True

        # --- State: START ---
        # On the first few executions, continuously send the 'home' (100) command.
        if self.state == "START":
            print(f"  -> Sending home signal (100)... Countdown: {self.start_cmd_counter}")
            Command_out.value = 100
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                # Once sent for enough cycles, move to the next state
                self.state = "WAITING_FOR_UNHOMED"
            return False

        # --- State: WAITING_FOR_UNHOMED ---
        # The robot's firmware should reset the homed status. We wait to see that happen.
        # During this time, we send 'idle' (255) to let the robot's controller take over.
        if self.state == "WAITING_FOR_UNHOMED":
            Command_out.value = 255
            # Check if at least one joint has started homing (is no longer homed)
            if any(h == 0 for h in Homed_in[:6]):
                print("  -> Homing sequence initiated by robot.")
                self.state = "WAITING_FOR_HOMED"
            # Check for timeout
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                print("  -> ERROR: Timeout waiting for robot to start homing sequence.")
                self.is_finished = True
            return self.is_finished

        # --- State: WAITING_FOR_HOMED ---
        # Now we wait for all joints to report that they are homed (all flags are 1).
        if self.state == "WAITING_FOR_HOMED":
            Command_out.value = 255
            # Check if all joints have finished homing
            if all(h == 1 for h in Homed_in[:6]):
                print("Homing sequence complete. All joints reported home.")
                self.is_finished = True
                Speed_out[:] = [0] * 6 # Ensure robot is stopped

        return self.is_finished

class JogCommand:
    """
    A non-blocking command to jog a joint for a specific duration or distance.
    It performs all safety and validity checks upon initialization.
    """
    def __init__(self, joint, speed_percentage=None, duration=None, distance_deg=None):
        """
        Initializes and validates the jog command. This is the SETUP phase.
        """
        self.is_valid = False
        self.is_finished = False
        self.mode = None
        self.command_step = 0

        # --- 1. Parameter Validation and Mode Selection ---
        if duration and distance_deg:
            self.mode = 'distance'
            print(f"Initializing Jog: Joint {joint}, Distance {distance_deg} deg, Duration {duration}s.")
        elif duration:
            self.mode = 'time'
            print(f"Initializing Jog: Joint {joint}, Speed {speed_percentage}%, Duration {duration}s.")
        elif distance_deg:
            self.mode = 'distance'
            print(f"Initializing Jog: Joint {joint}, Speed {speed_percentage}%, Distance {distance_deg} deg.")
        else:
            print("Error: JogCommand requires either 'duration', 'distance_deg', or both.")
            return

        # --- 2. Store parameters for deferred calculation ---
        self.joint = joint
        self.speed_percentage = speed_percentage
        self.duration = duration
        self.distance_deg = distance_deg

        # --- These will be calculated later ---
        self.direction = 1
        self.joint_index = 0
        self.speed_out = 0
        self.command_len = 0
        self.target_position = 0

        self.is_valid = True # Mark as valid for now; preparation step will confirm.


    def prepare_for_execution(self, current_position_in):
        """Pre-computes speeds and target positions using live data."""
        print(f"  -> Preparing for Jog command...")

        # Determine direction and joint index
        self.direction = 1 if 0 <= self.joint <= 5 else -1
        self.joint_index = self.joint if self.direction == 1 else self.joint - 6
        
        distance_steps = 0
        if self.distance_deg:
            distance_steps = int(PAROL6_ROBOT.DEG2STEPS(abs(self.distance_deg), self.joint_index))
            # --- MOVED LOGIC: Calculate target using the LIVE position ---
            self.target_position = current_position_in[self.joint_index] + (distance_steps * self.direction)
            
            min_limit, max_limit = PAROL6_ROBOT.Joint_limits_steps[self.joint_index]
            if not (min_limit <= self.target_position <= max_limit):
                print(f"  -> VALIDATION FAILED: Target position {self.target_position} is out of joint limits ({min_limit}, {max_limit}).")
                self.is_valid = False
                return

        # Calculate speed and duration
        speed_steps_per_sec = 0
        if self.mode == 'distance' and self.duration:
            speed_steps_per_sec = int(distance_steps / self.duration) if self.duration > 0 else 0
            max_joint_speed = PAROL6_ROBOT.Joint_max_speed[self.joint_index]
            if speed_steps_per_sec > max_joint_speed:
                print(f"  -> VALIDATION FAILED: Required speed ({speed_steps_per_sec} steps/s) exceeds joint's max speed ({max_joint_speed} steps/s).")
                self.is_valid = False
                return
        else:
            if self.speed_percentage is None:
                print("Error: 'speed_percentage' must be provided if not calculating automatically.")
                self.is_valid = False
                return
            # Map jog speed to jog caps scaled up slightly, clamped to absolute joint max
            max_cap = int(PAROL6_ROBOT.Joint_max_jog_speed[self.joint_index] * JOG_SPEED_SCALE)
            abs_max = PAROL6_ROBOT.Joint_max_speed[self.joint_index]
            max_joint_speed = min(abs_max, max_cap)
            speed_steps_per_sec = int(np.interp(abs(self.speed_percentage), [0, 100], [0, max_joint_speed]))

        self.speed_out = speed_steps_per_sec * self.direction
        self.command_len = int(self.duration / INTERVAL_S) if self.duration else float('inf')
        print("  -> Jog command is ready.")


    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.is_finished or not self.is_valid:
            return True

        stop_reason = None
        current_pos = Position_in[self.joint_index]

        if self.mode == 'time':
            if self.command_step >= self.command_len:
                stop_reason = "Timed jog finished."
        elif self.mode == 'distance':
            if (self.direction == 1 and current_pos >= self.target_position) or \
               (self.direction == -1 and current_pos <= self.target_position):
                stop_reason = "Distance jog finished."
        
        if not stop_reason:
            if (self.direction == 1 and current_pos >= PAROL6_ROBOT.Joint_limits_steps[self.joint_index][1]) or \
               (self.direction == -1 and current_pos <= PAROL6_ROBOT.Joint_limits_steps[self.joint_index][0]):
                stop_reason = f"Limit reached on joint {self.joint_index + 1}."

        if stop_reason:
            print(stop_reason)
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        else:
            Speed_out[:] = [0] * 6
            Speed_out[self.joint_index] = self.speed_out
            Command_out.value = 123
            self.command_step += 1
            return False
        
class MultiJogCommand:
    """
    A non-blocking command to jog multiple joints simultaneously for a specific duration.
    It performs all safety and validity checks upon initialization.
    """
    def __init__(self, joints, speed_percentages, duration):
        """
        Initializes and validates the multi-jog command.
        """
        self.is_valid = False
        self.is_finished = False
        self.command_step = 0

        # --- 1. Parameter Validation ---
        if not isinstance(joints, list) or not isinstance(speed_percentages, list):
            print("Error: MultiJogCommand requires 'joints' and 'speed_percentages' to be lists.")
            return

        if len(joints) != len(speed_percentages):
            print("Error: The number of joints must match the number of speed percentages.")
            return

        if not duration or duration <= 0:
            print("Error: MultiJogCommand requires a positive 'duration'.")
            return

        # ==========================================================
        # === NEW: Check for conflicting joint commands          ===
        # ==========================================================
        base_joints = set()
        for joint in joints:
            # Normalize the joint index to its base (0-5)
            base_joint = joint % 6
            # If the base joint is already in our set, it's a conflict.
            if base_joint in base_joints:
                print(f"  -> VALIDATION FAILED: Conflicting commands for Joint {base_joint + 1} (e.g., J1+ and J1-).")
                self.is_valid = False
                return
            base_joints.add(base_joint)
        # ==========================================================

        print(f"Initializing MultiJog for joints {joints} with speeds {speed_percentages}% for {duration}s.")

        # --- 2. Store parameters ---
        self.joints = joints
        self.speed_percentages = speed_percentages
        self.duration = duration
        self.command_len = int(self.duration / INTERVAL_S)

        # --- This will be calculated in the prepare step ---
        self.speeds_out = [0] * 6

        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Pre-computes the speeds for each joint."""
        print(f"  -> Preparing for MultiJog command...")

        for i, joint in enumerate(self.joints):
            # Determine direction and joint index (0-5 for positive, 6-11 for negative)
            direction = 1 if 0 <= joint <= 5 else -1
            joint_index = joint if direction == 1 else joint - 6
            speed_percentage = self.speed_percentages[i]

            # Check for joint index validity
            if not (0 <= joint_index < 6):
                print(f"  -> VALIDATION FAILED: Invalid joint index {joint_index}.")
                self.is_valid = False
                return

            # Calculate speed in steps/sec using scaled jog caps, clamped to absolute joint max
            max_cap = int(PAROL6_ROBOT.Joint_max_jog_speed[joint_index] * JOG_SPEED_SCALE)
            abs_max = PAROL6_ROBOT.Joint_max_speed[joint_index]
            max_joint_speed = min(abs_max, max_cap)
            speed_steps_per_sec = int(np.interp(speed_percentage, [0, 100], [0, max_joint_speed]))
            self.speeds_out[joint_index] = speed_steps_per_sec * direction

        print("  -> MultiJog command is ready.")


    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.is_finished or not self.is_valid:
            return True

        # Stop if the duration has elapsed
        if self.command_step >= self.command_len:
            print("Timed multi-jog finished.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        else:
            # Continuously check for joint limits during the jog
            for i in range(6):
                if self.speeds_out[i] != 0:
                    current_pos = Position_in[i]
                    # Hitting positive limit while moving positively
                    if self.speeds_out[i] > 0 and current_pos >= PAROL6_ROBOT.Joint_limits_steps[i][1]:
                         print(f"Limit reached on joint {i + 1}. Stopping jog.")
                         self.is_finished = True
                         Speed_out[:] = [0] * 6
                         Command_out.value = 255
                         return True
                    # Hitting negative limit while moving negatively
                    elif self.speeds_out[i] < 0 and current_pos <= PAROL6_ROBOT.Joint_limits_steps[i][0]:
                         print(f"Limit reached on joint {i + 1}. Stopping jog.")
                         self.is_finished = True
                         Speed_out[:] = [0] * 6
                         Command_out.value = 255
                         return True

            # If no limits are hit, apply the speeds
            Speed_out[:] = self.speeds_out
            Command_out.value = 123 # Jog command
            self.command_step += 1
            return False # Command is still running
        
# This dictionary maps descriptive axis names to movement vectors, which is cleaner.
# Format: ([x, y, z], [rx, ry, rz])
AXIS_MAP = {
    'X+': ([1, 0, 0], [0, 0, 0]), 'X-': ([-1, 0, 0], [0, 0, 0]),
    'Y+': ([0, 1, 0], [0, 0, 0]), 'Y-': ([0, -1, 0], [0, 0, 0]),
    'Z+': ([0, 0, 1], [0, 0, 0]), 'Z-': ([0, 0, -1], [0, 0, 0]),
    'RX+': ([0, 0, 0], [1, 0, 0]), 'RX-': ([0, 0, 0], [-1, 0, 0]),
    'RY+': ([0, 0, 0], [0, 1, 0]), 'RY-': ([0, 0, 0], [0, -1, 0]),
    'RZ+': ([0, 0, 0], [0, 0, 1]), 'RZ-': ([0, 0, 0], [0, 0, -1]),
}

class CartesianJogCommand:
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    This is the final, refactored version using clean, standard spatial math
    operations now that the core unit bug has been fixed.
    """
    def __init__(self, frame, axis, speed_percentage=50.0, duration=1.5, **kwargs):
        """
        Initializes and validates the Cartesian jog command.
        """
        self.is_valid = False
        self.is_finished = False
        print(f"Initializing CartesianJog: Frame {frame}, Axis {axis}...")

        if axis not in AXIS_MAP:
            print(f"  -> VALIDATION FAILED: Invalid axis '{axis}'.")
            return
        
        # Store all necessary parameters for use in execute_step
        self.frame = frame
        self.axis_vectors = AXIS_MAP[axis]
        self.is_rotation = any(self.axis_vectors[1])
        self.speed_percentage = speed_percentage
        self.duration = duration
        self.end_time = time.time() + self.duration
        
        self.is_valid = True
        print("  -> Command is valid and ready.")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        if self.is_finished or not self.is_valid:
            return True

        # --- A. Check for completion ---
        if time.time() >= self.end_time:
            print("Cartesian jog finished.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True

        # --- B. Calculate Target Pose using clean vector math ---
        Command_out.value = 123 # Set jog command
        
        q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
        T_current = PAROL6_ROBOT.robot.fkine(q_current)

        if not isinstance(T_current, SE3):
            return False # Wait for valid pose data

        # Calculate speed and displacement for this cycle
        linear_speed_ms = float(np.interp(self.speed_percentage, [0, 100], [PAROL6_ROBOT.Cartesian_linear_velocity_min_JOG, PAROL6_ROBOT.Cartesian_linear_velocity_max_JOG]))
        angular_speed_degs = float(np.interp(self.speed_percentage, [0, 100], [PAROL6_ROBOT.Cartesian_angular_velocity_min, PAROL6_ROBOT.Cartesian_angular_velocity_max]))

        delta_linear = linear_speed_ms * INTERVAL_S
        delta_angular_rad = np.deg2rad(angular_speed_degs * INTERVAL_S)

        # Create the small incremental transformation (delta_pose)
        trans_vec = np.array(self.axis_vectors[0]) * delta_linear
        rot_vec = np.array(self.axis_vectors[1]) * delta_angular_rad
        delta_pose = SE3.Rt(SE3.Eul(rot_vec).R, trans_vec)

        # Apply the transformation in the correct reference frame
        if self.frame == 'WRF':
            # Pre-multiply to apply the change in the World Reference Frame
            target_pose = delta_pose * T_current
        else: # TRF
            # Post-multiply to apply the change in the Tool Reference Frame
            target_pose = T_current * delta_pose
        
        # --- C. Solve IK and Calculate Velocities ---
        var = solve_ik_with_adaptive_tol_subdivision(PAROL6_ROBOT.robot, target_pose, q_current, jogging=True)

        if var.success:
            q_velocities = (var.q - q_current) / INTERVAL_S
            for i in range(6):
                Speed_out[i] = int(PAROL6_ROBOT.SPEED_RAD2STEP(q_velocities[i], i))
        else:
            print("IK Warning: Could not find solution for jog step. Stopping.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True

        # --- D. Speed Scaling ---
        max_scale_factor = 1.0
        for i in range(6):
            if abs(Speed_out[i]) > PAROL6_ROBOT.Joint_max_speed[i]:
                scale = abs(Speed_out[i]) / PAROL6_ROBOT.Joint_max_speed[i]
                if scale > max_scale_factor:
                    max_scale_factor = scale
        
        if max_scale_factor > 1.0:
            for i in range(6):
                Speed_out[i] = int(Speed_out[i] / max_scale_factor)

        return False # Command is still running

class MovePoseCommand:
    """
    A non-blocking command to move the robot to a specific Cartesian pose.
    The movement itself is a joint-space interpolation.
    """
    def __init__(self, pose, duration=None, velocity_percent=None, accel_percent=50, trajectory_type='poly'):
        self.is_valid = True  # Assume valid; preparation step will confirm.
        self.is_finished = False
        self.command_step = 0
        self.trajectory_steps = []

        print(f"Initializing MovePose to {pose}...")

        # --- MODIFICATION: Store parameters for deferred planning ---
        self.pose = pose
        self.duration = duration
        self.velocity_percent = velocity_percent
        self.accel_percent = accel_percent
        self.trajectory_type = trajectory_type

    """
        Initializes, validates, and pre-computes the trajectory for a move-to-pose command.

        Args:
            pose (list): A list of 6 values [x, y, z, r, p, y] for the target pose.
                         Positions are in mm, rotations are in degrees.
            duration (float, optional): The total time for the movement in seconds.
            velocity_percent (float, optional): The target velocity as a percentage (0-100).
            accel_percent (float, optional): The target acceleration as a percentage (0-100).
            trajectory_type (str, optional): The type of trajectory ('poly' or 'trap').
        """
    
    def prepare_for_execution(self, current_position_in):
        """Calculates the full trajectory just-in-time before execution."""
        print(f"  -> Preparing trajectory for MovePose to {self.pose}...")

        initial_pos_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(current_position_in)])
        target_pose = SE3(self.pose[0] / 1000.0, self.pose[1] / 1000.0, self.pose[2] / 1000.0) * SE3.RPY(self.pose[3:6], unit='deg', order='xyz')
        
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, target_pose, initial_pos_rad, ilimit=100)

        if not ik_solution.success:
            print("  -> VALIDATION FAILED: Inverse kinematics failed at execution time.")
            self.is_valid = False
            return

        target_pos_rad = ik_solution.q

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                print("  -> INFO: Both duration and velocity were provided. Using duration.")
            command_len = int(self.duration / INTERVAL_S)
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len)
            
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))

        elif self.velocity_percent is not None:
            try:
                accel_percent = self.accel_percent if self.accel_percent is not None else 50
                
                initial_pos_steps = np.array(current_position_in)
                target_pos_steps = np.array([int(PAROL6_ROBOT.RAD2STEPS(rad, i)) for i, rad in enumerate(target_pos_rad)])

                all_joint_times = []
                for i in range(6):
                    path_to_travel = abs(target_pos_steps[i] - initial_pos_steps[i])
                    if path_to_travel == 0:
                        all_joint_times.append(0)
                        continue
                    
                    v_max_joint = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Joint_min_speed[i], PAROL6_ROBOT.Joint_max_speed[i]])
                    a_max_rad = np.interp(accel_percent, [0, 100], [PAROL6_ROBOT.Joint_min_acc, PAROL6_ROBOT.Joint_max_acc])
                    a_max_steps = PAROL6_ROBOT.SPEED_RAD2STEP(a_max_rad, i)

                    if v_max_joint <= 0 or a_max_steps <= 0:
                        raise ValueError(f"Invalid speed/acceleration for joint {i+1}. Must be positive.")

                    t_accel = v_max_joint / a_max_steps
                    if path_to_travel < v_max_joint * t_accel:
                        t_accel = np.sqrt(path_to_travel / a_max_steps)
                        joint_time = 2 * t_accel
                    else:
                        joint_time = path_to_travel / v_max_joint + t_accel
                    all_joint_times.append(joint_time)
            
                total_time = max(all_joint_times)

                if total_time <= 0:
                    self.is_finished = True
                    return

                if total_time < (2 * INTERVAL_S):
                    total_time = 2 * INTERVAL_S

                execution_time = np.arange(0, total_time, INTERVAL_S)
                
                all_q, all_qd = [], []
                for i in range(6):
                    if abs(target_pos_steps[i] - initial_pos_steps[i]) == 0:
                        all_q.append(np.full(len(execution_time), initial_pos_steps[i]))
                        all_qd.append(np.zeros(len(execution_time)))
                    else:
                        joint_traj = rp.trapezoidal(initial_pos_steps[i], target_pos_steps[i], execution_time)
                        all_q.append(joint_traj.q)
                        all_qd.append(joint_traj.qd)

                self.trajectory_steps = list(zip(np.array(all_q).T.astype(int), np.array(all_qd).T.astype(int)))
                print(f"  -> Command is valid (duration calculated from speed: {total_time:.2f}s).")

            except Exception as e:
                print(f"  -> VALIDATION FAILED: Could not calculate velocity-based trajectory. Error: {e}")
                self.is_valid = False
                return

        else:
            print("  -> Using conservative values for MovePose.")
            command_len = 200
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len)
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))
        
        if not self.trajectory_steps:
             print(" -> Trajectory calculation resulted in no steps. Command is invalid.")
             self.is_valid = False
        else:
             print(f" -> Trajectory prepared with {len(self.trajectory_steps)} steps.")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        # This method remains unchanged.
        if self.is_finished or not self.is_valid:
            return True

        if self.command_step >= len(self.trajectory_steps):
            print(f"{type(self).__name__} finished.")
            self.is_finished = True
            Position_out[:] = Position_in[:]
            Speed_out[:] = [0] * 6
            Command_out.value = 156
            return True
        else:
            pos_step, _ = self.trajectory_steps[self.command_step]
            Position_out[:] = pos_step
            Speed_out[:] = [0] * 6
            Command_out.value = 156
            self.command_step += 1
            return False
        
class MoveJointCommand:
    """
    A non-blocking command to move the robot's joints to a specific configuration.
    It pre-calculates the entire trajectory upon initialization.
    """
    def __init__(self, target_angles, duration=None, velocity_percent=None, accel_percent=50, trajectory_type='poly'):
        self.is_valid = False  # Will be set to True after basic validation
        self.is_finished = False
        self.command_step = 0
        self.trajectory_steps = []

        print(f"Initializing MoveJoint to {target_angles}...")

        # --- MODIFICATION: Store parameters for deferred planning ---
        self.target_angles = target_angles
        self.duration = duration
        self.velocity_percent = velocity_percent
        self.accel_percent = accel_percent
        self.trajectory_type = trajectory_type

        # --- Perform only state-independent validation ---
        target_pos_rad = np.array([np.deg2rad(angle) for angle in self.target_angles])
        for i in range(6):
            min_rad, max_rad = PAROL6_ROBOT.Joint_limits_radian[i]
            if not (min_rad <= target_pos_rad[i] <= max_rad):
                print(f"  -> VALIDATION FAILED: Target for Joint {i+1} ({self.target_angles[i]} deg) is out of range.")
                return
        
        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Calculates the trajectory just before execution begins."""
        print(f"  -> Preparing trajectory for MoveJoint to {self.target_angles}...")
        
        initial_pos_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(current_position_in)])
        target_pos_rad = np.array([np.deg2rad(angle) for angle in self.target_angles])

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                print("  -> INFO: Both duration and velocity were provided. Using duration.")
            command_len = int(self.duration / INTERVAL_S)
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len)
            
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))

        elif self.velocity_percent is not None:
            try:
                accel_percent = self.accel_percent if self.accel_percent is not None else 50
                initial_pos_steps = np.array(current_position_in)
                target_pos_steps = np.array([int(PAROL6_ROBOT.RAD2STEPS(rad, i)) for i, rad in enumerate(target_pos_rad)])
                
                all_joint_times = []
                for i in range(6):
                    path_to_travel = abs(target_pos_steps[i] - initial_pos_steps[i])
                    if path_to_travel == 0:
                        all_joint_times.append(0)
                        continue

                    v_max_joint = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Joint_min_speed[i], PAROL6_ROBOT.Joint_max_speed[i]])
                    a_max_rad = np.interp(accel_percent, [0, 100], [PAROL6_ROBOT.Joint_min_acc, PAROL6_ROBOT.Joint_max_acc])
                    a_max_steps = PAROL6_ROBOT.SPEED_RAD2STEP(a_max_rad, i)

                    if v_max_joint <= 0 or a_max_steps <= 0:
                        raise ValueError(f"Invalid speed/acceleration for joint {i+1}. Must be positive.")

                    t_accel = v_max_joint / a_max_steps
                    if path_to_travel < v_max_joint * t_accel:
                        t_accel = np.sqrt(path_to_travel / a_max_steps)
                        joint_time = 2 * t_accel
                    else:
                        joint_time = path_to_travel / v_max_joint + t_accel
                    all_joint_times.append(joint_time)

                total_time = max(all_joint_times)

                if total_time <= 0:
                    self.is_finished = True
                    return

                if total_time < (2 * INTERVAL_S):
                    total_time = 2 * INTERVAL_S

                execution_time = np.arange(0, total_time, INTERVAL_S)
                
                all_q, all_qd = [], []
                for i in range(6):
                    if abs(target_pos_steps[i] - initial_pos_steps[i]) == 0:
                        all_q.append(np.full(len(execution_time), initial_pos_steps[i]))
                        all_qd.append(np.zeros(len(execution_time)))
                    else:
                        joint_traj = rp.trapezoidal(initial_pos_steps[i], target_pos_steps[i], execution_time)
                        all_q.append(joint_traj.q)
                        all_qd.append(joint_traj.qd)

                self.trajectory_steps = list(zip(np.array(all_q).T.astype(int), np.array(all_qd).T.astype(int)))
                print(f"  -> Command is valid (duration calculated from speed: {total_time:.2f}s).")

            except Exception as e:
                print(f"  -> VALIDATION FAILED: Could not calculate velocity-based trajectory. Error: {e}")
                print(f"  -> Please check Joint_min/max_speed and Joint_min/max_acc values in PAROL6_ROBOT.py.")
                self.is_valid = False
                return
        
        else:
            print("  -> Using conservative values for MoveJoint.")
            command_len = 200
            traj_generator = rp.tools.trajectory.jtraj(initial_pos_rad, target_pos_rad, command_len)
            for i in range(len(traj_generator.q)):
                pos_step = [int(PAROL6_ROBOT.RAD2STEPS(p, j)) for j, p in enumerate(traj_generator.q[i])]
                self.trajectory_steps.append((pos_step, None))
        
        if not self.trajectory_steps:
             print(" -> Trajectory calculation resulted in no steps. Command is invalid.")
             self.is_valid = False
        else:
             print(f" -> Trajectory prepared with {len(self.trajectory_steps)} steps.")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        # This method remains unchanged.
        if self.is_finished or not self.is_valid:
            return True

        if self.command_step >= len(self.trajectory_steps):
            print(f"{type(self).__name__} finished.")
            self.is_finished = True
            Position_out[:] = Position_in[:]
            Speed_out[:] = [0] * 6
            Command_out.value = 156
            return True
        else:
            pos_step, _ = self.trajectory_steps[self.command_step]
            Position_out[:] = pos_step
            Speed_out[:] = [0] * 6
            Command_out.value = 156
            self.command_step += 1
            return False
        
class MoveCartCommand:
    """
    A non-blocking command to move the robot's end-effector in a straight line
    in Cartesian space, completing the move in an exact duration.

    It works by:
    1. Pre-validating the final target pose.
    2. Interpolating the pose in Cartesian space in real-time.
    3. Solving Inverse Kinematics for each intermediate step to ensure path validity.
    """
    def __init__(self, pose, duration=None, velocity_percent=None):
        self.is_valid = False
        self.is_finished = False

        # --- MODIFICATION: Validate that at least one timing parameter is given ---
        if duration is None and velocity_percent is None:
            print("  -> VALIDATION FAILED: MoveCartCommand requires either 'duration' or 'velocity_percent'.")
            return
        if duration is not None and velocity_percent is not None:
            print("  -> INFO: Both duration and velocity_percent provided. Using duration.")
            self.velocity_percent = None # Prioritize duration
        else:
            self.velocity_percent = velocity_percent

        # --- Store parameters and set placeholders ---
        self.duration = duration
        self.pose = pose
        self.start_time = None
        self.initial_pose = None
        self.target_pose = None
        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Captures the initial state and validates the path just before execution."""
        print(f"  -> Preparing for MoveCart to {self.pose}...")
        
        # --- MOVED LOGIC: Capture initial state from live data ---
        initial_q_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(current_position_in)])
        self.initial_pose = PAROL6_ROBOT.robot.fkine(initial_q_rad)
        self.target_pose = SE3(self.pose[0]/1000.0, self.pose[1]/1000.0, self.pose[2]/1000.0) * SE3.RPY(self.pose[3:6], unit='deg', order='xyz')

        print("  -> Pre-validating final target pose...")
        ik_check = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, self.target_pose, initial_q_rad
        )

        if not ik_check.success:
            print("  -> VALIDATION FAILED: The final target pose is unreachable.")
            if ik_check.violations:
                print(f"     Reason: Solution violates joint limits: {ik_check.violations}")
            self.is_valid = False # Mark as invalid if path fails
            return

        # --- NEW BLOCK: Calculate duration from velocity if needed ---
        if self.velocity_percent is not None:
            print(f"  -> Calculating duration for {self.velocity_percent}% speed...")
            # Calculate the total distance for translation and rotation
            linear_distance = np.linalg.norm(self.target_pose.t - self.initial_pose.t)
            angular_distance_rad = self.initial_pose.angdist(self.target_pose)

            # Interpolate the target speeds from percentages, assuming constants exist in PAROL6_ROBOT
            target_linear_speed = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Cartesian_linear_velocity_min, PAROL6_ROBOT.Cartesian_linear_velocity_max])
            target_angular_speed = np.interp(self.velocity_percent, [0, 100], [PAROL6_ROBOT.Cartesian_angular_velocity_min, PAROL6_ROBOT.Cartesian_angular_velocity_max])
            target_angular_speed_rad = np.deg2rad(target_angular_speed)

            # Calculate time required for each component of the movement
            time_linear = linear_distance / target_linear_speed if target_linear_speed > 0 else 0
            time_angular = angular_distance_rad / target_angular_speed_rad if target_angular_speed_rad > 0 else 0

            # The total duration is the longer of the two times to ensure synchronization
            calculated_duration = max(time_linear, time_angular)

            if calculated_duration <= 0:
                print("  -> INFO: MoveCart has zero duration. Marking as finished.")
                self.is_finished = True
                self.is_valid = True # It's valid, just already done.
                return

            self.duration = calculated_duration
            print(f"  -> Calculated MoveCart duration: {self.duration:.2f}s")

        print("  -> Command is valid and ready for execution.")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        if self.is_finished or not self.is_valid:
            return True

        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        s = min(elapsed_time / self.duration, 1.0)
        s_scaled = quintic_scaling(s)

        current_target_pose = self.initial_pose.interp(self.target_pose, s_scaled)

        current_q_rad = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
        ik_solution = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, current_target_pose, current_q_rad
        )

        if not ik_solution.success:
            print("  -> ERROR: MoveCart failed. An intermediate point on the path is unreachable.")
            if ik_solution.violations:
                 print(f"     Reason: Path violates joint limits: {ik_solution.violations}")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True

        current_pos_rad = ik_solution.q

        # --- MODIFIED BLOCK ---
        # Send only the target position and let the firmware's P-controller handle speed.
        Position_out[:] = [int(PAROL6_ROBOT.RAD2STEPS(p, i)) for i, p in enumerate(current_pos_rad)]
        Speed_out[:] = [0] * 6 # Set feed-forward velocity to zero for smooth P-control.
        Command_out.value = 156
        # --- END MODIFIED BLOCK ---

        if s >= 1.0:
            print(f"MoveCart finished in ~{elapsed_time:.2f}s.")
            self.is_finished = True
            # The main loop will handle holding the final position.

        return self.is_finished
        
class GripperCommand:
    """
    A single, unified, non-blocking command to control all gripper functions.
    It internally selects the correct logic (position-based waiting, timed delay,
    or instantaneous) based on the specified action.
    """
    def __init__(self, gripper_type, action=None, position=100, speed=100, current=500, output_port=1):
        """
        Initializes the Gripper command and configures its internal state machine
        based on the requested action.
        """
        self.is_valid = True
        self.is_finished = False
        self.gripper_type = gripper_type.lower()
        self.action = action.lower() if action else 'move'
        self.state = "START"
        self.timeout_counter = 1000 # 10-second safety timeout for all waiting states

        # --- Configure based on Gripper Type and Action ---
        if self.gripper_type == 'electric':
            if self.action == 'move':
                self.target_position = position
                self.speed = speed
                self.current = current
                if not (0 <= position <= 255 and 0 <= speed <= 255 and 100 <= current <= 1000):
                    self.is_valid = False
            elif self.action == 'calibrate':
                self.wait_counter = 200 # 2-second fixed delay for calibration
            else:
                self.is_valid = False # Invalid action

        elif self.gripper_type == 'pneumatic':
            if self.action not in ['open', 'close']:
                self.is_valid = False
            self.state_to_set = 1 if self.action == 'open' else 0
            self.port_index = 2 if output_port == 1 else 3
        else:
            self.is_valid = False

        if not self.is_valid:
            print(f"  -> VALIDATION FAILED for GripperCommand with action: '{self.action}'")

    def execute_step(self, Gripper_data_out, InOut_out, Gripper_data_in, InOut_in, **kwargs):
        if self.is_finished or not self.is_valid:
            return True

        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            print(f"  -> ERROR: Gripper command timed out in state {self.state}.")
            self.is_finished = True
            return True

        # --- Pneumatic Logic (Instantaneous) ---
        if self.gripper_type == 'pneumatic':
            InOut_out[self.port_index] = self.state_to_set
            print("  -> Pneumatic gripper command sent.")
            self.is_finished = True
            return True

        # --- Electric Gripper Logic ---
        if self.gripper_type == 'electric':
            # On the first run, transition to the correct state for the action
            if self.state == "START":
                if self.action == 'calibrate':
                    self.state = "SEND_CALIBRATE"
                else: # 'move'
                    self.state = "WAIT_FOR_POSITION"
            
            # --- Calibrate Logic (Timed Delay) ---
            if self.state == "SEND_CALIBRATE":
                print("  -> Sending one-shot calibrate command...")
                Gripper_data_out[4] = 1 # Set mode to calibrate
                self.state = "WAITING_CALIBRATION"
                return False

            if self.state == "WAITING_CALIBRATION":
                self.wait_counter -= 1
                if self.wait_counter <= 0:
                    print("  -> Calibration delay finished.")
                    Gripper_data_out[4] = 0 # Reset to operation mode
                    self.is_finished = True
                    return True
                return False

            # --- Move Logic (Position-Based) ---
            if self.state == "WAIT_FOR_POSITION":
                # Persistently send the move command
                Gripper_data_out[0], Gripper_data_out[1], Gripper_data_out[2] = self.target_position, self.speed, self.current
                Gripper_data_out[4] = 0 # Operation mode
                bitfield = [1, 1, not InOut_in[4], 1, 0, 0, 0, 0]
                fused = PAROL6_ROBOT.fuse_bitfield_2_bytearray(bitfield)
                Gripper_data_out[3] = int(fused.hex(), 16)

                # Check for completion
                current_position = Gripper_data_in[1]
                if abs(current_position - self.target_position) <= 5:
                    print(f"  -> Gripper move complete.")
                    self.is_finished = True
                    # Set command back to idle
                    bitfield = [1, 0, not InOut_in[4], 1, 0, 0, 0, 0]
                    fused = PAROL6_ROBOT.fuse_bitfield_2_bytearray(bitfield)
                    Gripper_data_out[3] = int(fused.hex(), 16)
                    return True
                return False
        
        return self.is_finished

class DelayCommand:
    """
    A non-blocking command that pauses execution for a specified duration.
    During the delay, it ensures the robot remains idle by sending the
    appropriate commands.
    """
    def __init__(self, duration):
        """
        Initializes and validates the Delay command.

        Args:
            duration (float): The delay time in seconds.
        """
        self.is_valid = False
        self.is_finished = False

        # --- 1. Parameter Validation ---
        if not isinstance(duration, (int, float)) or duration <= 0:
            print(f"  -> VALIDATION FAILED: Delay duration must be a positive number, but got {duration}.")
            return

        print(f"Initializing Delay for {duration} seconds...")
        
        self.duration = duration
        self.end_time = None  # Will be set in prepare_for_execution
        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Set the end time when the command actually starts."""
        self.end_time = time.time() + self.duration
        print(f"  -> Delay starting for {self.duration} seconds...")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """
        Checks if the delay duration has passed and keeps the robot idle.
        This method is called on every loop cycle (~0.01s).
        """
        if self.is_finished or not self.is_valid:
            return True

        # --- A. Keep the robot idle during the delay ---
        Command_out.value = 255  # Set command to idle
        Speed_out[:] = [0] * 6   # Set all speeds to zero

        # --- B. Check for completion ---
        if self.end_time and time.time() >= self.end_time:
            print(f"Delay finished after {self.duration} seconds.")
            self.is_finished = True
        
        return self.is_finished
    
#########################################################################
# Smooth Motion Commands Start Here
#########################################################################
    
class BaseSmoothMotionCommand:
    """
    Base class for all smooth motion commands with proper error tracking.
    """
    
    def __init__(self, description="smooth motion"):
        self.description = description
        self.trajectory = None
        self.trajectory_command = None
        self.transition_command = None
        self.is_valid = True
        self.is_finished = False
        self.specified_start_pose = None
        self.transition_complete = False
        self.trajectory_prepared = False
        self.error_state = False
        self.error_message = ""
        self.trajectory_generated = False  # NEW: Track if trajectory is generated
        
    def create_transition_command(self, current_pose, target_pose):
        """Create a MovePose command for smooth transition to start position."""
        pos_error = np.linalg.norm(
            np.array(target_pose[:3]) - np.array(current_pose[:3])
        )
        
        # Lower threshold to 2mm for more aggressive transition generation
        if pos_error < 2.0:  # Changed from 5.0mm to 2.0mm
            print(f"  -> Already near start position (error: {pos_error:.1f}mm)")
            return None
        
        print(f"  -> Creating smooth transition to start ({pos_error:.1f}mm away)")
        
        # Calculate transition speed based on distance
        # Slower for short distances, faster for long distances
        if pos_error < 10:
            transition_speed = 20.0  # mm/s for short distances
        elif pos_error < 30:
            transition_speed = 30.0  # mm/s for medium distances
        else:
            transition_speed = 40.0  # mm/s for long distances
        
        transition_duration = max(pos_error / transition_speed, 0.5)  # Minimum 0.5s
        
        transition_cmd = MovePoseCommand(
            pose=target_pose,
            duration=transition_duration
        )
        
        return transition_cmd
    
    def get_current_pose_from_position(self, position_in):
        """Convert current position to pose [x,y,z,rx,ry,rz]"""
        current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                             for i, p in enumerate(position_in)])
        current_pose_se3 = PAROL6_ROBOT.robot.fkine(current_q)
        
        current_xyz = current_pose_se3.t * 1000  # Convert to mm
        current_rpy = current_pose_se3.rpy(unit='deg', order='xyz')
        return np.concatenate([current_xyz, current_rpy]).tolist()
        
    def prepare_for_execution(self, current_position_in):
        """Minimal preparation - just check if we need a transition."""
        print(f"  -> Preparing {self.description}...")
        
        # If there's a specified start pose, prepare transition
        if self.specified_start_pose:
            actual_current_pose = self.get_current_pose_from_position(current_position_in)
            self.transition_command = self.create_transition_command(
                actual_current_pose, self.specified_start_pose
            )
            
            if self.transition_command:
                self.transition_command.prepare_for_execution(current_position_in)
                if not self.transition_command.is_valid:
                    print(f"  -> ERROR: Cannot reach specified start position")
                    self.is_valid = False
                    self.error_state = True
                    self.error_message = "Cannot reach specified start position"
                    return
        else:
            self.transition_command = None
            
        # DON'T generate trajectory yet - wait until execution
        self.trajectory_generated = False
        self.trajectory_prepared = False
        print(f"  -> {self.description} preparation complete (trajectory will be generated at execution)")
            
    def generate_main_trajectory(self, effective_start_pose):
        """Override this in subclasses to generate the specific motion trajectory."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")
        
    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """Execute transition first if needed, then generate and execute trajectory."""
        if self.is_finished or not self.is_valid:
            return True
        
        # Execute transition first if needed
        if self.transition_command and not self.transition_complete:
            is_done = self.transition_command.execute_step(
                Position_in, Homed_in, Speed_out, Command_out, **kwargs
            )
            
            if is_done:
                print(f"  -> Transition complete")
                self.transition_complete = True
            return False
        
        # Generate trajectory on first execution step (not during preparation!)
        if not self.trajectory_generated:
            # Get ACTUAL current position NOW
            actual_current_pose = self.get_current_pose_from_position(Position_in)
            print(f"  -> Generating {self.description} from ACTUAL position: {[round(p, 1) for p in actual_current_pose[:3]]}")
            
            # Generate trajectory from where we ACTUALLY are
            self.trajectory = self.generate_main_trajectory(actual_current_pose)
            self.trajectory_command = SmoothTrajectoryCommand(
                self.trajectory, self.description
            )
            
            # Quick validation of first point only
            current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                                 for i, p in enumerate(Position_in)])
            first_pose = self.trajectory[0]
            target_se3 = SE3(first_pose[0]/1000, first_pose[1]/1000, first_pose[2]/1000) * \
                        SE3.RPY(first_pose[3:], unit='deg', order='xyz')
            
            ik_result = solve_ik_with_adaptive_tol_subdivision(
                PAROL6_ROBOT.robot, target_se3, current_q, ilimit=50, jogging=False
            )
            
            if not ik_result.success:
                print(f"  -> ERROR: Cannot reach first trajectory point")
                self.is_finished = True
                self.error_state = True
                self.error_message = "Cannot reach trajectory start"
                Speed_out[:] = [0] * 6
                Command_out.value = 255
                return True
                
            self.trajectory_generated = True
            self.trajectory_prepared = True
            
            # Verify first point is close to current
            distance = np.linalg.norm(first_pose[:3] - np.array(actual_current_pose[:3]))
            if distance > 5.0:
                print(f"  -> WARNING: First trajectory point {distance:.1f}mm from current!")
        
        # Execute main trajectory
        if self.trajectory_command and self.trajectory_prepared:
            is_done = self.trajectory_command.execute_step(
                Position_in, Homed_in, Speed_out, Command_out, **kwargs
            )
            
            # Check for errors in trajectory execution
            if hasattr(self.trajectory_command, 'error_state') and self.trajectory_command.error_state:
                self.error_state = True
                self.error_message = self.trajectory_command.error_message
            
            if is_done:
                self.is_finished = True
            
            return is_done
        else:
            self.is_finished = True
            return True

class SmoothTrajectoryCommand:
    """Command class for executing pre-generated smooth trajectories."""
    
    def __init__(self, trajectory, description="smooth motion"):
        self.trajectory = np.array(trajectory)
        self.description = description
        self.trajectory_index = 0
        self.is_valid = True
        self.is_finished = False
        self.first_step = True
        self.error_state = False
        self.error_message = ""
        
        print(f"Initializing smooth {description} with {len(trajectory)} points")
    
    def prepare_for_execution(self, current_position_in):
        """Skip validation - trajectory is already generated from correct position"""
        # No validation needed since trajectory was just generated from current position
        self.is_valid = True
        return
    
    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, Position_out=None, **kwargs):
        """Execute one step of the smooth trajectory"""
        if self.is_finished or not self.is_valid:
            return True
        
        # Get Position_out from kwargs if not provided
        if Position_out is None:
            Position_out = kwargs.get('Position_out', Position_in)
        
        if self.trajectory_index >= len(self.trajectory):
            print(f"Smooth {self.description} finished.")
            self.is_finished = True
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Get target pose for this step
        target_pose = self.trajectory[self.trajectory_index]
        
        # Convert to SE3
        target_se3 = SE3(target_pose[0]/1000, target_pose[1]/1000, target_pose[2]/1000) * \
                    SE3.RPY(target_pose[3:], unit='deg', order='xyz')
        
        # Get current joint configuration
        current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                            for i, p in enumerate(Position_in)])
        
        # Solve IK
        ik_result = solve_ik_with_adaptive_tol_subdivision(
            PAROL6_ROBOT.robot, target_se3, current_q, ilimit=50, jogging=False
        )
        
        if not ik_result.success:
            print(f"  -> IK failed at trajectory point {self.trajectory_index}")
            self.is_finished = True
            self.error_state = True
            self.error_message = f"IK failed at point {self.trajectory_index}/{len(self.trajectory)}"
            Speed_out[:] = [0] * 6
            Command_out.value = 255
            return True
        
        # Convert to steps
        target_steps = [int(PAROL6_ROBOT.RAD2STEPS(q, i)) 
                    for i, q in enumerate(ik_result.q)]
        
        # ADD VELOCITY LIMITING - This prevents violent movements
        if self.trajectory_index > 0:
            for i in range(6):
                step_diff = abs(target_steps[i] - Position_in[i])
                max_step_diff = PAROL6_ROBOT.Joint_max_speed[i] * 0.01  # Max steps in 10ms
                
                # Use 1.2x safety margin (not 2x as before)
                if step_diff > max_step_diff * 1.2:
                    #print(f"  -> WARNING: Joint {i+1} velocity limit exceeded at point {self.trajectory_index}")
                    #print(f"     Step difference: {step_diff}, Max allowed: {max_step_diff * 1.2:.1f}")
                    
                    # Clamp the motion
                    sign = 1 if target_steps[i] > Position_in[i] else -1
                    target_steps[i] = Position_in[i] + sign * int(max_step_diff)
        
        # Send position command
        Position_out[:] = target_steps
        Speed_out[:] = [0] * 6
        Command_out.value = 156
        
        # Advance to next point
        self.trajectory_index += 1
        
        return False

class SmoothCircleCommand(BaseSmoothMotionCommand):
    def __init__(self, center, radius, plane, duration, clockwise, frame='WRF', start_pose=None):
        super().__init__(f"circle (r={radius}mm, {frame})")
        self.center = center
        self.radius = radius
        self.plane = plane
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame  # Store reference frame
        self.specified_start_pose = start_pose
        self.normal_vector = None  # Will be set if TRF
        self.current_position_in = None  # Store for TRF transformation
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        # Store current position for potential use in generate_main_trajectory
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform parameters to WRF
            params = {
                'center': self.center,
                'plane': self.plane
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_CIRCLE', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.center = transformed['center']
            self.normal_vector = transformed.get('normal_vector')
            
            print(f"  -> TRF Circle: center {self.center[:3]} (WRF), normal {self.normal_vector}")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_CIRCLE', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        # Now do normal preparation with transformed parameters
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate circle starting from the actual start position."""
        motion_gen = CircularMotion()
        
        # Use transformed normal for TRF, or standard for WRF
        if self.normal_vector is not None:
            # TRF - use the transformed normal vector
            normal = np.array(self.normal_vector)
            print(f"    Using transformed normal: {normal.round(3)}")
        else:
            # WRF - use standard plane definition
            plane_normals = {'XY': [0, 0, 1], 'XZ': [0, 1, 0], 'YZ': [1, 0, 0]}
            normal = np.array(plane_normals.get(self.plane, [0, 0, 1]))  # Convert to numpy array
            print(f"    Using WRF plane {self.plane} with normal: {normal}")
        
        print(f"    Generating circle from position: {[round(p, 1) for p in effective_start_pose[:3]]}")
        print(f"    Circle center: {[round(c, 1) for c in self.center]}")
        
        # Add geometry validation
        center_np = np.array(self.center)
        start_np = np.array(effective_start_pose[:3])
        
        # Project start point onto circle plane to check distance
        to_start = start_np - center_np
        to_start_plane = to_start - np.dot(to_start, normal) * normal
        distance_to_center = np.linalg.norm(to_start_plane)
        
        if abs(distance_to_center - self.radius) > self.radius * 0.3:
            print(f"    WARNING: Robot is {distance_to_center:.1f}mm from center, but radius is {self.radius:.1f}mm")
            print(f"    Circle geometry will be auto-corrected")
        
        # Generate circle that starts at effective_start_pose
        trajectory = motion_gen.generate_circle_3d(
            self.center, 
            self.radius, 
            normal,  # This normal now correctly represents the plane
            start_point=effective_start_pose[:3],
            duration=self.duration
        )
        
        if self.clockwise:
            trajectory = trajectory[::-1]
        
        # Update orientations to match start pose
        for i in range(len(trajectory)):
            trajectory[i][3:] = effective_start_pose[3:]
        
        return trajectory
    
class SmoothArcCenterCommand(BaseSmoothMotionCommand):
    def __init__(self, end_pose, center, duration, clockwise, frame='WRF', start_pose=None):
        super().__init__(f"arc (center-based, {frame})")
        self.end_pose = end_pose
        self.center = center
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame
        self.specified_start_pose = start_pose
        self.normal_vector = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF."""
        if self.frame == 'TRF':
            params = {
                'end_pose': self.end_pose,
                'center': self.center
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_ARC_CENTER', params, 'TRF', current_position_in
            )
            self.end_pose = transformed['end_pose']
            self.center = transformed['center']
            self.normal_vector = transformed.get('normal_vector')
            
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_ARC_CENTER', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc from actual start to end."""
        motion_gen = CircularMotion()
        return motion_gen.generate_arc_3d(
            effective_start_pose, self.end_pose, self.center,
            normal=self.normal_vector,  # Use transformed normal if TRF
            clockwise=self.clockwise, duration=self.duration
        )
    
class SmoothArcParamCommand(BaseSmoothMotionCommand):
    def __init__(self, end_pose, radius, arc_angle, duration, clockwise, frame='WRF', start_pose=None):
        super().__init__(f"parametric arc (r={radius}mm, θ={arc_angle}°, {frame})")
        self.end_pose = end_pose
        self.radius = radius
        self.arc_angle = arc_angle
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame
        self.specified_start_pose = start_pose
        self.normal_vector = None  # Will be set if TRF
        self.current_position_in = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform parameters to WRF
            params = {
                'end_pose': self.end_pose,
                'plane': 'XY'  # Default plane for parametric arc
            }
            transformed = transform_command_params_to_wrf(
                'SMOOTH_ARC_PARAM', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.end_pose = transformed['end_pose']
            self.normal_vector = transformed.get('normal_vector')
            
            print(f"  -> TRF Parametric Arc: end {self.end_pose[:3]} (WRF)")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_ARC_PARAM', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc based on radius and angle from actual start."""
        # Get start and end positions
        start_xyz = np.array(effective_start_pose[:3])
        end_xyz = np.array(self.end_pose[:3])
        
        # If we have a transformed normal (TRF), use it to define the arc plane
        if self.normal_vector is not None:
            normal = np.array(self.normal_vector)
            
            # Project start and end onto the plane perpendicular to normal
            # This ensures the arc stays in the correct plane for TRF
            
            # Find center of arc using radius and angle
            chord_vec = end_xyz - start_xyz
            chord_length = np.linalg.norm(chord_vec)
            
            if chord_length > 2 * self.radius:
                print(f"  -> Warning: Points too far apart ({chord_length:.1f}mm) for radius {self.radius}mm")
                self.radius = chord_length / 2 + 1
            
            # Calculate center position using the normal vector
            chord_mid = (start_xyz + end_xyz) / 2
            
            # Height from chord midpoint to arc center
            if chord_length > 0:
                h = np.sqrt(max(0, self.radius**2 - (chord_length/2)**2))
                
                # Find perpendicular in the plane defined by normal
                chord_dir = chord_vec / chord_length
                perp_in_plane = np.cross(normal, chord_dir)
                if np.linalg.norm(perp_in_plane) > 0.001:
                    perp_in_plane = perp_in_plane / np.linalg.norm(perp_in_plane)
                else:
                    # Chord is parallel to normal (shouldn't happen)
                    perp_in_plane = np.array([1, 0, 0])
                
                # Arc center
                if self.clockwise:
                    center_3d = chord_mid - h * perp_in_plane
                else:
                    center_3d = chord_mid + h * perp_in_plane
            else:
                center_3d = start_xyz
            
        else:
            # WRF - use XY plane (standard behavior)
            normal = np.array([0, 0, 1])
            
            # Calculate arc center in XY plane
            start_xy = start_xyz[:2]
            end_xy = end_xyz[:2]
            
            # Midpoint
            mid = (start_xy + end_xy) / 2
            
            # Distance between points
            d = np.linalg.norm(end_xy - start_xy)
            
            if d > 2 * self.radius:
                print(f"  -> Warning: Points too far apart ({d:.1f}mm) for radius {self.radius}mm")
                self.radius = d / 2 + 1
            
            # Height of arc center from midpoint
            h = np.sqrt(max(0, self.radius**2 - (d/2)**2))
            
            # Perpendicular direction
            if d > 0:
                perp = np.array([-(end_xy[1] - start_xy[1]), end_xy[0] - start_xy[0]])
                perp = perp / np.linalg.norm(perp)
            else:
                perp = np.array([1, 0])
            
            # Arc center (choose based on clockwise)
            if self.clockwise:
                center_2d = mid - h * perp
            else:
                center_2d = mid + h * perp
            
            # Use average Z for center
            center_3d = [center_2d[0], center_2d[1], (start_xyz[2] + end_xyz[2]) / 2]
        
        # Generate arc trajectory from actual start
        motion_gen = CircularMotion()
        return motion_gen.generate_arc_3d(
            effective_start_pose, self.end_pose, center_3d.tolist(),
            normal=normal if self.normal_vector is not None else None,
            clockwise=self.clockwise, duration=self.duration
        )

class SmoothHelixCommand(BaseSmoothMotionCommand):
    def __init__(self, center, radius, pitch, height, duration, clockwise, frame='WRF', start_pose=None):
        super().__init__(f"helix (h={height}mm, {frame})")
        self.center = center
        self.radius = radius
        self.pitch = pitch
        self.height = height
        self.duration = duration
        self.clockwise = clockwise
        self.frame = frame
        self.specified_start_pose = start_pose
        self.helix_axis = None
        self.up_vector = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF."""
        if self.frame == 'TRF':
            params = {'center': self.center}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_HELIX', params, 'TRF', current_position_in
            )
            self.center = transformed['center']
            self.helix_axis = transformed.get('helix_axis', [0, 0, 1])
            self.up_vector = transformed.get('up_vector', [0, 1, 0])
            
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_HELIX', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate helix with proper axis orientation."""
        num_revolutions = self.height / self.pitch if self.pitch > 0 else 1
        num_points = int(self.duration * 100)  # 100Hz
        
        # Get helix axis (default Z for WRF, transformed for TRF)
        if self.helix_axis is not None:
            axis = np.array(self.helix_axis)
        else:
            axis = np.array([0, 0, 1])  # Default vertical
        
        # Create orthonormal basis for helix
        if self.up_vector is not None:
            up = np.array(self.up_vector)
        else:
            # Find perpendicular to axis
            if abs(axis[2]) < 0.9:
                up = np.array([0, 0, 1])
            else:
                up = np.array([1, 0, 0])
        
        # Ensure up is perpendicular to axis
        up = up - np.dot(up, axis) * axis
        up = up / np.linalg.norm(up)
        
        # Create right vector
        right = np.cross(axis, up)
        
        # Calculate starting angle based on actual start position
        to_start = np.array(effective_start_pose[:3]) - np.array(self.center)
        # Project onto plane perpendicular to axis
        to_start_plane = to_start - np.dot(to_start, axis) * axis
        
        if np.linalg.norm(to_start_plane) > 0.001:
            to_start_normalized = to_start_plane / np.linalg.norm(to_start_plane)
            start_angle = np.arctan2(np.dot(to_start_normalized, up), 
                                    np.dot(to_start_normalized, right))
        else:
            start_angle = 0
        
        trajectory = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            angle = start_angle + 2 * np.pi * num_revolutions * t
            if self.clockwise:
                angle = start_angle - 2 * np.pi * num_revolutions * t
            
            # Position on helix
            pos = np.array(self.center) + \
                  self.radius * (np.cos(angle) * right + np.sin(angle) * up) + \
                  self.height * t * axis
            
            # Use orientation from effective start
            trajectory.append(np.concatenate([pos, effective_start_pose[3:]]))
        
        return np.array(trajectory)

class SmoothSplineCommand(BaseSmoothMotionCommand):
    def __init__(self, waypoints, duration, frame='WRF', start_pose=None):
        super().__init__(f"spline ({len(waypoints)} points, {frame})")
        self.waypoints = waypoints
        self.duration = duration
        self.frame = frame
        self.specified_start_pose = start_pose
        self.current_position_in = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform waypoints to WRF
            params = {'waypoints': self.waypoints}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_SPLINE', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.waypoints = transformed['waypoints']
            
            print(f"  -> TRF Spline: transformed {len(self.waypoints)} waypoints to WRF")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_SPLINE', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate spline starting from actual position."""
        motion_gen = SplineMotion()
        
        # Always start from the effective start pose
        first_wp_error = np.linalg.norm(
            np.array(self.waypoints[0][:3]) - np.array(effective_start_pose[:3])
        )
        
        if first_wp_error > 5.0:
            # First waypoint is far, prepend the start position
            modified_waypoints = [effective_start_pose] + self.waypoints
            print(f"    Added start position as first waypoint (distance: {first_wp_error:.1f}mm)")
        else:
            # Replace first waypoint with actual start to ensure continuity
            modified_waypoints = [effective_start_pose] + self.waypoints[1:]
            print(f"    Replaced first waypoint with actual start position")
        
        timestamps = np.linspace(0, self.duration, len(modified_waypoints))
        
        # Generate the spline trajectory
        trajectory = motion_gen.generate_cubic_spline(modified_waypoints, timestamps)
        
        print(f"    Generated spline with {len(trajectory)} points")
        
        return trajectory

class SmoothBlendCommand(BaseSmoothMotionCommand):
    def __init__(self, segment_definitions, blend_time, frame='WRF', start_pose=None):
        super().__init__(f"blended ({len(segment_definitions)} segments, {frame})")
        self.segment_definitions = segment_definitions
        self.blend_time = blend_time
        self.frame = frame
        self.specified_start_pose = start_pose
        self.current_position_in = None
        
    def prepare_for_execution(self, current_position_in):
        """Transform parameters if in TRF, then prepare normally."""
        self.current_position_in = current_position_in
        
        if self.frame == 'TRF':
            # Transform all segment definitions to WRF
            params = {'segments': self.segment_definitions}
            transformed = transform_command_params_to_wrf(
                'SMOOTH_BLEND', params, 'TRF', current_position_in
            )
            
            # Update with transformed values
            self.segment_definitions = transformed['segments']
            
            print(f"  -> TRF Blend: transformed {len(self.segment_definitions)} segments to WRF")
            
            # Also transform start_pose if specified
            if self.specified_start_pose:
                params = {'start_pose': self.specified_start_pose}
                transformed = transform_command_params_to_wrf(
                    'SMOOTH_BLEND', params, 'TRF', current_position_in
                )
                self.specified_start_pose = transformed.get('start_pose')
        
        return super().prepare_for_execution(current_position_in)
        
    def generate_main_trajectory(self, effective_start_pose):
        """Generate blended trajectory starting from actual position."""
        trajectories = []
        motion_gen_circle = CircularMotion()
        motion_gen_spline = SplineMotion()
        
        # Always start from effective start pose
        last_end_pose = effective_start_pose
        
        for i, seg_def in enumerate(self.segment_definitions):
            seg_type = seg_def['type']
            
            # First segment always starts from effective_start_pose
            segment_start = effective_start_pose if i == 0 else last_end_pose
            
            if seg_type == 'LINE':
                end = seg_def['end']
                duration = seg_def['duration']
                
                # Generate line segment from actual position
                num_points = int(duration * 100)
                timestamps = np.linspace(0, duration, num_points)
                
                traj = []
                for t in timestamps:
                    s = t / duration if duration > 0 else 1
                    # Interpolate position
                    pos = [
                        segment_start[j] * (1-s) + end[j] * s
                        for j in range(3)
                    ]
                    # Interpolate orientation
                    orient = [
                        segment_start[j+3] * (1-s) + end[j+3] * s
                        for j in range(3)
                    ]
                    traj.append(pos + orient)
                
                trajectories.append(np.array(traj))
                last_end_pose = end
                
            elif seg_type == 'ARC':
                end = seg_def['end']
                center = seg_def['center']
                duration = seg_def['duration']
                clockwise = seg_def['clockwise']
                
                # Check if we have a transformed normal (from TRF)
                normal = seg_def.get('normal_vector', None)
                
                traj = motion_gen_circle.generate_arc_3d(
                    segment_start, end, center, 
                    normal=normal,  # Use transformed normal if available
                    clockwise=clockwise, duration=duration
                )
                trajectories.append(traj)
                last_end_pose = end
                
            elif seg_type == 'CIRCLE':
                center = seg_def['center']
                radius = seg_def['radius']
                plane = seg_def.get('plane', 'XY')
                duration = seg_def['duration']
                clockwise = seg_def['clockwise']
                
                # Use transformed normal if available (from TRF)
                if 'normal_vector' in seg_def:
                    normal = seg_def['normal_vector']
                else:
                    plane_normals = {'XY': [0, 0, 1], 'XZ': [0, 1, 0], 'YZ': [1, 0, 0]}
                    normal = plane_normals.get(plane, [0, 0, 1])
                
                traj = motion_gen_circle.generate_circle_3d(
                    center, radius, normal, 
                    start_point=segment_start[:3],
                    duration=duration
                )
                
                if clockwise:
                    traj = traj[::-1]
                    
                # Update orientations
                for j in range(len(traj)):
                    traj[j][3:] = segment_start[3:]
                    
                trajectories.append(traj)
                # Circle returns to start, so last pose is last point of trajectory
                last_end_pose = traj[-1].tolist()
                
            elif seg_type == 'SPLINE':
                waypoints = seg_def['waypoints']
                duration = seg_def['duration']
                
                # Check if first waypoint is close to segment start
                wp_error = np.linalg.norm(
                    np.array(waypoints[0][:3]) - np.array(segment_start[:3])
                )
                
                if wp_error > 5.0:
                    full_waypoints = [segment_start] + waypoints
                else:
                    full_waypoints = [segment_start] + waypoints[1:]
                
                timestamps = np.linspace(0, duration, len(full_waypoints))
                traj = motion_gen_spline.generate_cubic_spline(full_waypoints, timestamps)
                trajectories.append(traj)
                last_end_pose = waypoints[-1]
        
        # Blend all trajectories
        if len(trajectories) > 1:
            blender = MotionBlender(self.blend_time)
            blended = trajectories[0]
            blend_samples = int(self.blend_time * 100)
            
            for i in range(1, len(trajectories)):
                blended = blender.blend_trajectories(blended, trajectories[i], blend_samples)
            
            print(f"    Blended {len(trajectories)} segments into {len(blended)} points")
            return blended
        elif trajectories:
            return trajectories[0]
        else:
            raise ValueError("No trajectories generated in blend")
        
def calculate_duration_from_speed(trajectory_length: float, speed_percentage: float) -> float:
    """
    Calculate duration based on trajectory length and speed percentage.
    
    Args:
        trajectory_length: Total path length in mm
        speed_percentage: Speed as percentage (1-100)
        
    Returns:
        Duration in seconds
    """
    # Map speed percentage to mm/s (adjustable based on robot capabilities)
    # For example: 100% = 100mm/s, 50% = 50mm/s
    speed_mm_s = np.interp(speed_percentage, [0, 100], 
                          [PAROL6_ROBOT.Cartesian_linear_velocity_min * 1000,
                           PAROL6_ROBOT.Cartesian_linear_velocity_max * 1000])
    
    if speed_mm_s > 0:
        return trajectory_length / speed_mm_s
    else:
        return 5.0  # Default fallback
    
def parse_smooth_motion_commands(parts):
    """
    Parse smooth motion commands received via UDP and create appropriate command objects.
    All commands support:
    - Reference frame selection (WRF or TRF)
    - Optional start position (CURRENT or specified pose)
    - Both DURATION and SPEED timing modes
    
    Args:
        parts: List of command parts split by '|'
        
    Returns:
        Command object or None if parsing fails
    """
    command_type = parts[0]
    
    # Helper function for parsing optional start pose
    def parse_start_pose(start_str):
        """Parse start pose - returns None for CURRENT, or list of floats for specified pose."""
        if start_str == 'CURRENT' or start_str == 'NONE':
            return None
        else:
            try:
                return list(map(float, start_str.split(',')))
            except:
                print(f"Warning: Invalid start pose format: {start_str}")
                return None
    
    # Helper function for calculating duration from speed
    def calculate_duration_from_speed(trajectory_length: float, speed_percentage: float) -> float:
        """Calculate duration based on trajectory length and speed percentage."""
        # Map speed percentage to mm/s
        min_speed = PAROL6_ROBOT.Cartesian_linear_velocity_min * 1000  # Convert to mm/s
        max_speed = PAROL6_ROBOT.Cartesian_linear_velocity_max * 1000  # Convert to mm/s
        speed_mm_s = np.interp(speed_percentage, [0, 100], [min_speed, max_speed])
        
        if speed_mm_s > 0:
            return trajectory_length / speed_mm_s
        else:
            return 5.0  # Default fallback
    
    try:
        if command_type == 'SMOOTH_CIRCLE':
            # Format: SMOOTH_CIRCLE|center_x,center_y,center_z|radius|plane|frame|start_pose|timing_type|timing_value|clockwise
            center = list(map(float, parts[1].split(',')))
            radius = float(parts[2])
            plane = parts[3]
            frame = parts[4]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[5])
            timing_type = parts[6]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[7])
            clockwise = parts[8] == '1'
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Circle circumference
                path_length = 2 * np.pi * radius
                duration = calculate_duration_from_speed(path_length, timing_value)
            
            print(f"  -> Parsed circle: r={radius}mm, plane={plane}, frame={frame}, {timing_type}={timing_value}, duration={duration:.2f}s")
            
            # Return command object with frame parameter
            return SmoothCircleCommand(center, radius, plane, duration, clockwise, frame, start_pose)
            
        elif command_type == 'SMOOTH_ARC_CENTER':
            # Format: SMOOTH_ARC_CENTER|end_pose|center|frame|start_pose|timing_type|timing_value|clockwise
            end_pose = list(map(float, parts[1].split(',')))
            center = list(map(float, parts[2].split(',')))
            frame = parts[3]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[4])
            timing_type = parts[5]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[6])
            clockwise = parts[7] == '1'
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Estimate arc length (will be more accurate when we have actual positions)
                # Use a conservative estimate based on radius
                radius_estimate = np.linalg.norm(np.array(center) - np.array(end_pose[:3]))
                estimated_arc_angle = np.pi / 2  # 90 degrees estimate
                arc_length = radius_estimate * estimated_arc_angle
                duration = calculate_duration_from_speed(arc_length, timing_value)
            
            print(f"  -> Parsed arc (center): frame={frame}, {timing_type}={timing_value}, duration={duration:.2f}s")
            
            # Return command with frame
            return SmoothArcCenterCommand(end_pose, center, duration, clockwise, frame, start_pose)
            
        elif command_type == 'SMOOTH_ARC_PARAM':
            # Format: SMOOTH_ARC_PARAM|end_pose|radius|angle|frame|start_pose|timing_type|timing_value|clockwise
            end_pose = list(map(float, parts[1].split(',')))
            radius = float(parts[2])
            arc_angle = float(parts[3])
            frame = parts[4]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[5])
            timing_type = parts[6]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[7])
            clockwise = parts[8] == '1'
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Arc length = radius * angle (in radians)
                arc_length = radius * np.deg2rad(arc_angle)
                duration = calculate_duration_from_speed(arc_length, timing_value)
            
            print(f"  -> Parsed arc (param): r={radius}mm, θ={arc_angle}°, frame={frame}, duration={duration:.2f}s")
            
            # Return command object with frame
            return SmoothArcParamCommand(end_pose, radius, arc_angle, duration, clockwise, frame, start_pose)
            
        elif command_type == 'SMOOTH_SPLINE':
            # Format: SMOOTH_SPLINE|num_waypoints|frame|start_pose|timing_type|timing_value|waypoint1|waypoint2|...
            num_waypoints = int(parts[1])
            frame = parts[2]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[3])
            timing_type = parts[4]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[5])
            
            # Parse waypoints
            waypoints = []
            idx = 6
            for i in range(num_waypoints):
                wp = []
                for j in range(6):  # Each waypoint has 6 values (x,y,z,rx,ry,rz)
                    wp.append(float(parts[idx]))
                    idx += 1
                waypoints.append(wp)
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Calculate total path length
                total_dist = 0
                for i in range(1, len(waypoints)):
                    dist = np.linalg.norm(np.array(waypoints[i][:3]) - np.array(waypoints[i-1][:3]))
                    total_dist += dist
                
                duration = calculate_duration_from_speed(total_dist, timing_value)
            
            print(f"  -> Parsed spline: {num_waypoints} points, frame={frame}, duration={duration:.2f}s")
            
            # Return command object with frame
            return SmoothSplineCommand(waypoints, duration, frame, start_pose)
            
        elif command_type == 'SMOOTH_HELIX':
            # Format: SMOOTH_HELIX|center|radius|pitch|height|frame|start_pose|timing_type|timing_value|clockwise
            center = list(map(float, parts[1].split(',')))
            radius = float(parts[2])
            pitch = float(parts[3])
            height = float(parts[4])
            frame = parts[5]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[6])
            timing_type = parts[7]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[8])
            clockwise = parts[9] == '1'
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Calculate helix path length
                num_revolutions = height / pitch if pitch > 0 else 1
                horizontal_length = 2 * np.pi * radius * num_revolutions
                helix_length = np.sqrt(horizontal_length**2 + height**2)
                duration = calculate_duration_from_speed(helix_length, timing_value)
            
            print(f"  -> Parsed helix: h={height}mm, pitch={pitch}mm, frame={frame}, duration={duration:.2f}s")
            
            # Return command object with frame
            return SmoothHelixCommand(center, radius, pitch, height, duration, clockwise, frame, start_pose)
            
        elif command_type == 'SMOOTH_BLEND':
            # Format: SMOOTH_BLEND|num_segments|blend_time|frame|start_pose|timing_type|timing_value|segment1||segment2||...
            num_segments = int(parts[1])
            blend_time = float(parts[2])
            frame = parts[3]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[4])
            timing_type = parts[5]  # 'DEFAULT', 'DURATION', or 'SPEED'
            
            # Parse overall timing
            if timing_type == 'DEFAULT':
                # Use individual segment durations as-is
                overall_duration = None
                overall_speed = None
                segments_start_idx = 6
            else:
                timing_value = float(parts[6])
                if timing_type == 'DURATION':
                    overall_duration = timing_value
                    overall_speed = None
                else:  # SPEED
                    overall_speed = timing_value
                    overall_duration = None
                segments_start_idx = 7
            
            # Parse segments (separated by ||)
            segments_data = '|'.join(parts[segments_start_idx:])
            segment_strs = segments_data.split('||')
            
            # Parse segment definitions
            segment_definitions = []
            total_original_duration = 0
            total_estimated_length = 0
            
            for seg_str in segment_strs:
                if not seg_str:  # Skip empty segments
                    continue
                    
                seg_parts = seg_str.split('|')
                seg_type = seg_parts[0]
                
                if seg_type == 'LINE':
                    # Format: LINE|end_x,end_y,end_z,end_rx,end_ry,end_rz|duration
                    end = list(map(float, seg_parts[1].split(',')))
                    segment_duration = float(seg_parts[2])
                    total_original_duration += segment_duration
                    
                    # Estimate length (will be refined when we have actual start)
                    estimated_length = 100  # mm, conservative estimate
                    total_estimated_length += estimated_length
                    
                    segment_definitions.append({
                        'type': 'LINE',
                        'end': end,
                        'duration': segment_duration,
                        'original_duration': segment_duration
                    })
                    
                elif seg_type == 'CIRCLE':
                    # Format: CIRCLE|center_x,center_y,center_z|radius|plane|duration|clockwise
                    center = list(map(float, seg_parts[1].split(',')))
                    radius = float(seg_parts[2])
                    plane = seg_parts[3]
                    segment_duration = float(seg_parts[4])
                    total_original_duration += segment_duration
                    clockwise = seg_parts[5] == '1'
                    
                    # Circle circumference
                    estimated_length = 2 * np.pi * radius
                    total_estimated_length += estimated_length
                    
                    segment_definitions.append({
                        'type': 'CIRCLE',
                        'center': center,
                        'radius': radius,
                        'plane': plane,
                        'duration': segment_duration,
                        'original_duration': segment_duration,
                        'clockwise': clockwise
                    })
                    
                elif seg_type == 'ARC':
                    # Format: ARC|end_x,end_y,end_z,end_rx,end_ry,end_rz|center_x,center_y,center_z|duration|clockwise
                    end = list(map(float, seg_parts[1].split(',')))
                    center = list(map(float, seg_parts[2].split(',')))
                    segment_duration = float(seg_parts[3])
                    total_original_duration += segment_duration
                    clockwise = seg_parts[4] == '1'
                    
                    # Estimate arc length
                    estimated_radius = 50  # mm
                    estimated_arc_angle = np.pi / 2  # 90 degrees
                    estimated_length = estimated_radius * estimated_arc_angle
                    total_estimated_length += estimated_length
                    
                    segment_definitions.append({
                        'type': 'ARC',
                        'end': end,
                        'center': center,
                        'duration': segment_duration,
                        'original_duration': segment_duration,
                        'clockwise': clockwise
                    })
                    
                elif seg_type == 'SPLINE':
                    # Format: SPLINE|num_points|waypoint1;waypoint2;...|duration
                    num_points = int(seg_parts[1])
                    waypoints = []
                    wp_strs = seg_parts[2].split(';')
                    for wp_str in wp_strs:
                        waypoints.append(list(map(float, wp_str.split(','))))
                    segment_duration = float(seg_parts[3])
                    total_original_duration += segment_duration
                    
                    # Estimate spline length
                    estimated_length = 0
                    for i in range(1, len(waypoints)):
                        estimated_length += np.linalg.norm(
                            np.array(waypoints[i][:3]) - np.array(waypoints[i-1][:3])
                        )
                    total_estimated_length += estimated_length
                    
                    segment_definitions.append({
                        'type': 'SPLINE',
                        'waypoints': waypoints,
                        'duration': segment_duration,
                        'original_duration': segment_duration
                    })
            
            # Adjust segment durations if overall timing is specified
            if overall_duration is not None:
                # Scale all segment durations proportionally
                if total_original_duration > 0:
                    scale_factor = overall_duration / total_original_duration
                    for seg in segment_definitions:
                        seg['duration'] = seg['original_duration'] * scale_factor
                print(f"  -> Scaled blend segments to total duration: {overall_duration:.2f}s")
                        
            elif overall_speed is not None:
                # Calculate duration from speed and estimated path length
                overall_duration = calculate_duration_from_speed(total_estimated_length, overall_speed)
                if total_original_duration > 0:
                    scale_factor = overall_duration / total_original_duration
                    for seg in segment_definitions:
                        seg['duration'] = seg['original_duration'] * scale_factor
                print(f"  -> Calculated blend duration from speed: {overall_duration:.2f}s")
            else:
                print(f"  -> Using original segment durations (total: {total_original_duration:.2f}s)")
            
            print(f"  -> Parsed blend: {num_segments} segments, frame={frame}, blend_time={blend_time}s")
            
            # Return command with frame
            return SmoothBlendCommand(segment_definitions, blend_time, frame, start_pose)
            
    except Exception as e:
        print(f"Error parsing smooth motion command: {e}")
        print(f"Command parts: {parts}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"Warning: Unknown smooth motion command type: {command_type}")
    return None

def transform_command_params_to_wrf(command_type: str, params: dict, frame: str, current_position_in) -> dict:
    """
    Transform command parameters from TRF to WRF.
    Handles position, orientation, and directional vectors correctly.
    """
    if frame == 'WRF':
        return params
    
    # Get current tool pose
    current_q = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) 
                         for i, p in enumerate(current_position_in)])
    tool_pose = PAROL6_ROBOT.robot.fkine(current_q)
    
    transformed = params.copy()
    
    # SMOOTH_CIRCLE - Transform center and plane normal
    if command_type == 'SMOOTH_CIRCLE':
        if 'center' in params:
            center_trf = SE3(params['center'][0]/1000, 
                           params['center'][1]/1000, 
                           params['center'][2]/1000)
            center_wrf = tool_pose * center_trf
            transformed['center'] = (center_wrf.t * 1000).tolist()
        
        if 'plane' in params:
            plane_normals_trf = {
                'XY': [0, 0, 1],   # Tool's Z-axis
                'XZ': [0, 1, 0],   # Tool's Y-axis  
                'YZ': [1, 0, 0]    # Tool's X-axis
            }
            normal_trf = np.array(plane_normals_trf[params['plane']])
            normal_wrf = tool_pose.R @ normal_trf
            transformed['normal_vector'] = normal_wrf.tolist()
            print(f"  -> TRF circle plane {params['plane']} transformed to WRF")
    
    # SMOOTH_ARC_CENTER - Transform center, end_pose, and implied plane
    elif command_type == 'SMOOTH_ARC_CENTER':
        if 'center' in params:
            center_trf = SE3(params['center'][0]/1000, 
                           params['center'][1]/1000, 
                           params['center'][2]/1000)
            center_wrf = tool_pose * center_trf
            transformed['center'] = (center_wrf.t * 1000).tolist()
        
        if 'end_pose' in params:
            end_trf = SE3(params['end_pose'][0]/1000, 
                         params['end_pose'][1]/1000, 
                         params['end_pose'][2]/1000) * \
                      SE3.RPY(params['end_pose'][3:], unit='deg', order='xyz')
            end_wrf = tool_pose * end_trf
            transformed['end_pose'] = np.concatenate([
                end_wrf.t * 1000,
                end_wrf.rpy(unit='deg', order='xyz')
            ]).tolist()
        
        # Arc plane is determined by start, end, and center points
        # But we should transform any specified plane normal
        if 'plane' in params:
            # Similar to circle plane transformation
            plane_normals_trf = {
                'XY': [0, 0, 1],
                'XZ': [0, 1, 0],
                'YZ': [1, 0, 0]
            }
            normal_trf = np.array(plane_normals_trf[params['plane']])
            normal_wrf = tool_pose.R @ normal_trf
            transformed['normal_vector'] = normal_wrf.tolist()
    
    # SMOOTH_ARC_PARAM - Transform end_pose and arc plane
    elif command_type == 'SMOOTH_ARC_PARAM':
        if 'end_pose' in params:
            end_trf = SE3(params['end_pose'][0]/1000, 
                         params['end_pose'][1]/1000, 
                         params['end_pose'][2]/1000) * \
                      SE3.RPY(params['end_pose'][3:], unit='deg', order='xyz')
            end_wrf = tool_pose * end_trf
            transformed['end_pose'] = np.concatenate([
                end_wrf.t * 1000,
                end_wrf.rpy(unit='deg', order='xyz')
            ]).tolist()
        
        # For parametric arc, the plane is usually XY of the tool
        # Transform the assumed plane normal
        if 'plane' not in params:
            params['plane'] = 'XY'  # Default to XY plane
        
        plane_normals_trf = {
            'XY': [0, 0, 1],
            'XZ': [0, 1, 0],
            'YZ': [1, 0, 0]
        }
        normal_trf = np.array(plane_normals_trf[params.get('plane', 'XY')])
        normal_wrf = tool_pose.R @ normal_trf
        transformed['normal_vector'] = normal_wrf.tolist()
    
    # SMOOTH_HELIX - Transform center and helix axis
    elif command_type == 'SMOOTH_HELIX':
        if 'center' in params:
            center_trf = SE3(params['center'][0]/1000, 
                           params['center'][1]/1000, 
                           params['center'][2]/1000)
            center_wrf = tool_pose * center_trf
            transformed['center'] = (center_wrf.t * 1000).tolist()
        
        # Helix axis - default is Z-axis (vertical in TRF)
        # In TRF, helix rises along tool's Z-axis
        helix_axis_trf = np.array([0, 0, 1])  # Tool's Z-axis
        helix_axis_wrf = tool_pose.R @ helix_axis_trf
        transformed['helix_axis'] = helix_axis_wrf.tolist()
        
        # Also need to transform the "up" direction for proper orientation
        up_vector_trf = np.array([0, 1, 0])  # Tool's Y-axis
        up_vector_wrf = tool_pose.R @ up_vector_trf
        transformed['up_vector'] = up_vector_wrf.tolist()
    
    # SMOOTH_SPLINE - Transform all waypoints
    elif command_type == 'SMOOTH_SPLINE':
        if 'waypoints' in params:
            transformed_waypoints = []
            for wp in params['waypoints']:
                wp_trf = SE3(wp[0]/1000, wp[1]/1000, wp[2]/1000) * \
                         SE3.RPY(wp[3:], unit='deg', order='xyz')
                wp_wrf = tool_pose * wp_trf
                wp_transformed = np.concatenate([
                    wp_wrf.t * 1000,
                    wp_wrf.rpy(unit='deg', order='xyz')
                ]).tolist()
                transformed_waypoints.append(wp_transformed)
            transformed['waypoints'] = transformed_waypoints
    
    # SMOOTH_BLEND - Transform all segments recursively
    elif command_type == 'SMOOTH_BLEND':
        if 'segments' in params:
            transformed_segments = []
            for seg in params['segments']:
                seg_copy = seg.copy()
                seg_type = seg['type']
                
                if seg_type == 'LINE':
                    if 'end' in seg:
                        end_trf = SE3(seg['end'][0]/1000, 
                                    seg['end'][1]/1000, 
                                    seg['end'][2]/1000) * \
                                  SE3.RPY(seg['end'][3:], unit='deg', order='xyz')
                        end_wrf = tool_pose * end_trf
                        seg_copy['end'] = np.concatenate([
                            end_wrf.t * 1000,
                            end_wrf.rpy(unit='deg', order='xyz')
                        ]).tolist()
                
                elif seg_type == 'CIRCLE':
                    if 'center' in seg:
                        center_trf = SE3(seg['center'][0]/1000, 
                                       seg['center'][1]/1000, 
                                       seg['center'][2]/1000)
                        center_wrf = tool_pose * center_trf
                        seg_copy['center'] = (center_wrf.t * 1000).tolist()
                    
                    if 'plane' in seg:
                        plane_normals_trf = {
                            'XY': [0, 0, 1],
                            'XZ': [0, 1, 0],
                            'YZ': [1, 0, 0]
                        }
                        normal_trf = np.array(plane_normals_trf[seg['plane']])
                        normal_wrf = tool_pose.R @ normal_trf
                        seg_copy['normal_vector'] = normal_wrf.tolist()
                
                elif seg_type == 'ARC':
                    if 'center' in seg:
                        center_trf = SE3(seg['center'][0]/1000, 
                                       seg['center'][1]/1000, 
                                       seg['center'][2]/1000)
                        center_wrf = tool_pose * center_trf
                        seg_copy['center'] = (center_wrf.t * 1000).tolist()
                    
                    if 'end' in seg:
                        end_trf = SE3(seg['end'][0]/1000, 
                                    seg['end'][1]/1000, 
                                    seg['end'][2]/1000) * \
                                  SE3.RPY(seg['end'][3:], unit='deg', order='xyz')
                        end_wrf = tool_pose * end_trf
                        seg_copy['end'] = np.concatenate([
                            end_wrf.t * 1000,
                            end_wrf.rpy(unit='deg', order='xyz')
                        ]).tolist()
                
                elif seg_type == 'SPLINE':
                    if 'waypoints' in seg:
                        transformed_waypoints = []
                        for wp in seg['waypoints']:
                            wp_trf = SE3(wp[0]/1000, wp[1]/1000, wp[2]/1000) * \
                                     SE3.RPY(wp[3:], unit='deg', order='xyz')
                            wp_wrf = tool_pose * wp_trf
                            wp_transformed = np.concatenate([
                                wp_wrf.t * 1000,
                                wp_wrf.rpy(unit='deg', order='xyz')
                            ]).tolist()
                            transformed_waypoints.append(wp_transformed)
                        seg_copy['waypoints'] = transformed_waypoints
                
                transformed_segments.append(seg_copy)
            transformed['segments'] = transformed_segments
    
    # Transform start_pose if specified (common to all commands)
    if 'start_pose' in params and params['start_pose'] is not None:
        start_trf = SE3(params['start_pose'][0]/1000, 
                       params['start_pose'][1]/1000, 
                       params['start_pose'][2]/1000) * \
                    SE3.RPY(params['start_pose'][3:], unit='deg', order='xyz')
        start_wrf = tool_pose * start_trf
        transformed['start_pose'] = np.concatenate([
            start_wrf.t * 1000,
            start_wrf.rpy(unit='deg', order='xyz')
        ]).tolist()
    
    return transformed

#########################################################################
# Smooth Motion Commands and Robot Commands End Here
#########################################################################

# Acknowledgment system configuration
CLIENT_ACK_PORT = 5002  # Port where clients listen for acknowledgments
ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Command tracking
active_command_id = None
command_id_map = {}  # Maps command objects to their IDs

def send_acknowledgment(cmd_id: str, status: str, details: str = "", addr=None):
    """Send acknowledgment back to client"""
    if not cmd_id:
        return
        
    ack_message = f"ACK|{cmd_id}|{status}|{details}"
    
    # Send to the original sender if we have their address
    if addr:
        try:
            ack_socket.sendto(ack_message.encode('utf-8'), (addr[0], CLIENT_ACK_PORT))
        except Exception as e:
            print(f"Failed to send ack to {addr}: {e}")
    
    # Also broadcast to localhost in case the client is local
    try:
        ack_socket.sendto(ack_message.encode('utf-8'), ('127.0.0.1', CLIENT_ACK_PORT))
    except:
        pass

def parse_command_with_id(message: str) -> Tuple[Optional[str], str]:
    """
    Parse command ID if present.
    Format: [cmd_id|]COMMAND|params...
    Returns: (cmd_id or None, command_string)
    """
    # Clean up any logging artifacts
    if "ID:" in message or "):" in message:
        # Extract the actual command after these artifacts
        if "):" in message:
            message = message[message.rindex("):")+2:].strip()
        elif "ID:" in message:
            message = message[message.index("ID:")+3:].strip()
            # Remove any trailing parentheses or colons
            message = message.lstrip('):').strip()
    
    parts = message.split('|', 1)
    
    # Check if first part looks like a valid command ID (8 chars, alphanumeric)
    # IMPORTANT: Command IDs from uuid.uuid4()[:8] will contain lowercase letters/numbers
    # Actual commands are all uppercase, so exclude all-uppercase strings
    if (len(parts) > 1 and 
        len(parts[0]) == 8 and 
        parts[0].replace('-', '').isalnum() and 
        not parts[0].isupper()):  # This prevents "MOVEPOSE" from being treated as an ID
        return parts[0], parts[1]
    else:
        return None, message


# Create a new, empty command queue
command_queue = deque()

# --------------------------------------------------------------------------
# --- Test 1: Homing and Initial Setup
# --------------------------------------------------------------------------

# 1. Optionally start with the Home command (can be bypassed via PAROL6_NOAUTOHOME)
if not str(os.getenv("PAROL6_NOAUTOHOME", "0")).lower() in ("1", "true", "yes", "on"):
    command_queue.append(HomeCommand())
else:
    print("PAROL6_NOAUTOHOME is set; skipping auto-home on startup.")

# --- State variable for the currently running command ---
active_command = None
e_stop_active = False

# Use deque for an efficient FIFO queue
incoming_command_buffer = deque()
# Timestamp of the last processed network command
last_command_time = 0
# Cooldown period in seconds to prevent command flooding
COMMAND_COOLDOWN_S = 0.1 # 100ms

# Set interval
timer = Timer(interval=INTERVAL_S, warnings=False, precise=True)

# ============================================================================
# MODIFIED MAIN LOOP WITH ACKNOWLEDGMENTS
# ============================================================================

timer = Timer(interval=INTERVAL_S, warnings=False, precise=True)
prev_time = 0

while timer.elapsed_time < 1100000:
    
    # --- Connection Handling (non-blocking) ---
    if ser is None or not ser.is_open:
        now = time.time()
        if now - last_reconnect_attempt > 1.0:
            print("Serial port not open. Attempting to reconnect...")
            last_reconnect_attempt = now
            try:
                # Load port from com_port.txt if not already set
                if not com_port_str:
                    try:
                        with open("com_port.txt", "r") as f:
                            com_port_str = f.read().strip()
                    except FileNotFoundError:
                        com_port_str = None

                if com_port_str:
                    ser = serial.Serial(port=com_port_str, baudrate=3000000, timeout=0)
                    if ser.is_open:
                        print(f"Successfully reconnected to {com_port_str}")
            except serial.SerialException:
                ser = None
        # Do not block or continue; proceed to UDP handling every tick

    # =======================================================================
    # === NETWORK COMMAND RECEPTION WITH ID PARSING ===
    # =======================================================================
    try:
        while sock in select.select([sock], [], [], 0)[0]:
            data, addr = sock.recvfrom(1024)
            raw_message = data.decode('utf-8').strip()
            if raw_message:
                # Parse command ID if present
                cmd_id, message = parse_command_with_id(raw_message)
                
                parts = message.split('|')
                command_name = parts[0].upper()

                # Handle immediate response commands
                if command_name == 'STOP':
                    print("Received STOP command. Halting all motion and clearing queue.")
                    
                    # Cancel active command
                    if active_command and active_command_id:
                        send_acknowledgment(active_command_id, "CANCELLED", 
                                          "Stopped by user", addr)
                    active_command = None
                    active_command_id = None
                    
                    # Clear queue and notify about cancelled commands
                    for queued_cmd, (qid, qaddr) in list(command_id_map.items()):
                        if queued_cmd != active_command:
                            send_acknowledgment(qid, "CANCELLED", "Queue cleared by STOP", qaddr)
                    
                    command_queue.clear()
                    command_id_map.clear()
                    
                    # Stop robot
                    Command_out.value = 255
                    Speed_out[:] = [0] * 6
                    
                    # Send acknowledgment for STOP command itself
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Emergency stop executed", addr)

                elif command_name == 'GET_POSE':
                    q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
                    current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A
                    pose_flat = current_pose_matrix.flatten()
                    pose_str = ",".join(map(str, pose_flat))
                    response_message = f"POSE|{pose_str}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Pose data sent", addr)

                elif command_name == 'GET_ANGLES':
                    angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)]
                    angles_deg = np.rad2deg(angles_rad)
                    angles_str = ",".join(map(str, angles_deg))
                    response_message = f"ANGLES|{angles_str}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Angles data sent", addr)

                elif command_name == 'GET_IO':
                    io_status_str = ",".join(map(str, InOut_in[:5]))
                    response_message = f"IO|{io_status_str}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "IO data sent", addr)

                elif command_name == 'GET_GRIPPER':
                    gripper_status_str = ",".join(map(str, Gripper_data_in))
                    response_message = f"GRIPPER|{gripper_status_str}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Gripper data sent", addr)

                elif command_name == 'GET_SPEEDS':
                    speeds_str = ",".join(map(str, Speed_in))
                    response_message = f"SPEEDS|{speeds_str}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Speed data sent", addr)

                elif command_name == 'PING':
                    # Respond with PONG and ACK if cmd_id present
                    sock.sendto("PONG".encode('utf-8'), addr)
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "PONG", addr)

                elif command_name == 'GET_STATUS':
                    # Aggregate POSE, ANGLES, IO, GRIPPER into one frame
                    try:
                        q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
                        current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A
                        pose_flat = current_pose_matrix.flatten()
                        pose_str = ",".join(map(str, pose_flat))
                    except Exception:
                        pose_str = ",".join(["0"] * 16)

                    angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)]
                    angles_deg = np.rad2deg(angles_rad)
                    angles_str = ",".join(map(str, angles_deg))

                    io_status_str = ",".join(map(str, InOut_in[:5]))
                    gripper_status_str = ",".join(map(str, Gripper_data_in))

                    response_message = f"STATUS|POSE={pose_str}|ANGLES={angles_str}|IO={io_status_str}|GRIPPER={gripper_status_str}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Status data sent", addr)

                elif command_name == 'ENABLE':
                    enabled = True
                    disabled_reason = ""
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Controller enabled", addr)

                elif command_name == 'DISABLE':
                    enabled = False
                    disabled_reason = "Disabled by user"
                    # Cancel active command
                    if active_command and active_command_id:
                        send_acknowledgment(active_command_id, "CANCELLED", "Disabled by user", addr)
                    active_command = None
                    active_command_id = None
                    # Cancel queued commands
                    for queued_cmd, (qid, qaddr) in list(command_id_map.items()):
                        send_acknowledgment(qid, "CANCELLED", "Controller disabled", qaddr)
                    command_queue.clear()
                    command_id_map.clear()
                    # Stop robot motion
                    Command_out.value = 255
                    Speed_out[:] = [0] * 6
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Controller disabled", addr)

                elif command_name == 'CLEAR_ERROR':
                    soft_error = False
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "Errors cleared", addr)

                elif command_name == 'SET_PORT' and len(parts) >= 2:
                    new_port = parts[1].strip()
                    if new_port:
                        # Persist and trigger reconnection
                        try:
                            with open("com_port.txt", "w") as f:
                                f.write(new_port)
                        except Exception as e:
                            if cmd_id:
                                send_acknowledgment(cmd_id, "FAILED", f"Could not write com_port.txt: {e}", addr)
                            # Do not fall through to queuing
                            continue
                        try:
                            if ser and ser.is_open:
                                ser.close()
                        except Exception:
                            pass
                        com_port_str = new_port
                        ser = None  # main loop will reconnect
                        if cmd_id:
                            send_acknowledgment(cmd_id, "COMPLETED", f"Port set to {new_port}; reconnecting...", addr)
                    else:
                        if cmd_id:
                            send_acknowledgment(cmd_id, "FAILED", "No port specified", addr)

                else:
                    # Queue command for processing
                    incoming_command_buffer.append((raw_message, addr))

    except Exception as e:
        print(f"Network receive error: {e}")

    # =======================================================================
    # === PROCESS COMMANDS FROM BUFFER WITH ACKNOWLEDGMENTS ===
    # =======================================================================
    current_time = time.time()
    if incoming_command_buffer and (current_time - last_command_time) > COMMAND_COOLDOWN_S and not e_stop_active:
        raw_message, addr = incoming_command_buffer.popleft()
        last_command_time = current_time
        
        # Parse command ID
        cmd_id, message = parse_command_with_id(raw_message)
        print(f"Processing command{' (ID: ' + cmd_id + ')' if cmd_id else ''}: {message[:50]}...")
        
        parts = message.split('|')
        command_name = parts[0].upper()
        
        # Gate motion commands when controller is disabled
        if not enabled and command_name in {'MOVEPOSE','MOVEJOINT','MOVECART','JOG','MULTIJOG','CARTJOG',
                                            'SMOOTH_CIRCLE','SMOOTH_ARC_CENTER','SMOOTH_ARC_PARAM',
                                            'SMOOTH_SPLINE','SMOOTH_HELIX','SMOOTH_BLEND','HOME'}:
            if cmd_id:
                send_acknowledgment(cmd_id, "FAILED", f"Controller disabled{(': ' + disabled_reason) if disabled_reason else ''}", addr)
            # Skip processing this command
            continue
        
        # Variable to track if command was successfully queued
        command_queued = False
        command_obj = None
        error_details = ""

        # Parse and create command objects
        try:
            if command_name == 'MOVEPOSE' and len(parts) == 9:
                pose_vals = [float(p) for p in parts[1:7]]
                duration = None if parts[7].upper() == 'NONE' else float(parts[7])
                speed = None if parts[8].upper() == 'NONE' else float(parts[8])
                command_obj = MovePoseCommand(pose=pose_vals, duration=duration, velocity_percent=speed)
                command_queued = True
                
            elif command_name == 'MOVEJOINT' and len(parts) == 9:
                joint_vals = [float(p) for p in parts[1:7]]
                duration = None if parts[7].upper() == 'NONE' else float(parts[7])
                speed = None if parts[8].upper() == 'NONE' else float(parts[8])
                command_obj = MoveJointCommand(target_angles=joint_vals, duration=duration, velocity_percent=speed)
                command_queued = True
            
            elif command_name in ['SMOOTH_CIRCLE', 'SMOOTH_ARC_CENTER', 'SMOOTH_ARC_PARAM', 
                                 'SMOOTH_SPLINE', 'SMOOTH_HELIX', 'SMOOTH_BLEND']:
                command_obj = parse_smooth_motion_commands(parts)
                if command_obj:
                    command_queued = True
                else:
                    error_details = "Failed to parse smooth motion parameters"
            
            elif command_name == 'MOVECART' and len(parts) == 9:
                pose_vals = [float(p) for p in parts[1:7]]
                duration = None if parts[7].upper() == 'NONE' else float(parts[7])
                speed = None if parts[8].upper() == 'NONE' else float(parts[8])
                command_obj = MoveCartCommand(pose=pose_vals, duration=duration, velocity_percent=speed)
                command_queued = True
            
            elif command_name == 'DELAY' and len(parts) == 2:
                duration = float(parts[1])
                command_obj = DelayCommand(duration=duration)
                command_queued = True
            
            elif command_name == 'HOME':
                command_obj = HomeCommand()
                command_queued = True
                
            elif command_name == 'CARTJOG' and len(parts) == 5:
                frame, axis, speed, duration = parts[1].upper(), parts[2], float(parts[3]), float(parts[4])
                command_obj = CartesianJogCommand(frame=frame, axis=axis, speed_percentage=speed, duration=duration)
                command_queued = True
                
            elif command_name == 'JOG' and len(parts) == 5:
                joint_idx, speed = int(parts[1]), float(parts[2])
                duration = None if parts[3].upper() == 'NONE' else float(parts[3])
                distance = None if parts[4].upper() == 'NONE' else float(parts[4])
                command_obj = JogCommand(joint=joint_idx, speed_percentage=speed, duration=duration, distance_deg=distance)
                command_queued = True
            
            elif command_name == 'MULTIJOG' and len(parts) == 4:
                joint_indices = [int(j) for j in parts[1].split(',')]
                speeds = [float(s) for s in parts[2].split(',')]
                duration = float(parts[3])
                command_obj = MultiJogCommand(joints=joint_indices, speed_percentages=speeds, duration=duration)
                command_queued = True
                
            elif command_name == 'PNEUMATICGRIPPER' and len(parts) == 3:
                action, port = parts[1].lower(), int(parts[2])
                command_obj = GripperCommand(gripper_type='pneumatic', action=action, output_port=port)
                command_queued = True
                
            elif command_name == 'ELECTRICGRIPPER' and len(parts) == 5:
                action = None if parts[1].upper() == 'NONE' or parts[1].upper() == 'MOVE' else parts[1].lower()
                pos, spd, curr = int(parts[2]), int(parts[3]), int(parts[4])
                command_obj = GripperCommand(gripper_type='electric', action=action, position=pos, speed=spd, current=curr)
                command_queued = True
            
            else:
                error_details = f"Unknown or malformed command: {command_name}"
                
        except Exception as e:
            error_details = f"Error parsing command: {str(e)}"
            command_queued = False
        
        # Handle command queueing and acknowledgments
        if command_queued and command_obj:
            # Check if command is initially valid
            if hasattr(command_obj, 'is_valid') and not command_obj.is_valid:
                if cmd_id:
                    send_acknowledgment(cmd_id, "INVALID", 
                                       "Command failed validation", addr)
            else:
                # Add to queue
                command_queue.append(command_obj)
                if cmd_id:
                    command_id_map[command_obj] = (cmd_id, addr)
                    send_acknowledgment(cmd_id, "QUEUED", 
                                       f"Position {len(command_queue)} in queue", addr)
        else:
            # Command was not queued
            if cmd_id:
                send_acknowledgment(cmd_id, "INVALID", error_details, addr)
            print(f"Warning: {error_details}")

    # =======================================================================
    # === MAIN EXECUTION LOGIC WITH ACKNOWLEDGMENTS ===
    # =======================================================================
    try:
        # --- E-Stop Handling ---
        if InOut_in[4] == 0:  # E-Stop pressed
            if not e_stop_active:
                cancelled_command_info = "None"
                if active_command is not None:
                    cancelled_command_info = type(active_command).__name__
                    if active_command_id:
                        send_acknowledgment(active_command_id, "CANCELLED", 
                                          "E-Stop activated")
                
                # Cancel all queued commands
                for cmd_obj in command_queue:
                    if cmd_obj in command_id_map:
                        cmd_id, addr = command_id_map[cmd_obj]
                        send_acknowledgment(cmd_id, "CANCELLED", "E-Stop activated", addr)
                
                # Cancel all buffered but unprocessed commands
                for raw_message, addr in incoming_command_buffer:
                    cmd_id, _ = parse_command_with_id(raw_message)
                    if cmd_id:
                        send_acknowledgment(cmd_id, "CANCELLED", "E-Stop activated - command not processed", addr)
                
                print(f"E-STOP TRIGGERED! Active command '{cancelled_command_info}' cancelled.")
                print("Release E-Stop and press 'e' to re-enable.")
                e_stop_active = True
            
            Command_out.value = 102
            Speed_out[:] = [0] * 6
            Gripper_data_out[3] = 0
            active_command = None
            active_command_id = None
            command_queue.clear()
            command_id_map.clear()
            incoming_command_buffer.clear()
            
        elif e_stop_active:
            # Waiting for re-enable
            if keyboard and keyboard.is_pressed('e'):
                print("Re-enabling robot...")
                Command_out.value = 101
                e_stop_active = False
            else:
                Command_out.value = 255
                Speed_out[:] = [0] * 6
                Position_out[:] = Position_in[:]
                
        else:
            # --- Normal Command Processing ---
            
            # Start new command if none active
            if active_command is None and command_queue:
                new_command = command_queue.popleft()
                
                # Get command ID and address if tracked
                cmd_info = command_id_map.get(new_command, (None, None))
                new_cmd_id, new_addr = cmd_info
                
                # Initial validation
                if hasattr(new_command, 'is_valid') and not new_command.is_valid:
                    # Command was invalid from the start
                    if new_cmd_id:
                        send_acknowledgment(new_cmd_id, "INVALID", 
                                        "Initial validation failed", new_addr)
                    if new_command in command_id_map:
                        del command_id_map[new_command]
                    continue  # Skip to next command
                
                # Prepare command
                if hasattr(new_command, 'prepare_for_execution'):
                    try:
                        new_command.prepare_for_execution(current_position_in=Position_in)
                    except Exception as e:
                        print(f"Command preparation failed: {e}")
                        if hasattr(new_command, 'is_valid'):
                            new_command.is_valid = False
                        if hasattr(new_command, 'error_message'):
                            new_command.error_message = str(e)
                
                # Check if still valid after preparation
                if hasattr(new_command, 'is_valid') and not new_command.is_valid:
                    # Failed during preparation
                    error_msg = "Failed during preparation"
                    if hasattr(new_command, 'error_message'):
                        error_msg = new_command.error_message
                    
                    if new_cmd_id:
                        send_acknowledgment(new_cmd_id, "FAILED", error_msg, new_addr)
                    
                    # Clean up
                    if new_command in command_id_map:
                        del command_id_map[new_command]
                else:
                    # Command is valid, make it active
                    active_command = new_command
                    active_command_id = new_cmd_id
                    
                    if new_cmd_id:
                        send_acknowledgment(new_cmd_id, "EXECUTING", 
                                        f"Starting {type(new_command).__name__}", new_addr)
            
            # Execute active command
            if active_command:
                try:
                    is_done = active_command.execute_step(
                        Position_in=Position_in,
                        Homed_in=Homed_in,
                        Speed_out=Speed_out,
                        Command_out=Command_out,
                        Gripper_data_out=Gripper_data_out,
                        InOut_out=InOut_out,
                        InOut_in=InOut_in,
                        Gripper_data_in=Gripper_data_in,
                        Position_out=Position_out  # Add this if needed
                    )
                    
                    if is_done:
                        # Command completed
                        if active_command_id:
                            # Check for error state in smooth motion commands
                            if hasattr(active_command, 'error_state') and active_command.error_state:
                                error_msg = getattr(active_command, 'error_message', 'Command failed during execution')
                                send_acknowledgment(active_command_id, "FAILED", error_msg)
                            else:
                                send_acknowledgment(active_command_id, "COMPLETED", 
                                                f"{type(active_command).__name__} finished successfully")
                        
                        # Clean up
                        if active_command in command_id_map:
                            del command_id_map[active_command]
                        
                        active_command = None
                        active_command_id = None
                        
                except Exception as e:
                    # Command execution error
                    print(f"Command execution error: {e}")
                    if active_command_id:
                        send_acknowledgment(active_command_id, "FAILED", 
                                          f"Execution error: {str(e)}")
                    
                    # Clean up
                    if active_command in command_id_map:
                        del command_id_map[active_command]
                    
                    active_command = None
                    active_command_id = None
                    
            else:
                # No active command - idle
                Command_out.value = 255
                Speed_out[:] = [0] * 6
                Position_out[:] = Position_in[:]

        # --- Communication with Robot ---
        s = Pack_data(Position_out, Speed_out, Command_out.value, 
                     Affected_joint_out, InOut_out, Timeout_out, Gripper_data_out)
        if ser is not None and ser.is_open:
            for chunk in s:
                ser.write(chunk)
            Get_data(Position_in, Speed_in, Homed_in, InOut_in, Temperature_error_in, 
                    Position_error_in, Timeout_error, Timing_data_in, XTR_data, Gripper_data_in)
        else:
            # Serial not available; skip IO this cycle
            pass

    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        
        # Send failure acknowledgments for active command
        if active_command_id:
            send_acknowledgment(active_command_id, "FAILED", "Serial communication lost")
        
        if ser:
            ser.close()
        ser = None
        active_command = None
        active_command_id = None

    timer.checkpt()
