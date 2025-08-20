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
import keyboard
from spatialmath.base import trinterp
from collections import namedtuple, deque
import GUI.files.PAROL6_ROBOT as PAROL6_ROBOT

# Set interval
INTERVAL_S = 0.01
prev_time = 0

logging.basicConfig(level = logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
    datefmt='%H:%M:%S'
)
logging.disable(logging.DEBUG)


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
            try:
                com_port = input("Enter the COM port (e.g., COM9): ")
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

data_buffer = [None]*255 #Here save all data after data length byte
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
            if current_reach >= max_reach_threshold:
                print(f"Reach limit exceeded: {current_reach:.3f} >= {max_reach_threshold:.3f}")
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

    while (ser.inWaiting() > 0):
        input_byte = ser.read()

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
            speed_steps_per_sec = int(np.interp(abs(self.speed_percentage), [0, 100], [0, PAROL6_ROBOT.Joint_max_speed[self.joint_index] * 2]))

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

            # Calculate speed in steps/sec
            speed_steps_per_sec = int(np.interp(speed_percentage, [0, 100], [0, PAROL6_ROBOT.Joint_max_speed[joint_index]]))
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
    def __init__(self, frame, axis, speed_percentage=50, duration=1.5, **kwargs):
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
        self.end_time = time.time() + self.duration
        self.is_valid = True

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """
        Checks if the delay duration has passed and keeps the robot idle.
        This method is called on every loop cycle (~{INTERVAL_S}s).
        """
        if self.is_finished or not self.is_valid:
            return True

        # --- A. Keep the robot idle during the delay ---
        Command_out.value = 255 # Set command to idle
        Speed_out[:] = [0] * 6  # Set all speeds to zero

        # --- B. Check for completion ---
        if time.time() >= self.end_time:
            print(f"Delay finished after {self.duration} seconds.")
            self.is_finished = True
        
        return self.is_finished


# Create a new, empty command queue
command_queue = deque()

# --------------------------------------------------------------------------
# --- Test 1: Homing and Initial Setup
# --------------------------------------------------------------------------

# 1. Start with the mandatory Home command.
command_queue.append(lambda: HomeCommand())

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

# ==========================================================
# === MODIFIED MAIN LOOP WITH COMMAND QUEUE ================
# ==========================================================
timer = Timer(interval=INTERVAL_S, warnings=False, precise=True)
prev_time = 0

while timer.elapsed_time < 1100000:
    
    # --- Connection Handling ---
    if ser is None or not ser.is_open:
        # This block handles reconnections if the device is unplugged
        print("Serial port not open. Attempting to reconnect...")
        try:
            # CORRECTED LOGIC: Use the known, working com_port_str
            ser = serial.Serial(port=com_port_str, baudrate=3000000, timeout=0)
            if ser.is_open:
                print(f"Successfully reconnected to {com_port_str}")
        except serial.SerialException as e:
            # If reconnection fails, wait and try again on the next loop
            ser = None
            time.sleep(1) 
        continue # Skip the rest of this loop iteration until connected

    # =======================================================================
    # === UPDATED BLOCK to listen for network commands ===
    # =======================================================================
    try:
        # Use a while loop with select to read all pending data from the socket buffer
        while sock in select.select([sock], [], [], 0)[0]:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8').strip()
            if message: # Ensure we don't buffer empty messages
                #print(f"Buffering command from {addr}: {message}")
                # Append the full message and its origin address to the queue
                parts=message.split('|')
                command_name=parts[0].upper()

                if command_name == 'STOP':
                    print("Received STOP command. Halting all motion and clearing queue.")
                    # Cancel the currently running command
                    active_command = None
                    # Clear all pending commands
                    command_queue.clear()
                    # Immediately send an idle/stop command to the robot hardware
                    Command_out.value = 255 # Idle command
                    Speed_out[:] = [0] * 6  # Set all speeds to zero

                elif command_name == 'GET_POSE':
                    # Calculate the current pose from the robot's joint angles
                    q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)])
                    current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A # .A gets the 4x4 numpy array
                    
                    # Flatten the matrix into a 1D array and convert it to a comma-separated string
                    pose_flat = current_pose_matrix.flatten()
                    pose_str = ",".join(map(str, pose_flat))
                    response_message = f"POSE|{pose_str}"
                    
                    # Send the formatted pose string back to the address that sent the request
                    sock.sendto(response_message.encode('utf-8'), addr)
                    print(f"Responded with current pose to {addr}")
                    # Note: We don't queue a command for this, we just respond immediately.

                elif command_name == 'GET_ANGLES':
                    # Convert current joint positions (in steps) to degrees
                    angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)]
                    angles_deg = np.rad2deg(angles_rad)
                    
                    # Format the response string: "ANGLES|j1,j2,j3,..."
                    angles_str = ",".join(map(str, angles_deg))
                    response_message = f"ANGLES|{angles_str}"
                    
                    # Send the response back to the client
                    sock.sendto(response_message.encode('utf-8'), addr)
                    print(f"Responded with current joint angles to {addr}")

                elif command_name == 'GET_IO':
                    # Format the I/O data into a comma-separated string
                    # The InOut_in list is [IN1, IN2, OUT1, OUT2, ESTOP, ...]
                    io_status_str = ",".join(map(str, InOut_in[:5]))
                    response_message = f"IO|{io_status_str}"
                    
                    # Send the response back to the client
                    sock.sendto(response_message.encode('utf-8'), addr)
                    print(f"Responded with I/O status to {addr}")

                elif command_name == 'GET_GRIPPER':
                    # Format the gripper data into a comma-separated string
                    gripper_status_str = ",".join(map(str, Gripper_data_in))
                    response_message = f"GRIPPER|{gripper_status_str}"
                    
                    # Send the response back to the client
                    sock.sendto(response_message.encode('utf-8'), addr)
                    print(f"Responded with gripper status to {addr}")

                elif command_name == 'GET_SPEEDS':
                    # The Speed_in variable is already in steps/sec from the firmware
                    speeds_str = ",".join(map(str, Speed_in))
                    response_message = f"SPEEDS|{speeds_str}"
                    
                    # Send the response back to the client
                    sock.sendto(response_message.encode('utf-8'), addr)
                    print(f"Responded with current joint speeds to {addr}")

                else:
                    incoming_command_buffer.append((message, addr))

    except Exception as e:
        print(f"Network receive error: {e}")
    # Depending on the error, you might want to handle it more gracefully

    # =======================================================================
    # === REVISED BLOCK 2: PROCESS ONE COMMAND FROM THE BUFFER            ===
    # =======================================================================
    current_time = time.time()
    # Check if the buffer is not empty AND the cooldown period has passed
    if incoming_command_buffer and (current_time - last_command_time) > COMMAND_COOLDOWN_S:
        # --- PROCESS THE OLDEST COMMAND IN THE QUEUE ---
        message, addr = incoming_command_buffer.popleft() # Get the oldest command (FIFO)
        last_command_time = current_time # Reset the cooldown timer
        print(f"Processing command: {message}")
        
        parts = message.split('|')
        command_name = parts[0].upper()
        # --- PROCESS THE VALID COMMAND ---

        # --- Parse the command and add it to the queue ---
        # Protocol: MOVEPOSE|x|y|z|r|p|y|duration|speed_percentage
        if command_name == 'MOVEPOSE' and len(parts) == 9:
            pose_vals = [float(p) for p in parts[1:7]]
            duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            speed = None if parts[8].upper() == 'NONE' else float(parts[8])
            command_queue.append(
                lambda p=pose_vals, d=duration, s=speed: MovePoseCommand(pose=p, duration=d, velocity_percent=s)
            )
        # Protocol: MOVEJOINT|j1|j2|j3|j4|j5|j6|duration|speed_percentage
        elif command_name == 'MOVEJOINT' and len(parts) == 9:
            joint_vals = [float(p) for p in parts[1:7]]
            duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            speed = None if parts[8].upper() == 'NONE' else float(parts[8])
            command_queue.append(
                lambda j=joint_vals, d=duration, s=speed: MoveJointCommand(target_angles=j, duration=d, velocity_percent=s)
            )
        
        elif command_name == 'DELAY' and len(parts) == 2:
            duration = float(parts[1])
            command_queue.append(lambda d=duration: DelayCommand(duration=d))
        
        elif command_name == 'HOME':
            print("Queueing Home command.")
            command_queue.append(lambda: HomeCommand())

        elif command_name == 'CARTJOG' and len(parts) == 5:
            frame, axis, speed, duration = parts[1].upper(), parts[2], float(parts[3]), float(parts[4])
            print(f"Queueing CartesianJog: {frame} {axis}")
            command_queue.append(lambda f=frame, a=axis, s=speed, d=duration: CartesianJogCommand(frame=f, axis=a, speed_percentage=s, duration=d))

        elif command_name == 'JOG' and len(parts) == 5:
            joint_idx, speed = int(parts[1]), float(parts[2])
            duration = None if parts[3].upper() == 'NONE' else float(parts[3])
            distance = None if parts[4].upper() == 'NONE' else float(parts[4])
            print(f"Queueing Jog for joint {joint_idx+1}")
            command_queue.append(lambda j=joint_idx, s=speed, d=duration, dist=distance: JogCommand(joint=j, speed_percentage=s, duration=d, distance_deg=dist))
        
        # Protocol: MOVECART|x|y|z|r|p|y|duration|speed_percentage
        elif command_name == 'MOVECART' and len(parts) == 9:
            pose_vals = [float(p) for p in parts[1:7]]
            duration = None if parts[7].upper() == 'NONE' else float(parts[7])
            speed = None if parts[8].upper() == 'NONE' else float(parts[8])
            print(f"Queueing MoveCart to {pose_vals}.")
            command_queue.append(
                lambda p=pose_vals, d=duration, s=speed: MoveCartCommand(pose=p, duration=d, velocity_percent=s)
            )

        # Protocol: MULTIJOG|joint1,joint2|speed1,speed2|duration
        elif command_name == 'MULTIJOG' and len(parts) == 4:
            try:
                # Parse comma-separated strings into lists of numbers
                joint_indices = [int(j) for j in parts[1].split(',')]
                speeds = [float(s) for s in parts[2].split(',')]
                duration = float(parts[3])
                
                print(f"Queueing MultiJog for joints {joint_indices}")
                command_queue.append(
                    lambda js=joint_indices, spds=speeds, d=duration: MultiJogCommand(joints=js, speed_percentages=spds, duration=d)
                )
            except ValueError:
                print(f"Warning: Malformed MULTIJOG command: {message}")

        elif command_name == 'PNEUMATICGRIPPER' and len(parts) == 3:
            action, port = parts[1].lower(), int(parts[2])
            print(f"Queueing Pneumatic Gripper command: {action} on port {port}")
            command_queue.append(lambda act=action, p=port: GripperCommand(gripper_type='pneumatic', action=act, output_port=p))

        elif command_name == 'ELECTRICGRIPPER' and len(parts) == 5:
            action = None if parts[1].upper() == 'NONE' or parts[1].upper() == 'MOVE' else parts[1].lower()
            pos, spd, curr = int(parts[2]), int(parts[3]), int(parts[4])
            print(f"Queueing Electric Gripper command: action={action}, pos={pos}")
            command_queue.append(lambda act=action, p=pos, s=spd, c=curr: GripperCommand(gripper_type='electric', action=act, position=p, speed=s, current=c))

        else:
            print(f"Warning: Received unknown or malformed command: {message}")

    # --- Main Logic Block (only runs if ser is open) ---
    try:
        # --- A. High-Priority Safety and State Checks ---
        
        # Check 1: Is the E-Stop button currently pressed?
        if InOut_in[4] == 0:
            if not e_stop_active:
                # --- MODIFIED: Log the cancelled command ---
                cancelled_command_info = "None (robot was idle)"
                if active_command is not None:
                    # Get the class name of the command for logging
                    cancelled_command_info = type(active_command).__name__
                
                print(f"E-STOP TRIGGERED! Active command '{cancelled_command_info}' and all queued commands have been cancelled.")
                print("Release E-Stop and press 'e' to re-enable.")
                # --- END MODIFICATION ---

                e_stop_active = True
            
            # Continuously send the "Disable" command and clear any active tasks
            Command_out.value = 102
            Speed_out[:] = [0] * 6

            # --- ADDED LINE FOR GRIPPER SAFETY ---
            # Set the gripper command byte to 0 to deactivate it.
            # This corresponds to: [activate=0, action_status=0, ...]
            Gripper_data_out[3] = 0

            active_command = None
            command_queue.clear()
        
        # Check 2: Has the E-Stop been released, but we are still in a disabled state?
        elif e_stop_active:
            # The robot is now waiting for the user to explicitly re-enable it.
            if keyboard.is_pressed('e'):
                print("Re-enabling robot...")
                Command_out.value = 101  # Send the "Enable" command
                e_stop_active = False    # Return to normal operational state
            else:
                # While waiting, keep the robot idle.
                Command_out.value = 255
                Speed_out[:] = [0] * 6
        
        # Check 3: If not E-Stopped, run normal command processing.
        else:
            # --- B. Process Network Commands (if any) ---
            try:
                ready_to_read, _, _ = select.select([sock], [], [], 0)
                if ready_to_read:
                    data, addr = sock.recvfrom(1024)
                    message = data.decode('utf-8')
                    print(f"Received command from {addr}: {message}")
                    # ... (Your network command parsing logic goes here) ...

            except Exception as e:
                print(f"Network error: {e}")

            # --- C. Process Command Queue ---
            if active_command is None and command_queue:
                command_creator_function = command_queue.popleft()
                new_command_object = command_creator_function()

                # First, check if the command was validated successfully during its initialization.
                if new_command_object.is_valid:
                    # Only if it's valid, run the 'prepare' step (if it exists).
                    if hasattr(new_command_object, 'prepare_for_execution'):
                        new_command_object.prepare_for_execution(
                            current_position_in=Position_in
                        )

                    # After preparation, the command could have become invalid. Check again.
                    if new_command_object.is_valid:
                        active_command = new_command_object
                    else:
                        print("Skipping invalid command: failed during preparation step.")
                else:
                    # This handles commands that fail validation in their __init__ method.
                    print("Skipping invalid command: failed during initialization.")

            if active_command:
                is_done = active_command.execute_step(
                    Position_in=Position_in,
                    Homed_in=Homed_in,
                    Speed_out=Speed_out,
                    Command_out=Command_out,
                    Gripper_data_out=Gripper_data_out,
                    InOut_out=InOut_out,
                    InOut_in=InOut_in,  # Added for E-Stop status
                    Gripper_data_in=Gripper_data_in # The missing argument
                )
                if is_done:
                    active_command = None
            else:
                Command_out.value = 255
                Speed_out[:] = [0] * 6

        # --- D. Communication with Robot (This always runs) ---
        s = Pack_data(Position_out, Speed_out, Command_out.value, Affected_joint_out, InOut_out, Timeout_out, Gripper_data_out)
        for chunk in s:
            ser.write(chunk)
            
        Get_data(Position_in, Speed_in, Homed_in, InOut_in, Temperature_error_in, Position_error_in, Timeout_error, Timing_data_in, XTR_data, Gripper_data_in)

    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        if ser:
            ser.close()
        ser = None
        active_command = None

    # This enforces the 100Hz loop frequency, just like the original code.
    timer.checkpt()
