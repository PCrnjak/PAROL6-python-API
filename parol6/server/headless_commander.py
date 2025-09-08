'''
A full fledged "API" for the PAROL6 robot. To use this, you should pair it with the "robot_api.py" where you can import commands
from said file and use them anywhere within your code. This Python script will handle sending and performing all the commands
to the PAROL6 robot, as well as E-Stop functionality and safety limitations.
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
import argparse
import sys
import json
from typing import Optional, Tuple
from spatialmath.base import trinterp
from collections import namedtuple, deque
from pathlib import Path

# Ensure both package dir (parol6) and project root are on sys.path to import PAROL6_ROBOT and others
_pkg_dir = Path(__file__).parent.parent        # .../parol6
_root_dir = Path(__file__).parents[2]          # .../PAROL6-python-API
for _p in (str(_root_dir), str(_pkg_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Robust import of robot constants/kinematics
try:
    import PAROL6_ROBOT  # from project root
except ModuleNotFoundError:
    # Fallback: load directly from file path to handle non-standard execution contexts
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        _robot_path = (_root_dir / "PAROL6_ROBOT.py")
        _spec = spec_from_file_location("PAROL6_ROBOT", str(_robot_path))
        if _spec and _spec.loader:
            PAROL6_ROBOT = module_from_spec(_spec)
            sys.modules["PAROL6_ROBOT"] = PAROL6_ROBOT
            _spec.loader.exec_module(PAROL6_ROBOT)  # type: ignore[attr-defined]
        else:
            raise
    except Exception as e:
        print(f"[FATAL] Unable to import PAROL6_ROBOT from {_robot_path}: {e}", file=sys.stderr)
        raise

from smooth_motion import CircularMotion, SplineMotion, MotionBlender, SCurveProfile, QuinticPolynomial, MotionConstraints
from gcode import GcodeInterpreter

# Import all command classes from the modular commands directory
from commands import (
    # Helper class
    CommandValue,
    # Basic commands
    HomeCommand, JogCommand, MultiJogCommand,
    # Cartesian commands  
    CartesianJogCommand, MovePoseCommand, MoveCartCommand,
    # Joint commands
    MoveJointCommand,
    # Gripper commands
    GripperCommand,
    # Utility commands
    DelayCommand,
    # Smooth motion commands
    SmoothTrajectoryCommand, SmoothCircleCommand,
    SmoothArcCenterCommand, SmoothArcParamCommand,
    SmoothHelixCommand, SmoothSplineCommand,
    SmoothBlendCommand, SmoothWaypointsCommand
)

# Set interval
INTERVAL_S = 0.01
prev_time = 0

# Enable optional fake-serial simulation for hardware-free tests
FAKE_SERIAL = str(os.getenv("PAROL6_FAKE_SERIAL", "0")).lower() in ("1", "true", "yes", "on")

# Streaming toggle: STREAM|ON enables zero-queue latest-wins for JOG/CARTJOG; STREAM|OFF disables
stream_mode = False

# Global verbosity level - can be changed programmatically
GLOBAL_LOG_LEVEL = logging.INFO

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

# Logging configuration
def setup_logging(verbosity_level=None):
    """Configure logging with the specified verbosity level.
    
    Args:
        verbosity_level: Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' or None
    """
    global GLOBAL_LOG_LEVEL
    
    if verbosity_level:
        GLOBAL_LOG_LEVEL = getattr(logging, verbosity_level.upper(), logging.INFO)
    
    # Setup basic logging configuration
    logging.basicConfig(
        level=GLOBAL_LOG_LEVEL,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration if already configured
    )
    
    # Module-specific logger
    logger = logging.getLogger(__name__)
    logger.setLevel(GLOBAL_LOG_LEVEL)
    
    return logger

# Parse command-line arguments
def parse_arguments():
    """Parse command-line arguments for the headless commander."""
    parser = argparse.ArgumentParser(
        description='PAROL6 Headless Commander - Robot control server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Verbosity levels:
  DEBUG    : All messages including detailed debug information
  INFO     : Normal operation messages (default)
  WARNING  : Only warnings and errors
  ERROR    : Only error messages
  CRITICAL : Only critical error messages
  
Examples:
  python headless_commander.py --verbose              # Enable DEBUG level
  python headless_commander.py --log-level DEBUG      # Same as --verbose
  python headless_commander.py --log-level WARNING    # Only show warnings and above
  python headless_commander.py --quiet                # Minimal output (WARNING level)
        '''
    )
    
    # Verbosity options (mutually exclusive)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (DEBUG level)'
    )
    verbosity_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Minimize output (WARNING level)'
    )
    verbosity_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set specific logging level'
    )
    
    parser.add_argument(
        '--com-port',
        type=str,
        help='Specify COM port for robot connection (e.g., COM5 or /dev/ttyACM0)'
    )
    
    parser.add_argument(
        '--no-auto-home',
        action='store_true',
        help='Disable automatic homing on startup'
    )
    
    return parser.parse_args()

# Initialize command-line arguments and logging
args = parse_arguments()

# Set log level from command line args
if args.verbose:
    log_level = 'DEBUG'
elif args.quiet:
    log_level = 'WARNING'
elif args.log_level:
    log_level = args.log_level
else:
    log_level = 'INFO'

# Setup logging with determined level
logger = setup_logging(log_level)


my_os = platform.system()
if my_os == "Windows":
    # Load COM port from saved configuration
    try:
        with open("com_port.txt", "r") as f:
            com_port_str = f.read().strip()
            ser = serial.Serial(port=com_port_str, baudrate=3000000, timeout=0)
            logger.info(f"Connected to saved COM port: {com_port_str}")
    except (FileNotFoundError, serial.SerialException):
        # Fallback to user input for COM port
        while True:
            try:
                com_port = input("Enter the COM port (e.g., COM9): ")
                ser = serial.Serial(port=com_port, baudrate=3000000, timeout=0)
                logger.info(f"Successfully connected to {com_port}")
                # Cache successful port for future runs
                with open("com_port.txt", "w") as f:
                    f.write(com_port)
                break
            except serial.SerialException:
                logger.error(f"Could not open port {com_port}. Please try again.")

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
input_byte = 0 # Serial byte buffer

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

#Setup IP address and Simulator port
ip = "127.0.0.1" #Loopback address
port = 5001
# UDP server setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))
logger.info(f'Start listening to {ip}:{port}')
START_TIME = time.time()

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
    #logging.debug(test_list)

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
    # One-shot command reset for calibration and error clearing
    # ==========================================================
    # Reset mode to normal after calibration/error clear commands
    if Gripper_data_out[4] == 1 or Gripper_data_out[4] == 2:
        Gripper_data_out[4] = 0
    # ==========================================================
    
    # Gripper ID
    test_list.append(bytes([Gripper_data[5]]))
 
    # CRC byte
    test_list.append(bytes([CRC_byte]))

    # END bytes
    test_list.append((end_bytes))
    
    #logging.debug(test_list)
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
                #logging.debug("data len we got from robot packet= ")
                #logging.debug(input_byte)
                #logging.debug("good start for DATA that we received at PC")
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
            # Valid start sequence detected
            data_buffer[data_counter] = input_byte
            if (data_counter == data_len - 1):

                #logging.debug("Data len PC")
                #logging.debug(data_len)
                #logging.debug("End bytes are:")
                #logging.debug(data_buffer[data_len -1])
                #logging.debug(data_buffer[data_len -2])

                # End sequence validation and data processing 
                if (data_buffer[data_len -1] == end_cond2_byte and data_buffer[data_len - 2] == end_cond1_byte):

                    #logging.debug("GOOD END CONDITION PC")
                    #logging.debug("I UNPACKED RAW DATA RECEIVED FROM THE ROBOT")
                    Unpack_data(data_buffer, Position_in,Speed_in,Homed_in,InOut_in,Temperature_error_in,Position_error_in,Timeout_error,Timing_data_in,
                    XTR_data,Gripper_data_in)
                    #logging.debug("DATA UNPACK FINISHED")
                    # ako su dobri izračunaj crc
                    # if crc dobar raspakiraj podatke
                    # ako je dobar paket je dobar i spremi ga u nove variable!
                
                # Print every byte
                #logging.debug("podaci u data bufferu su:")
                #for i in range(data_len):
                    #logging.debug(data_buffer[i])

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

# Find first occurrence of value 1 in list 
# If yes return its index, if no element is 1 return -1
def check_elements(lst):
    for i, element in enumerate(lst):
        if element == 1:
            return i
    return -1  # Return -1 if no element is 1

def simulate_robot_step(dt: float) -> None:
    """
    Simulate firmware feedback at 100 Hz when no serial is present.
    Updates Position_in/Speed_in based on Command_out and Speed_out/Position_out.
    """
    # Mark robot as homed and ensure E-Stop released bit
    for i in range(6):
        Homed_in[i] = 1
    if len(InOut_in) > 4:
        InOut_in[4] = 1  # 1 = not pressed; main loop treats 0 as E-Stop

    # Integrate motion according to current command type
    if Command_out.value == 123:
        # Jog/speed control: integrate Speed_out (steps/sec) over dt
        for i in range(6):
            v = int(Speed_out[i])
            max_v = int(PAROL6_ROBOT.Joint_max_speed[i])
            if v > max_v:
                v = max_v
            elif v < -max_v:
                v = -max_v
            new_pos = int(Position_in[i] + v * dt)
            # Clamp to limits
            jmin, jmax = PAROL6_ROBOT.Joint_limits_steps[i]
            if new_pos < jmin:
                new_pos = jmin
                v = 0
            elif new_pos > jmax:
                new_pos = jmax
                v = 0
            Speed_in[i] = v
            Position_in[i] = new_pos

    elif Command_out.value == 156:
        # Position control: move toward Position_out with capped delta per tick
        for i in range(6):
            err = int(Position_out[i] - Position_in[i])
            if err == 0:
                Speed_in[i] = 0
                continue
            max_step = int(PAROL6_ROBOT.Joint_max_speed[i] * dt)
            if max_step < 1:
                max_step = 1
            step = max(-max_step, min(max_step, err))
            new_pos = Position_in[i] + step
            # Clamp to limits
            jmin, jmax = PAROL6_ROBOT.Joint_limits_steps[i]
            if new_pos < jmin:
                new_pos = jmin
                step = 0
            elif new_pos > jmax:
                new_pos = jmax
                step = 0
            Position_in[i] = int(new_pos)
            Speed_in[i] = int(step / dt) if dt > 0 else 0

    else:
        # Idle/other: hold position
        for i in range(6):
            Speed_in[i] = 0
    
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
    
    # Parse optional trajectory start pose
    def parse_start_pose(start_str):
        """Parse start pose - returns None for CURRENT, or list of floats for specified pose."""
        if start_str == 'CURRENT' or start_str == 'NONE':
            return None
        else:
            try:
                return list(map(float, start_str.split(',')))
            except:
                logger.error(f"Warning: Invalid start pose format: {start_str}")
                return None
    
    # Convert speed percentage to trajectory duration
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
            # Format: SMOOTH_CIRCLE|center_x,center_y,center_z|radius|plane|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk_limit]
            center = list(map(float, parts[1].split(',')))
            radius = float(parts[2])
            plane = parts[3]
            frame = parts[4]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[5])
            timing_type = parts[6]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[7])
            clockwise = parts[8] == '1'
            
            # Parse trajectory type (new parameter)
            trajectory_type = 'cubic'  # default
            jerk_limit = None
            if len(parts) > 9:
                trajectory_type = parts[9]
                if trajectory_type == 's_curve' and len(parts) > 10 and parts[10] != 'DEFAULT':
                    try:
                        jerk_limit = float(parts[10])
                    except ValueError:
                        pass
            
            # Parse center_mode and entry_mode (new parameters)
            center_mode = 'ABSOLUTE'  # default
            entry_mode = 'NONE'  # default
            if len(parts) > 11:
                center_mode = parts[11]
                if len(parts) > 12:
                    entry_mode = parts[12]
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Circle circumference
                path_length = 2 * np.pi * radius
                duration = calculate_duration_from_speed(path_length, timing_value)
            
            logger.info(f"  -> Parsed circle: r={radius}mm, plane={plane}, frame={frame}, trajectory={trajectory_type}, center_mode={center_mode}, entry_mode={entry_mode}, duration={duration:.2f}s")
            
            # Return command object with frame parameter and trajectory type
            return SmoothCircleCommand(center, radius, plane, duration, clockwise, frame, start_pose, trajectory_type, jerk_limit, center_mode, entry_mode)
            
        elif command_type == 'SMOOTH_ARC_CENTER':
            # Format: SMOOTH_ARC_CENTER|end_pose|center|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk_limit]
            end_pose = list(map(float, parts[1].split(',')))
            center = list(map(float, parts[2].split(',')))
            frame = parts[3]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[4])
            timing_type = parts[5]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[6])
            clockwise = parts[7] == '1'
            
            # Parse trajectory type (new parameter)
            trajectory_type = 'cubic'  # default
            jerk_limit = None
            if len(parts) > 8:
                trajectory_type = parts[8]
                if trajectory_type == 's_curve' and len(parts) > 9 and parts[9] != 'DEFAULT':
                    try:
                        jerk_limit = float(parts[9])
                    except ValueError:
                        pass
            
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
            
            logger.info(f"  -> Parsed arc (center): frame={frame}, trajectory={trajectory_type}, duration={duration:.2f}s")
            
            # Return command with frame and trajectory type
            return SmoothArcCenterCommand(end_pose, center, duration, clockwise, frame, start_pose, trajectory_type, jerk_limit)
            
        elif command_type == 'SMOOTH_ARC_PARAM':
            # Format: SMOOTH_ARC_PARAM|end_pose|radius|angle|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk_limit]
            end_pose = list(map(float, parts[1].split(',')))
            radius = float(parts[2])
            arc_angle = float(parts[3])
            frame = parts[4]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[5])
            timing_type = parts[6]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[7])
            clockwise = parts[8] == '1'
            
            # Parse trajectory type (optional, defaults to cubic)
            trajectory_type = 'cubic'
            jerk_limit = None
            if len(parts) > 9:
                trajectory_type = parts[9]
            if len(parts) > 10 and parts[10] != 'DEFAULT':
                try:
                    jerk_limit = float(parts[10])
                except (ValueError, IndexError):
                    pass
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Arc length = radius * angle (in radians)
                arc_length = radius * np.deg2rad(arc_angle)
                duration = calculate_duration_from_speed(arc_length, timing_value)
            
            logger.info(f"  -> Parsed arc (param): r={radius}mm, θ={arc_angle}°, frame={frame}, trajectory={trajectory_type}, duration={duration:.2f}s")
            
            # Return command object with frame and trajectory type
            return SmoothArcParamCommand(end_pose, radius, arc_angle, duration, clockwise, frame, start_pose, trajectory_type, jerk_limit)
            
        elif command_type == 'SMOOTH_SPLINE':
            # Format: SMOOTH_SPLINE|num_waypoints|frame|start_pose|timing_type|timing_value|trajectory_type|[jerk_limit]|waypoint1|waypoint2|...
            num_waypoints = int(parts[1])
            frame = parts[2]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[3])
            timing_type = parts[4]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[5])
            
            # Parse trajectory type (new parameter)
            idx = 6
            trajectory_type = 'cubic'  # default
            jerk_limit = None
            if idx < len(parts) and parts[idx] in ['cubic', 'quintic', 's_curve']:
                trajectory_type = parts[idx]
                idx += 1
                # Check for jerk limit if s_curve
                if trajectory_type == 's_curve' and idx < len(parts) and parts[idx] != 'DEFAULT':
                    try:
                        jerk_limit = float(parts[idx])
                        idx += 1
                    except ValueError:
                        idx += 1  # Skip if not a number
                elif trajectory_type == 's_curve':
                    idx += 1  # Skip DEFAULT
            
            # Parse waypoints
            waypoints = []
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
            
            logger.info(f"  -> Parsed spline: {num_waypoints} points, frame={frame}, trajectory={trajectory_type}, duration={duration:.2f}s")
            
            # Return command object with frame and trajectory type
            return SmoothSplineCommand(waypoints, duration, frame, start_pose, trajectory_type, jerk_limit)
            
        elif command_type == 'SMOOTH_HELIX':
            # Format: SMOOTH_HELIX|center|radius|pitch|height|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk_limit]
            center = list(map(float, parts[1].split(',')))
            radius = float(parts[2])
            pitch = float(parts[3])
            height = float(parts[4])
            frame = parts[5]  # 'WRF' or 'TRF'
            start_pose = parse_start_pose(parts[6])
            timing_type = parts[7]  # 'DURATION' or 'SPEED'
            timing_value = float(parts[8])
            clockwise = parts[9] == '1'
            
            # Parse trajectory type (new parameter)
            trajectory_type = 'cubic'  # default
            jerk_limit = None
            if len(parts) > 10:
                trajectory_type = parts[10]
                if trajectory_type == 's_curve' and len(parts) > 11 and parts[11] != 'DEFAULT':
                    try:
                        jerk_limit = float(parts[11])
                    except ValueError:
                        pass
            
            # Calculate duration
            if timing_type == 'DURATION':
                duration = timing_value
            else:  # SPEED
                # Calculate helix path length
                num_revolutions = height / pitch if pitch > 0 else 1
                horizontal_length = 2 * np.pi * radius * num_revolutions
                helix_length = np.sqrt(horizontal_length**2 + height**2)
                duration = calculate_duration_from_speed(helix_length, timing_value)
            
            logger.info(f"  -> Parsed helix: h={height}mm, pitch={pitch}mm, frame={frame}, trajectory={trajectory_type}, duration={duration:.2f}s")
            
            # Return command object with frame and trajectory type
            return SmoothHelixCommand(center, radius, pitch, height, duration, clockwise, frame, start_pose, trajectory_type, jerk_limit)
            
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
                logger.info(f"  -> Scaled blend segments to total duration: {overall_duration:.2f}s")
                        
            elif overall_speed is not None:
                # Calculate duration from speed and estimated path length
                overall_duration = calculate_duration_from_speed(total_estimated_length, overall_speed)
                if total_original_duration > 0:
                    scale_factor = overall_duration / total_original_duration
                    for seg in segment_definitions:
                        seg['duration'] = seg['original_duration'] * scale_factor
                logger.info(f"  -> Calculated blend duration from speed: {overall_duration:.2f}s")
            else:
                logger.info(f"  -> Using original segment durations (total: {total_original_duration:.2f}s)")
            
            logger.info(f"  -> Parsed blend: {num_segments} segments, frame={frame}, blend_time={blend_time}s")
            
            # For now, use default trajectory type (backward compatibility)
            # TODO: Add trajectory type parsing when needed
            trajectory_type = 'cubic'  # Default
            jerk_limit = None
            
            # Return command with frame
            return SmoothBlendCommand(segment_definitions, blend_time, frame, start_pose, trajectory_type, jerk_limit)
        
        elif command_type == 'SMOOTH_WAYPOINTS':
            # Format: SMOOTH_WAYPOINTS|num_waypoints|blend_radii|blend_mode|via_modes|max_vel|max_acc|traj_type|frame|duration|waypoints
            num_waypoints = int(parts[1])
            blend_radii_str = parts[2]
            blend_mode = parts[3]
            via_modes_str = parts[4]
            max_vel_str = parts[5]
            max_acc_str = parts[6]
            trajectory_type = parts[7]
            frame = parts[8]
            duration_str = parts[9]
            
            # Parse blend radii
            if blend_radii_str == 'auto':
                blend_radii = 'auto'
            else:
                blend_radii = [float(r) for r in blend_radii_str.split(',')]
            
            # Parse via modes
            via_modes = via_modes_str.split(',')
            
            # Parse constraints
            max_velocity = None if max_vel_str == 'default' else float(max_vel_str)
            max_acceleration = None if max_acc_str == 'default' else float(max_acc_str)
            duration = None if duration_str == 'auto' else float(duration_str)
            
            # Parse waypoints (remaining parts joined by |)
            waypoints = []
            waypoint_parts = parts[10:]  # All remaining parts are waypoint data
            for wp_str in waypoint_parts:
                if wp_str:  # Skip empty parts
                    wp_values = [float(v) for v in wp_str.split(',')]
                    waypoints.append(wp_values)
            
            logger.info(f"  -> Parsed waypoints: {num_waypoints} points, {blend_mode} blending, frame={frame}")
            
            # Return command object
            return SmoothWaypointsCommand(
                waypoints, blend_radii, blend_mode, via_modes,
                max_velocity, max_acceleration, trajectory_type,
                frame, duration
            )
            
    except Exception as e:
        logger.error(f"Error parsing smooth motion command: {e}")
        logger.info(f"Command parts: {parts}")
        import traceback
        traceback.print_exc()
        return None
    
    logger.warning(f"Warning: Unknown smooth motion command type: {command_type}")
    return None

# Acknowledgment system configuration
CLIENT_ACK_PORT = int(os.getenv("PAROL6_ACK_PORT", "5002"))  # Port where clients listen for acknowledgments
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
            logger.error(f"Failed to send ack to {addr}: {e}")
    
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

# Initialize GCODE interpreter
gcode_interpreter = GcodeInterpreter()
logger.info("GCODE interpreter initialized")

# --------------------------------------------------------------------------
# --- Test 1: Homing and Initial Setup
# --------------------------------------------------------------------------

# 1. Optionally start with the Home command (can be bypassed via PAROL6_NOAUTOHOME or --no-auto-home)
skip_auto_home = (
    str(os.getenv("PAROL6_NOAUTOHOME", "0")).lower() in ("1", "true", "yes", "on") or
    args.no_auto_home
)
if not skip_auto_home:
    command_queue.append(HomeCommand())
else:
    reason = "command line flag" if args.no_auto_home else "environment variable"
    logging.info(f"Auto-home disabled via {reason}; skipping auto-home on startup.")

# --- State variable for the currently running command ---
active_command = None
e_stop_active = False

# Use deque for an efficient FIFO queue
incoming_command_buffer = deque()
# Timestamp of the last processed network command
last_command_time = 0
# Cooldown period in seconds to prevent command flooding
COMMAND_COOLDOWN_S = 0.01 # 10ms

# Set interval
timer = Timer(interval=INTERVAL_S, warnings=False, precise=True)

# ============================================================================
# MODIFIED MAIN LOOP WITH ACKNOWLEDGMENTS
# ============================================================================

timer = Timer(interval=INTERVAL_S, warnings=False, precise=True)
prev_time = 0

while timer.elapsed_time < 1100000:
    
    # --- Connection Handling (non-blocking) ---
    if not FAKE_SERIAL and (ser is None or not ser.is_open):
        now = time.time()
        if now - last_reconnect_attempt > 1.0:
            logging.warning("Serial port not open. Attempting to reconnect...")
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
                    logging.info(f"Trying: {com_port_str}")
                    ser = serial.Serial(port=com_port_str, baudrate=3000000, timeout=0)
                    if ser.is_open:
                        logging.info(f"Successfully reconnected to {com_port_str}")
            except serial.SerialException:
                ser = None
        # Do not block or continue; proceed to UDP handling every tick
    elif FAKE_SERIAL:
        # In FAKE_SERIAL mode, pretend we always have a connection
        # This prevents the constant "Serial port not open" warnings
        pass

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

                # Immediate command dispatch
                if command_name == 'STOP':
                    logger.info("Received STOP command. Halting all motion and clearing queue.")
                    
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

                elif command_name == 'GET_GCODE_STATUS':
                    # Get GCODE interpreter status
                    gcode_status = gcode_interpreter.get_status()
                    status_json = json.dumps(gcode_status)
                    response_message = f"GCODE_STATUS|{status_json}"
                    sock.sendto(response_message.encode('utf-8'), addr)
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "GCODE status sent", addr)
                
                elif command_name == 'GCODE_STOP':
                    # Stop GCODE program execution
                    gcode_interpreter.stop_program()
                    logger.info("GCODE program stopped")
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "GCODE program stopped", addr)
                
                elif command_name == 'GCODE_PAUSE':
                    # Pause GCODE program execution
                    gcode_interpreter.pause_program()
                    logger.info("GCODE program paused")
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "GCODE program paused", addr)
                
                elif command_name == 'GCODE_RESUME':
                    # Resume GCODE program execution
                    gcode_interpreter.start_program()  # start_program resumes if already loaded
                    logger.info("GCODE program resumed")
                    
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "GCODE program resumed", addr)

                elif command_name == 'PING':
                    # Respond with PONG and ACK if cmd_id present
                    sock.sendto("PONG".encode('utf-8'), addr)
                    if cmd_id:
                        send_acknowledgment(cmd_id, "COMPLETED", "PONG", addr)

                elif command_name == 'GET_SERVER_STATE':
                    # Structured server state for external managers
                    try:
                        state = {
                            "listening": {"transport": "udp", "address": f"{ip}:{port}"},
                            "serial_connected": bool(ser and getattr(ser, "is_open", False)),
                            "homed": any(bool(h) for h in Homed_in) if isinstance(Homed_in, list) else False,
                            "queue_depth": len(command_queue) if command_queue is not None else 0,
                            "active_command": type(active_command).__name__ if active_command is not None else None,
                            "stream_mode": bool(stream_mode),
                            "uptime_s": float(time.time() - START_TIME) if 'START_TIME' in globals() else 0.0,
                        }
                        payload = f"SERVER_STATE|{json.dumps(state)}"
                        sock.sendto(payload.encode('utf-8'), addr)
                        if cmd_id:
                            send_acknowledgment(cmd_id, "COMPLETED", "Server state sent", addr)
                    except Exception as e:
                        if cmd_id:
                            send_acknowledgment(cmd_id, "FAILED", f"State error: {e}", addr)

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

                elif command_name == 'STREAM' and len(parts) >= 2:
                    arg = parts[1].strip().upper()
                    if arg == 'ON':
                        stream_mode = True
                        if cmd_id:
                            send_acknowledgment(cmd_id, "COMPLETED", "Stream mode ON", addr)
                    elif arg == 'OFF':
                        stream_mode = False
                        if cmd_id:
                            send_acknowledgment(cmd_id, "COMPLETED", "Stream mode OFF", addr)
                    else:
                        if cmd_id:
                            send_acknowledgment(cmd_id, "FAILED", "Expected ON or OFF", addr)

                elif command_name == 'JOG' and stream_mode and len(parts) == 5:
                    # Streaming JOG: create JogCommand instance with latest-wins semantics
                    try:
                        joint_index = int(parts[1])
                        speed_percent = float(parts[2])
                        duration = None if parts[3].upper() == 'NONE' else float(parts[3])
                        distance = None if parts[4].upper() == 'NONE' else float(parts[4])
                        
                        # Use default duration if none specified
                        if duration is None:
                            duration = 0.2
                        
                        # Create command instance
                        command_obj = JogCommand(joint=joint_index, speed_percentage=speed_percent, 
                                               duration=duration, distance_deg=distance)
                        command_obj.prepare_for_execution(current_position_in=Position_in)
                        # Cancel any active jog-type command (latest-wins)
                        if active_command and isinstance(active_command, (JogCommand, CartesianJogCommand, MultiJogCommand)):
                            if active_command_id:
                                send_acknowledgment(active_command_id, "CANCELLED", "Replaced by new jog command", addr)
                            if active_command in command_id_map:
                                del command_id_map[active_command]
                        
                        # Purge queued jog commands
                        for queued_cmd in list(command_queue):
                            if isinstance(queued_cmd, (JogCommand, CartesianJogCommand, MultiJogCommand)):
                                command_queue.remove(queued_cmd)
                                if queued_cmd in command_id_map:
                                    qid, qaddr = command_id_map[queued_cmd]
                                    send_acknowledgment(qid, "CANCELLED", "Replaced by streaming jog", qaddr)
                                    del command_id_map[queued_cmd]
                        
                        # Activate command immediately
                        active_command = command_obj
                        active_command_id = cmd_id
                        if cmd_id:
                            command_id_map[command_obj] = (cmd_id, addr)
                            send_acknowledgment(cmd_id, "EXECUTING", "Streaming jog started", addr)
                        
                    except Exception as e:
                        logging.error(e)
                        if cmd_id:
                            send_acknowledgment(cmd_id, "FAILED", "Malformed JOG (stream)", addr)

                elif command_name == 'CARTJOG' and stream_mode and len(parts) == 5:
                    # Streaming CARTJOG: create CartesianJogCommand instance with latest-wins semantics
                    try:
                        frame = parts[1].upper()
                        axis = parts[2]
                        speed_percent = float(parts[3])
                        timeout_s = float(parts[4]) if parts[4].upper() != 'NONE' else 0.2
                        
                        # Create command instance
                        command_obj = CartesianJogCommand(frame=frame, axis=axis, 
                                                        speed_percentage=speed_percent, duration=timeout_s)
                        
                        # Cancel any active jog-type command (latest-wins)
                        if active_command and isinstance(active_command, (JogCommand, CartesianJogCommand, MultiJogCommand)):
                            if active_command_id:
                                send_acknowledgment(active_command_id, "CANCELLED", "Replaced by new cartesian jog", addr)
                            if active_command in command_id_map:
                                del command_id_map[active_command]
                        
                        # Purge queued jog commands
                        for queued_cmd in list(command_queue):
                            if isinstance(queued_cmd, (JogCommand, CartesianJogCommand, MultiJogCommand)):
                                command_queue.remove(queued_cmd)
                                if queued_cmd in command_id_map:
                                    qid, qaddr = command_id_map[queued_cmd]
                                    send_acknowledgment(qid, "CANCELLED", "Replaced by streaming cartesian jog", qaddr)
                                    del command_id_map[queued_cmd]
                        
                        # Activate command immediately
                        active_command = command_obj
                        active_command_id = cmd_id
                        if cmd_id:
                            command_id_map[command_obj] = (cmd_id, addr)
                            send_acknowledgment(cmd_id, "EXECUTING", "Streaming cartesian jog started", addr)
                        
                    except Exception as e:
                        logging.error(e)
                        if cmd_id:
                            send_acknowledgment(cmd_id, "FAILED", "Malformed CARTJOG (stream)", addr)

                else:
                    # Queue command for processing (coalesce jog-type commands to avoid backlog)
                    cmd_upper = parts[0].upper()
                    if cmd_upper in {'JOG', 'CARTJOG', 'MULTIJOG'}:
                        filtered = []
                        for m, a in incoming_command_buffer:
                            _, m2 = parse_command_with_id(m)
                            c2 = m2.split('|', 1)[0].upper()
                            if c2 not in {'JOG', 'CARTJOG', 'MULTIJOG'}:
                                filtered.append((m, a))
                        incoming_command_buffer.clear()
                        incoming_command_buffer.extend(filtered)
                    incoming_command_buffer.append((raw_message, addr))

    except Exception as e:
        logger.error(f"Network receive error: {e}")

    # =======================================================================
    # === PROCESS COMMANDS FROM BUFFER WITH ACKNOWLEDGMENTS ===
    # =======================================================================
    current_time = time.time()
    if incoming_command_buffer and (current_time - last_command_time) > COMMAND_COOLDOWN_S and not e_stop_active:
        raw_message, addr = incoming_command_buffer.popleft()
        last_command_time = current_time
        
        # Parse command ID
        cmd_id, message = parse_command_with_id(raw_message)
        logger.info(f"Processing command{' (ID: ' + cmd_id + ')' if cmd_id else ''}: {message[:50]}...")
        
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
                                 'SMOOTH_SPLINE', 'SMOOTH_HELIX', 'SMOOTH_BLEND', 'SMOOTH_WAYPOINTS']:
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
            
            elif command_name == 'GCODE' and len(parts) >= 2:
                # Single GCODE line execution
                gcode_line = '|'.join(parts[1:])  # Rejoin in case GCODE has | characters
                try:
                    # Update interpreter position with current robot position
                    current_angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(Position_in)]
                    current_pose_matrix = PAROL6_ROBOT.robot.fkine(current_angles_rad).A
                    current_xyz = current_pose_matrix[:3, 3]
                    # Convert from meters to millimeters
                    gcode_interpreter.state.update_position({
                        'X': current_xyz[0] * 1000,
                        'Y': current_xyz[1] * 1000, 
                        'Z': current_xyz[2] * 1000
                    })
                    
                    # Parse GCODE line and get robot commands
                    robot_commands = gcode_interpreter.parse_line(gcode_line)
                    
                    if robot_commands:
                        # Process each generated robot command
                        for robot_cmd_str in robot_commands:
                            # Parse the generated robot command string
                            cmd_parts = robot_cmd_str.split('|')
                            cmd_name = cmd_parts[0].upper()
                            
                            # Create command objects from the generated strings
                            if cmd_name == 'MOVEPOSE' and len(cmd_parts) == 9:
                                pose_vals = [float(p) for p in cmd_parts[1:7]]
                                duration = None if cmd_parts[7].upper() == 'NONE' else float(cmd_parts[7])
                                speed = None if cmd_parts[8].upper() == 'NONE' else float(cmd_parts[8])
                                cmd = MovePoseCommand(pose=pose_vals, duration=duration, velocity_percent=speed)
                                command_queue.append(cmd)
                                if cmd_id:
                                    command_id_map[cmd] = (cmd_id, addr)
                            
                            elif cmd_name == 'MOVECART' and len(cmd_parts) == 9:
                                pose_vals = [float(p) for p in cmd_parts[1:7]]
                                duration = None if cmd_parts[7].upper() == 'NONE' else float(cmd_parts[7])
                                speed = None if cmd_parts[8].upper() == 'NONE' else float(cmd_parts[8])
                                cmd = MoveCartCommand(pose=pose_vals, duration=duration, velocity_percent=speed)
                                command_queue.append(cmd)
                                if cmd_id:
                                    command_id_map[cmd] = (cmd_id, addr)
                                    
                            elif cmd_name == 'DELAY' and len(cmd_parts) == 2:
                                duration = float(cmd_parts[1])
                                cmd = DelayCommand(duration=duration)
                                command_queue.append(cmd)
                                if cmd_id:
                                    command_id_map[cmd] = (cmd_id, addr)
                                    
                            elif cmd_name == 'HOME':
                                cmd = HomeCommand()
                                command_queue.append(cmd)
                                if cmd_id:
                                    command_id_map[cmd] = (cmd_id, addr)
                                    
                            elif cmd_name == 'PNEUMATICGRIPPER' and len(cmd_parts) == 3:
                                action, port = cmd_parts[1].lower(), int(cmd_parts[2])
                                cmd = GripperCommand(gripper_type='pneumatic', action=action, output_port=port)
                                command_queue.append(cmd)
                                if cmd_id:
                                    command_id_map[cmd] = (cmd_id, addr)
                        
                        command_queued = True
                        logger.info(f"GCODE '{gcode_line}' generated {len(robot_commands)} command(s)")
                    else:
                        # GCODE line didn't generate any motion commands (might be modal state change)
                        command_queued = True
                        logger.debug(f"GCODE '{gcode_line}' updated state only")
                        
                except Exception as e:
                    error_details = f"GCODE parsing error: {str(e)}"
                    command_queued = False
            
            elif command_name == 'GCODE_PROGRAM' and len(parts) >= 2:
                # Load and execute GCODE program
                program_type = parts[1].upper()
                
                try:
                    if program_type == 'FILE' and len(parts) == 3:
                        # Load from file
                        filepath = parts[2]
                        if gcode_interpreter.load_file(filepath):
                            logger.info(f"Loaded GCODE file: {filepath}")
                            # Start program execution
                            gcode_interpreter.start_program()
                            command_queued = True
                        else:
                            error_details = f"Failed to load GCODE file: {filepath}"
                            command_queued = False
                            
                    elif program_type == 'INLINE':
                        # Load inline program (lines separated by semicolons)
                        program_lines = '|'.join(parts[2:]).split(';')
                        if gcode_interpreter.load_program(program_lines):
                            logger.info(f"Loaded inline GCODE program ({len(program_lines)} lines)")
                            # Start program execution
                            gcode_interpreter.start_program()
                            command_queued = True
                        else:
                            error_details = "Failed to load inline GCODE program"
                            command_queued = False
                    else:
                        error_details = f"Invalid GCODE_PROGRAM format"
                        command_queued = False
                        
                except Exception as e:
                    error_details = f"GCODE program error: {str(e)}"
                    command_queued = False
            
            else:
                error_details = f"Unknown or malformed command: {command_name}"
                
        except Exception as e:
            error_details = f"Error parsing command: {str(e)}"
            command_queued = False
        
        # Command queue management with ACK
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
            logger.error(f"Warning: {error_details}")

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
                
                logger.info(f"E-STOP TRIGGERED! Active command '{cancelled_command_info}' cancelled.")
                logger.info("Release E-Stop and press 'e' to re-enable.")
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
            if keyboard.is_pressed('e'):
                logger.info("Re-enabling robot...")
                Command_out.value = 101
                e_stop_active = False
            else:
                Command_out.value = 255
                Speed_out[:] = [0] * 6
                Position_out[:] = Position_in[:]
                
        else:
            # --- Normal Command Processing ---
            
            # Check if GCODE program is running and fetch next command
            if active_command is None and not command_queue and gcode_interpreter.is_running:
                # Get next command from GCODE program
                next_gcode_cmd = gcode_interpreter.get_next_command()
                if next_gcode_cmd:
                    # Parse the generated robot command string
                    cmd_parts = next_gcode_cmd.split('|')
                    cmd_name = cmd_parts[0].upper()
                    
                    # Create command object from the generated string
                    if cmd_name == 'MOVEPOSE' and len(cmd_parts) == 9:
                        pose_vals = [float(p) for p in cmd_parts[1:7]]
                        duration = None if cmd_parts[7].upper() == 'NONE' else float(cmd_parts[7])
                        speed = None if cmd_parts[8].upper() == 'NONE' else float(cmd_parts[8])
                        cmd = MovePoseCommand(pose=pose_vals, duration=duration, velocity_percent=speed)
                        command_queue.append(cmd)
                    
                    elif cmd_name == 'MOVECART' and len(cmd_parts) == 9:
                        pose_vals = [float(p) for p in cmd_parts[1:7]]
                        duration = None if cmd_parts[7].upper() == 'NONE' else float(cmd_parts[7])
                        speed = None if cmd_parts[8].upper() == 'NONE' else float(cmd_parts[8])
                        cmd = MoveCartCommand(pose=pose_vals, duration=duration, velocity_percent=speed)
                        command_queue.append(cmd)
                        
                    elif cmd_name == 'DELAY' and len(cmd_parts) == 2:
                        duration = float(cmd_parts[1])
                        cmd = DelayCommand(duration=duration)
                        command_queue.append(cmd)
                        
                    elif cmd_name == 'HOME':
                        cmd = HomeCommand()
                        command_queue.append(cmd)
                        
                    elif cmd_name == 'PNEUMATICGRIPPER' and len(cmd_parts) == 3:
                        action, port = cmd_parts[1].lower(), int(cmd_parts[2])
                        cmd = GripperCommand(gripper_type='pneumatic', action=action, output_port=port)
                        command_queue.append(cmd)
            
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
                        logger.error(f"Command preparation failed: {e}")
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
                    logger.error(f"Command execution error: {e}")
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
            # Serial not available; optionally simulate plant
            if FAKE_SERIAL:
                simulate_robot_step(INTERVAL_S)
            else:
                pass

    except serial.SerialException as e:
        logger.error(f"Serial communication error: {e}")
        
        # Send failure acknowledgments for active command
        if active_command_id:
            send_acknowledgment(active_command_id, "FAILED", "Serial communication lost")
        
        if ser:
            ser.close()
        ser = None
        active_command = None
        active_command_id = None

    timer.checkpt()


def main():
    """
    Main entry point for the headless commander.
    This function wraps the main execution loop for CLI usage.
    """
    # The main loop is already implemented above as a module-level script
    # This function exists to provide a clean entry point for the CLI
    pass


if __name__ == "__main__":
    main()
