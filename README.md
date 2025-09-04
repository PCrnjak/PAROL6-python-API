# PAROL6 Headless Commander Documentation

## 1. Important Notes & Disclaimers
* **Software Origin**: This control system is based on the `experimental_kinematics` branch of the `PAROL_commander_software` repository. The core communication functions were derived from the `Serial_sender_good_latest.py` file; however, the approach to motion planning has been altered from the original implementation. This system was created by editing the `Commander_minimal_version.py` file, which was used as a base.
* **Automatic Homing on Startup**: By default, the `headless_commander.py` script will immediately command the robot to home itself upon startup. This is done for convenience but can be disabled. To prevent automatic homing, comment out or delete the corresponding line in `headless_commander.py`.
* **AI-Assisted Development**: This code was developed with significant AI assistance. While the core logic has been corrected and improved, it has not been exhaustively tested in all scenarios. Users should proceed with caution and verify functionality for their specific needs.

## 2. Safety Precautions & Disclaimer
This control software includes several built-in safety features designed to prevent damage to the robot and ensure predictable operation:
* **E-Stop Monitoring**: The system constantly checks the physical E-Stop button. If triggered, all motion is immediately halted, the command queue is cleared, and the robot is disabled. The system must be manually re-enabled by pressing the `'e'` key after the E-Stop is released.
* **Synchronized Speed Calculation**: For moves defined by a speed percentage (`MoveJoint`, `MovePose`), the system now calculates the maximum possible synchronized speed for all joints involved. This prevents individual joints from exceeding their limits and ensures predictable, smooth motion.
* **Inverse Kinematics (IK) Validation**: The system verifies that a valid kinematic solution exists for any pose-based command. If the target pose is unreachable, the command will fail safely before any motion occurs.

> **WARNING**: These are software-based safety measures and are not a substitute for responsible operation and a safe work environment. The user assumes all responsibility for safe operation. Always be attentive when the robot is active, ensure you have immediate access to the physical E-Stop, and operate the robot in a clear area.

## 3. Installation

### Base Software Installation
Follow the official PAROL commander software installation guide:
- Repository: [PAROL Commander Software](https://github.com/PCrnjak/PAROL-commander-software)
- Branch: Use the `experimental_kinematics` branch
- Installation Guide: [Official Instructions](https://github.com/PCrnjak/PAROL-commander-software)

### Additional Dependencies for Headless Commander
After installing the base software, install these additional packages:

```bash
# Install Python 3 and pip (if not already installed)
# https://www.python.org/downloads/

# Install Git (if not already installed)
# https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

# Core dependencies (from official installation)
pip3 install roboticstoolbox-python==1.0.3
pip3 install numpy==1.23.4
pip3 install scipy==1.11.4
pip3 install spatialmath
pip3 install pyserial
pip3 install oclock
pip3 install keyboard
```
## 4. System Architecture

### Client-Server Design
The system uses a UDP-based client-server architecture that separates robot control from command generation:

* **The Robot Controller (`headless_commander.py`)**: 
  - Runs on the computer physically connected to the robot via USB/Serial
  - Maintains a high-frequency control loop (100Hz) for real-time robot control
  - Handles all complex calculations (inverse kinematics, trajectory planning)
  - Requires heavy dependencies (roboticstoolbox, numpy, scipy)
  - Listens for UDP commands on port 5001

* **The Remote Client (`robot_api.py`)**: 
  - Can run on any computer (same or different from controller)
  - Sends simple text commands via UDP
  - Requires minimal dependencies (mostly Python standard library)
  - Extremely lightweight - can run on resource-constrained devices
  - Optionally receives acknowledgments on port 5002

* **Support Modules**:
  - `smooth_motion.py`: Advanced trajectory generation algorithms
  - `PAROL6_ROBOT.py`: Robot-specific parameters and kinematic model

### Why UDP?
The UDP protocol was chosen for several reasons:
- **Simplicity**: No connection management overhead
- **Low Latency**: Direct message passing without handshaking
- **Lightweight Client**: Client only needs to send text strings
- **Cross-Platform**: Works on any OS with network support
- **Flexible Deployment**: Client can run anywhere on the network

### Command Flow
1. Client calls API function (e.g., `move_robot_joints()`)
2. API formats command as text string (e.g., `"MOVEJOINT|90|-45|90|0|45|180|5.5|None"`)
3. Command sent via UDP to controller
4. Controller queues and executes command
5. Optional: Acknowledgment sent back to client
6. Optional: Client checks status using command ID

### Command Acknowledgment System
The system includes an optional acknowledgment tracking feature that provides feedback on command execution:
* **Tracking States**: Commands can report status as `QUEUED`, `EXECUTING`, `COMPLETED`, `FAILED`, `CANCELLED`, or `INVALID`
* **Zero Overhead**: When not used, the tracking system has zero resource overhead - no threads or sockets are created
* **Non-Blocking Mode**: Commands can be sent with `non_blocking=True` to return immediately with a command ID, allowing asynchronous operation
* **Status Checking**: Use `check_command_status(command_id)` to poll command status later

Example of non-blocking usage:
```python
# Send command and get ID immediately
cmd_id = move_robot_joints([90, -45, 90, 0, 45, 180], 
                          duration=5, 
                          wait_for_ack=True, 
                          non_blocking=True)

# Do other work...
time.sleep(1)

# Check status later
status = check_command_status(cmd_id)
if status and status['completed']:
    print(f"Command finished with status: {status['status']}")
```

## 5. Command Reference & API Usage

### Motion Commands

#### `home_robot()`
* **Purpose**: Initiates the robot's built-in homing sequence.
* **Parameters**: 
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 30.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import home_robot
    home_robot()  # Simple usage
    home_robot(wait_for_ack=True, timeout=30)  # With tracking
    ```

#### `move_robot_joints()`
* **Purpose**: Moves joints to a target configuration (in degrees).
* **Parameters**:
    * `joint_angles` (List[float]): List of 6 target angles in degrees for joints 1-6
    * `duration` (float, optional): Total time for the movement in seconds
    * `speed_percentage` (int, optional): Speed as percentage (0-100)
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import move_robot_joints

    # Simple move by speed
    move_robot_joints([90, -45, 90, 0, 45, 180], speed_percentage=75)

    # Move with acknowledgment tracking
    result = move_robot_joints([0, -90, 180, 0, 0, 180], 
                               duration=5.5, 
                               wait_for_ack=True)
    if result['status'] == 'COMPLETED':
        print("Move completed successfully")
    ```

#### `move_robot_pose()`
* **Purpose**: Moves the end-effector to a Cartesian pose via a joint-based path.
* **Parameters**:
    * `pose` (List[float]): Target pose [x, y, z, Rx, Ry, Rz] in mm and degrees
    * `duration` (float, optional): Total time for the movement in seconds
    * `speed_percentage` (int, optional): Speed as percentage (0-100)
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import move_robot_pose
    move_robot_pose([250, 0, 200, 180, 0, 90], speed_percentage=50)
    ```

#### `move_robot_cartesian()`
* **Purpose**: Moves the end-effector to a target pose in a guaranteed straight-line path.
* **Parameters**:
    * `pose` (List[float]): Target pose [x, y, z, Rx, Ry, Rz] in mm and degrees
    * `duration` (float, optional): Time for the movement in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import move_robot_cartesian
    move_robot_cartesian([200, -50, 180, 180, 0, 135], duration=4.0)
    ```

### Jogging Commands

#### `jog_robot_joint()`
* **Purpose**: Jogs a single joint by time or angular distance.
* **Parameters**:
    * `joint_index` (int): Joint to move (0-5 for positive direction, 6-11 for negative)
    * `speed_percentage` (int): Jog speed as percentage (0-100)
    * `duration` (float, optional): Time to jog in seconds
    * `distance_deg` (float, optional): Distance to jog in degrees
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `distance_deg`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import jog_robot_joint
    # Jog joint 1 for 2 seconds
    jog_robot_joint(joint_index=0, speed_percentage=40, duration=2.0)
    # Jog joint 3 backwards by 15 degrees
    jog_robot_joint(joint_index=8, speed_percentage=60, distance_deg=15)
    ```

#### `jog_multiple_joints()`
* **Purpose**: Jogs multiple joints simultaneously.
* **Parameters**:
    * `joints` (List[int]): List of joint indices (0-5 positive, 6-11 negative)
    * `speeds` (List[float]): List of corresponding speeds (1-100%)
    * `duration` (float): Duration in seconds
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import jog_multiple_joints
    # Jog joints 1, 4, and 6 simultaneously
    jog_multiple_joints([0, 3, 5], [70, 40, 60], 1.2)
    ```

#### `jog_cartesian()`
* **Purpose**: Jogs the end-effector continuously along an axis.
* **Parameters**:
    * `frame` (str): Reference frame ('TRF' for Tool, 'WRF' for World)
    * `axis` (str): Axis and direction ('X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-', 'RX+', 'RX-', 'RY+', 'RY-', 'RZ+', 'RZ-')
    * `speed_percentage` (int): Jog speed as percentage (0-100)
    * `duration` (float): Time to jog in seconds
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import jog_cartesian
    jog_cartesian(frame='TRF', axis='Z+', speed_percentage=50, duration=1.5)
    ```

### Smooth Motion Commands

These commands create smooth, curved trajectories with continuous velocity profiles. All commands support reference frame selection via the `frame` parameter:

- **WRF (World Reference Frame)**: Default. All coordinates are interpreted relative to the robot's base coordinate system.
- **TRF (Tool Reference Frame)**: All coordinates are interpreted relative to the tool's current position and orientation. This means:
  - Positions are relative to the tool's origin
  - Planes (XY, XZ, YZ) are the tool's local planes, not world planes
  - If the tool is rotated, the entire motion rotates with it

#### `smooth_circle()`
* **Purpose**: Execute a smooth circular motion.
* **Parameters**:
    * `center` (List[float]): Center point [x, y, z] in mm
    * `radius` (float): Circle radius in mm
    * `plane` (str, optional): Plane of circle ('XY', 'XZ', or 'YZ'). Default: 'XY'
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF'). Default: 'WRF'
    * `start_pose` (List[float], optional): Starting pose [x, y, z, rx, ry, rz], or None for current position. Default: None
    * `duration` (float, optional): Time to complete circle in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `clockwise` (bool, optional): Direction of motion. Default: False
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 10.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import smooth_circle
    
    # Draw a 50mm radius circle in world XY plane
    smooth_circle(center=[200, 0, 200], radius=50, plane='XY', duration=5.0)
    
    # Draw a circle in tool's XY plane (follows tool orientation)
    smooth_circle(center=[0, 30, 0], radius=25, plane='XY', frame='TRF', duration=4.0)
    ```

#### `smooth_arc_center()`
* **Purpose**: Execute a smooth arc motion defined by center point.
* **Parameters**:
    * `end_pose` (List[float]): End pose [x, y, z, rx, ry, rz] in mm and degrees
    * `center` (List[float]): Arc center point [x, y, z] in mm
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF'). Default: 'WRF'
    * `start_pose` (List[float], optional): Starting pose, or None for current position. Default: None
    * `duration` (float, optional): Time to complete arc in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `clockwise` (bool, optional): Direction of motion. Default: False
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 10.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import smooth_arc_center
    
    # Arc in world coordinates
    smooth_arc_center(end_pose=[250, 50, 200, 0, 0, 90], 
                     center=[200, 0, 200], 
                     duration=3.0)
    
    # Arc in tool coordinates (relative to tool position/orientation)
    smooth_arc_center(end_pose=[30, 30, 0, 0, 0, 45],
                     center=[15, 15, 0],
                     frame='TRF',
                     duration=3.0)
    ```

#### `smooth_arc_parametric()`
* **Purpose**: Execute a smooth arc motion defined by radius and angle.
* **Parameters**:
    * `end_pose` (List[float]): End pose [x, y, z, rx, ry, rz] in mm and degrees
    * `radius` (float): Arc radius in mm
    * `arc_angle` (float): Arc angle in degrees
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF'). Default: 'WRF'
    * `start_pose` (List[float], optional): Starting pose, or None for current position. Default: None
    * `duration` (float, optional): Time to complete arc in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `clockwise` (bool, optional): Direction of motion. Default: False
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 10.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import smooth_arc_parametric
    
    # Parametric arc in world frame
    smooth_arc_parametric(end_pose=[250, 50, 200, 0, 0, 90],
                         radius=50, arc_angle=90, duration=3.0)
    
    # Parametric arc in tool frame
    smooth_arc_parametric(end_pose=[40, 0, 0, 0, 0, 60],
                         radius=25, arc_angle=60,
                         frame='TRF',
                         speed_percentage=40)
    ```

#### `smooth_spline()`
* **Purpose**: Create smooth spline through waypoints.
* **Parameters**:
    * `waypoints` (List[List[float]]): List of poses [x, y, z, rx, ry, rz] to pass through
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF'). Default: 'WRF'
    * `start_pose` (List[float], optional): Starting pose, or None for current position. Default: None
    * `duration` (float, optional): Total time for motion in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 10.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import smooth_spline
    
    # Spline through world coordinates
    waypoints = [
        [200, 0, 100, 0, 0, 0],
        [250, 50, 150, 0, 15, 45],
        [200, 100, 200, 0, 30, 90]
    ]
    smooth_spline(waypoints, duration=8.0)
    
    # Spline through tool-relative coordinates
    tool_waypoints = [
        [20, 0, 0, 0, 0, 0],
        [20, 20, 10, 0, 0, 30],
        [0, 20, 20, 0, 0, 60]
    ]
    smooth_spline(tool_waypoints, frame='TRF', duration=6.0)
    ```

#### `smooth_helix()`
* **Purpose**: Execute helical motion.
* **Parameters**:
    * `center` (List[float]): Helix center point [x, y, z] in mm
    * `radius` (float): Helix radius in mm
    * `pitch` (float): Vertical distance per revolution in mm
    * `height` (float): Total height of helix in mm
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF'). Default: 'WRF'
    * `start_pose` (List[float], optional): Starting pose, or None for current position. Default: None
    * `duration` (float, optional): Time to complete helix in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `clockwise` (bool, optional): Direction of motion. Default: False
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 10.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* > *Note: You must provide either `duration` or `speed_percentage`, but not both.*
* > *Note: In TRF mode, the helix rises along the tool's Z-axis, not the world Z-axis.*
* **Python API Usage**:
    ```python
    from robot_api import smooth_helix
    
    # Vertical helix in world frame
    smooth_helix(center=[200, 0, 150], radius=30, pitch=20, 
                height=100, duration=10.0)
    
    # Helix along tool's Z-axis (follows tool orientation)
    smooth_helix(center=[0, 30, 0], radius=20, pitch=15,
                height=60, frame='TRF', duration=8.0)
    ```

#### `smooth_blend()`
* **Purpose**: Blend multiple motion segments smoothly.
* **Parameters**:
    * `segments` (List[Dict]): List of segment dictionaries defining the motion path
    * `blend_time` (float, optional): Time for blending between segments in seconds. Default: 0.5
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF') for all segments. Default: 'WRF'
    * `start_pose` (List[float], optional): Starting pose, or None for current position. Default: None
    * `duration` (float, optional): Total time for entire motion in seconds
    * `speed_percentage` (float, optional): Speed as percentage (1-100)
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 15.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import smooth_blend
    
    # Blend in world coordinates
    segments = [
        {'type': 'LINE', 'end': [250, 0, 200, 0, 0, 0], 'duration': 2.0},
        {'type': 'CIRCLE', 'center': [250, 0, 200], 'radius': 50, 
         'plane': 'XY', 'duration': 4.0, 'clockwise': False},
        {'type': 'LINE', 'end': [200, 0, 200, 0, 0, 0], 'duration': 2.0}
    ]
    smooth_blend(segments, blend_time=0.5, duration=10.0)
    
    # Blend in tool coordinates (all segments relative to tool)
    tool_segments = [
        {'type': 'LINE', 'end': [30, 0, 0, 0, 0, 0], 'duration': 2.0},
        {'type': 'CIRCLE', 'center': [30, 20, 0], 'radius': 20, 
         'plane': 'XY', 'duration': 4.0, 'clockwise': False},
        {'type': 'LINE', 'end': [0, 20, 0, 0, 0, 0], 'duration': 2.0}
    ]
    smooth_blend(tool_segments, frame='TRF', blend_time=0.5, duration=10.0)
    ```
### Gripper Commands

#### `control_pneumatic_gripper()`
* **Purpose**: Controls the pneumatic gripper.
* **Parameters**:
    * `action` (str): Action to perform ('open' or 'close')
    * `port` (int): Digital output port (1 or 2)
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import control_pneumatic_gripper
    control_pneumatic_gripper(action='open', port=1)
    ```

#### `control_electric_gripper()`
* **Purpose**: Controls the electric gripper.
* **Parameters**:
    * `action` (str): Action to perform ('move' or 'calibrate')
    * `position` (int, optional): Target position (0-255). Default: 255
    * `speed` (int, optional): Movement speed (0-255). Default: 150
    * `current` (int, optional): Max motor current in mA (100-1000). Default: 500
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import control_electric_gripper
    control_electric_gripper(action='calibrate')
    control_electric_gripper(action='move', position=200, speed=150)
    ```

### Utility Commands

#### `delay_robot()`
* **Purpose**: Pauses command queue execution.
* **Parameters**:
    * `duration` (float): Pause time in seconds
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import delay_robot
    delay_robot(2.5)
    ```

#### `stop_robot_movement()`
* **Purpose**: Immediately stops all motion and clears command queue.
* **Parameters**:
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: False
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 2.0
    * `non_blocking` (bool, optional): Return immediately with command ID. Default: False
* **Python API Usage**:
    ```python
    from robot_api import stop_robot_movement
    stop_robot_movement()
    ```

### Query Commands

These commands request current robot state without moving the robot:

#### `get_robot_pose()`
* **Purpose**: Get current end-effector pose.
* **Parameters**: None
* **Returns**: List [x, y, z, Rx, Ry, Rz] in mm and degrees, or None if failed
* **Python API Usage**:
    ```python
    from robot_api import get_robot_pose
    pose = get_robot_pose()
    if pose:
        print(f"Current pose: {pose}")
    ```

#### `get_robot_joint_angles()`
* **Purpose**: Get current joint angles.
* **Parameters**: None
* **Returns**: List of 6 angles in degrees, or None if failed
* **Python API Usage**:
    ```python
    from robot_api import get_robot_joint_angles
    angles = get_robot_joint_angles()
    ```

#### `get_robot_joint_speeds()`
* **Purpose**: Get current joint speeds.
* **Parameters**: None
* **Returns**: List of 6 speeds in steps/sec, or None if failed

#### `get_robot_io()`
* **Purpose**: Get digital I/O status.
* **Parameters**:
    * `verbose` (bool, optional): Print formatted status to console. Default: False
* **Returns**: List [IN1, IN2, OUT1, OUT2, ESTOP] (0 or 1 values), or None if failed

#### `get_electric_gripper_status()`
* **Purpose**: Get electric gripper status.
* **Parameters**:
    * `verbose` (bool, optional): Print formatted status to console. Default: False
* **Returns**: List [ID, Position, Speed, Current, StatusByte, ObjectDetected], or None if failed

#### `get_robot_pose_matrix()`
* **Purpose**: Get robot pose as transformation matrix.
* **Parameters**: None
* **Returns**: 4x4 numpy array, or None if failed

#### `is_robot_stopped()`
* **Purpose**: Check if robot has stopped moving.
* **Parameters**:
    * `threshold_speed` (float, optional): Speed threshold in steps/sec. Default: 2.0
* **Returns**: Boolean (True if stopped, False otherwise)

#### `is_estop_pressed()`
* **Purpose**: Check if E-stop is currently pressed.
* **Parameters**: None
* **Returns**: Boolean (True if pressed, False otherwise)

#### `get_robot_status()`
* **Purpose**: Get comprehensive robot status.
* **Parameters**: None
* **Returns**: Dictionary containing pose, angles, speeds, IO, gripper status, stopped state, and E-stop state

### Helper Functions

#### `execute_trajectory()`
* **Purpose**: High-level trajectory execution with best method selection.
* **Parameters**:
    * `trajectory` (List[List[float]]): List of poses [x, y, z, rx, ry, rz]
    * `timing_mode` (str, optional): Either 'duration' or 'speed'. Default: 'duration'
    * `timing_value` (float, optional): Duration in seconds or speed percentage. Default: 5.0
    * `motion_type` (str, optional): Either 'spline' or 'linear'. Default: 'spline'
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF') for spline motion. Default: 'WRF'
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: True
    * `timeout` (float, optional): Timeout for acknowledgment in seconds. Default: 30.0
* > *Note: The `frame` parameter only applies when `motion_type='spline'`. Linear motions are always in WRF.*
* **Python API Usage**:
    ```python
    from robot_api import execute_trajectory
    
    # Execute trajectory in world frame
    trajectory = [[200, 0, 200, 0, 0, 0], 
                  [250, 50, 200, 0, 0, 45],
                  [200, 100, 200, 0, 0, 90]]
    execute_trajectory(trajectory, timing_mode='duration', 
                      timing_value=10.0, motion_type='spline')
    
    # Execute trajectory in tool frame (spline only)
    tool_trajectory = [[20, 0, 0, 0, 0, 0],
                      [20, 20, 0, 0, 0, 30],
                      [0, 20, 10, 0, 0, 60]]
    execute_trajectory(tool_trajectory, frame='TRF',
                      timing_mode='speed', 
                      timing_value=40, motion_type='spline')
    ```


#### `wait_for_robot_stopped()`
* **Purpose**: Wait for robot to stop moving.
* **Parameters**:
    * `timeout` (float, optional): Maximum time to wait in seconds. Default: 10.0
    * `poll_rate` (float, optional): How often to check in seconds. Default: 0.1
* **Returns**: Boolean (True if robot stopped, False if timeout)

#### `safe_move_with_retry()`
* **Purpose**: Execute move with automatic retry on failure.
* **Parameters**:
    * `move_func` (callable): The movement function to call
    * `*args`: Arguments for the movement function
    * `max_retries` (int, optional): Maximum number of retry attempts. Default: 3
    * `retry_delay` (float, optional): Delay between retries in seconds. Default: 1.0
    * `**kwargs`: Keyword arguments for the movement function
* **Returns**: Result from the movement function or error dictionary

#### `chain_smooth_motions()`
* **Purpose**: Chain multiple smooth motions with automatic continuity.
* **Parameters**:
    * `motions` (List[Dict]): List of motion dictionaries
    * `ensure_continuity` (bool, optional): Automatically set start_pose for continuity. Default: True
    * `frame` (str, optional): Reference frame ('WRF' or 'TRF') for all motions. Default: 'WRF'
    * `wait_for_ack` (bool, optional): Enable command tracking. Default: True
    * `timeout` (float, optional): Timeout per motion in seconds. Default: 30.0
* **Returns**: List of results for each motion
* **Python API Usage**:
    ```python
    from robot_api import chain_smooth_motions
    
    # Chain motions in world frame (default)
    motions = [
        {'type': 'circle', 'center': [200, 0, 200], 'radius': 50, 'duration': 5},
        {'type': 'arc', 'end_pose': [250, 50, 200, 0, 0, 90], 
         'center': [225, 25, 200], 'duration': 3}
    ]
    chain_smooth_motions(motions, ensure_continuity=True)
    
    # Chain motions in tool frame
    tool_motions = [
        {'type': 'circle', 'center': [0, 30, 0], 'radius': 25, 'duration': 4},
        {'type': 'arc', 'end_pose': [30, 30, 0, 0, 0, 45], 
         'center': [15, 15, 0], 'duration': 3}
    ]
    chain_smooth_motions(tool_motions, frame='TRF', ensure_continuity=True)
    ```

#### `check_command_status()`
* **Purpose**: Check status of a previously sent command.
* **Parameters**:
    * `command_id` (str): The command ID returned from a non-blocking command
* **Returns**: Dictionary with status information, or None if tracker not initialized
* **Dictionary Contents**:
    * `status` (str): Current status ('QUEUED', 'EXECUTING', 'COMPLETED', 'FAILED', 'CANCELLED', 'INVALID')
    * `details` (str): Additional status details
    * `completed` (bool): Whether command has finished
    * `sent_time` (datetime): When command was sent
    * `ack_time` (datetime, optional): When acknowledgment was received

#### `is_tracking_active()`
* **Purpose**: Check if command tracking system is active.
* **Parameters**: None
* **Returns**: Boolean (True if tracking is active, False otherwise)

#### `get_tracking_stats()`
* **Purpose**: Get resource usage statistics for tracking system.
* **Parameters**: None
* **Returns**: Dictionary with tracking statistics
* **Dictionary Contents**:
    * `active` (bool): Whether tracking is active
    * `commands_tracked` (int): Number of commands being tracked
    * `memory_bytes` (int): Approximate memory usage
    * `thread_active` (bool): Whether tracking thread is running

## 6. Setup & Operation

### Dependencies

The system is designed with a client-server architecture where most dependencies are only needed on the server (robot controller) side. The client API (`robot_api.py`) uses only standard Python libraries for UDP communication, making it lightweight and portable.

#### Server Dependencies (for `headless_commander.py`)
Install Python 3 and the following packages on the computer connected to the robot:

```bash
# Core robotics libraries
pip3 install roboticstoolbox-python==1.0.3
pip3 install numpy==1.23.4
pip3 install scipy==1.11.4
pip3 install spatialmath

# Serial communication and timing
pip3 install pyserial
pip3 install oclock

# User input
pip3 install keyboard
```

#### Client Dependencies (for `robot_api.py`)
The client API is designed to be lightweight with minimal dependencies:

```bash
# Only needed for get_robot_pose() matrix conversion
pip3 install numpy==1.23.4
pip3 install spatialmath

# All other functionality uses only Python standard library:
# socket, threading, time, uuid, datetime, collections, typing
```

**Note**: If you only need to send commands and don't use `get_robot_pose()`, the client requires NO external dependencies - only Python's built-in libraries.

### File Structure

#### Server Side (Robot Controller Computer)
Required files in the same folder:
* `headless_commander.py` - Main server/controller
* `PAROL6_ROBOT.py` - Robot configuration and kinematic model  
* `smooth_motion.py` - Advanced trajectory generation
* `GUI/files/` folder structure - For imports to work correctly

Optional:
* `com_port.txt` - Contains the USB COM port (e.g., COM5)

#### Client Side (Any Computer)
Only required file:
* `robot_api.py` - Client API for sending commands

The client can run on any computer on the same network as the server, or on the same computer in a different process.

### How to Operate

#### Starting the Server (Robot Controller)

1. **Connect Robot**: Ensure the robot is connected via USB to the controller computer.

2. **Start Controller**: On the robot controller computer, navigate to the folder containing the server files and run:
    ```bash
    python headless_commander.py
    ```
    The controller will:
    - Connect to the robot via serial port (prompts if `com_port.txt` not found)
    - Start listening for UDP commands on port 5001
    - Optionally home the robot on startup (unless disabled)

#### Sending Commands (Client)

Commands can be sent from:
- **Same Computer**: Run Python scripts or interactive sessions in another terminal
- **Different Computer**: Ensure network connectivity and update `SERVER_IP` in `robot_api.py`

3. **Send Commands**: Use the API functions from `robot_api.py`:
    ```python
    from robot_api import *
    
    # Example sequence
    home_robot()
    move_robot_joints([90, -90, 160, 12, 12, 180], duration=5.5)
    delay_robot(0.5)
    
    # Smooth motion example
    smooth_circle([200, 0, 200], radius=50, duration=5.0)
    
    # Non-blocking example with status checking
    cmd_id = move_robot_pose([250, 0, 200, 180, 0, 90], 
                             speed_percentage=50,
                             wait_for_ack=True, 
                             non_blocking=True)
    
    # Check status after some time
    import time
    time.sleep(2)
    status = check_command_status(cmd_id)
    if status:
        print(f"Command status: {status['status']}")
    ```

#### Network Configuration

If running client and server on different computers:

1. **Update Server IP**: In `robot_api.py`, modify the `SERVER_IP` variable:
    ```python
    SERVER_IP = "192.168.1.100"  # Replace with robot controller's IP
    SERVER_PORT = 5001  # Default port (usually no change needed)
    ```

2. **Firewall Settings**: Ensure UDP port 5001 is open on the robot controller computer.

3. **Network Requirements**: 
    - Both computers must be on the same network
    - Low latency recommended for real-time control
    - Command acknowledgments use port 5002 (optional feature)

### Advanced Usage with Acknowledgments

The acknowledgment system allows for sophisticated command management:

```python
from robot_api import *
import time

# Send multiple commands non-blocking
cmd1 = move_robot_joints([90, -45, 90, 0, 45, 180], 
                         duration=3, 
                         wait_for_ack=True, 
                         non_blocking=True)

cmd2 = smooth_circle([200, 0, 200], radius=30, 
                    duration=5, 
                    wait_for_ack=True, 
                    non_blocking=True)

# Monitor both commands
while True:
    status1 = check_command_status(cmd1)
    status2 = check_command_status(cmd2)
    
    if status1 and status1['completed'] and status2 and status2['completed']:
        print("Both commands completed!")
        break
    
    time.sleep(0.1)
```

## 7. Troubleshooting

* **Serial Connection Issues**: Check COM port in Device Manager (Windows) and update `com_port.txt`
* **Command Not Executing**: Verify robot is homed and E-stop is not pressed
* **Tracking Not Working**: Ensure `wait_for_ack=True` is set for commands
* **IK Failures**: Target pose may be unreachable; check robot workspace limits
* **Smooth Motion Errors**: Verify waypoints are reachable and properly formatted

For additional support, refer to the [PAROL commander software repository](https://github.com/PCrnjak/PAROL-commander-software).

Or you can head over to the [PAROL6 Discord channel](https://discord.com/invite/prjUvjmGpZ) for extra support
