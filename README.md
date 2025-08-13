# PAROL6 Headless Commander Documentation

## 1. Important Notes & Disclaimers
* **Software Origin**: This control system is based on the `experimental_kinematics` branch of the `PAROL_commander_software` repository. The core communication functions were derived from the `Serial_sender_good_latest.py` file; however, the approach to motion planning has been altered from the original implementation. This system was created by editing the `Commander_minimal_version.py` file, which was used as a base.
* **Automatic Homing on Startup**: By default, the `headless_commander.py` script will immediately command the robot to home itself upon startup. This is done for convenience but can be disabled. To prevent automatic homing, comment out or delete line 1556 (`command_queue.append(lambda: HomeCommand())`) in `headless_commander.py`.
* **AI-Assisted Development**: This code was developed with significant AI assistance. While the core logic has been corrected and improved, it has not been exhaustively tested in all scenarios. Users should proceed with caution and verify functionality for their specific needs.
* **Untested Gripper Functionality**: The `GripperCommand` for both pneumatic and electric grippers has been implemented based on the reference code but has not been tested on physical hardware. Please report any issues you encounter.

## 2. Safety Precautions & Disclaimer
This control software includes several built-in safety features designed to prevent damage to the robot and ensure predictable operation:
* **E-Stop Monitoring**: The system constantly checks the physical E-Stop button. If triggered, all motion is immediately halted, the command queue is cleared, and the robot is disabled. The system must be manually re-enabled by pressing the `'e'` key after the E-Stop is released.
* **Synchronized Speed Calculation**: For moves defined by a speed percentage (`MoveJoint`, `MovePose`), the system now calculates the maximum possible synchronized speed for all joints involved. This prevents individual joints from exceeding their limits and ensures predictable, smooth motion.
* **Inverse Kinematics (IK) Validation**: The system verifies that a valid kinematic solution exists for any pose-based command. If the target pose is unreachable, the command will fail safely before any motion occurs.

> **WARNING**: These are software-based safety measures and are not a substitute for responsible operation and a safe work environment. The user assumes all responsibility for safe operation. Always be attentive when the robot is active, ensure you have immediate access to the physical E-Stop, and operate the robot in a clear area.

## Instalation
Intructions are the same as for [commander software](https://github.com/PCrnjak/PAROL-commander-software) Follow the guide located [here](https://github.com/PCrnjak/PAROL-commander-software)

## 3. System Architecture
The system uses a server/client model to separate robot operation from command generation.
* **The Robot Controller (`headless_commander.py`)**: This script is the server and runs on the computer connected to the robot. Its high-frequency main loop handles a command queue that processes a sequence of object-oriented commands one at a time. A non-blocking UDP server listens for remote commands and adds them to the queue without interrupting the current operation.
* **The Remote Client (`robot_api.py`)**: The functions in this script act as the client, sending command messages to the robot controller via UDP. This decoupled design allows control from other programs or computers.

## 4. Command Reference & API Usage
This section details each command, its parameters, and provides a Python example using the `robot_api.py` functions.

### Motion & Logic Commands

#### `home_robot()`
* **Purpose**: Initiates the robot's built-in homing sequence.
* **Parameters**: None.
* **Python API Usage**:
    ```python
    from robot_api import home_robot
    home_robot()
    ```

#### `move_robot_joints()`
* **Purpose**: Moves joints to a target configuration (in degrees).
* **Parameters**:
    * `joint_angles` (`List[float]`): A list of 6 target angles in degrees for joints 1-6.
    * `duration` (`Optional[float]`): The total time for the movement in seconds.
    * `speed_percentage` (`Optional[int]`): The desired speed as a percentage (0-100) of the robot's synchronized maximum.
* > *Note: You must provide either `duration` or `speed_percentage`, but not both. If both are given, `duration` will take precedence.*
* **Python API Usage**:
    ```python
    from robot_api import move_robot_joints

    # Move by speed (75% of max)
    move_robot_joints([90, -45, 90, 0, 45, 180], speed_percentage=75)

    # Move by duration (in 5.5 seconds)
    move_robot_joints([0, -90, 180, 0, 0, 180], duration=5.5)
    ```
    
#### `move_robot_pose()`
* **Purpose**: Moves the end-effector to a Cartesian pose via a joint-based path.
* **Parameters**:
    * `pose` (`List[float]`): A list of 6 values for the target pose `[x, y, z, Rx, Ry, Rz]`, with positions in millimeters and rotations in degrees.
    * `duration` (`Optional[float]`): The total time for the movement in seconds.
    * `speed_percentage` (`Optional[int]`): The desired speed as a percentage (0-100).
* > *Note: You must provide either `duration` or `speed_percentage`, but not both. If both are given, `duration` will take precedence.*
* **Python API Usage**:
    ```python
    from robot_api import move_robot_pose

    # Move by speed (50% of max)
    move_robot_pose([250, 0, 200, 180, 0, 90], speed_percentage=50)

    # Move by duration (in 3 seconds)
    move_robot_pose([150, 100, 250, 180, 0, 90], duration=3.0)
    ```

#### `move_robot_cartesian()`
* **Purpose**: Moves the end-effector to a target pose in a guaranteed straight-line path.
* **Parameters**:
    * `pose` (`List[float]`): A list of 6 values for the target pose `[x, y, z, Rx, Ry, Rz]`.
    * `duration` (`float`): The required total time for the movement in seconds. Must be a realistic value.
* **Python API Usage**:
    ```python
    from robot_api import move_robot_cartesian

    # Move to pose in a straight line over 4 seconds
    move_robot_cartesian([200, -50, 180, 180, 0, 135], duration=4.0)
    ```

#### `jog_cartesian()`
* **Purpose**: Jogs the end-effector continuously along an axis.
* **Parameters**:
    * `frame` (`str`): The reference frame, either `'TRF'` (Tool) or `'WRF'` (World).
    * `axis` (`str`): The axis and direction (e.g., `'X+'`, `'Y-'`, `'RZ+'`).
    * `speed_percentage` (`int`): The jog speed as a percentage (0-100).
    * `duration` (`float`): The time in seconds to jog for.
* **Python API Usage**:
    ```python
    from robot_api import jog_cartesian

    # Jog in the tool's Z+ direction for 1.5s at 50% speed
    jog_cartesian(frame='TRF', axis='Z+', speed_percentage=50, duration=1.5)
    ```

#### `jog_robot_joint()`
* **Purpose**: Jogs a single joint by time or angular distance.
* **Parameters**:
    * `joint_index` (`int`): The joint to move. 0-5 for positive direction (J1-J6), 6-11 for negative direction.
    * `speed_percentage` (`int`): The jog speed as a percentage (0-100).
    * `duration` (`Optional[float]`): The time in seconds to jog for.
    * `distance_deg` (`Optional[float]`): The distance in degrees to jog.
* > *Note: You must provide either `duration` or `distance_deg`, but not both.*
* **Python API Usage**:
    ```python
    from robot_api import jog_robot_joint

    # Jog joint 1 (index 0) for 2 seconds
    jog_robot_joint(joint_index=0, speed_percentage=40, duration=2.0)

    # Jog joint 3 backwards by 15 degrees (index 2 -> negative is 2+6=8)
    jog_robot_joint(joint_index=8, speed_percentage=60, distance_deg=15)
    ```

#### `jog_multiple_joints()`
* **Purpose**: Allows you to jog multiple joints at the same time.
* **Parameters**:
  * `joints` (`List[int]`): A list of joint indices. Use 0-5 for positive direction and 6-11 for negative direction (e.g., 6 is J1-).
  * `speeds` (`List[float]`): A list of corresponding speeds (1-100%). The number of speeds must match the number of joints.
  * `duration (float)`: The duration of the jog in seconds.
* **Python API Usage**:
    ```python
    from robot_api import jog_multiple_joints

    # Jog joints J1, J4 and J6 at a speed of 70%, 40% and 60% respectively for a duration of 0.6 seconds
    jog_multiple_joints([0, 3, 5], [70, 40, 60], 0.6)

    # Jog joints J1, J4 and J6 in the opposite direction at a speed of 70%, 40% and 60% respectively for a duration of 1.2 seconds
    jog_multiple_joints([(0+6), (3+6), (5+6)], [70, 40, 60], 1.2)
    # Equivalent to jog_multiple_joints([6, 9, 11], [70, 40, 60], 1.2)

#### `delay_robot()`
* **Purpose**: Pauses the command queue execution.
* **Parameters**:
    * `duration` (`float`): The pause time in seconds.
* **Python API Usage**:
    ```python
    from robot_api import delay_robot

    # Pause for 2.5 seconds
    delay_robot(2.5)
    ```

#### `control_pneumatic_gripper()` / `control_electric_gripper()`
* **Purpose**: Controls the pneumatic or electric gripper. (Untested)
* **Parameters (Pneumatic)**:
    * `action` (`str`): The action to perform, either `'open'` or `'close'`.
    * `port` (`int`): The digital output port to use, either `1` or `2`.
* **Parameters (Electric)**:
    * `action` (`str`): The action to perform, either `'move'` or `'calibrate'`.
    * `position` (`Optional[int]`): Target position (0-255).
    * `speed` (`Optional[int]`): Movement speed (0-255).
    * `current` (`Optional[int]`): Max motor current (100-1000 mA).
* **Python API Usage**:
    ```python
    from robot_api import control_pneumatic_gripper, control_electric_gripper

    # Pneumatic
    control_pneumatic_gripper(action='open', port=1)

    # Electric Move
    control_electric_gripper(action='move', position=200, speed=150)

    # Electric Calibrate
    control_electric_gripper(action='calibrate')
    ```

### State Query Commands

#### `get_robot_pose()`
* **Purpose**: Queries the robot for its current end-effector pose.
* **Parameters**: None.
* **Returns**: A list of 6 values `[x, y, z, Rx, Ry, Rz]` or `None` on failure.
* **Python API Usage**:
    ```python
    from robot_api import get_robot_pose

    current_pose = get_robot_pose()
    if current_pose:
        print(f"Current Pose (x,y,z,r,p,y): {current_pose}")
    ```

#### `get_robot_joint_angles()`
* **Purpose**: Queries the robot for its current joint angles.
* **Parameters**: None.
* **Returns**: A list of 6 angles in degrees `[j1, j2, j3, j4, j5, j6]` or `None` on failure.
* **Python API Usage**:
    ```python
    from robot_api import get_robot_joint_angles

    angles = get_robot_joint_angles()
    if angles:
        print(f"Current Angles (deg): {angles}")
    ```
    
#### `get_robot_io()`
* **Purpose**: Requests the robot's current digital I/O status.
* **Parameters**: None.
* **Returns**: Returns a list [IN1, IN2, OUT1, OUT2, ESTOP] or None if it fails.
* **Python API Usage**:
    ```python
    from robot_api import get_robot_io

    data = get_robot_io()
    print(data)
    ```

#### `get_electric_gripper_status()`
* **Purpose**: Requests the electric gripper's current status.
* **Parameters**: None.
* **Returns**: Returns a list [ID, Position, Speed, Current, StatusByte, ObjectDetected] or None.
* **Python API Usage**:
    ```python
    from robot_api import get_electric_gripper_status

    data = get_electric_gripper_status()
    print(data)
    ```

## 5. Setup & Operation
### Dependencies
Before running the system, ensure you have the required Python packages installed. You can install them using pip:

pip install roboticstoolbox-python numpy oclock pyserial keyboard

File Structure 
For the system to work correctly, the following script files must all be located in the 
same folder: 
● headless_commander.py (The main server/controller) 
● robot_api.py (The client API for sending commands) 
● PAROL6_ROBOT.py (The robot's specific configuration and kinematic model) 

> [!NOTE]
> com_port.txt is optional and it needs to have a single element and that is your USB com port of the robot. For example COM5
>

As long as these three files are kept together, their parent folder can be located <br />
anywhere on your computer. <br />
How to Operate the System  <br />
1. Start Controller: In a terminal, navigate to your folder and run the main controller <br />
script: <br />
```python
python headless_commander.py
```
2. Send Commands: In a separate script or terminal, you can import and use the 
functions from robot_api.py to send commands to the running controller. For 
example: 

```python
from robot_api import move_robot_joints
move_robot_joints([90, -90, 160, 12, 12, 180], duration=5.5)
delay_robot(0.2)
move_robot_joints([50, -60, 180, -12, 32, 0], duration=5.5)
delay_robot(0.2)
```


Or use test_script.py <br />

* **Python API Usage**:
    ```python
    # your_script.py 
    from robot_api import move_robot_joints, home_robot, delay_robot, get_robot_joint_angles, control_pneumatic_gripper,get_robot_pose, control_electric_gripper, move_robot_pose,move_robot_cartesian,get_electric_gripper_status,get_robot_io
    import time
    print("Homing robot...") 
    time.sleep(2)
    control_electric_gripper(action = "calibrate")
    time.sleep(2)
    control_electric_gripper(action='move', position=100, speed=150, current = 200) 
    time.sleep(2)
    control_electric_gripper(action='move', position=200, speed=150, current = 200) 
    time.sleep(2)
    print(get_robot_joint_angles())
    print(get_robot_pose())
    print("Moving to new position...") 
    control_pneumatic_gripper("open",1)
    time.sleep(0.3)
    control_pneumatic_gripper("close",1)
    time.sleep(0.3)
    control_pneumatic_gripper("open",1)
    time.sleep(0.3)
    control_pneumatic_gripper("close",1)
    time.sleep(0.3)
    move_robot_joints([90, -90, 160, 12, 12, 180], duration=5.5)
    time.sleep(6)
    move_robot_joints([50, -60, 180, -12, 32, 0], duration=5.5)
    time.sleep(6)
    move_robot_joints([90, -90, 160, 12, 12, 180], duration=5.5)
    time.sleep(6)
    move_robot_pose([7, 250, 200, -100, 0, -90], duration=5.5) 
    time.sleep(6)
    move_robot_cartesian([7, 250, 150, -100, 0, -90], speed_percentage=50) 
    delay_robot(0.2)
    print(get_electric_gripper_status())
    print(get_robot_io())
    ```
