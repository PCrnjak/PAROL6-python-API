"""
Hardware tests for manual robot movements.

These tests require actual robot hardware and human confirmation.
They are marked with @pytest.mark.hardware and require the --run-hardware flag.

SAFETY NOTICE: These tests will move the physical robot. Ensure the robot
workspace is clear and E-stop is within reach before running.
"""

import os
import sys
import time

import pytest

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Removed direct robot_api import - using client fixture from conftest.py

# Define the safe starting joint configuration for all hardware tests
# This ensures consistency and repeatability for each test
# Angles: [J1, J2, J3, J4, J5, J6] in degrees
SAFE_SMOOTH_START_JOINTS = [42.697, -89.381, 144.831, -0.436, 31.528, 180.0]


def initialize_hardware_position(client, human_prompt) -> list[float] | None:
    """
    Moves the robot to the predefined safe starting joint angles.

    Args:
        client: RobotClient fixture from conftest.py
        human_prompt: Fixture for human confirmation

    Returns:
        Robot's Cartesian pose after moving, or None if failed
    """

    print(f"Moving to safe starting position: {SAFE_SMOOTH_START_JOINTS}")

    # Move to the joint position
    result = client.move_joints(SAFE_SMOOTH_START_JOINTS, duration=4, wait_for_ack=True, timeout=15)
    print(f"Move command result: {result}")

    # Wait until robot stops
    if client.wait_until_stopped(timeout=15):
        print("Robot has reached the starting position.")
        time.sleep(1)
        start_pose = client.get_pose_rpy()  # Get [x,y,z,rx,ry,rz] format
        if start_pose:
            print(f"Starting pose: {[round(p, 2) for p in start_pose]}")
            return start_pose
        else:
            print("ERROR: Could not retrieve robot pose after moving.")
            return None
    else:
        print("ERROR: Robot did not stop in time.")
        return None


@pytest.mark.hardware
@pytest.mark.slow
class TestHardwareBasicMoves:
    """Test basic robot movements with hardware."""

    def test_hardware_homing(self, client, human_prompt):
        """Test robot homing sequence."""
        if not human_prompt(
            "Ready to test robot homing?\n"
            "This will home all joints to their reference positions.\n"
            "Ensure robot workspace is completely clear."
        ):
            pytest.skip("User declined homing test")

        # Check E-stop status first
        if client.is_estop_pressed():
            pytest.fail("E-stop is pressed. Release E-stop before testing.")

        print("Starting homing sequence...")
        result = client.home(wait_for_ack=True, timeout=60)

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        # Wait for homing to complete - use client's built-in wait
        if client.wait_until_stopped(timeout=90, show_progress=True):
            print("Homing completed successfully")
        else:
            pytest.fail("Robot homing did not complete within timeout")

    def test_small_joint_movements(self, client, human_prompt):
        """Test small joint movements for safety verification."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test small joint movements?\n"
            "Robot will move each joint individually by small amounts.\n"
            "Watch for smooth, controlled motion."
        ):
            pytest.skip("User declined joint movement test")

        # Test small movements on each joint
        for joint_idx in range(6):
            print(f"Testing joint {joint_idx + 1} movement...")

            # Small positive movement
            result = client.jog_joint(
                joint_idx, speed_percentage=20, duration=1.0, wait_for_ack=True
            )

            assert isinstance(result, dict)
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

            client.wait_until_stopped(timeout=5)

            # Small negative movement (return) - use joint_idx+6 for reverse direction
            result = client.jog_joint(
                joint_idx + 6,  # +6 indicates reverse direction
                speed_percentage=20,
                duration=1.0,
                wait_for_ack=True,
            )

            client.wait_until_stopped(timeout=5)

        print("All joint movements completed successfully")

    def test_small_cartesian_movements(self, client, human_prompt):
        """Test small Cartesian movements in different axes."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test small Cartesian movements?\n"
            "Robot will move end-effector in X, Y, Z directions.\n"
            "Movements will be small (10mm) and slow (20% speed)."
        ):
            pytest.skip("User declined Cartesian movement test")

        # Test movements in each axis
        axes = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]

        for axis in axes:
            print(f"Testing Cartesian jog in {axis} direction...")

            result = client.jog_cartesian(
                frame="WRF", axis=axis, speed_percentage=20, duration=1.0, wait_for_ack=True
            )

            assert isinstance(result, dict)
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

            time.sleep(2.0)
            client.wait_until_stopped(timeout=5)

        print("All Cartesian movements completed successfully")


@pytest.mark.hardware
@pytest.mark.slow
class TestHardwareSmoothMotion:
    """Test smooth motion commands with actual hardware."""

    def test_hardware_smooth_circle(self, client, human_prompt):
        """Test smooth circular motion on hardware."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test smooth circular motion?\n"
            "Robot will execute a 30mm radius circle in XY plane.\n"
            "Motion will be slow and controlled (5 second duration).\n"
            "Watch for smooth, continuous motion without stops."
        ):
            pytest.skip("User declined circle test")

        # Execute smooth circle
        center_point = [start_pose[0], start_pose[1] + 30, start_pose[2]]

        print(f"Executing circle: center={center_point}, radius=30mm")
        result = client.smooth_circle(
            center=center_point,
            radius=30.0,
            plane="XY",
            duration=5.0,
            clockwise=False,
            wait_for_ack=True,
            timeout=15,
        )

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        # Wait for completion
        if client.wait_until_stopped(timeout=15):
            print("Circle motion completed successfully")

            if not human_prompt(
                "Did the robot execute a smooth circular motion?\n"
                "Motion should have been continuous without stops or jerks."
            ):
                pytest.fail("User reported motion was not smooth")
        else:
            pytest.fail("Robot did not stop after circle motion timeout")

    def test_hardware_smooth_arc(self, client, human_prompt):
        """Test smooth arc motion on hardware."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test smooth arc motion?\n"
            "Robot will execute an arc from current position to a new pose.\n"
            "Motion will be controlled and smooth."
        ):
            pytest.skip("User declined arc test")

        # Define arc motion
        end_pose = [
            start_pose[0] + 40,
            start_pose[1] + 20,
            start_pose[2],
            start_pose[3],
            start_pose[4],
            start_pose[5] + 45,
        ]
        center = [start_pose[0] + 20, start_pose[1], start_pose[2]]

        print(f"Executing arc: end={end_pose[:3]}, center={center}")
        result = client.smooth_arc_center(
            end_pose=end_pose,
            center=center,
            duration=4.0,
            clockwise=True,
            wait_for_ack=True,
            timeout=12,
        )

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        if client.wait_until_stopped(timeout=12):
            print("Arc motion completed successfully")

            if not human_prompt(
                "Did the robot execute a smooth arc motion?\n"
                "Path should have been curved, not straight lines."
            ):
                pytest.fail("User reported arc motion was not smooth")
        else:
            pytest.fail("Robot did not stop after arc motion timeout")

    def test_hardware_smooth_spline(self, client, human_prompt):
        """Test smooth spline motion through multiple waypoints."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test smooth spline motion?\n"
            "Robot will move through multiple waypoints with smooth transitions.\n"
            "Motion should be continuous without stops at waypoints."
        ):
            pytest.skip("User declined spline test")

        # Define spline waypoints
        waypoints = [
            [
                start_pose[0] + 20,
                start_pose[1] + 10,
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5],
            ],
            [
                start_pose[0] + 35,
                start_pose[1] + 25,
                start_pose[2] + 10,
                start_pose[3],
                start_pose[4],
                start_pose[5] + 20,
            ],
            [
                start_pose[0] + 20,
                start_pose[1] + 35,
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5] + 40,
            ],
            [
                start_pose[0] + 5,
                start_pose[1] + 20,
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5],
            ],
        ]

        print(f"Executing spline through {len(waypoints)} waypoints")
        result = client.smooth_spline(
            waypoints=waypoints, duration=6.0, frame="WRF", wait_for_ack=True, timeout=20
        )

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        if client.wait_until_stopped(timeout=20):
            print("Spline motion completed successfully")

            if not human_prompt(
                "Did the robot move smoothly through all waypoints?\n"
                "Motion should have been continuous without stops at intermediate points."
            ):
                pytest.fail("User reported spline motion was not smooth")
        else:
            pytest.fail("Robot did not stop after spline motion timeout")


@pytest.mark.hardware
@pytest.mark.slow
class TestHardwareAdvancedSmooth:
    """Test advanced smooth motion features with hardware."""

    def test_hardware_helix_motion(self, client, human_prompt):
        """Test helical motion on hardware."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test helical motion?\n"
            "Robot will execute a helical (screw-like) motion.\n"
            "Motion combines circular movement with vertical progression.\n"
            "Radius: 25mm, Height: 40mm, 3 revolutions."
        ):
            pytest.skip("User declined helix test")

        # Execute helix motion
        center = [start_pose[0], start_pose[1] + 25, start_pose[2] - 20]

        print(f"Executing helix: center={center}, radius=25mm, height=40mm")
        result = client.smooth_helix(
            center=center,
            radius=25.0,
            pitch=13.0,  # 40mm / ~3 revolutions
            height=40.0,
            duration=8.0,
            clockwise=False,
            wait_for_ack=True,
            timeout=20,
        )

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        if client.wait_until_stopped(timeout=20):
            print("Helix motion completed successfully")

            if not human_prompt("Did the robot execute a smooth helical motion?\n"):
                pytest.fail("User reported helix motion was incorrect")
        else:
            pytest.fail("Robot did not stop after helix motion timeout")

    def test_hardware_reference_frame_comparison(self, client, human_prompt):
        """Test motion in different reference frames (WRF vs TRF)."""
        start_pose = initialize_hardware_position(client, human_prompt)
        if not start_pose:
            pytest.skip("Failed to reach starting position")

        if not human_prompt(
            "Ready to test reference frame comparison?\n"
            "Robot will execute similar motions in World frame (WRF) and Tool frame (TRF).\n"
            "You should observe different motion patterns."
        ):
            pytest.skip("User declined reference frame test")

        # Test 1: Circle in World Reference Frame
        print("Executing circle in World Reference Frame (WRF)...")
        result_wrf = client.smooth_circle(
            center=[start_pose[0], start_pose[1] + 30, start_pose[2]],
            radius=20,
            duration=4.0,
            frame="WRF",
            plane="XY",
            wait_for_ack=True,
            timeout=12,
        )

        assert isinstance(result_wrf, dict)
        assert result_wrf.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        client.wait_until_stopped(timeout=12)
        time.sleep(2)

        # Test 2: Circle in Tool Reference Frame
        print("Executing circle in Tool Reference Frame (TRF)...")
        result_trf = client.smooth_circle(
            center=[0, 30, 0],  # Relative to tool position
            radius=20,
            duration=4.0,
            frame="TRF",
            plane="XY",  # Tool's XY plane
            wait_for_ack=True,
            timeout=12,
        )

        assert isinstance(result_trf, dict)
        assert result_trf.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        client.wait_until_stopped(timeout=12)

        if not human_prompt(
            "Did you observe different motion patterns?\n"
            "WRF motion should follow world coordinates.\n"
            "TRF motion should follow tool orientation."
        ):
            pytest.fail("User reported motion patterns were not as expected")


@pytest.mark.hardware
@pytest.mark.slow
class TestHardwareSafety:
    """Test hardware safety features and error conditions."""

    def test_joint_limit_safety(self, client, human_prompt):
        """Test joint limit safety (if supported by controller)."""
        if not human_prompt(
            "Ready to test joint limit safety?\n"
            "This will attempt to move a joint near its limit to test safety systems.\n"
            "Controller should prevent unsafe movements.\n"
            "This test is informational only."
        ):
            pytest.skip("User declined joint limit test")

        # Try to move to a potentially extreme position (should be rejected or limited)
        extreme_joints = [180.0, -180.0, 180.0, -180.0, 180.0, -180.0]  # Extreme angles as floats

        print("Testing extreme joint angles (should be rejected or limited)...")
        result = client.move_joints(
            extreme_joints,
            speed_percentage=5,  # Very slow for safety
            wait_for_ack=True,
            timeout=10,
        )

        print(f"Result of extreme joint command: {result}")

        # This test is informational - we just want to see how the system responds
        time.sleep(5.0)

        # Return to safe position
        initialize_hardware_position(client, human_prompt)


@pytest.mark.hardware
@pytest.mark.slow
class TestHardwareLegacySequence:
    """Test the exact sequence from the legacy test_script.py for verified safe operation."""

    def test_legacy_script_safe_sequence(self, client, human_prompt):
        """
        Reproduce the exact sequence from test_script.py with verified safe waypoints.

        This test uses the exact same joint angles, poses, and parameters that were
        manually verified to be safe in the original test script.
        """
        if not human_prompt(
            "Ready to execute the legacy safe test sequence?\n"
            "This will reproduce the exact movements from test_script.py:\n"
            "- Electric gripper calibration and moves (pos 100, then 200)\n"
            "- Pneumatic gripper open/close sequence\n"
            "- Joint moves: [90,-90,160,12,12,180] -> [50,-60,180,-12,32,0] -> back\n"
            "- Pose move: [7,250,200,-100,0,-90]\n"
            "- Cartesian move: [7,250,150,-100,0,-90]\n"
            "These waypoints were verified safe in the original script."
        ):
            pytest.skip("User declined legacy sequence test")

        # Check E-stop status first
        if client.is_estop_pressed():
            pytest.fail("E-stop is pressed. Release E-stop before testing.")

        print("Starting legacy test sequence with verified safe waypoints...")

        # Electric gripper calibration and moves
        print("Calibrating electric gripper...")
        result = client.control_electric_gripper(action="calibrate", wait_for_ack=True, timeout=10)
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(2)

        print("Moving electric gripper to position 100...")
        result = client.control_electric_gripper(
            action="move", position=100, speed=150, current=200, wait_for_ack=True, timeout=10
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(2)

        print("Moving electric gripper to position 200...")
        result = client.control_electric_gripper(
            action="move", position=200, speed=150, current=200, wait_for_ack=True, timeout=10
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(2)

        # Get and verify initial status
        print("Getting robot joint angles and pose...")
        angles = client.get_angles()
        pose = client.get_pose_rpy()
        assert angles is not None
        assert pose is not None
        print(f"Initial angles: {angles}")
        print(f"Initial pose: {pose}")

        # Pneumatic gripper sequence (exact timing from test_script.py)
        print("Testing pneumatic gripper sequence...")
        client.control_pneumatic_gripper("open", 1)
        time.sleep(0.3)
        client.control_pneumatic_gripper("close", 1)
        time.sleep(0.3)
        client.control_pneumatic_gripper("open", 1)
        time.sleep(0.3)
        client.control_pneumatic_gripper("close", 1)
        time.sleep(0.3)

        # Joint movement sequence (exact waypoints and timing from test_script.py)
        print("Moving to first joint position: [90, -90, 160, 12, 12, 180]...")
        result = client.move_joints(
            [90, -90, 160, 12, 12, 180], duration=5.5, wait_for_ack=True, timeout=15
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(6)

        print("Moving to second joint position: [50, -60, 180, -12, 32, 0]...")
        result = client.move_joints(
            [50, -60, 180, -12, 32, 0], duration=5.5, wait_for_ack=True, timeout=15
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(6)

        print("Moving back to first joint position: [90, -90, 160, 12, 12, 180]...")
        result = client.move_joints(
            [90, -90, 160, 12, 12, 180], duration=5.5, wait_for_ack=True, timeout=15
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(6)

        # Pose movement (exact waypoint from test_script.py)
        print("Moving to pose: [7, 250, 200, -100, 0, -90]...")
        result = client.move_pose(
            [7, 250, 200, -100, 0, -90], duration=5.5, wait_for_ack=True, timeout=15
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]
        time.sleep(6)

        # Cartesian movement (exact waypoint from test_script.py)
        print("Moving cartesian to: [7, 250, 150, -100, 0, -90]...")
        result = client.move_cartesian(
            [7, 250, 150, -100, 0, -90], speed_percentage=50, wait_for_ack=True, timeout=15
        )
        if isinstance(result, dict):
            assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        # Final status checks (exact from test_script.py)
        print("Getting final gripper and IO status...")
        gripper_status = client.get_gripper_status()
        io_status = client.get_io()

        assert gripper_status is not None
        assert io_status is not None

        print(f"Final gripper status: {gripper_status}")
        print(f"Final IO status: {io_status}")

        if not human_prompt(
            "Legacy test sequence completed successfully.\n"
            "Did all movements execute safely and as expected?\n"
            "This sequence replicates the verified safe waypoints from the original test_script.py."
        ):
            pytest.fail("User reported legacy sequence did not execute correctly")

        print("Legacy safe sequence test completed successfully")


@pytest.mark.hardware
@pytest.mark.slow
class TestHardwareGripper:
    """Test gripper functionality with hardware."""

    def test_pneumatic_gripper(self, client, human_prompt):
        """Test pneumatic gripper operation."""
        if not human_prompt(
            "Ready to test pneumatic gripper?\n"
            "Ensure gripper is connected and air pressure is available.\n"
            "Gripper will open and close on port 1."
        ):
            pytest.skip("User declined gripper test")

        # Test gripper open
        print("Opening pneumatic gripper...")
        result = client.control_pneumatic_gripper("open", 1, wait_for_ack=True, timeout=5)

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        time.sleep(2.0)

        if not human_prompt("Did the gripper open? Confirm before continuing."):
            pytest.fail("User reported gripper did not open")

        # Test gripper close
        print("Closing pneumatic gripper...")
        result = client.control_pneumatic_gripper("close", 1, wait_for_ack=True, timeout=5)

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        time.sleep(2.0)

        if not human_prompt("Did the gripper close? Confirm operation."):
            pytest.fail("User reported gripper did not close")

        print("Pneumatic gripper test completed successfully")

    def test_electric_gripper(self, client, human_prompt):
        """Test electric gripper operation including calibration."""
        if not human_prompt(
            "Ready to test electric gripper?\n"
            "Ensure electric gripper is connected and powered.\n"
            "Gripper will calibrate, then move to different positions."
        ):
            pytest.skip("User declined electric gripper test")

        # Get current gripper status
        gripper_status = client.get_gripper_status()
        if gripper_status:
            print(f"Initial gripper status: {gripper_status}")

        # Test gripper calibration (from legacy test_script.py)
        print("Calibrating electric gripper...")
        result = client.control_electric_gripper(action="calibrate", wait_for_ack=True, timeout=10)

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        time.sleep(2.0)

        # Test gripper movement
        print("Moving electric gripper to position 100...")
        result = client.control_electric_gripper(
            "move", position=100, speed=100, current=400, wait_for_ack=True, timeout=10
        )

        assert isinstance(result, dict)
        assert result.get("status") in ["COMPLETED", "QUEUED", "EXECUTING"]

        time.sleep(3.0)

        # Check new position
        new_status = client.get_gripper_status()
        if new_status:
            print(f"Gripper status after move: {new_status}")

        if not human_prompt(
            "Did the electric gripper move to the commanded position?\n"
            "Check gripper position and movement quality."
        ):
            pytest.fail("User reported electric gripper did not move correctly")

        print("Electric gripper test completed successfully")


if __name__ == "__main__":
    pytest.main([__file__])
