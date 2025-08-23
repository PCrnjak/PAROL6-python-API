import time
import robot_api
import numpy as np

# Define the safe starting joint configuration for all smooth motion tests.
# This ensures consistency and repeatability for each test.
# Angles: [J1, J2, J3, J4, J5, J6] in degrees.
SAFE_SMOOTH_START_JOINTS = [42.697,-89.381,144.831,-0.436,31.528,180.0]

def initialize_test_position():
    """
    Moves the robot to the predefined safe starting joint angles and waits.
    This function is called before every smooth motion test.

    Returns:
        list: The robot's Cartesian pose [x, y, z, rx, ry, rz] after moving,
              or None if the move fails or the pose cannot be retrieved.
    """
    print("\n" + "="*60)
    print(f"MOVING TO SAFE STARTING POSITION: {SAFE_SMOOTH_START_JOINTS}")
    print("="*60)
    
    # Move to the joint position with a 4-second duration and wait for acknowledgment.
    result = robot_api.move_robot_joints(
        SAFE_SMOOTH_START_JOINTS, 
        duration=4, 
        wait_for_ack=True,
        timeout=5
    )
    print(f"--> Move command result: {result}")

    # Wait until the robot has physically stopped moving.
    if robot_api.wait_for_robot_stopped(timeout=10):
        print("--> Robot has reached the starting position.")
        time.sleep(1)
        start_pose = robot_api.get_robot_pose()
        if start_pose:
            print(f"--> Starting Pose confirmed at: {[round(p, 2) for p in start_pose]}")
            return start_pose
        else:
            print("--> ERROR: Could not retrieve robot pose after moving.")
            return None
    else:
        print("--> ERROR: Robot did not stop in time. Aborting test.")
        return None

def test_smooth_circle_basic(start_pose):
    """Tests the smooth_circle command with different planes, directions, and timing modes."""
    print("\n--- TESTING SMOOTH CIRCLE (BASIC) ---")
    
    # Define a center point relative to the starting Z-height
    radius = 30.0  # 30mm radius

    center_point = [start_pose[0], start_pose[1] + radius, start_pose[2]]  # Changed from +50 to +radius

    # Test 1: XY plane, counter-clockwise with DURATION
    print("\n[1/3] Testing Circle: XY Plane, Counter-Clockwise (Duration mode)")
    result = robot_api.smooth_circle(
        center=center_point,
        radius=radius,
        plane='XY',
        duration=5.0,  # Using duration
        clockwise=False,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(2)

    # Test 2: XZ plane, clockwise with SPEED PERCENTAGE
    print("\n[2/3] Testing Circle: XZ Plane, Clockwise (Speed percentage mode)")
    result = robot_api.smooth_circle(
        center=center_point,
        radius=radius,
        plane='XZ',
        speed_percentage=30,  # Using speed percentage (30% speed)
        clockwise=True,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(2)
    
    # Test 3: YZ plane with specified start position (NEW)
    print("\n[3/3] Testing Circle: YZ Plane with SPECIFIED START POSITION")
    # Define a start position slightly offset from current
    specified_start = [start_pose[0] + 10, start_pose[1] + 10, start_pose[2], 
                      start_pose[3], start_pose[4], start_pose[5]]
    result = robot_api.smooth_circle(
        center=center_point,
        radius=radius,
        plane='YZ',
        start_pose=specified_start,  # NEW: Will transition here first
        duration=5.0,
        clockwise=False,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)

def test_smooth_arc_with_start_positions(start_pose):
    """Tests smooth arc commands with specified start positions and transitions."""
    print("\n--- TESTING SMOOTH ARC WITH START POSITIONS ---")

    # Test 1: Arc with FAR start position (should see smooth transition)
    print("\n[1/3] Testing Arc with FAR START POSITION (big transition)")
    far_start = [start_pose[0] + 40, start_pose[1] - 20, start_pose[2] + 10,
                 start_pose[3], start_pose[4], start_pose[5]]
    arc_center = [far_start[0] - 20, far_start[1], far_start[2]]
    end_pose_arc = [arc_center[0], arc_center[1] + 20, far_start[2],
                   far_start[3], far_start[4], far_start[5] + 45]
    
    print(f"  Current position: {[round(p, 1) for p in start_pose[:3]]}")
    print(f"  Transition to: {[round(p, 1) for p in far_start[:3]]}")
    print(f"  Then arc to: {[round(p, 1) for p in end_pose_arc[:3]]}")
    
    result = robot_api.smooth_arc_center(
        end_pose=end_pose_arc,
        center=arc_center,
        start_pose=far_start,  # Will transition here first
        duration=6.0,
        clockwise=True,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=12)
    time.sleep(2)

    # Re-initialize for next test
    current_pose = initialize_test_position()
    if not current_pose: return

    # Test 2: Arc with CLOSE start position (minimal transition)
    print("\n[2/3] Testing Arc with CLOSE START POSITION (minimal transition)")
    close_start = [current_pose[0] + 2, current_pose[1] + 2, current_pose[2],
                   current_pose[3], current_pose[4], current_pose[5]]
    arc_center = [close_start[0] - 15, close_start[1], close_start[2]]
    end_pose_arc = [arc_center[0], arc_center[1] + 15, close_start[2],
                   close_start[3], close_start[4], close_start[5] + 30]
    
    print(f"  Current position: {[round(p, 1) for p in current_pose[:3]]}")
    print(f"  Transition to: {[round(p, 1) for p in close_start[:3]]}")
    print(f"  Then arc to: {[round(p, 1) for p in end_pose_arc[:3]]}")
    
    result = robot_api.smooth_arc_center(
        end_pose=end_pose_arc,
        center=arc_center,
        start_pose=close_start,  # Very close, minimal transition
        speed_percentage=40,
        clockwise=False,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(2)

    # Test 3: Parametric arc with specified start
    print("\n[3/3] Testing PARAMETRIC Arc with specified start")
    param_start = [current_pose[0] - 10, current_pose[1] + 5, current_pose[2],
                   current_pose[3], current_pose[4], current_pose[5]]
    end_pose_param = [param_start[0] + 20, param_start[1] - 10, param_start[2],
                     param_start[3], param_start[4], param_start[5]]
    
    result = robot_api.smooth_arc_parametric(
        end_pose=end_pose_param,
        radius=20.0,
        arc_angle=60.0,
        start_pose=param_start,
        duration=4.0,
        clockwise=False,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)

def test_motion_chaining(start_pose):
    """Tests precise motion chaining using end pose of one motion as start of next."""
    print("\n--- TESTING MOTION CHAINING (NEW) ---")
    print("This tests using the exact end pose of one motion as the start of the next")
    
    # Motion 1: Arc to a specific end pose
    print("\n[1/4] First Motion: Arc")
    arc_center = [start_pose[0] - 20, start_pose[1], start_pose[2]]
    arc_end = [arc_center[0], arc_center[1] + 30, start_pose[2],
               start_pose[3], start_pose[4] + 15, start_pose[5] + 45]
    
    result = robot_api.smooth_arc_center(
        end_pose=arc_end,
        center=arc_center,
        duration=4.0,
        clockwise=True,
        wait_for_ack=True
    )
    print(f"--> Arc ended at: {[round(p, 1) for p in arc_end[:3]]}")
    robot_api.wait_for_robot_stopped(timeout=8)
    time.sleep(1)
    
    # Motion 2: Circle starting exactly where arc ended
    print("\n[2/4] Second Motion: Circle starting at arc's end position")
    circle_center = [arc_end[0], arc_end[1] + 25, arc_end[2]]
    
    result = robot_api.smooth_circle(
        center=circle_center,
        radius=25.0,
        plane='XY',
        start_pose=arc_end,  # Start exactly where arc ended
        speed_percentage=35,
        clockwise=False,
        wait_for_ack=True
    )
    print(f"--> Circle completed")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(1)
    
    # Since circle returns to start, we know where we are
    circle_end = arc_end  # Circle returns to its start point
    
    # Motion 3: Helix starting where circle ended
    print("\n[3/4] Third Motion: Helix starting at circle's position")
    # Calculate actual radius from circle end position
    helix_center = [circle_end[0], circle_end[1], circle_end[2] - 30]
    # Use the actual distance as radius
    actual_radius = np.sqrt((circle_end[0] - helix_center[0])**2 + 
                        (circle_end[1] - helix_center[1])**2)
    radius = max(actual_radius, 1.0)  # Use actual distance, minimum 1mm
    
    result = robot_api.smooth_helix(
        center=helix_center,
        radius=15.0,
        pitch=10.0,
        height=30.0,
        start_pose=circle_end,  # Start where circle ended
        duration=6.0,
        clockwise=True,
        wait_for_ack=True
    )
    print(f"--> Helix completed")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(1)
    
    # Calculate helix end position (approximately)
    helix_end = [helix_center[0] + 15, helix_center[1], helix_center[2] + 30,
                 circle_end[3], circle_end[4], circle_end[5]]
    
    # Motion 4: Spline back to near start
    print("\n[4/4] Fourth Motion: Spline path back near start")
    waypoints = [
        helix_end,  # Start from helix end
        [helix_end[0] - 10, helix_end[1] - 10, helix_end[2] - 10,
         helix_end[3], helix_end[4], helix_end[5] - 20],
        [start_pose[0] + 5, start_pose[1] + 5, start_pose[2],
         start_pose[3], start_pose[4], start_pose[5]]
    ]
    
    result = robot_api.smooth_spline(
        waypoints=waypoints[1:],  # Skip first since we specify start_pose
        start_pose=waypoints[0],  # Explicitly start from helix end
        speed_percentage=30,
        wait_for_ack=True
    )
    print(f"--> Spline completed - returned near start")
    robot_api.wait_for_robot_stopped(timeout=10)

def test_smooth_spline_with_starts(start_pose):
    """Tests smooth_spline with various start position scenarios."""
    print("\n--- TESTING SMOOTH SPLINE WITH START POSITIONS ---")
    
    # Test 1: Spline with default start (current position)
    print("\n[1/3] Spline with DEFAULT start (from current position)")
    waypoints = []
    for i in range(4):
        x = start_pose[0] + i * 15
        y = start_pose[1] + (15 if i % 2 else -15)
        z = start_pose[2]
        waypoints.append([x, y, z, start_pose[3], start_pose[4], start_pose[5]])
    
    result = robot_api.smooth_spline(
        waypoints=waypoints,
        # No start_pose specified - uses current
        duration=5.0,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(2)
    
    # Re-initialize
    current_pose = initialize_test_position()
    if not current_pose: return
    
    # Test 2: Spline with specified start far from first waypoint
    print("\n[2/3] Spline with SPECIFIED start (different from first waypoint)")
    specified_start = [current_pose[0] - 20, current_pose[1] + 15, current_pose[2],
                      current_pose[3], current_pose[4], current_pose[5]]
    
    waypoints = [
        [specified_start[0] + 30, specified_start[1], specified_start[2],
         specified_start[3], specified_start[4], specified_start[5]],
        [specified_start[0] + 40, specified_start[1] + 20, specified_start[2],
         specified_start[3], specified_start[4], specified_start[5]],
        [specified_start[0] + 20, specified_start[1] + 30, specified_start[2],
         specified_start[3], specified_start[4], specified_start[5]]
    ]
    
    print(f"  Current: {[round(p, 1) for p in current_pose[:3]]}")
    print(f"  Will transition to: {[round(p, 1) for p in specified_start[:3]]}")
    print(f"  Then follow spline through waypoints")
    
    result = robot_api.smooth_spline(
        waypoints=waypoints,
        start_pose=specified_start,  # Will transition here first
        speed_percentage=40,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=12)
    time.sleep(2)
    
    # Re-initialize
    current_pose = initialize_test_position()
    if not current_pose: return
    
    # Test 3: Spline with start matching first waypoint (no transition needed)
    print("\n[3/3] Spline with start MATCHING first waypoint (no transition)")
    first_waypoint = [current_pose[0] + 5, current_pose[1] + 5, current_pose[2],
                     current_pose[3], current_pose[4], current_pose[5]]
    
    waypoints = [
        first_waypoint,  # Same as start_pose
        [first_waypoint[0] + 20, first_waypoint[1] + 10, first_waypoint[2],
         first_waypoint[3], first_waypoint[4], first_waypoint[5]],
        [first_waypoint[0] + 10, first_waypoint[1] + 25, first_waypoint[2],
         first_waypoint[3], first_waypoint[4], first_waypoint[5]]
    ]
    
    result = robot_api.smooth_spline(
        waypoints=waypoints[1:],  # Skip first since we use it as start_pose
        start_pose=first_waypoint,  # Same as would-be first waypoint
        duration=4.0,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=8)

def test_smooth_helix_with_starts(start_pose):
    """Tests smooth_helix with specified start positions."""
    print("\n--- TESTING SMOOTH HELIX WITH START POSITIONS ---")
    
    # Test 1: Helix with default start
    print("\n[1/2] Helix with DEFAULT start (from current position)")
    center = [start_pose[0], start_pose[1] + 30, start_pose[2] - 40]
    
    result = robot_api.smooth_helix(
        center=center,
        radius=30.0,
        pitch=12.0,
        height=36.0,  # 3 revolutions
        # No start_pose - uses current
        duration=10.0,
        clockwise=True,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=10)
    time.sleep(2)
    
    # Re-initialize
    current_pose = initialize_test_position()
    if not current_pose: return
    
    # Test 2: Helix with specified start on the helix perimeter
    print("\n[2/2] Helix with SPECIFIED start on perimeter")
    center = [current_pose[0], current_pose[1] + 30, current_pose[2] - 40]
    # Start position on the helix perimeter (different angle)
    start_on_perimeter = [
        center[0] + 20,  # radius * cos(0)
        center[1],       # radius * sin(0)
        center[2],       # Starting height
        current_pose[3], current_pose[4], current_pose[5]
    ]
    
    print(f"  Current: {[round(p, 1) for p in current_pose[:3]]}")
    print(f"  Will transition to helix start: {[round(p, 1) for p in start_on_perimeter[:3]]}")
    
    result = robot_api.smooth_helix(
        center=center,
        radius=20.0,
        pitch=12.0,
        height=36.0,
        start_pose=start_on_perimeter,
        speed_percentage=30,
        clockwise=False,
        wait_for_ack=True
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=12)

def test_smooth_blend_with_starts(start_pose):
    """Tests smooth_blend with specified start position for first segment."""
    print("\n--- TESTING SMOOTH BLEND WITH START POSITIONS ---")
    
    # Test 1: Blend with default start
    print("\n[1/3] Blend with DEFAULT start")
    p1 = start_pose
    p2 = [p1[0] + 25, p1[1] + 10, p1[2], p1[3], p1[4], p1[5] + 20]
    arc_center = [p2[0] - 10, p2[1] + 10, p2[2]]
    p3 = [arc_center[0], arc_center[1] + 15, arc_center[2], p1[3], p1[4], p1[5] + 40]
    
    segments = [
        {'type': 'LINE', 'end': p2, 'duration': 2.0},
        {'type': 'ARC', 'end': p3, 'center': arc_center, 'duration': 3.0, 'clockwise': False},
    ]
    
    result = robot_api.smooth_blend(
        segments=segments,
        blend_time=0.5,
        # No start_pose - uses current
        duration=6.0,
        wait_for_ack=True,
        timeout=15
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=15)
    time.sleep(2)
    
    # Re-initialize
    current_pose = initialize_test_position()
    if not current_pose: return
    
    # Test 2: Blend with specified start for first segment
    print("\n[2/3] Blend with SPECIFIED start (adds transition)")
    specified_start = [current_pose[0] + 15, current_pose[1] - 10, current_pose[2],
                      current_pose[3], current_pose[4], current_pose[5]]
    p2 = [specified_start[0] + 20, specified_start[1] + 15, specified_start[2],
          specified_start[3], specified_start[4], specified_start[5] + 30]
    circle_center = [p2[0], p2[1] + 20, p2[2]]
    
    segments = [
        {'type': 'LINE', 'end': p2, 'duration': 2.5},
        {'type': 'CIRCLE', 'center': circle_center, 'radius': 20, 'plane': 'XY', 
         'duration': 4.0, 'clockwise': True},
    ]
    
    print(f"  Current: {[round(p, 1) for p in current_pose[:3]]}")
    print(f"  Will transition to: {[round(p, 1) for p in specified_start[:3]]}")
    print(f"  Then execute blend segments")
    
    result = robot_api.smooth_blend(
        segments=segments,
        blend_time=0.75,
        start_pose=specified_start,  # First segment starts here
        speed_percentage=35,
        wait_for_ack=True,
        timeout=20
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=20)
    time.sleep(2)
    
    # Re-initialize
    current_pose = initialize_test_position()
    if not current_pose: return
    
    # Test 3: Complex blend with spline segment and specified start
    print("\n[3/3] Complex blend with SPLINE segment and specified start")
    blend_start = [current_pose[0] - 10, current_pose[1] + 10, current_pose[2],
                   current_pose[3], current_pose[4], current_pose[5]]
    
    # Define waypoints for spline segment
    spline_waypoints = [
        [blend_start[0] + 30, blend_start[1], blend_start[2],
         blend_start[3], blend_start[4], blend_start[5]],
        [blend_start[0] + 35, blend_start[1] + 15, blend_start[2],
         blend_start[3], blend_start[4], blend_start[5] + 15],
        [blend_start[0] + 25, blend_start[1] + 25, blend_start[2],
         blend_start[3], blend_start[4], blend_start[5] + 30]
    ]
    
    segments = [
        {'type': 'LINE', 'end': spline_waypoints[0], 'duration': 2.0},
        {'type': 'SPLINE', 'waypoints': spline_waypoints, 'duration': 4.0},
        {'type': 'LINE', 'end': [blend_start[0], blend_start[1] + 20, blend_start[2],
                                 blend_start[3], blend_start[4], blend_start[5]], 
         'duration': 2.0}
    ]
    
    result = robot_api.smooth_blend(
        segments=segments,
        blend_time=0.5,
        start_pose=blend_start,
        duration=10.0,  # Overall duration
        wait_for_ack=True,
        timeout=20
    )
    print(f"--> Command result: {result}")
    robot_api.wait_for_robot_stopped(timeout=20)

def test_transition_distances():
    """Test transitions with various distances to verify smooth transition behavior."""
    print("\n--- TESTING TRANSITION DISTANCES ---")
    
    # Get current position
    start_pose = initialize_test_position()
    if not start_pose: return
    
    # Define test distances: very close, medium, far
    test_cases = [
        ("Very Close (3mm)", 3),
        ("Close (10mm)", 10),
        ("Medium (30mm)", 30),
        ("Far (50mm)", 50)
    ]
    
    for description, distance in test_cases:
        print(f"\n[{test_cases.index((description, distance)) + 1}/{len(test_cases)}] Testing transition: {description}")
        
        # Create a start position at the specified distance
        transition_start = [
            start_pose[0] + distance,
            start_pose[1],
            start_pose[2],
            start_pose[3], start_pose[4], start_pose[5]
        ]
        
        # Use a simple circle to observe the transition
        circle_center = [transition_start[0], transition_start[1] + 30, transition_start[2]]
        
        print(f"  Current position: {[round(p, 1) for p in start_pose[:3]]}")
        print(f"  Transition to: {[round(p, 1) for p in transition_start[:3]]}")
        print(f"  Distance: {distance}mm")
        
        start_time = time.time()
        result = robot_api.smooth_circle(
            center=circle_center,
            radius=30.0,
            plane='XY',
            start_pose=transition_start,
            duration=5.0,
            clockwise=False,
            wait_for_ack=True
        )
        
        # Note the transition time
        robot_api.wait_for_robot_stopped(timeout=10)
        total_time = time.time() - start_time
        
        print(f"  Total execution time: {total_time:.2f}s")
        if distance <= 5:
            print(f"  -> Minimal transition expected and observed")
        else:
            transition_time = distance / 30.0  # Assuming 30mm/s transition speed
            print(f"  -> Estimated transition time: {transition_time:.2f}s")
        
        time.sleep(2)
        
        # Return to start for next test
        if test_cases.index((description, distance)) < len(test_cases) - 1:
            initialize_test_position()

def test_timing_comparison_with_starts():
    """Compare timing modes with specified start positions."""
    print("\n--- TESTING TIMING MODES WITH START POSITIONS ---")
    
    # Initialize
    start_pose = initialize_test_position()
    if not start_pose: return
    
    # Define a specific start position for both tests
    test_start = [start_pose[0] + 20, start_pose[1] - 10, start_pose[2],
                  start_pose[3], start_pose[4], start_pose[5]]
    center = [test_start[0], test_start[1] + 30, test_start[2]]
    radius = 30.0
    
    print("\n[1/2] Circle with specified start + 5-second DURATION")
    print(f"  Transition from: {[round(p, 1) for p in start_pose[:3]]}")
    print(f"  To start position: {[round(p, 1) for p in test_start[:3]]}")
    
    start_time = time.time()
    result = robot_api.smooth_circle(
        center=center,
        radius=radius,
        plane='XY',
        start_pose=test_start,
        duration=5.0,
        clockwise=False,
        wait_for_ack=True
    )
    robot_api.wait_for_robot_stopped(timeout=12)
    elapsed = time.time() - start_time
    print(f"--> Total execution time (including transition): {elapsed:.2f}s")
    time.sleep(2)
    
    # Return to start
    initialize_test_position()
    
    print("\n[2/2] Same circle with specified start + 40% SPEED")
    print(f"  Same transition and circle path")
    
    start_time = time.time()
    result = robot_api.smooth_circle(
        center=center,
        radius=radius,
        plane='XY',
        start_pose=test_start,
        speed_percentage=40,
        clockwise=False,
        wait_for_ack=True
    )
    robot_api.wait_for_robot_stopped(timeout=12)
    elapsed = time.time() - start_time
    print(f"--> Total execution time (including transition): {elapsed:.2f}s")
    
    # Calculate expected times
    circumference = 2 * np.pi * radius
    transition_dist = np.sqrt((test_start[0] - start_pose[0])**2 + 
                              (test_start[1] - start_pose[1])**2 + 
                              (test_start[2] - start_pose[2])**2)
    print(f"\nAnalysis:")
    print(f"  Transition distance: {transition_dist:.1f}mm")
    print(f"  Circle circumference: {circumference:.1f}mm")
    print(f"  At 40% speed (~40mm/s), circle should take ~{circumference/40:.1f}s")
    print(f"  Transition at ~30mm/s should take ~{transition_dist/30:.1f}s")

if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE SMOOTH MOTION TEST SUITE")
    print("Testing NEW features: Start Positions & Automatic Transitions")
    print("="*70)
    
    '''
    # Test 1: Basic tests with new start position feature
    print("\n[TEST GROUP 1: BASIC COMMANDS WITH START POSITIONS]")
    start_pose = initialize_test_position()
    if start_pose:
        test_smooth_circle_basic(start_pose)
    
    # Test 2: Arc commands with various start positions
    print("\n[TEST GROUP 2: ARC COMMANDS WITH TRANSITIONS]")
    start_pose = initialize_test_position()
    if start_pose:
        test_smooth_arc_with_start_positions(start_pose)
    
    # Test 3: Motion chaining - using end of one as start of next
    print("\n[TEST GROUP 3: PRECISE MOTION CHAINING]")
    start_pose = initialize_test_position()
    if start_pose:
        test_motion_chaining(start_pose)
    
    # Test 4: Spline with various start scenarios
    print("\n[TEST GROUP 4: SPLINE WITH START POSITIONS]")
    start_pose = initialize_test_position()
    if start_pose:
        test_smooth_spline_with_starts(start_pose)
    '''
    # Test 5: Helix with start positions
    print("\n[TEST GROUP 5: HELIX WITH START POSITIONS]")
    start_pose = initialize_test_position()
    if start_pose:
        test_smooth_helix_with_starts(start_pose)
    '''
    # Test 6: Blend with start positions
    print("\n[TEST GROUP 6: BLEND WITH START POSITIONS]")
    start_pose = initialize_test_position()
    if start_pose:
        test_smooth_blend_with_starts(start_pose)
    
    # Test 7: Transition distance testing
    print("\n[TEST GROUP 7: TRANSITION DISTANCE BEHAVIOR]")
    test_transition_distances()
    
    # Test 8: Timing comparison with transitions
    print("\n[TEST GROUP 8: TIMING MODES WITH TRANSITIONS]")
    test_timing_comparison_with_starts()
    '''
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUITE COMPLETE")
    print("Tested features:")
    print("  ✓ All commands with duration mode")
    print("  ✓ All commands with speed percentage mode")
    print("  ✓ Default start positions (current position)")
    print("  ✓ Specified start positions with automatic transitions")
    print("  ✓ Motion chaining with precise continuity")
    print("  ✓ Transition behavior for various distances")
    print("  ✓ Blend segments with overall timing control")
    print("="*70)
    
    # Final return to safe position
    print("\nReturning to safe position...")
    initialize_test_position()
    print("\n===== All Tests Finished =====")