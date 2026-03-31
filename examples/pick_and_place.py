"""Pick-and-place cycle with tool actions.

Picks three parts from the left side of the workspace and places them
on the right, using approach/retract moves and gripper open/close between
each transfer. Runs in the built-in simulator.

Run:
    python examples/pick_and_place.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

ORIENTATION = [90, 0, 90]
PICK_BASE = [-20, 220, 210]
PLACE_BASE = [20, 220, 210]
APPROACH_HEIGHT = 25
PART_SPACING = 15
NUM_PARTS = 3
SPEED = 0.6

with Robot(host=HOST, port=PORT, normalize_logs=True):
    rbt = RobotClient(host=HOST, port=PORT, timeout=2.0)
    rbt.wait_ready(timeout=5.0)
    rbt.simulator(True)
    rbt.select_tool("SSG-48")

    print("Homing...")
    rbt.home(wait=True)

    for part in range(NUM_PARTS):
        x_offset = part * PART_SPACING
        pick = [PICK_BASE[0] + x_offset, PICK_BASE[1], PICK_BASE[2]] + ORIENTATION
        pick_above = [pick[0], pick[1], pick[2] + APPROACH_HEIGHT] + ORIENTATION
        place = [PLACE_BASE[0] + x_offset, PLACE_BASE[1], PLACE_BASE[2]] + ORIENTATION
        place_above = [place[0], place[1], place[2] + APPROACH_HEIGHT] + ORIENTATION

        print(f"Part {part + 1}/{NUM_PARTS}:")

        # Open gripper and approach pick location
        rbt.tool_action("SSG-48", "open")
        rbt.move_l(pick_above, speed=SPEED, wait=True)

        # Descend and grab
        rbt.move_l(pick, speed=SPEED * 0.5, wait=True)
        print(f"  Picking at X={pick[0]:.0f}mm...")
        rbt.tool_action("SSG-48", "close", wait=True)

        # Retract, move to place, descend
        rbt.move_l(pick_above, speed=SPEED, wait=True)
        rbt.move_l(place_above, speed=SPEED, wait=True)
        rbt.move_l(place, speed=SPEED * 0.5, wait=True)

        # Release and retract
        print(f"  Placing at X={place[0]:.0f}mm...")
        rbt.tool_action("SSG-48", "open", wait=True)
        rbt.move_l(place_above, speed=SPEED, wait=True)

    rbt.home(wait=True)
    print("Done!")
