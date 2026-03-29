"""Pick-and-place cycle with electric gripper.

Demonstrates gripper control, approach/retract patterns, and looping
over multiple parts. Starts the controller in simulator mode for safety.

Run:
    python examples/pick_and_place.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

# Positions (mm + deg)
ORIENTATION = [90, 0, 90]
PICK_BASE = [-60, 300, 100]
PLACE_BASE = [60, 300, 100]
APPROACH_HEIGHT = 60
PART_SPACING = 30
NUM_PARTS = 3
SPEED = 0.6


def main() -> None:
    with Robot(host=HOST, port=PORT, normalize_logs=True):
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as rbt:
            rbt.wait_ready(timeout=5.0)
            rbt.simulator_on()

            print("Homing...")
            rbt.home(wait=True)

            for part in range(NUM_PARTS):
                x_offset = part * PART_SPACING
                pick = [
                    PICK_BASE[0] + x_offset,
                    PICK_BASE[1],
                    PICK_BASE[2],
                ] + ORIENTATION
                pick_above = [pick[0], pick[1], pick[2] + APPROACH_HEIGHT] + ORIENTATION
                place = [
                    PLACE_BASE[0] + x_offset,
                    PLACE_BASE[1],
                    PLACE_BASE[2],
                ] + ORIENTATION
                place_above = [
                    place[0],
                    place[1],
                    place[2] + APPROACH_HEIGHT,
                ] + ORIENTATION

                print(f"Part {part + 1}/{NUM_PARTS}:")

                # Open gripper and approach pick location
                rbt.control_electric_gripper(
                    "move", position=0.0, speed=0.5, current=500
                )
                rbt.moveL(pick_above, speed=SPEED, wait=True)

                # Descend and grab
                rbt.moveL(pick, speed=SPEED * 0.5, wait=True)
                print(f"  Picking at X={pick[0]:.0f}mm...")
                rbt.control_electric_gripper(
                    "move", position=1.0, speed=0.5, current=500, wait=True
                )

                # Retract, move to place, descend
                rbt.moveL(pick_above, speed=SPEED, wait=True)
                rbt.moveL(place_above, speed=SPEED, wait=True)
                rbt.moveL(place, speed=SPEED * 0.5, wait=True)

                # Release and retract
                print(f"  Placing at X={place[0]:.0f}mm...")
                rbt.control_electric_gripper(
                    "move", position=0.0, speed=0.5, current=500, wait=True
                )
                rbt.moveL(place_above, speed=SPEED, wait=True)

            print("Homing...")
            rbt.home(wait=True)
            print("Done!")


if __name__ == "__main__":
    main()
