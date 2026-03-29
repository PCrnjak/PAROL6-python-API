"""Curved motion commands: moveC, moveS, and moveP.

Draws circles and shapes using the robot's built-in curved motion
commands, progressing from simplest (one moveC) to most flexible
(computed waypoints with moveS/moveP). Runs in simulator mode.

Run:
    python examples/draw_circle.py
"""

import math

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

# Circle: vertical (XZ plane), in front of robot
CX, CY, CZ = 0, 280, 200  # center (mm)
RADIUS = 60  # mm
ORIENTATION = [90, 0, 90]
SPEED = 0.4


def circle_point(angle_deg: float) -> list[float]:
    """Point on the circle at a given angle (0=right, 90=top)."""
    a = math.radians(angle_deg)
    return [CX + RADIUS * math.cos(a), CY, CZ + RADIUS * math.sin(a)] + ORIENTATION


def main() -> None:
    with Robot(host=HOST, port=PORT, normalize_logs=True):
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as rbt:
            rbt.wait_ready(timeout=5.0)
            rbt.simulator_on()

            print("Homing...")
            rbt.home(wait=True)

            # -- Part 1: Full circle with a single moveC --
            print("\n--- Full circle (single moveC) ---")
            print(f"Center: ({CX}, {CY}, {CZ})  Radius: {RADIUS}mm")

            rbt.moveL(circle_point(0), speed=SPEED, wait=True)
            rbt.moveC(
                via=circle_point(180), end=circle_point(0), speed=SPEED, wait=True
            )

            # -- Part 2: Two half-circle arcs --
            print("\n--- Two half-circle arcs ---")

            rbt.moveL(circle_point(90), speed=SPEED, wait=True)
            rbt.moveC(
                via=circle_point(0), end=circle_point(270), speed=SPEED, wait=True
            )
            rbt.moveC(
                via=circle_point(180), end=circle_point(90), speed=SPEED, wait=True
            )

            # -- Part 3: Circle via moveS (spline through 8 waypoints) --
            print("\n--- Circle via moveS (8-point spline) ---")

            waypoints = [circle_point(i * 45) for i in range(8)]
            waypoints.append(waypoints[0])  # close the loop

            rbt.moveL(waypoints[0], speed=SPEED, wait=True)
            rbt.moveS(waypoints, speed=SPEED, wait=True)

            # -- Part 4: Hexagon via moveP (constant TCP speed) --
            print("\n--- Hexagon via moveP (constant TCP speed) ---")

            hex_points = [circle_point(i * 60) for i in range(6)]
            hex_points.append(hex_points[0])  # close the shape

            rbt.moveL(hex_points[0], speed=SPEED, wait=True)
            rbt.moveP(hex_points, speed=SPEED, wait=True)

            print("Done!")


if __name__ == "__main__":
    main()
