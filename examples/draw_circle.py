"""Draw circles with different curved motion commands.

Draws two circles side by side, each using a different curve method
to show the range of options:
  1. Two move_c arcs (half-circles joined)
  2. move_p through computed waypoints (constant TCP speed)
Then a move_s spline threads through both circles.
Runs in the built-in simulator.

Run:
    python examples/draw_circle.py
"""

import math

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

ORIENTATION = [90, 0, 90]
RADIUS = 20
SPEED = 0.4

# Two circles near center of workspace (close to home position)
CENTERS = [(-25, 240, 240), (25, 240, 240)]


def pt(cx: float, cz: float, angle_deg: float) -> list[float]:
    a = math.radians(angle_deg)
    return [cx + RADIUS * math.cos(a), 240, cz + RADIUS * math.sin(a)] + ORIENTATION


with Robot(host=HOST, port=PORT, normalize_logs=True):
    rbt = RobotClient(host=HOST, port=PORT, timeout=2.0)
    rbt.wait_ready(timeout=5.0)
    rbt.simulator(True)

    print("Homing...")
    rbt.home(wait=True)

    # -- Circle 1: two move_c arcs (half-circles) --
    cx, _, cz = CENTERS[0]
    print(f"\nCircle 1 (two move_c arcs) at X={cx}mm")
    rbt.move_l(pt(cx, cz, 0), speed=SPEED, wait=True)
    rbt.move_c(via=pt(cx, cz, 90), end=pt(cx, cz, 180), speed=SPEED, wait=True)
    rbt.move_c(via=pt(cx, cz, 270), end=pt(cx, cz, 0), speed=SPEED, wait=True)

    # -- Circle 2: move_p through 12 waypoints (constant TCP speed) --
    cx, _, cz = CENTERS[1]
    print(f"Circle 2 (move_p, 12 waypoints) at X={cx}mm")
    waypoints = [pt(cx, cz, i * 30) for i in range(12)]
    waypoints.append(waypoints[0])
    rbt.move_l(waypoints[0], speed=SPEED, wait=True)
    rbt.move_p(waypoints, speed=SPEED, wait=True)

    # -- Finale: move_s spline threading through both circles --
    print("Spline (move_s) threading through both circles...")
    spline = []
    for cx, _, cz in CENTERS:
        for angle in range(0, 360, 45):
            spline.append(pt(cx, cz, angle))
    spline.append(spline[0])

    rbt.move_l(spline[0], speed=SPEED, wait=True)
    rbt.move_s(spline, speed=SPEED, wait=True)

    rbt.home(wait=True)
    print("Done!")
