"""Zig-zag raster scan with blended corners.

Generates a raster scan pattern using Python for-loops and the blend
radius (r) parameter for smooth continuous motion. Moves are queued
without waiting so the controller blends them into one fluid path.
Runs in the built-in simulator.

Run:
    python examples/zigzag_scan.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

ROWS = 5
X_MIN, X_MAX = -80, 80
Z_MIN, Z_MAX = 150, 250
Y = 280
ORIENTATION = [90, 0, 90]
BLEND_R = 15  # mm
SPEED = 0.5

with Robot(host=HOST, port=PORT, normalize_logs=True):
    rbt = RobotClient(host=HOST, port=PORT, timeout=2.0)
    rbt.wait_ready(timeout=5.0)
    rbt.simulator(True)

    print("Homing...")
    rbt.home(wait=True)

    start = [X_MIN, Y, Z_MAX + 30] + ORIENTATION
    rbt.move_l(start, speed=SPEED, wait=True)

    z_step = (Z_MAX - Z_MIN) / (ROWS - 1)
    print(f"Running {ROWS}-row zig-zag scan (blend radius={BLEND_R}mm)...")

    for row in range(ROWS):
        z = Z_MAX - row * z_step
        is_last = row == ROWS - 1

        if row % 2 == 0:
            x_start, x_end = X_MIN, X_MAX
        else:
            x_start, x_end = X_MAX, X_MIN

        # Queue moves without waiting — controller blends them at the corners.
        rbt.move_l([x_start, Y, z] + ORIENTATION, speed=SPEED, r=BLEND_R, wait=False)
        rbt.move_l(
            [x_end, Y, z] + ORIENTATION,
            speed=SPEED,
            r=0 if is_last else BLEND_R,
            wait=False,
        )

        direction = "left->right" if row % 2 == 0 else "right->left"
        print(f"  Row {row + 1}/{ROWS}: Z={z:.0f}mm  {direction}")

    rbt.wait_motion()
    print("Done!")
