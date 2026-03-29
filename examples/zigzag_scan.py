"""Zig-zag raster scan with blended corners.

Demonstrates using Python for-loops to generate a scan pattern and the
blend radius (r) parameter for smooth continuous motion through waypoints.
Runs in simulator mode.

Run:
    python examples/zigzag_scan.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

# Scan area parameters
ROWS = 5
X_MIN, X_MAX = -80, 80  # scan width (mm)
Z_MIN, Z_MAX = 150, 250  # scan height (mm)
Y = 280  # fixed forward distance (mm)
ORIENTATION = [90, 0, 90]
BLEND_R = 15  # blend radius for smooth corners (mm)
SPEED = 0.5


def main() -> None:
    with Robot(host=HOST, port=PORT, normalize_logs=True):
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as rbt:
            rbt.wait_ready(timeout=5.0)
            rbt.simulator_on()

            print("Homing...")
            rbt.home(wait=True)

            # Move to above the scan area
            start = [X_MIN, Y, Z_MAX + 30] + ORIENTATION
            print("Moving to start position...")
            rbt.moveL(start, speed=SPEED, wait=True)

            z_step = (Z_MAX - Z_MIN) / (ROWS - 1)

            print(f"Running {ROWS}-row zig-zag scan (blend radius={BLEND_R}mm)...")

            for row in range(ROWS):
                z = Z_MAX - row * z_step

                # Alternate sweep direction each row
                if row % 2 == 0:
                    x_start, x_end = X_MIN, X_MAX
                else:
                    x_start, x_end = X_MAX, X_MIN

                is_last_row = row == ROWS - 1

                # Queue moves without waiting so the controller can blend them.
                # Use r=BLEND_R for smooth corners; r=0 on the last move to flush.
                rbt.moveL(
                    [x_start, Y, z] + ORIENTATION, speed=SPEED, r=BLEND_R, wait=False
                )
                rbt.moveL(
                    [x_end, Y, z] + ORIENTATION,
                    speed=SPEED,
                    r=0 if is_last_row else BLEND_R,
                    wait=False,
                )

                direction = "left->right" if row % 2 == 0 else "right->left"
                print(f"  Row {row + 1}/{ROWS}: Z={z:.0f}mm  {direction}")

            rbt.wait_motion_complete()
            print("Done!")


if __name__ == "__main__":
    main()
