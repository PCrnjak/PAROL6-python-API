"""Zig-zag raster scan with blended corners.

Queues linear moves with blend radius `r` so the controller rounds
the corners into one smooth, continuous path. Each row alternates
sweep direction to minimise repositioning.

Run:
    python examples/zigzag_scan.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

ZZ_ORI = [-180, -90, -180]
ROWS = 6
Y_MIN, Y_MAX = 0, 160
Z_MIN, Z_MAX = 200, 300
X = 280
BLEND = 15

with Robot(host=HOST, port=PORT, normalize_logs=True):
    rbt = RobotClient(host=HOST, port=PORT, timeout=2.0)
    rbt.wait_ready(timeout=5.0)
    rbt.simulator(True)

    rbt.home(wait=True)

    # Approach the scan area
    rbt.move_j(pose=[X, 0, 334] + ZZ_ORI, speed=0.5, wait=True)
    rbt.move_l([X, Y_MIN, Z_MAX + 30] + ZZ_ORI, speed=0.5, wait=True)

    # Raster scan — queue every move without waiting so the controller
    # can blend adjacent segments through the blend radius.
    z_step = (Z_MAX - Z_MIN) / (ROWS - 1)
    for row in range(ROWS):
        z = Z_MAX - row * z_step
        is_last = row == ROWS - 1
        y_start, y_end = (Y_MIN, Y_MAX) if row % 2 == 0 else (Y_MAX, Y_MIN)
        rbt.move_l([X, y_start, z] + ZZ_ORI, speed=0.5, r=BLEND, wait=False)
        rbt.move_l(
            [X, y_end, z] + ZZ_ORI, speed=0.5, r=0 if is_last else BLEND, wait=False
        )

    rbt.wait_motion()
    print("Done!")
