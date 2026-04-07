"""Speed and motion profile comparison.

Runs the same L-shaped path at different speeds and with different
trajectory profiles, timing each run. Useful for tuning speed and
profile selection for your application. Runs in the built-in simulator.

Run:
    python examples/speed_comparison.py
"""

import time

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

ORIENTATION = [90, 0, 90]
START = [0, 280, 300] + ORIENTATION
PATH = [
    [80, 280, 300] + ORIENTATION,
    [80, 280, 150] + ORIENTATION,
    [-80, 280, 150] + ORIENTATION,
]


def run_path(rbt: RobotClient, speed: float) -> float:
    rbt.move_l(START, speed=1.0, wait=True)
    t0 = time.time()
    for wp in PATH:
        rbt.move_l(wp, speed=speed, wait=True)
    return time.time() - t0


with Robot(host=HOST, port=PORT, normalize_logs=True):
    rbt = RobotClient(host=HOST, port=PORT, timeout=2.0)
    rbt.wait_ready(timeout=5.0)
    rbt.simulator(True)

    print("Homing...")
    rbt.home(wait=True)

    # -- Speed comparison --
    print("\n--- Speed Comparison (TOPPRA) ---\n")
    rbt.select_profile("TOPPRA")

    speed_results = []
    for speed in [0.3, 0.6, 1.0]:
        elapsed = run_path(rbt, speed)
        speed_results.append((speed, elapsed))
        print(f"  speed={speed:.1f}  time={elapsed:.2f}s")

    # -- Profile comparison --
    print("\n--- Profile Comparison (speed=0.5) ---\n")

    profile_results = []
    for profile in ["TOPPRA", "TRAPEZOID"]:
        rbt.select_profile(profile)
        elapsed = run_path(rbt, 0.5)
        profile_results.append((profile, elapsed))
        print(f"  {profile:12s}  time={elapsed:.2f}s")

    rbt.select_profile("TOPPRA")

    print("\n--- Summary ---")
    print(
        f"  Speed 0.3 vs 1.0: {speed_results[0][1] / speed_results[2][1]:.1f}x slower"
    )
    print(
        f"  TOPPRA vs TRAPEZOID: {profile_results[0][1] / profile_results[1][1]:.2f}x ratio"
    )

    rbt.home(wait=True)
    print("Done!")
