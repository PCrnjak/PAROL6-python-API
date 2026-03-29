"""Speed and motion profile comparison.

Runs the same L-shaped path at different speeds and with different motion
profiles, timing each run to show the effect of these parameters.
Runs in simulator mode.

Run:
    python examples/speed_comparison.py
"""

import time

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001

ORIENTATION = [90, 0, 90]

# L-shaped path
START = [0, 280, 300] + ORIENTATION
PATH = [
    [80, 280, 300] + ORIENTATION,  # right
    [80, 280, 150] + ORIENTATION,  # down
    [-80, 280, 150] + ORIENTATION,  # left
]


def run_path(rbt: RobotClient, speed: float) -> float:
    """Run the L-shaped path and return elapsed time."""
    rbt.moveL(START, speed=1.0, wait=True)
    t0 = time.time()
    for wp in PATH:
        rbt.moveL(wp, speed=speed, wait=True)
    return time.time() - t0


def main() -> None:
    with Robot(host=HOST, port=PORT, normalize_logs=True):
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as rbt:
            rbt.wait_ready(timeout=5.0)
            rbt.simulator_on()

            print("Homing...")
            rbt.home(wait=True)

            # -- Speed comparison (TOPPRA profile) --
            print("\n--- Speed Comparison (TOPPRA) ---\n")
            rbt.set_profile("TOPPRA")

            speed_results = []
            for speed in [0.3, 0.6, 1.0]:
                elapsed = run_path(rbt, speed)
                speed_results.append((speed, elapsed))
                print(f"  speed={speed:.1f}  time={elapsed:.2f}s")

            # -- Profile comparison (speed=0.5) --
            print("\n--- Profile Comparison (speed=0.5) ---\n")

            profile_results = []
            for profile in ["TOPPRA", "TRAPEZOID"]:
                rbt.set_profile(profile)
                elapsed = run_path(rbt, 0.5)
                profile_results.append((profile, elapsed))
                print(f"  {profile:12s}  time={elapsed:.2f}s")

            # Reset to default
            rbt.set_profile("TOPPRA")

            # -- Summary --
            print("\n--- Summary ---")
            print(
                f"  Speed 0.3 vs 1.0: {speed_results[0][1] / speed_results[2][1]:.1f}x slower"
            )
            print(
                f"  TOPPRA vs TRAPEZOID: {profile_results[0][1] / profile_results[1][1]:.2f}x ratio"
            )

            print("\nHoming...")
            rbt.home(wait=True)
            print("Done!")


if __name__ == "__main__":
    main()
