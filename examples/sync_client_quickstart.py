"""
Sync client quickstart for PAROL6.
- Starts a local headless controller
- Enables simulator mode for safety
- Performs ping, queries, and a small move

Run from the repository root:
    python examples/sync_client_quickstart.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001


def main() -> None:
    with Robot(host=HOST, port=PORT, normalize_logs=True):
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as client:
            ready = client.wait_ready(timeout=5.0)
            print(f"server ready: {ready}")
            if not ready:
                raise SystemExit(1)

            client.simulator(True)
            print("ping:", client.ping())
            print("pose xyz:", client.pose()[:3])
            print("angles:", client.angles())

            # Small relative move (safe in simulator)
            moved = client.move_l([0, 0, 5, 0, 0, 0], rel=True, duration=1.0)
            print("move_l ->", moved)


if __name__ == "__main__":
    main()
