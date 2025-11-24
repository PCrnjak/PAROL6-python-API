"""
Sync client quickstart for PAROL6.
- Assumes a controller is already running at 127.0.0.1:5001
- Does not enable simulator in this example
- Performs ping and basic queries

Run from the repository root:
    python external/PAROL6-python-API/examples/sync_client_quickstart.py
"""

from parol6 import RobotClient

HOST = "127.0.0.1"
PORT = 5001


def main() -> None:
    with RobotClient(host=HOST, port=PORT, timeout=2.0) as client:
        ready = client.wait_for_server_ready(timeout=3.0)
        print(f"server ready: {ready}")
        print("ping:", client.ping())
        print("pose xyz:", client.get_pose_xyz())
        print("angles:", client.get_angles())
        code = 0 if ready else 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
