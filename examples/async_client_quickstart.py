"""
Async client quickstart for PAROL6.
- Starts a local headless controller at 127.0.0.1:5001
- Enables simulator mode for safety
- Shows basic queries and status streaming

Run from the repository root:
    python external/PAROL6-python-API/examples/async_client_quickstart.py
"""

import asyncio
from parol6 import AsyncRobotClient
from parol6.client.manager import managed_server

HOST = "127.0.0.1"
PORT = 5001


async def run_client() -> int:
    async with AsyncRobotClient(host=HOST, port=PORT, timeout=2.0) as client:
        ready = await client.wait_for_server_ready(timeout=5.0)
        print(f"server ready: {ready}")
        if not ready:
            return 1

        # Safety: enable simulator for this demo
        ok = await client.simulator_on()
        print(f"simulator_on: {ok}")

        print("ping:", await client.ping())
        pose_xyz = await client.get_pose_xyz()
        print("pose xyz:", pose_xyz)

        # Consume one status broadcast
        print("one status frame speeds:")
        async for status in client.status_stream():
            print(status.get("speeds"))
            break

        # Small relative TRF move (safe in simulator)
        # Move +5mm in Z over 1.0s
        moved = await client.move_cartesian_rel_trf([0, 0, 5, 0, 0, 0], duration=1.0)
        print("move_cartesian_rel_trf ->", moved)

        return 0


def main() -> None:
    # Auto-start and stop controller for this example
    with managed_server(host=HOST, port=PORT, normalize_logs=True):
        code = asyncio.run(run_client())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
