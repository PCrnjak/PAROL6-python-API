"""
Manage server lifecycle demonstration for PAROL6.
- Starts a local controller at 127.0.0.1:5001
- Waits until ready, then connects with RobotClient
- Toggles simulator ON for a safe demo motion, then OFF
- Stops the controller on exit

Run from the repository root:
    python external/PAROL6-python-API/examples/manage_server_demo.py
"""

from parol6 import Robot, RobotClient

HOST = "127.0.0.1"
PORT = 5001


def main() -> None:
    robot = Robot(host=HOST, port=PORT, normalize_logs=True)
    robot.start()
    try:
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as client:
            ready = client.wait_ready(timeout=5.0)
            print(f"server ready: {ready}")
            if not ready:
                raise SystemExit(1)

            print("ping:", client.ping())

            # Enable simulator for a safe motion
            sim_on = client.simulator_on()
            print("simulator_on:", sim_on)

            if sim_on:
                # Small relative move: +3mm in Z over 0.8s
                moved = client.moveL([0, 0, 3, 0, 0, 0], rel=True, duration=0.8)
                print("moveL ->", moved)

            # Demonstrate toggling simulator off again (no motion follows)
            sim_off = client.simulator_off()
            print("simulator_off:", sim_off)

            raise SystemExit(0)
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
