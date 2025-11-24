"""
Manage server lifecycle demonstration for PAROL6.
- Starts a local headless controller at 127.0.0.1:5001
- Waits until ready, then connects with RobotClient
- Toggles simulator ON for a safe demo motion, then OFF
- Stops the controller on exit

Run from the repository root:
    python external/PAROL6-python-API/examples/manage_server_demo.py
"""

from parol6 import manage_server, RobotClient

HOST = "127.0.0.1"
PORT = 5001


def main() -> None:
    mgr = manage_server(host=HOST, port=PORT, normalize_logs=True)
    try:
        with RobotClient(host=HOST, port=PORT, timeout=2.0) as client:
            ready = client.wait_for_server_ready(timeout=5.0)
            print(f"server ready: {ready}")
            if not ready:
                raise SystemExit(1)

            print("ping:", client.ping())

            # Enable simulator for a safe motion
            sim_on = client.simulator_on()
            print("simulator_on:", sim_on)

            if sim_on:
                # Small relative TRF move: +3mm in Z over 0.8s
                moved = client.move_cartesian_rel_trf([0, 0, 3, 0, 0, 0], duration=0.8)
                print("move_cartesian_rel_trf ->", moved)

            # Demonstrate toggling simulator off again (no motion follows)
            sim_off = client.simulator_off()
            print("simulator_off:", sim_off)

            raise SystemExit(0)
    finally:
        mgr.stop_controller()


if __name__ == "__main__":
    main()
