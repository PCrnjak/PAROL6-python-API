"""
Integration test for servo Cartesian move accuracy.

Tests the servo_l path used for TCP dragging.
Catches bugs where reference pose gets corrupted (e.g., aliasing with FK cache).
"""

import time

import numpy as np
import pytest


def angle_diff(a: float, b: float) -> float:
    """Compute smallest angle difference considering wrapping."""
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def stream_servo_l(client, target: list[float], timeout: float = 10.0) -> None:
    """Drive ``servo_l`` the way a dragging client does: refresh the target
    every 50 ms until the tool is on it, then let the stream run out. A
    stream whose client goes silent brakes and holds, so a single datagram
    is never a move."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert client.servo_l(target, speed=1.0) > 0
        time.sleep(0.05)
        pose = client.pose()
        on_target = np.linalg.norm(
            np.array(pose[:3]) - np.array(target[:3])
        ) < 0.5 and all(angle_diff(pose[3 + i], target[3 + i]) < 0.5 for i in range(3))
        if on_target:
            break
    assert client.wait_motion(timeout=5.0)


def assert_pose_accuracy(
    final_pose: list[float],
    target: list[float],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 1.0,
    context: str = "",
) -> None:
    """Assert that final pose matches target within tolerances."""
    # Position check
    pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target[:3]))
    assert pos_error < pos_tol_mm, (
        f"{context}Position error {pos_error:.3f}mm exceeds {pos_tol_mm}mm tolerance. "
        f"Target: {target[:3]}, Final: {final_pose[:3]}"
    )

    # Orientation check
    for i, axis in enumerate(["RX", "RY", "RZ"]):
        ori_error = angle_diff(final_pose[3 + i], target[3 + i])
        assert ori_error < ori_tol_deg, (
            f"{context}{axis} error {ori_error:.3f}° exceeds {ori_tol_deg}° tolerance. "
            f"Target: {target[3 + i]:.1f}°, Final: {final_pose[3 + i]:.1f}°"
        )


@pytest.mark.integration
class TestServoCartesianAccuracy:
    """Test that servo cartesian moves reach correct targets."""

    def test_servo_l_reaches_target(self, client, server_proc):
        """servo_l move should arrive at the requested target.

        Tests the servo Cartesian path (replaces old stream_on + move_cartesian).
        """
        assert client.reset() > 0
        assert client.home() >= 0
        assert client.wait_motion(timeout=15.0)

        # Get starting pose
        start_pose = client.pose()
        print(f"\nStart pose: {start_pose}")

        # Target: offset from start (like beginning of a TCP drag)
        target = list(start_pose)
        target[0] += 30.0  # +30mm in X

        print(f"Target pose: {target}")

        stream_servo_l(client, target)

        # Verify final pose
        final_pose = client.pose()
        print(f"Final pose:  {final_pose}")

        assert_pose_accuracy(final_pose, target)

    def test_servo_l_sequential_targets(self, client, server_proc):
        """Sequential servo moves should each reach their target.

        Simulates TCP dragging behavior where multiple servo_l commands
        are sent in sequence.
        """
        assert client.reset() > 0
        assert client.home() >= 0
        assert client.wait_motion(timeout=15.0)

        start_pose = client.pose()
        print(f"\nStart pose: {start_pose}")

        # Simulate a drag path: series of small incremental moves
        # This pattern catches bugs where reference pose gets corrupted
        # between moves (like the FK cache aliasing bug)
        offsets = [
            (30.0, 0.0, 0.0),  # +30mm X
            (30.0, 30.0, 0.0),  # +30mm X, +30mm Y
            (30.0, 30.0, -30.0),  # +30mm X, +30mm Y, -30mm Z
            (0.0, 0.0, 0.0),  # hold position
        ]

        for i, (dx, dy, dz) in enumerate(offsets):
            target = list(start_pose)
            target[0] += dx
            target[1] += dy
            target[2] += dz

            print(f"\n--- Move {i + 1}/{len(offsets)} ---")
            print(f"Target: {target[:3]}")

            stream_servo_l(client, target)

            final_pose = client.pose()
            start_pose = final_pose
            print(f"Final:  {final_pose[:3]}")

            assert_pose_accuracy(final_pose, target, context=f"Move {i + 1}: ")
            print(f"Move {i + 1} accurate")

        print("\nAll sequential servo moves reached targets accurately")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


@pytest.mark.integration
def test_a_jog_l_moves_the_tcp_straight_and_ends_where_the_preview_does(
    client, server_proc
):
    """A jog_l is a straight TCP line: with an angular part as well, the tool
    turns about the TCP while the TCP holds its line. The dry run previews
    the pose a full-scale diagonal ends at, held to the same speed ceiling
    as on the arm. (Near the wrist singularity at standby the joint speed
    ceilings slow the tool on the arm, which a preview does not model, so
    the diagonal starts clear of it.)"""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT
    from parol6.client.dry_run_client import DryRunRobotClient
    from parol6.config import LIMITS

    standby = [float(v) for v in PAROL6_ROBOT.joint.standby_deg]
    linear = float(LIMITS.cart.jog.velocity.linear)
    angular = float(LIMITS.cart.jog.velocity.angular)
    clear_of_the_wrist = [90.0, -80.0, 190.0, 0.0, 30.0, 180.0]
    for begin, axes, speeds, duration, previewed in (
        (standby, ["X", "RZ"], [0.05 / linear, 0.5 / angular], 2.0, False),
        (clear_of_the_wrist, ["X", "Y", "Z"], [1.0, -1.0, -1.0], 0.6, True),
    ):
        preview = DryRunRobotClient(initial_joints_deg=begin)
        assert preview.jog_l("WRF", axes=axes, speeds_list=speeds, duration=duration)
        assert preview.plan().blocks[0].error is None
        expected = preview.pose()

        assert client.teleport(begin) == 1
        start = np.asarray(client.pose()[:3])
        direction = np.zeros(3)
        for axis, speed in zip(axes, speeds):
            if axis in ("X", "Y", "Z"):
                direction["XYZ".index(axis)] = speed
        direction /= np.linalg.norm(direction)

        assert (
            client.jog_l("WRF", axes=axes, speeds_list=speeds, duration=duration) == 1
        )
        worst = 0.0
        # The jog runs its duration, then brakes: sample the whole of it.
        end = time.monotonic() + duration + 1.0
        while time.monotonic() < end:
            offset = np.asarray(client.pose()[:3]) - start
            worst = max(
                worst,
                float(np.linalg.norm(offset - np.dot(offset, direction) * direction)),
            )
            time.sleep(0.02)
        assert worst < 1.0, (
            f"{axes} at {speeds}: the TCP left its line by {worst:.1f} mm"
        )
        if previewed:
            assert_pose_accuracy(
                client.pose(),
                expected,
                pos_tol_mm=2.0,
                ori_tol_deg=1.0,
                context=f"{axes} at {speeds}, previewed vs run: ",
            )
