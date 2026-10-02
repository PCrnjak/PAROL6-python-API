"""Plans at the edges of what a host sends: a pose read back from an arm
parked on a joint limit, commanded again, and a dry run of a blended chain
whose last move cannot be reached."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_a_pose_read_back_on_a_joint_limit_can_be_commanded_again(client, server_proc):
    """A host replays what it read: the arm's own angles, a motor step's
    rounding of where it was sent, or a preview's rows, carried as
    float32. Parked on a joint limit, either lands a hair past the limit's
    decimal; the arm is there, so teleporting or servoing to it is
    accepted."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT
    from parol6.client.dry_run_client import DryRunRobotClient
    from parol6.config import LIMITS

    standby = [float(v) for v in PAROL6_ROBOT.joint.standby_deg]
    limits = LIMITS.joint.position.deg
    for joint, side in ((1, 0), (2, 1), (4, 0), (4, 1), (5, 1)):
        parked = list(standby)
        parked[joint] = float(limits[joint, side])
        assert client.teleport(parked) == 1
        read_back = client.angles()
        assert read_back is not None
        assert client.teleport(read_back) == 1, f"J{joint + 1} at {read_back[joint]}"
        assert client.servo_j(read_back) > 0, f"J{joint + 1} at {read_back[joint]}"

    on_limit = list(standby)
    on_limit[4] = float(limits[4, 1])
    preview = DryRunRobotClient(initial_joints_deg=standby)
    index = preview.move_j(on_limit, speed=0.5)
    # The hold after the move carries the pose the arm stands at, whichever
    # tick the move's last row falls on.
    preview.delay(0.1)
    record = preview.plan()
    assert record.blocks[index].error is None
    row = np.degrees(np.asarray(record.joints_rad[-1], dtype=np.float64)).tolist()
    assert client.teleport(row) == 1, f"J5 at {row[4]}"
    assert client.servo_j(row) > 0, f"J5 at {row[4]}"
    angles = client.angles()
    assert angles is not None
    assert abs(angles[4] - on_limit[4]) < 0.01


def test_a_failing_blended_chain_previews_its_moves_from_where_they_start():
    """A dry run of a blended chain whose last move is out of reach still
    draws the moves ahead of it where they run: the first before the
    second, and the second, a relative move, from where the first ends
    rather than from where the chain began."""
    from parol6.client.dry_run_client import DryRunRobotClient

    r = 10.0
    preview = DryRunRobotClient(
        initial_joints_deg=[90.0, -80.0, 190.0, 0.0, 30.0, 180.0]
    )
    s = np.asarray(preview.pose()[:3])
    first = preview.move_l([0.0, 0.0, -20.0, 0.0, 0.0, 0.0], rel=True, r=r, speed=0.5)
    second = preview.move_l([0.0, 30.0, 0.0, 0.0, 0.0, 0.0], rel=True, r=r, speed=0.5)
    unreachable = preview.move_l([0.0, 0.0, 2000.0, 0.0, 0.0, 0.0], rel=True, speed=0.5)
    record = preview.plan()
    assert record.blocks[unreachable].error is not None

    a, b = record.blocks[first], record.blocks[second]
    drawn = (
        np.asarray(record.tcp[a.start_row : b.start_row + b.rows, :3], dtype=np.float64)
        * 1000.0
    )
    assert len(drawn) > 0, "the preview drew neither move ahead of the failure"
    first_end = s + np.array([0.0, 0.0, -20.0])
    second_end = first_end + np.array([0.0, 30.0, 0.0])
    to_first = np.linalg.norm(drawn - first_end, axis=1)
    to_second = np.linalg.norm(drawn - second_end, axis=1)
    # A blend zone rounds a junction by up to its radius.
    assert to_first.min() < r + 0.5, (
        f"the preview passed {to_first.min():.0f} mm from the first move's end"
    )
    assert to_second.min() < r + 0.5, (
        f"the preview passed {to_second.min():.0f} mm from the second move's end"
    )
    assert np.argmin(to_first) < np.argmin(to_second)
