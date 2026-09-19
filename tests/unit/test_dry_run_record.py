"""The commanded record a parol6 dry run returns: one block per command on
one row axis, with delays, tool travel and refusals all on it."""

import numpy as np
import pytest
from waldoctl import following_error

from parol6.client.dry_run_client import DryRunRobotClient
from tests.conftest import rows_for
from parol6.tools import get_registry

HOME = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
W1 = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
W2 = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]


def _span(record, block):
    return slice(block.start_row, block.start_row + block.rows)


def test_delay_holds_the_pose_for_its_rows():
    client = DryRunRobotClient(initial_joints_deg=HOME)
    index = client.delay(2.0)
    record = client.plan()
    block = record.blocks[index]
    assert block.command == index and block.rows == rows_for(2.0) == 100
    held = record.joints_rad[_span(record, block)]
    # Motor-step quantisation moves the pose by well under a hundredth of a
    # degree; what matters is that every row holds the same pose.
    np.testing.assert_allclose(
        np.degrees(held), np.broadcast_to(HOME, held.shape), atol=0.01
    )
    assert np.ptp(held, axis=0).max() == 0
    assert block.error is None and block.move_type is None
    assert record.duration_s == pytest.approx(2.0, abs=record.row_dt_s)
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            client.delay(bad)


def test_gripper_close_ramps_the_jaws_over_the_tools_travel():
    client = DryRunRobotClient(initial_joints_deg=HOME)
    assert client.select_tool("SSG-48") == 1
    index = client.tool.close()
    record = client.plan()
    block = record.blocks[index]
    expected = get_registry().get("SSG-48").estimate_duration("close", [])
    assert expected > 0
    assert block.rows == pytest.approx(rows_for(expected), abs=1)
    closed = record.tool_closed[_span(record, block)]
    assert closed[0] == pytest.approx(0.0, abs=0.05)
    assert np.all(np.diff(closed) >= 0) and closed[-1] > 0.9
    # The arm holds still while the jaws move.
    assert np.ptp(record.joints_rad[_span(record, block)], axis=0).max() == 0
    # Once the action is over the jaws are closed: the row grid may skip the
    # ramp's final tick, but everything after it holds the closed state.
    after = client.delay(0.1)
    record = client.plan()
    assert np.all(record.tool_closed[_span(record, record.blocks[after])] == 1.0)


def test_a_refused_move_keeps_its_place_and_its_error():
    client = DryRunRobotClient(initial_joints_deg=[0.0] * 6, initial_homed=False)
    first = client.move_j(W1, speed=0.5)
    assert first == 0 and client.wait_command(first) is False
    record = client.plan()
    assert record.stop == "failed"
    assert "not homed" in str(record.blocks[first].error)
    assert record.blocks[first].rows == 0
    # A later command still lands after it, in order.
    homed = client.home()
    assert client.wait_command(homed)
    later = client.move_j(W1, speed=0.5)
    assert client.wait_command(later)
    record = client.plan()
    assert [b.command for b in record.blocks] == [0, 1, 2]
    assert record.blocks[later].rows > 1 and record.blocks[later].move_type == "joints"


def test_simulate_is_the_plan_on_a_planner_only_backend():
    client = DryRunRobotClient(initial_joints_deg=HOME)
    client.move_j(W1, speed=0.5)
    client.delay(0.5)
    client.move_j(W2, speed=0.5)
    plan = client.plan()
    predicted = client.simulate()
    assert predicted.digest == plan.digest and predicted.rows == plan.rows
    assert not following_error(plan, predicted).any()
    assert [b.move_type for b in plan.blocks] == ["joints", None, "joints"]
    assert sum(b.rows for b in plan.blocks) == plan.rows
    assert plan.blocks[2].start_row == plan.blocks[1].start_row + plan.blocks[1].rows


def test_execution_speed_stretches_the_rows_not_the_path():
    normal = DryRunRobotClient(initial_joints_deg=HOME)
    slow = DryRunRobotClient(initial_joints_deg=HOME)
    n = normal.move_j(W1, duration=2)
    assert slow.set_execution_speed(0.5) == 1
    s = slow.move_j(W1, duration=2)
    nb = normal.plan().blocks[n]
    sb = slow.plan().blocks[s]
    assert sb.rows == pytest.approx(2 * nb.rows, abs=1)
    np.testing.assert_allclose(
        slow.plan().joints_rad[sb.start_row + sb.rows - 1],
        normal.plan().joints_rad[nb.start_row + nb.rows - 1],
        atol=1e-5,
    )


def test_budget_truncates_the_record_and_says_so():
    client = DryRunRobotClient(initial_joints_deg=HOME)
    client.delay(2.0)
    client.move_j(W1, speed=0.5)
    full = client.plan()
    cut = client.plan(max_seconds=1.0)
    assert cut.stop == "budget_exhausted"
    assert cut.rows == rows_for(1.0) < full.rows
    assert cut.blocks[0].rows == cut.rows and cut.blocks[1].rows == 0
    assert full.stop == "completed"
