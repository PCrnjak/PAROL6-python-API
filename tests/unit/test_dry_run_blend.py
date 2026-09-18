"""Unit tests for DryRunRobotClient blend buffering."""

import numpy as np
import pytest

from parol6.client.dry_run_client import DryRunRobotClient
from waldoctl.skills import UnresolvedPreview

# Valid PAROL6 joint angles (deg) within limits:
#   J1: [-123, 123], J2: [-145, -3.375], J3: [107.9, 287.9],
#   J4: [-105, 105], J5: [-90, 90], J6: [0, 360]
# Home/standby is [90, -90, 180, 0, 0, 180].
W0 = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
W1 = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
W2 = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]
W3 = [60.0, -60.0, 210.0, 30.0, 30.0, 210.0]


@pytest.fixture
def client():
    return DryRunRobotClient()


class TestDryRunBlend:
    """Blend buffering: a chain lands under its head command."""

    def test_blend_chain_lands_under_its_head(self, client):
        """3x move_j with r > 0 buffer; the head block owns the chain's rows
        and the folded commands keep their place at zero rows."""
        first = client.move_j(angles=W1, speed=0.5, r=10)
        second = client.move_j(angles=W2, speed=0.5, r=10)
        third = client.move_j(angles=W3, speed=0.5, r=0)
        assert (first, second, third) == (0, 1, 2)
        record = client.plan()
        assert record.blocks[first].rows > 0
        assert record.blocks[second].rows == 0 and record.blocks[third].rows == 0
        assert record.tcp.shape == (record.rows, 6)
        assert all(b.error is None for b in record.blocks)
        np.testing.assert_allclose(np.degrees(record.joints_rad[-1]), W3, atol=0.5)

    def test_no_blend_without_radius(self, client):
        index = client.move_j(angles=W1, speed=0.5, r=0)
        block = client.plan().blocks[index]
        assert block.rows > 0 and block.error is None

    def test_flush_plans_the_pending_chain(self, client):
        client.move_j(angles=W1, speed=0.5, r=10)
        client.move_j(angles=W2, speed=0.5, r=10)
        assert client.program_length == 2
        client.flush()
        assert client.plan().blocks[0].rows > 0

    def test_empty_program_is_an_empty_record(self, client):
        assert client.flush() is None
        record = client.plan()
        assert record.rows == 0 and record.blocks == ()

    def test_blended_chain_is_longer_than_one_move(self, client):
        single = DryRunRobotClient()
        single.move_j(angles=W3, speed=0.3, r=0)
        client.move_j(angles=W1, speed=0.3, r=10)
        client.move_j(angles=W2, speed=0.3, r=10)
        client.move_j(angles=W3, speed=0.3, r=0)
        assert client.plan().duration_s > single.plan().duration_s

    def test_state_updated_after_blend(self, client):
        client.move_j(angles=W1, speed=0.5, r=10)
        client.move_j(angles=W2, speed=0.5, r=0)
        np.testing.assert_allclose(client.angles(), W2, atol=0.5)

    def test_pause_holds_the_program_until_resume(self):
        slow = DryRunRobotClient(initial_joints_deg=W0)
        assert slow.set_execution_speed(0.5) == 1
        assert slow.move_j(W1, duration=2) >= 0
        assert slow.pause() == 1
        assert slow.set_execution_speed(0.3) == 1
        assert slow.execution_speed().paused
        for operation in (lambda: slow.move_j(W2, duration=2), lambda: slow.delay(1)):
            with pytest.raises(UnresolvedPreview, match="paused"):
                operation()
        np.testing.assert_allclose(slow.angles(), W1, atol=0.05)
        assert slow.resume() == 1
        assert slow.execution_speed().applied_scale == 0.3
        assert slow.wait_command(slow.move_j(W2, duration=2))
        for invalid in (0, True, 2, float("nan")):
            with pytest.raises(ValueError):
                slow.set_execution_speed(invalid)
