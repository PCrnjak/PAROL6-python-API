"""Verify that all motion commands users write in scripts work through the dry run client.

The dry run client uses __getattr__ + build_cmd to dispatch calls by mapping
kwargs to wire struct fields. If the client API param names don't match the
struct field names, the kwargs get silently dropped and the command fails.

This test calls every user-facing motion method with the same signatures shown
in the docs / editor auto-complete, ensuring the dry run path doesn't diverge
from the real client.
"""

import numpy as np
import pytest
from waldoctl import CommandKind, command_table

from parol6.client.dry_run_client import _CMD_STRUCTS, DryRunRobotClient
from tests.conftest import rows_for

HOME = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
POSE_A = [0.0, 280.0, 200.0, 90.0, 0.0, 90.0]
POSE_B = [50.0, 280.0, 200.0, 90.0, 0.0, 90.0]
POSE_C = [50.0, 280.0, 250.0, 90.0, 0.0, 90.0]
ANGLES_A = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
ANGLES_B = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]


@pytest.fixture
def client():
    return DryRunRobotClient()


class TestDryRunScriptCompat:
    """Every call signature a user can write in a script must work in dry run."""

    def _planned(self, client, index):
        assert isinstance(index, int) and index >= 0
        assert client.wait_command(index)
        return client.plan().blocks[index]

    def test_home(self, client):
        self._planned(client, client.home())

    def test_move_j_positional(self, client):
        self._planned(client, client.move_j(ANGLES_A, speed=0.5))

    def test_move_j_angles_kwarg(self, client):
        self._planned(client, client.move_j(angles=ANGLES_A, speed=0.5))

    def test_move_j_with_accel(self, client):
        self._planned(client, client.move_j(ANGLES_A, speed=0.5, accel=0.8))

    def test_move_j_with_duration(self, client):
        self._planned(client, client.move_j(ANGLES_A, duration=2.0))

    def test_move_j_relative(self, client):
        assert client.move_j(ANGLES_A, speed=0.5, rel=True) >= 0

    def test_move_l_positional(self, client):
        self._planned(client, client.move_l(POSE_A, speed=0.5))

    def test_move_l_with_frame(self, client):
        assert client.move_l(POSE_A, speed=0.5, frame="WRF") >= 0

    def test_move_c(self, client):
        client.move_l(POSE_A, speed=0.5)
        assert client.move_c(via=POSE_B, end=POSE_A, speed=0.5) >= 0

    def test_move_s(self, client):
        client.move_l(POSE_A, speed=0.5)
        waypoints = [POSE_A, POSE_B, POSE_C, POSE_A]
        assert client.move_s(waypoints=waypoints, speed=0.5) >= 0

    def test_move_p(self, client):
        client.move_l(POSE_A, speed=0.5)
        waypoints = [POSE_A, POSE_B, POSE_C, POSE_A]
        assert client.move_p(waypoints=waypoints, speed=0.5) >= 0

    def test_move_j_blend_radius(self, client):
        """A blend radius buffers the move; the index comes back at once and
        the rows land under the chain's head once r=0 closes it."""
        r1 = client.move_j(ANGLES_A, speed=0.5, r=10)
        r2 = client.move_j(ANGLES_B, speed=0.5, r=0)
        assert (r1, r2) == (0, 1)
        record = client.plan()
        assert record.blocks[r1].rows > 0 and record.blocks[r2].rows == 0

    def test_move_l_blend_radius(self, client):
        r1 = client.move_l(POSE_A, speed=0.5, r=15)
        r2 = client.move_l(POSE_B, speed=0.5, r=0)
        assert (r1, r2) == (0, 1)
        assert client.plan().blocks[r1].rows > 0

    def test_angles(self, client):
        angles = client.angles()
        assert isinstance(angles, list)
        assert len(angles) == 6

    def test_pose(self, client):
        pose = client.pose()
        assert isinstance(pose, list)
        assert len(pose) == 6

    def test_flush(self, client):
        assert client.flush() is None

    def test_delay(self, client):
        index = client.delay(1.0)
        assert client.plan().blocks[index].rows == rows_for(1.0)

    def test_wait_motion(self, client):
        client.move_j(ANGLES_A, speed=0.5)
        assert client.wait_motion() is True


class TestDryRunHomedGate:
    """The dry run mirrors the live unhomed-motion gate: seeded from an
    unhomed robot, planned moves are refused with the actionable not-homed
    error (not a garbage collision prediction from unreferenced positions);
    a home() in the script establishes references and later moves plan
    cleanly."""

    def test_unhomed_seed_gates_planned_moves_until_home(self):
        client = DryRunRobotClient(initial_joints_deg=[0.0] * 6, initial_homed=False)

        refused = client.move_j(ANGLES_A, speed=0.5)
        assert client.wait_command(refused) is False
        assert "not homed" in str(client.plan().blocks[refused].error)

        # home() snaps to the home pose and establishes references —
        # the first move after it must NOT error.
        assert client.wait_command(client.home())
        assert client.wait_command(client.move_j(ANGLES_A, speed=0.5))

    def test_homed_seed_plans_immediately(self):
        client = DryRunRobotClient(initial_joints_deg=HOME, initial_homed=True)
        assert client.wait_command(client.move_j(ANGLES_A, speed=0.5))

    def test_referenced_home_previews_as_return_move(self):
        client = DryRunRobotClient(initial_joints_deg=ANGLES_A, initial_homed=True)
        index = client.home()
        record = client.plan()
        block = record.blocks[index]
        assert block.error is None and block.rows > 1
        assert np.allclose(
            np.degrees(record.joints_rad[block.start_row + block.rows - 1]),
            HOME,
            atol=0.5,
        )

        # Unreferenced seed keeps the instant snap — the switch-seek can't
        # be previewed from unreferenced positions — as one row at home.
        client = DryRunRobotClient(initial_joints_deg=[0.0] * 6, initial_homed=False)
        index = client.home()
        record = client.plan()
        block = record.blocks[index]
        assert block.error is None and block.rows == 1
        assert np.allclose(
            np.degrees(record.joints_rad[block.start_row]), HOME, atol=0.5
        )

    def test_snap_carries_the_pending_blend_chain(self):
        """A blended move still buffered when the script homes with calibrate
        (or teleports) is planned under its own command before the snap —
        the live controller runs it before the snap, so the preview must
        show it."""
        client = DryRunRobotClient(initial_joints_deg=HOME, initial_homed=True)
        chain = client.move_j(ANGLES_A, speed=0.5, r=10)
        snap = client.home(calibrate=True)
        record = client.plan()
        assert record.blocks[chain].rows > 1 and record.blocks[chain].error is None
        assert record.blocks[snap].rows == 1
        assert np.allclose(np.degrees(record.joints_rad[-1]), HOME, atol=0.5)
        assert client.flush() is None


def test_jogs_take_the_live_clients_arguments(client):
    """`rbt.jog_j(0, 0.5, 1.0)` and `rbt.jog_l("WRF", "X", 0.5, 1.0)` are the
    forms the docs show; the preview must plan them, not the wire struct's
    field order."""
    before = np.asarray(client.angles())
    assert client.jog_j(0, 0.5, 1.0) == 1
    record = client.plan()
    block = record.blocks[client.program_length - 1]
    after = np.degrees(record.joints_rad[block.start_row + block.rows - 1])
    assert after[0] > before[0] + 1.0
    assert np.allclose(after[1:], before[1:], atol=1e-6)

    client = DryRunRobotClient()
    x_before = client.pose()[0]
    assert client.jog_l("WRF", "X", 0.5, 1.0) == 1
    record = client.plan()
    block = record.blocks[client.program_length - 1]
    assert record.tcp[block.start_row + block.rows - 1][0] * 1000.0 > x_before + 1.0
    assert client.pose()[0] > x_before + 1.0
    with pytest.raises(ValueError, match="joint="):
        client.jog_j(speed=0.5)


_STATE_ARGS = {
    "reset": (),
    "reset_state": (),
    "set_status_rate": (50,),
    "simulator": (True,),
    "teleport": (HOME,),
    "set_shapes": ([],),
    "select_profile": ("RUCKIG",),
    "select_tool": ("NONE",),
    "set_tcp_offset": (0.0, 0.0, 0.0),
    "connect_hardware": ("/dev/null",),
    "stop": (),
    "estop": (),
    "pause": (),
    "resume": (),
    "set_execution_speed": (0.5,),
}


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n, s in command_table().items()
        if s.kind in (CommandKind.SYSTEM, CommandKind.CONTROL) and n in _CMD_STRUCTS
    ),
)
def test_state_commands_answer_with_the_live_clients_int_codes(client, name):
    """`if rbt.stop() < 0:` must read the same in preview as on the arm: a
    system or control command returns 1/0/negative, never a planner result."""
    assert name in _STATE_ARGS, f"add sample arguments for {name}"
    result = getattr(client, name)(*_STATE_ARGS[name])
    assert isinstance(result, int) and not isinstance(result, bool)
    assert result == 1
