import math

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.base import CommandBase, ExecutionStatus
from parol6.config import INTERVAL_S


class DummyCommand(CommandBase):
    def match(self, parts):
        return True, None

    def setup(self, state, *, udp_transport=None, addr=None, gcode_interpreter=None):
        # No setup needed for dummy
        return

    def execute_step(self, state) -> ExecutionStatus:
        return ExecutionStatus.completed("ok")


def test_interval_constant_value():
    assert isinstance(INTERVAL_S, float)
    assert INTERVAL_S > 0
    # Default as documented
    assert math.isclose(INTERVAL_S, 0.01, rel_tol=0, abs_tol=1e-9)


def test_lifecycle_flags():
    base = DummyCommand()
    assert base.is_valid is True
    assert base.is_finished is False
    assert base.error_state is False
    assert base.error_message == ""

    base.finish()
    assert base.is_finished is True
    # fail() should mark invalid + finished and capture message
    base = DummyCommand()
    base.fail("boom")
    assert base.is_valid is False
    assert base.is_finished is True
    assert base.error_state is True
    assert base.error_message == "boom"


def test_within_limits_and_clamp():
    j = 0
    mn, mx = PAROL6_ROBOT.Joint_limits_steps[j]
    assert CommandBase.within_limits(j, mn) is True
    assert CommandBase.within_limits(j, mx) is True
    assert CommandBase.within_limits(j, (mn + mx) // 2) is True
    assert CommandBase.within_limits(j, mn - 1) is False
    assert CommandBase.within_limits(j, mx + 1) is False

    assert CommandBase.clamp_to_limits(j, mn - 123) == mn
    assert CommandBase.clamp_to_limits(j, mx + 456) == mx
    mid = (mn + mx) // 2
    assert CommandBase.clamp_to_limits(j, mid) == mid


def test_joint_dir_and_index():
    # Positive direction selectors
    d, idx = CommandBase.joint_dir_and_index(0)
    assert d == 1 and idx == 0
    d, idx = CommandBase.joint_dir_and_index(5)
    assert d == 1 and idx == 5

    # Negative direction selectors
    d, idx = CommandBase.joint_dir_and_index(6)
    assert d == -1 and idx == 0
    d, idx = CommandBase.joint_dir_and_index(11)
    assert d == -1 and idx == 5
