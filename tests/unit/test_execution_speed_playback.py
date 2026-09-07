"""Observe firmware position commands from the real planner/player pipeline."""

import numpy as np

from parol6.config import INTERVAL_S, LIMITS, deg_to_steps, steps_to_rad
from parol6.protocol.wire import MoveJCmd, PauseCmd, SetExecutionSpeedCmd
from parol6.server.command_registry import create_command_from_struct
from parol6.server.motion_planner import MotionPlanner, PlanCommand, PlannerWorker
from parol6.server.segment_player import SegmentPlayer
from parol6.server.state import ControllerState


def test_override_transitions_bound_commanded_acceleration(monkeypatch):
    import parol6.server.segment_player as module

    quantum = np.empty(6)
    steps_to_rad(np.ones(6, dtype=np.int32), quantum)
    # Each emitted position has at most half a step of rounding error;
    # the second difference has coefficients 1, -2, 1.
    rounding = 2 * quantum / INTERVAL_S**2
    limits = np.asarray(LIMITS.joint.hard.acceleration)
    target = [105.0, -75.0, 195.0, 15.0, 15.0, 195.0]
    for transition in (1.0, 0.03):
        monkeypatch.setattr(module, "EXECUTION_OVERRIDE_TRANSITION_S", transition)
        planner = MotionPlanner()
        worker = PlannerWorker(planner._segment_queue)
        state = ControllerState()
        deg_to_steps(np.array([90.0, -90.0, 180.0, 0.0, 0.0, 180.0]), state.Position_in)
        state.Position_out[:] = state.Position_in
        worker.process_command(
            PlanCommand(
                command_index=1,
                params=MoveJCmd(angles=target, speed=1.0),
                position_in=state.Position_in.copy(),
                homed=True,
            )
        )
        assert planner._segment_queue._reader.poll(3), "planner produced no result"
        player = SegmentPlayer(planner)
        trace = []
        held_at = None

        def apply(params):
            command, _, error = create_command_from_struct(params)
            assert command is not None, error
            command.setup(state)
            command.tick(state)

        try:
            for k in range(round(20 / INTERVAL_S)):
                if k == round(0.15 / INTERVAL_S):
                    apply(SetExecutionSpeedCmd(0.1))
                if k == round(0.35 / INTERVAL_S):
                    apply(PauseCmd(True))
                if held_at is not None and k == held_at + round(0.2 / INTERVAL_S):
                    apply(PauseCmd(False))
                    apply(SetExecutionSpeedCmd(0.6))
                active = player.tick(state)
                assert state.error is None
                q = np.empty(6)
                steps_to_rad(state.Position_out, q)
                trace.append(q)
                state.Position_in[:] = state.Position_out
                if state.execution_paused and state.execution_applied_speed == 0:
                    if held_at is None:
                        held_at = k
                    else:
                        np.testing.assert_array_equal(q, trace[held_at])
                if not active and state.completed_command_index == 1:
                    break
            assert state.completed_command_index == 1 and held_at is not None
            np.testing.assert_allclose(
                trace[-1], np.radians(target), atol=quantum.max()
            )
            acceleration = np.diff(np.asarray(trace), n=2, axis=0) / INTERVAL_S**2
            excess = np.maximum(0.0, np.abs(acceleration) - rounding) / limits
            assert excess.max() <= 1.01, (
                f"{transition}s override exceeded acceleration: "
                f"{excess.max(axis=0).tolist()} times the joint limits after quantization"
            )
        finally:
            planner.stop()
