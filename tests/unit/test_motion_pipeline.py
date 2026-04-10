"""Unit tests for the motion pipeline (MotionPlanner + SegmentPlayer).

Tests the PlannerWorker directly (no subprocess) and the SegmentPlayer
state machine with mock segments.
"""

import multiprocessing
import time

import numpy as np
import pytest

from parol6.config import deg_to_steps
from parol6.protocol.wire import (
    CheckpointCmd,
    DelayCmd,
    HomeCmd,
    MoveJCmd,
    SelectToolCmd,
)
from parol6.server.motion_planner import (
    InlineSegment,
    MotionPlanner,
    PlanCommand,
    PlannerWorker,
    TrajectorySegment,
)
from parol6.server.segment_player import SegmentPlayer
from parol6.server.state import ControllerState

# Valid PAROL6 joint angles within limits (same as test_dry_run_blend.py)
HOME = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
W1 = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
W2 = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]
W3 = [60.0, -60.0, 210.0, 30.0, 30.0, 210.0]


def _home_steps() -> np.ndarray:
    """Get home position in motor steps."""
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT

    buf = np.zeros(6, dtype=np.int32)
    home_deg = np.array(PAROL6_ROBOT.joint.standby_deg, dtype=np.float64)
    deg_to_steps(home_deg, buf)
    return buf


def _deg_to_steps(angles: list[float]) -> np.ndarray:
    buf = np.zeros(6, dtype=np.int32)
    deg_to_steps(np.array(angles, dtype=np.float64), buf)
    return buf


def _make_movej_cmd(
    angles: list[float], speed: float = 0.5, r: float = 0.0
) -> MoveJCmd:
    return MoveJCmd(angles=angles, speed=speed, accel=1.0, r=r)


# ---------------------------------------------------------------------------
# Planner unit tests (direct PlannerWorker calls, no subprocess)
# ---------------------------------------------------------------------------


class TestPlannerDirect:
    """Test the planner's command routing and trajectory generation."""

    @pytest.fixture
    def worker(self):
        q = multiprocessing.Queue()
        w = PlannerWorker(q)
        w.state.Position_in[:] = _home_steps()
        return w

    @pytest.fixture
    def segment_queue(self, worker):
        return worker._segment_queue

    def test_single_movej_produces_trajectory_segment(self, worker, segment_queue):
        """A single MoveJ command should produce a TrajectorySegment."""
        params = _make_movej_cmd(W1)
        msg = PlanCommand(command_index=0, params=params)

        worker.process_command(msg)

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, TrajectorySegment)
        assert seg.command_index == 0
        assert seg.trajectory_steps.shape[0] > 0
        assert seg.trajectory_steps.shape[1] == 6
        assert seg.trajectory_steps.dtype == np.int32
        assert seg.duration > 0
        assert seg.blend_consumed_indices == []

    def test_settool_produces_inline_segment(self, worker, segment_queue):
        """SelectTool should produce an InlineSegment."""
        params = SelectToolCmd(tool_name="PNEUMATIC")
        msg = PlanCommand(command_index=0, params=params)

        worker.process_command(msg)

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, InlineSegment)
        assert seg.command_index == 0
        assert isinstance(seg.params, SelectToolCmd)

    def test_home_produces_inline_and_updates_position(self, worker, segment_queue):
        """Home command should produce InlineSegment and update planner position."""
        # Start from non-home position
        worker.state.Position_in[:] = _deg_to_steps(W1)
        home_steps = _home_steps()

        params = HomeCmd()
        msg = PlanCommand(command_index=0, params=params)

        worker.process_command(msg)

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, InlineSegment)
        # Planner's position should now be home
        np.testing.assert_array_equal(worker.state.Position_in, home_steps)

    def test_checkpoint_produces_inline_segment(self, worker, segment_queue):
        """Checkpoint should produce an InlineSegment."""
        params = CheckpointCmd(label="step1")
        msg = PlanCommand(command_index=5, params=params)

        worker.process_command(msg)

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, InlineSegment)
        assert seg.command_index == 5

    def test_delay_produces_inline_segment(self, worker, segment_queue):
        """Delay should produce an InlineSegment."""
        params = DelayCmd(seconds=1.5)
        msg = PlanCommand(command_index=3, params=params)

        worker.process_command(msg)

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, InlineSegment)
        assert seg.command_index == 3

    def test_ordering_preserved_movej_settool_movej(self, worker, segment_queue):
        """MoveJ -> SelectTool -> MoveJ should produce segments in order."""
        # 1. MoveJ to W1
        worker.process_command(PlanCommand(command_index=0, params=_make_movej_cmd(W1)))

        # 2. SelectTool (inline)
        worker.process_command(
            PlanCommand(command_index=1, params=SelectToolCmd(tool_name="PNEUMATIC"))
        )

        # 3. MoveJ to W2
        worker.process_command(PlanCommand(command_index=2, params=_make_movej_cmd(W2)))

        # Verify ordering
        seg1 = segment_queue.get(timeout=1.0)
        seg2 = segment_queue.get(timeout=1.0)
        seg3 = segment_queue.get(timeout=1.0)

        assert isinstance(seg1, TrajectorySegment)
        assert seg1.command_index == 0

        assert isinstance(seg2, InlineSegment)
        assert seg2.command_index == 1

        assert isinstance(seg3, TrajectorySegment)
        assert seg3.command_index == 2

    def test_blend_chain_produces_single_segment(self, worker, segment_queue):
        """MoveJ r>0, MoveJ r>0, MoveJ r=0 should produce a blended segment."""
        # r=10 -> buffer
        worker.process_command(
            PlanCommand(command_index=0, params=_make_movej_cmd(W1, r=10.0))
        )
        assert len(worker._planner._blend_buffer) == 1

        # r=10 -> buffer
        worker.process_command(
            PlanCommand(command_index=1, params=_make_movej_cmd(W2, r=10.0))
        )
        assert len(worker._planner._blend_buffer) == 2

        # r=0 -> flush
        worker.process_command(
            PlanCommand(command_index=2, params=_make_movej_cmd(W3, r=0.0))
        )
        assert len(worker._planner._blend_buffer) == 0

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, TrajectorySegment)
        assert seg.command_index == 0
        assert seg.trajectory_steps.shape[0] > 0
        # Should have consumed commands 1 and 2
        assert len(seg.blend_consumed_indices) > 0

    def test_stale_blend_buffer_flushed_on_inline(self, worker, segment_queue):
        """A pending blend buffer should be flushed when an inline command arrives."""
        # MoveJ with r=10 -> buffers
        worker.process_command(
            PlanCommand(command_index=0, params=_make_movej_cmd(W1, r=10.0))
        )
        assert len(worker._planner._blend_buffer) == 1

        # Inline command -> flush blend buffer first
        worker.process_command(
            PlanCommand(command_index=1, params=SelectToolCmd(tool_name="PNEUMATIC"))
        )

        # Should get: trajectory segment (flushed from buffer), then inline segment
        seg1 = segment_queue.get(timeout=1.0)
        seg2 = segment_queue.get(timeout=1.0)
        assert isinstance(seg1, TrajectorySegment)
        assert seg1.command_index == 0
        assert isinstance(seg2, InlineSegment)
        assert seg2.command_index == 1

    def test_stale_blend_flushed_on_timeout(self, worker, segment_queue):
        """flush_stale_blend should emit buffered trajectory."""
        worker.process_command(
            PlanCommand(command_index=0, params=_make_movej_cmd(W1, r=10.0))
        )
        assert len(worker._planner._blend_buffer) == 1

        worker.flush_stale_blend()
        assert len(worker._planner._blend_buffer) == 0

        seg = segment_queue.get(timeout=1.0)
        assert isinstance(seg, TrajectorySegment)
        assert seg.command_index == 0

    def test_cancel_clears_blend_buffer(self, worker, segment_queue):
        """cancel() should clear the blend buffer."""
        worker.process_command(
            PlanCommand(command_index=0, params=_make_movej_cmd(W1, r=10.0))
        )
        assert len(worker._planner._blend_buffer) == 1

        worker.cancel()
        assert len(worker._planner._blend_buffer) == 0


# ---------------------------------------------------------------------------
# Planner subprocess test
# ---------------------------------------------------------------------------


class TestPlannerSubprocess:
    """Test the planner running in a real subprocess."""

    def test_planner_start_stop(self):
        """Planner starts and stops cleanly."""
        planner = MotionPlanner()
        planner.start()
        assert planner.alive
        planner.stop()
        assert not planner.alive

    def test_planner_produces_segment(self):
        """Submit a MoveJ and get a TrajectorySegment back."""
        planner = MotionPlanner()
        planner.start()
        try:
            position_in = _home_steps()
            planner.submit(
                PlanCommand(
                    command_index=0,
                    params=_make_movej_cmd(W1),
                    position_in=position_in,
                )
            )

            # Poll for result (with timeout)
            seg = None
            deadline = time.time() + 10.0
            while time.time() < deadline:
                seg = planner.poll_segment()
                if seg is not None:
                    break
                time.sleep(0.05)

            assert seg is not None, "Planner should produce a segment"
            assert isinstance(seg, TrajectorySegment)
            assert seg.command_index == 0
            assert seg.trajectory_steps.shape[0] > 0
        finally:
            planner.stop()

    def test_planner_cancel(self):
        """CancelAll clears planner state."""
        planner = MotionPlanner()
        planner.start()
        try:
            planner.cancel()
            time.sleep(0.2)
            # Should still be alive after cancel
            assert planner.alive
        finally:
            planner.stop()

    def test_planner_sync_profile(self):
        """SyncProfile message is accepted without error."""
        planner = MotionPlanner()
        planner.start()
        try:
            planner.sync_profile("QUINTIC")
            # Submit a command to verify the planner is still working
            planner.submit(
                PlanCommand(
                    command_index=0,
                    params=_make_movej_cmd(W1),
                    position_in=_home_steps(),
                )
            )
            seg = None
            deadline = time.time() + 10.0
            while time.time() < deadline:
                seg = planner.poll_segment()
                if seg is not None:
                    break
                time.sleep(0.05)
            assert seg is not None
        finally:
            planner.stop()

    def test_planner_inline_command(self):
        """Submit a SelectTool command, get InlineSegment back."""
        planner = MotionPlanner()
        planner.start()
        try:
            planner.submit(
                PlanCommand(
                    command_index=0,
                    params=SelectToolCmd(tool_name="PNEUMATIC"),
                    position_in=_home_steps(),
                )
            )

            seg = None
            deadline = time.time() + 5.0
            while time.time() < deadline:
                seg = planner.poll_segment()
                if seg is not None:
                    break
                time.sleep(0.05)

            assert seg is not None
            assert isinstance(seg, InlineSegment)
            assert seg.command_index == 0
        finally:
            planner.stop()


# ---------------------------------------------------------------------------
# SegmentPlayer unit tests
# ---------------------------------------------------------------------------


class TestSegmentPlayer:
    """Test the SegmentPlayer state machine with mock data."""

    @pytest.fixture
    def state(self):
        s = ControllerState()
        s.Position_in[:] = _home_steps()
        return s

    def test_trajectory_playback(self, state):
        """SegmentPlayer should play trajectory waypoints at 1 step per tick."""
        planner = MotionPlanner()  # not started -- we'll inject segments manually

        player = SegmentPlayer(planner)

        # Create a trajectory with 5 waypoints
        steps = np.tile(_home_steps(), (5, 1))
        for i in range(5):
            steps[i, 0] += i * 100  # vary J1

        seg = TrajectorySegment(
            command_index=0,
            trajectory_steps=steps,
            duration=0.05,
        )
        player._buffer.append(seg)

        # Tick 5 times -- should output each waypoint
        for i in range(5):
            assert player.tick(state) is True
            np.testing.assert_array_equal(state.Position_out, steps[i])
            # Simulate Position_in converging to Position_out (firmware tracking)
            state.Position_in[:] = state.Position_out

        # 6th tick -- settling detects convergence, completes -> returns False
        assert player.tick(state) is False
        assert state.completed_command_index == 0

    def test_inline_command_execution(self, state):
        """SegmentPlayer should execute inline commands via setup()/tick()."""
        planner = MotionPlanner()
        player = SegmentPlayer(planner)

        # Checkpoint is an instant inline command (completes in 1 tick)
        seg = InlineSegment(
            command_index=1,
            params=CheckpointCmd(label="test_checkpoint"),
        )
        player._buffer.append(seg)

        # Should complete immediately (returns False since no more segments)
        result = player.tick(state)
        assert result is False
        assert state.completed_command_index == 1
        assert state.last_checkpoint == "test_checkpoint"

    def test_segment_ordering(self, state):
        """Segments should be processed in FIFO order."""
        planner = MotionPlanner()
        player = SegmentPlayer(planner)

        # Trajectory segment
        steps = np.tile(_home_steps(), (3, 1))
        traj = TrajectorySegment(command_index=0, trajectory_steps=steps, duration=0.03)

        # Inline segment
        inline = InlineSegment(
            command_index=1, params=CheckpointCmd(label="after_move")
        )

        player._buffer.append(traj)
        player._buffer.append(inline)

        # Play through trajectory (3 ticks)
        for _ in range(3):
            assert player.tick(state) is True

        # Next tick processes inline, then returns False (empty)
        assert player.tick(state) is False
        assert state.completed_command_index == 1
        assert state.last_checkpoint == "after_move"

    def test_cancel_clears_everything(self, state):
        """Cancel should clear active segment and buffer."""
        planner = MotionPlanner()
        player = SegmentPlayer(planner)

        steps = np.tile(_home_steps(), (100, 1))
        seg = TrajectorySegment(command_index=0, trajectory_steps=steps, duration=1.0)
        player._buffer.append(seg)

        # Start playing
        assert player.tick(state) is True

        # Cancel
        player.cancel(state)
        assert player._active is None
        assert len(player._buffer) == 0

        # Should return False now
        assert player.tick(state) is False

    def test_blend_consumed_indices(self, state):
        """Blended segment should report max consumed index on completion."""
        planner = MotionPlanner()
        player = SegmentPlayer(planner)

        steps = np.tile(_home_steps(), (2, 1))
        seg = TrajectorySegment(
            command_index=0,
            trajectory_steps=steps,
            duration=0.02,
            blend_consumed_indices=[1, 2],
        )
        player._buffer.append(seg)

        # Play 2 ticks
        assert player.tick(state) is True
        assert player.tick(state) is True

        # Complete
        assert player.tick(state) is False
        assert state.completed_command_index == 2  # max of 0, 1, 2

    def test_active_property(self, state):
        """active should reflect whether there's work to do."""
        planner = MotionPlanner()
        player = SegmentPlayer(planner)

        assert player.active is False

        steps = np.tile(_home_steps(), (2, 1))
        player._buffer.append(
            TrajectorySegment(command_index=0, trajectory_steps=steps, duration=0.02)
        )
        assert player.active is True

        # Play through
        player.tick(state)
        player.tick(state)
        player.tick(state)
        assert player.active is False
