"""
Tests for trajectory execution:
1. Precomputed trajectories go directly to controller (no streaming executor)
2. QUINTIC/TRAPEZOID profiles validate velocity/acceleration limits
3. TOPPRA/RUCKIG profiles are inherently limit-respecting
"""

import numpy as np
import pytest

from parol6.commands.joint_commands import MoveJCommand
from parol6.config import rad_to_steps, steps_to_rad
from parol6.motion import JointPath, TrajectoryBuilder, ProfileType

pytestmark = pytest.mark.unit


# Minimal ControllerState mock
class MockState:
    def __init__(self):
        # Start position: actual home position from config
        from parol6.config import HOME_ANGLES_DEG

        home_rad = np.deg2rad(HOME_ANGLES_DEG)
        self.Position_in = np.zeros(6, dtype=np.int32)
        rad_to_steps(home_rad, self.Position_in)
        self.Position_out = np.zeros(6, dtype=np.int32)
        self.Speed_out = np.zeros(6, dtype=np.int32)
        self.Command_out = 0
        self.motion_profile = "TOPPRA"


class TestPrecomputedTrajectories:
    """Tests for precomputed trajectory execution (no streaming executor)."""

    def test_ruckig_trajectory_builds_successfully(self):
        """RUCKIG trajectory builder creates valid trajectory."""
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        end_rad = np.deg2rad([10, -50, 180, 15, 10, 5]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.RUCKIG,
            velocity_frac=0.5,
            accel_frac=0.5,
        )

        trajectory = builder.build()

        assert len(trajectory.steps) > 0, "Trajectory should have steps"
        # Trajectory has no smooth attribute anymore - goes direct to controller

    def test_toppra_trajectory_builds_successfully(self):
        """TOPPRA trajectory builder creates valid trajectory."""
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        end_rad = np.deg2rad([10, -50, 180, 15, 10, 5]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.TOPPRA,
            velocity_frac=0.5,
            accel_frac=0.5,
        )

        trajectory = builder.build()

        assert len(trajectory.steps) > 0, "Trajectory should have steps"

    def test_linear_trajectory_builds_successfully(self):
        """LINEAR trajectory builder creates valid trajectory."""
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        end_rad = np.deg2rad([10, -50, 180, 15, 10, 5]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.LINEAR,
            velocity_frac=0.5,
            accel_frac=0.5,
        )

        trajectory = builder.build()

        assert len(trajectory.steps) > 0, "Trajectory should have steps"


class TestLimitValidation:
    """Tests for QUINTIC/TRAPEZOID limit validation."""

    def test_quintic_extends_duration_when_explicit_duration_too_short(self, caplog):
        """QUINTIC automatically extends duration when user-specified 0.5s would violate joint velocity limits.

        Scenario: Large joint move (80° on J1, 40° on J2) with explicit 0.5s duration.
        Expected: Builder detects velocity violation, extends duration, logs warning about
        exceeding the user-requested duration.
        """
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        end_rad = np.deg2rad([10, -50, 180, 15, 10, 5]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.QUINTIC,
            velocity_frac=0.5,
            accel_frac=0.5,
            duration=0.5,  # Too short for this move - will be extended
        )

        trajectory = builder.build()
        # Duration should have been extended beyond the requested 0.5s
        assert trajectory.duration > 0.5
        # Should have logged a warning about extension
        assert "Extending duration" in caplog.text

    def test_quintic_accepts_safe_velocity(self):
        """QUINTIC trajectory builds successfully with safe parameters."""
        # Small move with long duration stays within limits
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        # Small move - just 5 degrees
        end_rad = np.deg2rad([90, -85, 180, 0, 0, 180]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.QUINTIC,
            velocity_frac=0.5,
            accel_frac=0.5,
            duration=2.0,  # Long duration for small move
        )

        trajectory = builder.build()
        assert len(trajectory.steps) > 0

    def test_trapezoid_extends_duration_when_explicit_duration_too_short(self, caplog):
        """TRAPEZOID automatically extends duration when user-specified 0.3s would violate joint velocity limits.

        Scenario: Large joint move (80° on J1, 40° on J2) with explicit 0.3s duration.
        Expected: Builder detects velocity violation, extends duration, logs warning about
        exceeding the user-requested duration.
        """
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        end_rad = np.deg2rad([10, -50, 180, 15, 10, 5]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.TRAPEZOID,
            velocity_frac=0.5,
            accel_frac=0.5,
            duration=0.3,  # Too short for this move - will be extended
        )

        trajectory = builder.build()
        # Duration should have been extended beyond the requested 0.3s
        assert trajectory.duration > 0.3
        # Should have logged a warning about extension
        assert "Extending duration" in caplog.text


class TestRuckigExecution:
    """Tests for RUCKIG joint move execution."""

    def test_ruckig_reaches_target(self):
        """RUCKIG trajectory reaches target accurately."""
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        target_angles = [10, -50, 180, 15, 10, 5]
        end_rad = np.deg2rad(target_angles).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.RUCKIG,
            velocity_frac=1.0,
            accel_frac=1.0,
        )

        trajectory = builder.build()

        # Verify final step reaches target
        final_steps = trajectory.steps[-1]
        final_rad = np.zeros(6, dtype=np.float64)
        steps_to_rad(final_steps, final_rad)

        error_deg = np.rad2deg(np.abs(final_rad - end_rad))
        max_error = np.max(error_deg)

        assert max_error < 0.1, (
            f"RUCKIG should reach target within 0.1 deg, got {max_error:.3f} deg"
        )

    def test_ruckig_joint_move_command_setup(self):
        """MoveJCommand with RUCKIG profile sets up trajectory."""
        from parol6.protocol.wire import MoveJCmd

        # Create params struct with speed (duration=0 means use speed)
        params = MoveJCmd(
            angles=[10.0, -50.0, 180.0, 15.0, 10.0, 5.0],
            speed=0.5,
            accel=0.5,
        )
        cmd = MoveJCommand(params)

        state = MockState()
        state.motion_profile = "RUCKIG"

        cmd.do_setup(state)

        # Command should have trajectory steps ready for direct execution
        assert hasattr(cmd, "trajectory_steps")
        assert len(cmd.trajectory_steps) > 0


class TestQuinticGeometry:
    """Tests for QUINTIC trajectory geometry."""

    def test_quintic_samples_path_correctly(self):
        """QUINTIC trajectory samples all path waypoints correctly."""
        # Use a small move that respects limits
        start_rad = np.deg2rad([90, -90, 180, 0, 0, 180]).astype(np.float64)
        end_rad = np.deg2rad([90, -85, 180, 0, 0, 180]).astype(np.float64)

        joint_path = JointPath.interpolate(start_rad, end_rad, n_samples=100)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=ProfileType.QUINTIC,
            velocity_frac=0.5,
            duration=3.0,  # Long duration for safe velocity
        )

        trajectory = builder.build()

        # Check intermediate points follow the path geometry
        n_steps = len(trajectory.steps)
        mid_idx = n_steps // 2

        mid_steps = trajectory.steps[mid_idx]
        mid_rad = np.zeros(6, dtype=np.float64)
        steps_to_rad(mid_steps, mid_rad)

        # Quintic timing at t=0.5 gives s = 10*(0.5)^3 - 15*(0.5)^4 + 6*(0.5)^5 = 0.5
        # So midpoint should be halfway along path
        expected_mid_rad = (start_rad + end_rad) / 2
        error_deg = np.rad2deg(np.abs(mid_rad - expected_mid_rad))
        max_error = np.max(error_deg)

        # Allow some error from step quantization
        assert max_error < 2.0, (
            f"QUINTIC midpoint should be near path midpoint, error={max_error:.2f} deg"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
