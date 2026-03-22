"""Test that ServoL joint-velocity clamping doesn't cause TCP lurching.

When IK requires joint velocities that exceed hardware limits, ServoLCommand
clamps the joint deltas proportionally.  The per-tick CSE position correction
keeps the Cartesian planner in sync with the actual robot joints, preventing
gap accumulation and lurch on clamp exit.

This test verifies that:
1. Clamping actually occurs (the test is meaningful)
2. No single tick produces a TCP displacement larger than limits allow
3. The final position converges accurately despite clamping
"""

import math

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.servo_commands import ServoLCommand
from parol6.config import INTERVAL_S, LIMITS, rad_to_steps, steps_to_rad
from parol6.protocol.wire import ServoLCmd
from parol6.server.state import ControllerState


_q_rad_buf = np.zeros(6, dtype=np.float64)
_T_buf = np.asfortranarray(np.eye(4, dtype=np.float64))
_rpy_buf = np.zeros(3, dtype=np.float64)


def _home_steps() -> np.ndarray:
    """Home position in step space."""
    home_rad = np.deg2rad(
        np.ascontiguousarray(PAROL6_ROBOT.joint.standby_deg, dtype=np.float64)
    )
    out = np.zeros(6, dtype=np.int32)
    rad_to_steps(home_rad, out)
    return out


def _fk_mm_rpy(q_steps: np.ndarray) -> np.ndarray:
    """FK from steps -> [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
    from pinokin import so3_rpy

    steps_to_rad(np.asarray(q_steps, dtype=np.int32), _q_rad_buf)
    _q_rad_buf_c = np.ascontiguousarray(_q_rad_buf)
    PAROL6_ROBOT.robot.fkine_into(_q_rad_buf_c, _T_buf)
    so3_rpy(_T_buf[:3, :3], _rpy_buf)
    return np.array(
        [
            _T_buf[0, 3] * 1000.0,
            _T_buf[1, 3] * 1000.0,
            _T_buf[2, 3] * 1000.0,
            math.degrees(_rpy_buf[0]),
            math.degrees(_rpy_buf[1]),
            math.degrees(_rpy_buf[2]),
        ]
    )


class TestServoLClampResync:
    """Verify no TCP lurch when joint-velocity clamping is active."""

    def test_no_lurch_and_accurate_convergence(self) -> None:
        """A linear target that triggers clamping should converge accurately.

        A +30mm X offset from home triggers clamping because the IK solution
        requires joint velocities exceeding hardware limits.  We verify that:
        1. Clamping actually occurs (the test is meaningful)
        2. No single tick produces a TCP displacement larger than limits allow
        3. The motion converges to <1mm of the target
        """
        state = ControllerState()
        home = _home_steps()
        state.Position_in[:] = home

        home_pose = _fk_mm_rpy(home)
        target_pose = list(home_pose)
        target_pose[0] += 30.0  # +30mm X triggers clamping

        cmd = ServoLCommand(ServoLCmd(pose=target_pose, speed=1.0, accel=1.0))
        cmd.do_setup(state)

        # Max displacement per tick (generous 3x margin for Ruckig jerk overshoot)
        max_lin_per_tick = LIMITS.cart.jog.velocity.linear * INTERVAL_S * 3.0
        max_ang_per_tick = LIMITS.cart.jog.velocity.angular * INTERVAL_S * 3.0

        prev_pose_mm_rpy: np.ndarray | None = None
        clamped_any = False

        for tick in range(2000):
            cmd.do_setup(state)
            status = cmd.execute_step(state)

            state.Position_in[:] = state.Position_out

            current_pose = _fk_mm_rpy(state.Position_in)

            if prev_pose_mm_rpy is not None:
                lin_delta = np.linalg.norm(current_pose[:3] - prev_pose_mm_rpy[:3])
                ang_deltas = [
                    abs(
                        (current_pose[3 + i] - prev_pose_mm_rpy[3 + i] + 180) % 360
                        - 180
                    )
                    for i in range(3)
                ]
                ang_delta = max(ang_deltas)

                assert lin_delta < max_lin_per_tick * 1000.0, (
                    f"Tick {tick}: linear TCP jump {lin_delta:.3f}mm exceeds "
                    f"limit {max_lin_per_tick * 1000:.3f}mm"
                )
                assert ang_delta < math.degrees(max_ang_per_tick), (
                    f"Tick {tick}: angular TCP jump {ang_delta:.3f}° exceeds "
                    f"limit {math.degrees(max_ang_per_tick):.3f}°"
                )

            if cmd._clamped:
                clamped_any = True

            prev_pose_mm_rpy = current_pose.copy()

            if status.name == "COMPLETED":
                break

        assert clamped_any, (
            "Test did not trigger joint-velocity clamping — "
            "adjust target to force clamping"
        )

        # Verify final accuracy
        final_pose = _fk_mm_rpy(state.Position_in)
        pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target_pose[:3]))
        assert pos_error < 1.0, (
            f"Position error {pos_error:.3f}mm exceeds 1.0mm tolerance"
        )
