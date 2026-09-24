"""
Streaming executors for online robot motion using Ruckig.

Provides jerk-limited motion execution for real-time control:
- StreamingExecutor: Joint-space jogging and streaming
- CartesianStreamingExecutor: Cartesian-space jogging and streaming

Precomputed trajectories bypass these executors and go directly to the controller,
since they're already time-optimal (TOPPRA/RUCKIG) or validated (QUINTIC/TRAPEZOID).
"""

import logging
import math
from abc import ABC, abstractmethod

import numpy as np
from numba import njit
from numpy.typing import NDArray
from ruckig import (  # type: ignore[unresolved-import, ty:unresolved-import]
    ControlInterface,
    InputParameter,
    OutputParameter,
    Result,
    Ruckig,
    Synchronization,
)

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import INTERVAL_S, LIMITS
from pinokin import so3_exp, so3_log

logger = logging.getLogger(__name__)

_IDENTITY_SE3: np.ndarray = np.eye(4, dtype=np.float64)
_IDENTITY_SE3.flags.writeable = False


@njit(cache=True)
def _pose_to_tangent_jit(
    ref_pose: np.ndarray,
    pose: np.ndarray,
    rel_rot: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
) -> None:
    """Coordinates of ``pose`` against ``ref_pose``: its translation in the
    reference's axes, then the axis-angle of its rotation relative to the
    reference. The two are independent, so a straight line in these
    coordinates is a straight line for the TCP with the tool turning about
    one fixed axis — not the screw an SE3 twist traces when both change.

    Args:
        ref_pose: Reference pose (4x4 SE3)
        pose: Pose to convert (4x4 SE3)
        rel_rot: Workspace for the relative rotation (3x3)
        out: Output coordinates (6,) [x, y, z, wx, wy, wz]
        omega_ws: Workspace for the axis-angle (3,)
    """
    for i in range(3):
        acc = 0.0
        for k in range(3):
            acc += ref_pose[k, i] * (pose[k, 3] - ref_pose[k, 3])
        out[i] = acc
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += ref_pose[k, i] * pose[k, j]
            rel_rot[i, j] = acc
    so3_log(rel_rot, omega_ws)
    out[3] = omega_ws[0]
    out[4] = omega_ws[1]
    out[5] = omega_ws[2]


@njit(cache=True)
def _tangent_to_pose_jit(
    ref_pose: np.ndarray,
    tangent: np.ndarray,
    rel_rot: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
) -> None:
    """The pose at ``tangent`` against ``ref_pose``; the inverse of
    :func:`_pose_to_tangent_jit`.

    Args:
        ref_pose: Reference pose (4x4 SE3)
        tangent: Coordinates (6,) [x, y, z, wx, wy, wz]
        rel_rot: Workspace for the relative rotation (3x3)
        out: Output pose (4x4 SE3)
        omega_ws: Workspace for the axis-angle (3,)
    """
    omega_ws[0] = tangent[3]
    omega_ws[1] = tangent[4]
    omega_ws[2] = tangent[5]
    so3_exp(omega_ws, rel_rot)
    for i in range(3):
        acc = ref_pose[i, 3]
        for k in range(3):
            acc += ref_pose[i, k] * tangent[k]
        out[i, 3] = acc
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += ref_pose[i, k] * rel_rot[k, j]
            out[i, j] = acc
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


def cap_twist(twist: np.ndarray, linear_max: float, angular_max: float) -> None:
    """Scale a twist's linear and angular parts down to their ceilings in
    place, each keeping its direction. The live jog and its preview cap
    alike through this."""
    lin = math.sqrt(twist[0] * twist[0] + twist[1] * twist[1] + twist[2] * twist[2])
    if lin > linear_max:
        k = linear_max / lin
        for i in range(3):
            twist[i] *= k
    ang = math.sqrt(twist[3] * twist[3] + twist[4] * twist[4] + twist[5] * twist[5])
    if ang > angular_max:
        k = angular_max / ang
        for i in range(3, 6):
            twist[i] *= k


# Module-level constant avoids tuple creation per error check.
_RUCKIG_ERRORS = (Result.Error, Result.ErrorInvalidInput)


# =============================================================================
# Base Class
# =============================================================================


class RuckigExecutorBase(ABC):
    """
    Abstract base class for Ruckig-based streaming executors.

    Provides common infrastructure for jerk-limited motion execution:
    - Velocity/acceleration limit scaling
    - Common tick() error handling
    - Graceful stop implementation

    Subclasses implement space-specific logic (joint vs Cartesian).

    Note: Position/velocity buffers returned by tick() are reused across calls
    to minimize allocations in the 250Hz control loop. Callers must copy data
    if they need to retain values across ticks.
    """

    def __init__(self, num_dofs: int, dt: float = INTERVAL_S):
        self.num_dofs = num_dofs
        self.dt = dt
        self.ruckig = Ruckig(num_dofs, dt)
        self.inp = InputParameter(num_dofs)
        self.out = OutputParameter(num_dofs)
        self.active = False
        self._vel_scale: float = 1.0
        self._acc_scale: float = 1.0

        # Reused every tick to avoid allocations.
        self._pos_out = np.zeros(num_dofs)
        self._vel_out = np.zeros(num_dofs)
        self._zeros_np = np.zeros(num_dofs)
        self._zeros: list[float] = [0.0] * num_dofs

        self._init_limits()
        self._init_state()

    @abstractmethod
    def _init_limits(self) -> None:
        """Initialize hardware limits from config. Called by __init__."""
        ...

    @abstractmethod
    def _init_state(self) -> None:
        """Initialize Ruckig input parameters. Called by __init__."""
        ...

    @abstractmethod
    def _apply_limits(self) -> None:
        """Apply current limits (with scaling) to Ruckig parameters."""
        ...

    def set_limits(self, velocity_frac: float = 1.0, accel_frac: float = 1.0) -> None:
        """Set velocity/acceleration as fraction of limits (0.0-1.0)."""
        self._vel_scale = max(0.01, min(1.0, velocity_frac))
        self._acc_scale = max(0.01, min(1.0, accel_frac))
        self._apply_limits()

    def _tick_ruckig(self) -> tuple[Result, np.ndarray, np.ndarray]:
        """
        Common Ruckig update logic.

        Returns:
            (result, new_position, new_velocity) - uses pre-allocated buffers
        """
        result = self.ruckig.update(self.inp, self.out)
        if result in _RUCKIG_ERRORS:
            logger.error(f"Ruckig error: {result}")
            self.active = False
        else:
            self.out.pass_to_input(self.inp)
        # Copy into pre-allocated buffers to avoid a list() allocation per tick.
        self._pos_out[:] = self.out.new_position
        self._vel_out[:] = self.out.new_velocity
        return result, self._pos_out, self._vel_out

    def stop(self) -> None:
        """Request graceful stop - decelerate to zero velocity."""
        # Whole-array assignment: ruckig hands out a copy of its targets, so
        # writing into an element of one changes nothing.
        self.inp.control_interface = ControlInterface.Velocity
        self.inp.target_velocity = self._zeros
        self.inp.target_acceleration = self._zeros


# =============================================================================
# Joint-Space Executor
# =============================================================================


class StreamingExecutor(RuckigExecutorBase):
    """
    Streaming execution layer for online robot motion in joint space.

    Used only for jogging and streaming commands. Precomputed trajectories
    bypass this executor entirely.

    Key features:
    - Jerk-limited smoothing via Ruckig
    - Automatic Cartesian velocity limiting when enabled
    - Smooth motion to position targets
    - Online modification: changing targets mid-motion produces smooth blending
    - Preserves velocity/acceleration state across ticks
    """

    def __init__(self, num_dofs: int = 6, dt: float = INTERVAL_S):
        """
        Initialize streaming executor.

        Args:
            num_dofs: Number of degrees of freedom (joints)
            dt: Control cycle time in seconds
        """
        # Cartesian velocity limit (mm/s), None = no cart limiting.
        # Must be set before super().__init__ calls _init_limits/_init_state.
        self._cart_vel_limit: float | None = None

        # Pre-allocated buffers for cart velocity limit calculations (avoids per-call allocations).
        self._q_current_buf = np.zeros(num_dofs, dtype=np.float64)
        self._q_target_buf = np.zeros(num_dofs, dtype=np.float64)
        self._dq_buf = np.zeros(num_dofs, dtype=np.float64)
        self._jacob0_buf = np.zeros((6, num_dofs), dtype=np.float64, order="F")

        # Each Ruckig-parameter buffer below has ONE semantic purpose, never reused across roles.
        self._sync_pos_buf: list[float] = [0.0] * num_dofs
        self._max_vel_buf: list[float] = [0.0] * num_dofs
        self._max_acc_buf: list[float] = [0.0] * num_dofs
        self._max_jerk_buf: list[float] = [0.0] * num_dofs
        self._target_vel_buf: list[float] = [0.0] * num_dofs

        super().__init__(num_dofs, dt)

    def _init_limits(self) -> None:
        """Initialize hardware limits from centralized config."""
        self._hardware_v_max = LIMITS.joint.hard.velocity
        self._hardware_a_max = LIMITS.joint.hard.acceleration
        self._hardware_j_max = LIMITS.joint.hard.jerk
        self._jog_v_max = LIMITS.joint.jog.velocity

    def _init_state(self) -> None:
        """Initialize Ruckig input parameters."""
        n = self.num_dofs
        self.inp.current_position = [0.0] * n
        self.inp.current_velocity = [0.0] * n
        self.inp.current_acceleration = [0.0] * n
        self.inp.target_position = [0.0] * n
        self.inp.target_velocity = [0.0] * n
        self.inp.target_acceleration = [0.0] * n
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Apply current limits (with scaling) to Ruckig parameters."""
        self._apply_scaled_vel_limit()
        for i in range(self.num_dofs):
            self._max_acc_buf[i] = self._hardware_a_max[i] * self._acc_scale
            self._max_jerk_buf[i] = self._hardware_j_max[i]
        self.inp.max_acceleration = self._max_acc_buf
        self.inp.max_jerk = self._max_jerk_buf

    def _apply_scaled_vel_limit(self) -> None:
        for i in range(self.num_dofs):
            self._max_vel_buf[i] = self._hardware_v_max[i] * self._vel_scale
        self.inp.max_velocity = self._max_vel_buf

    def set_cart_velocity_limit(self, limit_mm_s: float | None) -> None:
        """
        Set Cartesian velocity limit for subsequent position targets.

        When set, joint velocity limits are dynamically adjusted based on
        the direction to target to ensure TCP velocity stays within limit.

        Args:
            limit_mm_s: Cartesian velocity limit in mm/s, or None to disable
        """
        self._cart_vel_limit = limit_mm_s

    def sync_position(self, pos: list[float] | np.ndarray) -> None:
        """
        Sync current position from robot feedback.

        Call this when idle to ensure executor starts from the actual robot position.
        Should not be called while active (mid-motion).

        Args:
            pos: Current joint positions in radians
        """
        if not self.active:
            self._sync_pos_buf[:] = pos
            self.inp.current_position = self._sync_pos_buf
            self.inp.current_velocity = self._zeros
            self.inp.current_acceleration = self._zeros
            self.inp.target_position = self._sync_pos_buf

    def set_position_target(self, q_target: list[float]) -> None:
        """
        Set position target with automatic Cartesian velocity limiting.

        If cart_vel_limit is set, joint velocity limits are dynamically
        adjusted based on the direction to target using the local tangent
        method to ensure TCP velocity stays within the Cartesian limit.

        Args:
            q_target: Target joint positions in radians
        """
        if self._cart_vel_limit is not None and self._cart_vel_limit > 0:
            self._apply_cart_velocity_limit(q_target)
        else:
            self._apply_scaled_vel_limit()

        self.inp.synchronization = Synchronization.Time
        self.inp.control_interface = ControlInterface.Position
        self._sync_pos_buf[:] = q_target
        self.inp.target_position = self._sync_pos_buf
        self.inp.target_velocity = self._zeros  # Stop at target
        self.active = True

    def set_jog_velocity(self, joint_velocities: NDArray[np.float64]) -> None:
        """
        Set target velocity for jogging using Ruckig velocity control.

        Ruckig will smoothly accelerate/decelerate to reach target velocity.
        Call with [0,0,0,0,0,0] to smoothly stop.

        Args:
            joint_velocities: Desired velocity for each joint in rad/s (signed)
        """
        # Jog uses its own velocity limits (~80% of hardware) rather than the hardware caps.
        for i in range(self.num_dofs):
            self._max_vel_buf[i] = self._jog_v_max[i] * self._vel_scale
            self._max_acc_buf[i] = self._hardware_a_max[i] * self._acc_scale
        self.inp.max_velocity = self._max_vel_buf
        self.inp.max_acceleration = self._max_acc_buf

        # Each joint brakes on its own profile: synchronized, a joint whose
        # limit stops it would be stretched to finish with one still
        # ramping, and carried past the stopping distance its lookahead
        # measured.
        self.inp.synchronization = Synchronization.No
        self.inp.control_interface = ControlInterface.Velocity
        self._target_vel_buf[:] = joint_velocities
        self.inp.target_velocity = self._target_vel_buf
        self.inp.target_acceleration = self._zeros
        self.active = True

    def _apply_cart_velocity_limit(self, q_target: list[float]) -> None:
        """
        Compute and apply Cartesian-aware joint velocity limits.

        Uses the local tangent method: computes joint velocity limits that
        ensure TCP velocity along the direction to target stays within the
        Cartesian velocity limit.
        """
        self._q_current_buf[:] = self.inp.current_position
        self._q_target_buf[:] = q_target
        np.subtract(self._q_target_buf, self._q_current_buf, out=self._dq_buf)

        # Linear part of the Jacobian is the first 3 rows.
        PAROL6_ROBOT.robot.jacob0_into(self._q_current_buf, self._jacob0_buf)
        J_lin = self._jacob0_buf[:3, :]

        # Cartesian velocity per unit "scale" along the dq direction.
        cart_vel_per_scale = np.linalg.norm(J_lin @ self._dq_buf)

        if cart_vel_per_scale > 1e-6:
            v_max_m_s = (
                self._cart_vel_limit / 1000.0 if self._cart_vel_limit else 0.0
            )  # mm/s to m/s
            max_scale = v_max_m_s / cart_vel_per_scale

            for j in range(self.num_dofs):
                # Joint velocity = dq[j] * scale, so max joint vel = |dq[j]| * max_scale.
                q_dot_max = min(
                    abs(self._dq_buf[j]) * max_scale,
                    self._hardware_v_max[j] * self._vel_scale,
                )
                # Non-zero minimum avoids Ruckig issues with zero limits.
                self._max_vel_buf[j] = max(q_dot_max, 1e-6)

            self.inp.max_velocity = self._max_vel_buf
        else:
            # Near-zero motion: fall back to the scaled hardware limits.
            self._apply_scaled_vel_limit()

    def tick(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Execute one control cycle.

        Warning: Returned arrays are reused across calls. Copy if needed across ticks.

        Returns:
            Tuple of (position, velocity, finished):
            - position: Current commanded position in radians
            - velocity: Current commanded velocity in rad/s
            - finished: True if target reached (position mode) or velocity reached (velocity mode)
        """
        if not self.active:
            self._pos_out[:] = self.inp.current_position
            return self._pos_out, self._zeros_np, True

        result, pos, vel = self._tick_ruckig()

        if result in _RUCKIG_ERRORS:
            return self._pos_out, self._zeros_np, True

        return pos, vel, result == Result.Finished

    def reset_limits(self) -> None:
        """Reset velocity, acceleration, and jerk limits to hardware defaults."""
        self._vel_scale = 1.0
        self._acc_scale = 1.0
        self._apply_limits()

    def reset(self) -> None:
        """Reset executor state."""
        self._vel_scale = 1.0
        self._acc_scale = 1.0
        self.active = False
        self._cart_vel_limit = None
        self._init_state()

    @property
    def cart_vel_limit(self) -> float | None:
        """Get current Cartesian velocity limit in mm/s, or None if disabled."""
        return self._cart_vel_limit


# =============================================================================
# Cartesian-Space Executor
# =============================================================================


class CartesianStreamingExecutor(RuckigExecutorBase):
    """
    Cartesian-space Ruckig executor for smooth TCP motion.

    Uses SE3 Lie algebra representation internally:
    - Position: [x, y, z] in meters
    - Orientation: [wx, wy, wz] as axis-angle vector (radians)

    Ruckig operates on this 6D tangent space representation, ensuring
    smooth interpolation without gimbal lock issues.

    Key features:
    - Jerk-limited smoothing via Ruckig in Cartesian space
    - Position mode for MOVECART (straight-line TCP motion)
    - Velocity mode for JOGL (6-DOF twist jogging)
    - WRF/TRF frame support for jogging
    """

    def __init__(self, dt: float = INTERVAL_S):
        """
        Initialize Cartesian streaming executor.

        Args:
            dt: Control cycle time in seconds
        """
        # Reference pose for tangent space computations.
        # Must be set before super().__init__ calls _init_limits/_init_state.
        self.reference_pose: np.ndarray | None = None

        # Ruckig copies values on assignment, so these are updated in-place then re-assigned.
        # Must exist before super().__init__(), which calls _apply_limits() during init.
        self._max_velocity_arr = np.zeros(6, dtype=np.float64)
        self._max_acceleration_arr = np.zeros(6, dtype=np.float64)
        self._max_jerk_arr = np.zeros(6, dtype=np.float64)
        self._target_velocity_arr = np.zeros(6, dtype=np.float64)
        self._target_acceleration_arr = np.zeros(6, dtype=np.float64)

        # Unit direction of the motion the limits are being applied to,
        # and the tangent the last tick reached. _apply_limits reads the
        # direction, and super().__init__() calls it, so both exist first.
        self._direction = np.zeros(6, dtype=np.float64)
        self._cur_tangent = np.zeros(6, dtype=np.float64)
        self._delta_tangent = np.zeros(6, dtype=np.float64)
        self._last_target = np.zeros(6, dtype=np.float64)
        self._has_target = False

        super().__init__(num_dofs=6, dt=dt)  # 6-DOF: [x, y, z, wx, wy, wz]

        self._tangent_buf = np.zeros(6, dtype=np.float64)
        self._vel_np_buf = np.zeros(6, dtype=np.float64)

        # Ruckig's default (Time) only makes the six components FINISH
        # together; each still takes its own time-optimal route there, so
        # the coordinates bow and the TCP leaves the straight line by
        # millimetres. Phase holds them to one shared profile, which keeps
        # the TCP on its line and the tool on its axis. Ruckig falls back
        # to time synchronization by itself when the limits make a shared
        # profile impossible.
        self.inp.synchronization = Synchronization.Phase

        # Workspace buffers let the JIT pose conversions run with zero allocation.
        self._result_pose_buf = np.zeros((4, 4), dtype=np.float64)
        self._omega_ws = np.zeros(3, dtype=np.float64)
        self._R_ws = np.zeros((3, 3), dtype=np.float64)

    def _init_limits(self) -> None:
        """Initialize Cartesian velocity/acceleration/jerk limits from centralized config."""
        # Use jog limits for streaming (servo/jog share the same executor)
        # Linear limits (SI: m/s, m/s², m/s³)
        self._v_lin_max = LIMITS.cart.jog.velocity.linear
        self._a_lin_max = LIMITS.cart.jog.acceleration.linear
        self._j_lin_max = LIMITS.cart.jog.jerk.linear
        # Angular limits (SI: rad/s, rad/s², rad/s³)
        self._v_ang_max = LIMITS.cart.jog.velocity.angular
        self._a_ang_max = LIMITS.cart.jog.acceleration.angular
        self._j_ang_max = LIMITS.cart.jog.jerk.angular

    def _init_state(self) -> None:
        """Initialize Ruckig input parameters."""
        self.inp.current_position = self._zeros
        self.inp.current_velocity = self._zeros
        self.inp.current_acceleration = self._zeros
        self.inp.target_position = self._zeros
        self.inp.target_velocity = self._zeros
        self.inp.target_acceleration = self._zeros
        self._apply_limits()

    def _set_direction(self, vec: np.ndarray) -> None:
        """Point the envelope along `vec`, leaving it as it was when
        `vec` is too small to take a direction from.

        A servo stream retargets every tick and the remaining delta
        shrinks to nothing as the move lands; snapping back to isotropic
        there would change the limits under a move still running.
        """
        norm = 0.0
        for i in range(6):
            norm += vec[i] * vec[i]
        if norm <= 1e-24:
            return
        inv = 1.0 / math.sqrt(norm)
        for i in range(6):
            self._direction[i] = vec[i] * inv

    def _direction_scale(self, lo: int, hi: int) -> float:
        """Ratio turning a per-component ceiling into a TCP-norm one.

        The configured ceilings are TCP speeds -- what the tool may
        travel at, not what each axis may. Ruckig bounds each component
        on its own, so an isotropic envelope lets a diagonal run the norm
        up to sqrt(3) times the ceiling: a 200 mm/s limit reaches 269
        mm/s on a three-axis move. Under phase synchronization the six
        components share one profile, so the tangent runs along a fixed
        direction at some scalar rate -- the component Ruckig binds on is
        the largest |d|, and the norm is |d| over the half. Their ratio
        makes the two agree.
        """
        norm = 0.0
        largest = 0.0
        for i in range(lo, hi):
            v = self._direction[i]
            norm += v * v
            a = abs(v)
            if a > largest:
                largest = a
        if norm <= 0.0:
            return 1.0
        return largest / math.sqrt(norm)

    def _apply_limits(self) -> None:
        """Apply current limits (with scaling) to Ruckig parameters.

        Uses pre-allocated numpy arrays to avoid per-tick allocations.
        """
        lin = self._direction_scale(0, 3)
        ang = self._direction_scale(3, 6)

        self._max_velocity_arr[:3] = self._v_lin_max * self._vel_scale * lin
        self._max_velocity_arr[3:] = self._v_ang_max * self._vel_scale * ang
        self.inp.max_velocity = self._max_velocity_arr

        self._max_acceleration_arr[:3] = self._a_lin_max * self._acc_scale * lin
        self._max_acceleration_arr[3:] = self._a_ang_max * self._acc_scale * ang
        self.inp.max_acceleration = self._max_acceleration_arr

        self._max_jerk_arr[:3] = self._j_lin_max * lin
        self._max_jerk_arr[3:] = self._j_ang_max * ang
        self.inp.max_jerk = self._max_jerk_arr

    def sync_pose(self, current_pose: np.ndarray) -> None:
        """
        Sync current pose from robot feedback.

        Call this when idle to ensure executor starts from actual robot pose.
        Sets the reference pose for tangent space computations.

        Args:
            current_pose: Current TCP pose as 4x4 SE3 matrix
        """
        self.reference_pose = current_pose.copy()  # avoid aliasing with cached FK
        self._cur_tangent.fill(0.0)
        self._has_target = False
        # Reset Ruckig state to origin (relative to reference)
        self.inp.current_position = self._zeros
        self.inp.current_velocity = self._zeros
        self.inp.current_acceleration = self._zeros
        self.inp.target_position = self._zeros
        self.active = False

    def _pose_to_tangent(self, pose: np.ndarray) -> np.ndarray:
        """
        Coordinates of an SE3 pose relative to the reference:
        [x, y, z, wx, wy, wz], the translation in the reference's axes and
        the relative rotation's axis-angle (see ``_pose_to_tangent_jit``).

        Args:
            pose: 4x4 SE3 matrix to convert

        Returns:
            Pre-allocated 6D buffer (reused across calls; Ruckig copies on assignment)
        """
        if self.reference_pose is None:
            self._tangent_buf.fill(0.0)
            return self._tangent_buf
        _pose_to_tangent_jit(
            self.reference_pose,
            pose,
            self._R_ws,
            self._tangent_buf,
            self._omega_ws,
        )
        return self._tangent_buf

    def _tangent_to_pose(self, tangent: np.ndarray) -> np.ndarray:
        """
        Convert 6D tangent vector back to SE3 pose.

        Args:
            tangent: 6D tangent vector [x, y, z, wx, wy, wz]

        Returns:
            4x4 SE3 matrix
        """
        if self.reference_pose is None:
            return np.eye(4, dtype=np.float64)
        self._tangent_buf[:] = tangent
        _tangent_to_pose_jit(
            self.reference_pose,
            self._tangent_buf,
            self._R_ws,
            self._result_pose_buf,
            self._omega_ws,
        )
        return self._result_pose_buf

    def set_pose_target(self, target_pose: np.ndarray) -> None:
        """
        Set target pose for position mode (MOVECART).

        Ruckig will smoothly interpolate from current pose to target
        along a straight line in Cartesian space.

        Args:
            target_pose: Target TCP pose as SE3
        """
        target_tangent = self._pose_to_tangent(target_pose)

        # Re-planning a target Ruckig is already tracking costs the phase
        # synchronization that keeps the TCP on its line. A re-plan tests
        # the current velocity and acceleration against the new profile
        # and falls back to time synchronization unless they line up
        # exactly; once it has, the state drifts further out of phase and
        # the next tick fails the test again. A servo stream repeats its
        # target at the tick rate, so this is the common case, not an
        # edge one: the same move retargeted every tick left the line by
        # 4.7 mm, and left by none at all when set once.
        if self._has_target:
            same = True
            for i in range(6):
                if self._last_target[i] != target_tangent[i]:
                    same = False
                    break
            if same:
                self.active = True
                return
        self._last_target[:] = target_tangent
        self._has_target = True

        # The envelope is direction-dependent (see _apply_limits), and
        # the direction is the one from where the limiter is to the
        # target, not the target's own bearing from the reference.
        np.subtract(target_tangent, self._cur_tangent, out=self._delta_tangent)
        self._set_direction(self._delta_tangent)

        self.inp.control_interface = ControlInterface.Position
        self.inp.target_position = target_tangent
        self.inp.target_velocity = self._zeros  # Stop at target

        self._apply_limits()
        self.active = True

    def set_jog_twist(self, twist: np.ndarray, wrf: bool) -> None:
        """Drive the TCP at a 6-DOF velocity `[vx, vy, vz, wx, wy, wz]`
        (m/s, rad/s), in world axes when `wrf` else in the tool's.

        Velocity mode: Ruckig ramps to the twist under the envelope,
        which follows the twist's direction so a diagonal runs at the
        configured TCP ceiling rather than sqrt(3) times it; a twist
        asking for more than the ceiling is scaled down to it, direction
        kept, since Ruckig's velocity interface does not bound the
        target itself. An all-zero twist is a brake. Needs
        `reference_pose`, which `sync_pose` sets.
        """
        if self.reference_pose is None:
            logger.warning("set_jog_twist called without reference_pose")
            return
        t = self._target_velocity_arr
        if wrf:
            # The coordinates are in the reference's axes: Rᵀ · world.
            R = self.reference_pose
            for i in range(3):
                t[i] = R[0, i] * twist[0] + R[1, i] * twist[1] + R[2, i] * twist[2]
                t[3 + i] = R[0, i] * twist[3] + R[1, i] * twist[4] + R[2, i] * twist[5]
        else:
            t[:] = twist
        cap_twist(
            t, self._v_lin_max * self._vel_scale, self._v_ang_max * self._vel_scale
        )

        self._has_target = False
        self._set_direction(self._target_velocity_arr)
        self.inp.control_interface = ControlInterface.Velocity
        self.inp.target_velocity = self._target_velocity_arr
        self._target_acceleration_arr.fill(0.0)
        self.inp.target_acceleration = self._target_acceleration_arr

        self._apply_limits()
        self.active = True

    def tick(self) -> tuple[np.ndarray, NDArray[np.float64], bool]:
        """
        Execute one control cycle.

        Warning: Returned pose and velocity arrays are reused across calls.
        Copy if needed across ticks.

        Returns:
            Tuple of (smoothed_pose, velocity, finished):
            - smoothed_pose: The smoothed Cartesian pose for this tick (buffer, reused)
            - velocity: Current 6D velocity [vx, vy, vz, wx, wy, wz] (buffer, reused)
            - finished: True if target reached (position mode) or
                       target velocity reached (velocity mode)
        """
        if not self.active or self.reference_pose is None:
            self._vel_np_buf.fill(0.0)
            return (
                self.reference_pose
                if self.reference_pose is not None
                else _IDENTITY_SE3,
                self._vel_np_buf,
                True,
            )

        result, pos, vel = self._tick_ruckig()

        if result in _RUCKIG_ERRORS:
            self._vel_np_buf.fill(0.0)
            return (
                self.reference_pose
                if self.reference_pose is not None
                else _IDENTITY_SE3,
                self._vel_np_buf,
                True,
            )

        smoothed_pose = self._tangent_to_pose(pos)
        self._cur_tangent[:] = pos
        self._vel_np_buf[:] = vel

        # Don't auto-deactivate in velocity mode - caller controls via set_jog_velocity(0)
        return smoothed_pose, self._vel_np_buf, result == Result.Finished

    def correct_position(self, actual_pose: np.ndarray) -> None:
        """Correct CSE position to match actual TCP after joint-space clamping.

        Only updates position — velocity and acceleration are left as-is
        since the robot is still moving.  Ruckig handles the slight
        state inconsistency and re-plans smoothly.
        """
        self._pose_to_tangent(actual_pose)
        self.inp.current_position = self._tangent_buf

    def reset(self) -> None:
        """Reset executor state."""
        self._vel_scale = 1.0
        self._acc_scale = 1.0
        self.reference_pose = None
        self.active = False
        self._init_state()
