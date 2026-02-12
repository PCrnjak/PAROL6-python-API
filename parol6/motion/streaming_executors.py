"""
Streaming executors for online robot motion using Ruckig.

Provides jerk-limited motion execution for real-time control:
- StreamingExecutor: Joint-space jogging and streaming
- CartesianStreamingExecutor: Cartesian-space jogging and streaming

Precomputed trajectories bypass these executors and go directly to the controller,
since they're already time-optimal (TOPPRA/RUCKIG) or validated (QUINTIC/TRAPEZOID).
"""

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import LIMITS
from pinokin import se3_exp_ws, se3_inverse, se3_log_ws, se3_mul

logger = logging.getLogger(__name__)


@njit(cache=True)
def _pose_to_tangent_jit(
    ref_pose: np.ndarray,
    pose: np.ndarray,
    ref_inv: np.ndarray,
    delta: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_inv_ws: np.ndarray,
) -> None:
    """Convert SE3 pose to 6D tangent vector relative to reference.

    Uses workspace variants for zero internal allocation.

    Args:
        ref_pose: Reference pose (4x4 SE3)
        pose: Pose to convert (4x4 SE3)
        ref_inv: Workspace buffer for reference inverse (4x4)
        delta: Workspace buffer for delta transform (4x4)
        out: Output tangent vector (6,) [vx, vy, vz, wx, wy, wz]
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_inv_ws: Workspace buffer for V inverse matrix (3,3)
    """
    se3_inverse(ref_pose, ref_inv)
    se3_mul(ref_inv, pose, delta)
    se3_log_ws(delta, out, omega_ws, R_ws, V_inv_ws)


@njit(cache=True)
def _tangent_to_pose_jit(
    ref_pose: np.ndarray,
    tangent: np.ndarray,
    delta: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """Convert 6D tangent vector back to SE3 pose.

    Uses workspace variants for zero internal allocation.

    Args:
        ref_pose: Reference pose (4x4 SE3)
        tangent: Tangent vector (6,) [vx, vy, vz, wx, wy, wz]
        delta: Workspace buffer for delta transform (4x4)
        out: Output pose (4x4 SE3)
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    se3_exp_ws(tangent, delta, omega_ws, R_ws, V_ws)
    se3_mul(ref_pose, delta, out)


# Module-level constant for error checking (avoids tuple creation per check)
_RUCKIG_ERRORS = (Result.Error, Result.ErrorInvalidInput)


# =============================================================================
# Base Class
# =============================================================================


class RuckigExecutorBase(ABC):
    """
    Abstract base class for Ruckig-based streaming executors.

    Provides common infrastructure for jerk-limited motion execution:
    - Thread-safe Ruckig state management
    - Velocity/acceleration limit scaling
    - Common tick() error handling
    - Graceful stop implementation

    Subclasses implement space-specific logic (joint vs Cartesian).

    Note: Position/velocity buffers returned by tick() are reused across calls
    to minimize allocations in the 250Hz control loop. Callers must copy data
    if they need to retain values across ticks.
    """

    def __init__(self, num_dofs: int, dt: float = 0.004):
        self._lock = threading.Lock()
        self.num_dofs = num_dofs
        self.dt = dt
        self.ruckig = Ruckig(num_dofs, dt)
        self.inp = InputParameter(num_dofs)
        self.out = OutputParameter(num_dofs)
        self.active = False
        self._vel_scale: float = 1.0
        self._acc_scale: float = 1.0

        # Pre-allocated output buffers (reused every tick to avoid allocations)
        self._pos_out: list[float] = [0.0] * num_dofs
        self._vel_out: list[float] = [0.0] * num_dofs
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

    def _tick_ruckig(self) -> tuple[Result, list[float], list[float]]:
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
        # Copy into pre-allocated buffers (avoids list() allocation per tick)
        self._pos_out[:] = self.out.new_position
        self._vel_out[:] = self.out.new_velocity
        return result, self._pos_out, self._vel_out

    def stop(self) -> None:
        """Request graceful stop - decelerate to zero velocity."""
        with self._lock:
            self.inp.control_interface = ControlInterface.Velocity
            for i in range(self.num_dofs):
                self.inp.target_velocity[i] = 0.0
                self.inp.target_acceleration[i] = 0.0


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

    def __init__(self, num_dofs: int = 6, dt: float = 0.004):
        """
        Initialize streaming executor.

        Args:
            num_dofs: Number of degrees of freedom (joints)
            dt: Control cycle time in seconds (default 0.004 = 250Hz)
        """
        # Cartesian velocity limit (mm/s), None = no cart limiting
        # Must be set before super().__init__ calls _init_limits/_init_state
        self._cart_vel_limit: float | None = None

        # Pre-allocated buffers for cart velocity limit calculations (avoids per-call allocations)
        self._q_current_buf = np.zeros(num_dofs, dtype=np.float64)
        self._q_target_buf = np.zeros(num_dofs, dtype=np.float64)
        self._dq_buf = np.zeros(num_dofs, dtype=np.float64)
        self._jacob0_buf = np.zeros((6, num_dofs), dtype=np.float64, order="F")

        # Pre-allocated buffers for Ruckig parameters - each has ONE semantic purpose
        # Position sync (current/target position share same values)
        self._sync_pos_buf: list[float] = [0.0] * num_dofs
        # Limit parameters (max_velocity, max_acceleration, max_jerk)
        self._max_vel_buf: list[float] = [0.0] * num_dofs
        self._max_acc_buf: list[float] = [0.0] * num_dofs
        self._max_jerk_buf: list[float] = [0.0] * num_dofs
        # Target velocity for jogging
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
        for i in range(self.num_dofs):
            self._max_vel_buf[i] = self._hardware_v_max[i] * self._vel_scale
            self._max_acc_buf[i] = self._hardware_a_max[i] * self._acc_scale
            self._max_jerk_buf[i] = self._hardware_j_max[i]
        self.inp.max_velocity = self._max_vel_buf
        self.inp.max_acceleration = self._max_acc_buf
        self.inp.max_jerk = self._max_jerk_buf

    def set_cart_velocity_limit(self, limit_mm_s: float | None) -> None:
        """
        Set Cartesian velocity limit for subsequent position targets.

        When set, joint velocity limits are dynamically adjusted based on
        the direction to target to ensure TCP velocity stays within limit.

        Args:
            limit_mm_s: Cartesian velocity limit in mm/s, or None to disable
        """
        self._cart_vel_limit = limit_mm_s

    def sync_position(self, pos: list[float]) -> None:
        """
        Sync current position from robot feedback.

        Call this when idle to ensure executor starts from the actual robot position.
        Should not be called while active (mid-motion).

        Args:
            pos: Current joint positions in radians
        """
        with self._lock:
            if not self.active:
                self._sync_pos_buf[:] = pos
                self.inp.current_position = self._sync_pos_buf
                self.inp.current_velocity = (
                    self._zeros
                )  # Constant zeros buffer from base
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
        with self._lock:
            # Apply Cartesian velocity limiting if enabled
            if self._cart_vel_limit is not None and self._cart_vel_limit > 0:
                self._apply_cart_velocity_limit(q_target)
            else:
                # Reset to hardware limits (reuse buffer)
                self._max_vel_buf[:] = self._hardware_v_max
                self.inp.max_velocity = self._max_vel_buf

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
        with self._lock:
            # Use jog-specific velocity limits (~80% of hardware limits) - reuse buffers
            for i in range(self.num_dofs):
                self._max_vel_buf[i] = self._jog_v_max[i] * self._vel_scale
                self._max_acc_buf[i] = self._hardware_a_max[i] * self._acc_scale
            self.inp.max_velocity = self._max_vel_buf
            self.inp.max_acceleration = self._max_acc_buf

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
        # Use pre-allocated buffers to avoid allocations
        self._q_current_buf[:] = self.inp.current_position
        self._q_target_buf[:] = q_target
        np.subtract(self._q_target_buf, self._q_current_buf, out=self._dq_buf)

        # Get the linear part of the Jacobian (first 3 rows)
        PAROL6_ROBOT.robot.jacob0_into(self._q_current_buf, self._jacob0_buf)
        J_lin = self._jacob0_buf[:3, :]

        # Compute Cartesian velocity per unit "scale" along dq direction
        cart_vel_per_scale = np.linalg.norm(J_lin @ self._dq_buf)

        if cart_vel_per_scale > 1e-6:
            v_max_m_s = (
                self._cart_vel_limit / 1000.0 if self._cart_vel_limit else 0.0
            )  # mm/s to m/s
            max_scale = v_max_m_s / cart_vel_per_scale

            # Reuse pre-allocated buffer for velocity limits
            for j in range(self.num_dofs):
                # Joint velocity = dq[j] * scale, so max joint vel = |dq[j]| * max_scale
                q_dot_max = min(
                    abs(self._dq_buf[j]) * max_scale, self._hardware_v_max[j]
                )
                # Ensure non-zero minimum to avoid Ruckig issues
                self._max_vel_buf[j] = max(q_dot_max, 1e-6)

            self.inp.max_velocity = self._max_vel_buf
        else:
            # Near-zero motion, use hardware limits (reuse buffer)
            self._max_vel_buf[:] = self._hardware_v_max
            self.inp.max_velocity = self._max_vel_buf

    def tick(self) -> tuple[list[float], list[float], bool]:
        """
        Execute one control cycle.

        Warning: Returned lists are reused across calls. Copy if needed across ticks.

        Returns:
            Tuple of (position, velocity, finished):
            - position: Current commanded position in radians
            - velocity: Current commanded velocity in rad/s
            - finished: True if target reached (position mode) or velocity reached (velocity mode)
        """
        with self._lock:
            if not self.active:
                # Sync _pos_out with current position for inactive state
                self._pos_out[:] = self.inp.current_position
                return self._pos_out, self._zeros, True

            result, pos, vel = self._tick_ruckig()

            if result in _RUCKIG_ERRORS:
                return self._pos_out, self._zeros, True

            # pos/vel are pre-allocated buffers from _tick_ruckig
            return pos, vel, result == Result.Finished

    def reset_limits(self) -> None:
        """Reset velocity, acceleration, and jerk limits to hardware defaults."""
        with self._lock:
            self._vel_scale = 1.0
            self._acc_scale = 1.0
            self._apply_limits()

    def reset(self) -> None:
        """Reset executor state."""
        with self._lock:
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
    - Velocity mode for CARTJOG (1-DOF jogging)
    - WRF/TRF frame support for jogging
    """

    def __init__(self, dt: float = 0.004):
        """
        Initialize Cartesian streaming executor.

        Args:
            dt: Control cycle time in seconds (default 0.004 = 250Hz)
        """
        # Reference pose for tangent space computations
        # Must be set before super().__init__ calls _init_limits/_init_state
        self.reference_pose: np.ndarray | None = None

        # Pre-allocated arrays for Ruckig parameters (reused to avoid per-tick allocations)
        # Ruckig copies values on assignment, so we update in-place then assign same array
        # Must be created before super().__init__() since _apply_limits() is called during init
        self._max_velocity_arr = np.zeros(6, dtype=np.float64)
        self._max_acceleration_arr = np.zeros(6, dtype=np.float64)
        self._max_jerk_arr = np.zeros(6, dtype=np.float64)
        self._target_velocity_arr = np.zeros(6, dtype=np.float64)
        self._target_acceleration_arr = np.zeros(6, dtype=np.float64)

        super().__init__(num_dofs=6, dt=dt)  # 6-DOF: [x, y, z, wx, wy, wz]

        # Pre-allocated numpy arrays for hot path (avoids allocations per tick)
        self._tangent_buf = np.zeros(6, dtype=np.float64)
        self._vel_np_buf = np.zeros(6, dtype=np.float64)
        self._world_vel_buf = np.zeros(6, dtype=np.float64)

        # SE3 workspace buffers for JIT functions (avoids allocations in pose conversions)
        self._ref_inv_buf = np.zeros((4, 4), dtype=np.float64)
        self._delta_buf = np.zeros((4, 4), dtype=np.float64)
        self._result_pose_buf = np.zeros((4, 4), dtype=np.float64)
        # Additional workspace for se3_log_ws/se3_exp_ws (zero internal allocation)
        self._omega_ws = np.zeros(3, dtype=np.float64)
        self._R_ws = np.zeros((3, 3), dtype=np.float64)
        self._V_ws = np.zeros((3, 3), dtype=np.float64)  # Reused for V and V_inv

    def _init_limits(self) -> None:
        """Initialize Cartesian velocity/acceleration/jerk limits from centralized config."""
        # Linear limits (SI: m/s, m/s², m/s³)
        self._v_lin_max = LIMITS.cart.hard.velocity.linear
        self._a_lin_max = LIMITS.cart.hard.acceleration.linear
        self._j_lin_max = LIMITS.cart.hard.jerk.linear
        # Angular limits (SI: rad/s, rad/s², rad/s³)
        self._v_ang_max = LIMITS.cart.hard.velocity.angular
        self._a_ang_max = LIMITS.cart.hard.acceleration.angular
        self._j_ang_max = LIMITS.cart.hard.jerk.angular

    def _init_state(self) -> None:
        """Initialize Ruckig input parameters."""
        self.inp.current_position = [0.0] * 6
        self.inp.current_velocity = [0.0] * 6
        self.inp.current_acceleration = [0.0] * 6
        self.inp.target_position = [0.0] * 6
        self.inp.target_velocity = [0.0] * 6
        self.inp.target_acceleration = [0.0] * 6
        self._apply_limits()

    def _apply_limits(self) -> None:
        """Apply current limits (with scaling) to Ruckig parameters.

        Uses pre-allocated numpy arrays to avoid per-tick allocations.
        """
        # Update velocity limits in-place
        self._max_velocity_arr[:3] = self._v_lin_max * self._vel_scale
        self._max_velocity_arr[3:] = self._v_ang_max * self._vel_scale
        self.inp.max_velocity = self._max_velocity_arr

        # Update acceleration limits in-place
        self._max_acceleration_arr[:3] = self._a_lin_max * self._acc_scale
        self._max_acceleration_arr[3:] = self._a_ang_max * self._acc_scale
        self.inp.max_acceleration = self._max_acceleration_arr

        # Update jerk limits in-place
        self._max_jerk_arr[:3] = self._j_lin_max
        self._max_jerk_arr[3:] = self._j_ang_max
        self.inp.max_jerk = self._max_jerk_arr

    def sync_pose(self, current_pose: np.ndarray) -> None:
        """
        Sync current pose from robot feedback.

        Call this when idle to ensure executor starts from actual robot pose.
        Sets the reference pose for tangent space computations.

        Args:
            current_pose: Current TCP pose as 4x4 SE3 matrix
        """
        with self._lock:
            self.reference_pose = (
                current_pose.copy()
            )  # Copy to avoid aliasing with cached FK
            # Reset Ruckig state to origin (relative to reference)
            self.inp.current_position = [0.0] * 6
            self.inp.current_velocity = [0.0] * 6
            self.inp.current_acceleration = [0.0] * 6
            self.inp.target_position = [0.0] * 6
            self.active = False

    def _pose_to_tangent(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert SE3 pose to 6D tangent vector relative to reference.

        The tangent vector is the Lie algebra representation (twist):
        [vx, vy, vz, wx, wy, wz] where v is linear and w is angular.

        Args:
            pose: 4x4 SE3 matrix to convert

        Returns:
            Pre-allocated 6D buffer (reused across calls; Ruckig copies on assignment)
        """
        if self.reference_pose is None:
            self._tangent_buf.fill(0.0)
            return self._tangent_buf
        # Use JIT function with pre-allocated workspace buffers (zero allocation)
        _pose_to_tangent_jit(
            self.reference_pose,
            pose,
            self._ref_inv_buf,
            self._delta_buf,
            self._tangent_buf,
            self._omega_ws,
            self._R_ws,
            self._V_ws,
        )
        return self._tangent_buf

    def _tangent_to_pose(self, tangent: list[float]) -> np.ndarray:
        """
        Convert 6D tangent vector back to SE3 pose.

        Args:
            tangent: 6D tangent vector [x, y, z, wx, wy, wz]

        Returns:
            4x4 SE3 matrix
        """
        if self.reference_pose is None:
            return np.eye(4, dtype=np.float64)
        # Copy tangent to buffer and use JIT function with pre-allocated workspace
        self._tangent_buf[:] = tangent
        _tangent_to_pose_jit(
            self.reference_pose,
            self._tangent_buf,
            self._delta_buf,
            self._result_pose_buf,
            self._omega_ws,
            self._R_ws,
            self._V_ws,
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
        with self._lock:
            target_tangent = self._pose_to_tangent(target_pose)

            self.inp.control_interface = ControlInterface.Position
            self.inp.target_position = target_tangent
            self.inp.target_velocity = self._zeros  # Stop at target

            self._apply_limits()
            self.active = True

    def set_jog_velocity_1dof(
        self, axis: int, velocity: float, is_rotation: bool
    ) -> None:
        """
        Set 1-DOF jog velocity in body frame (TRF - Tool Reference Frame).

        The tangent space is relative to reference_pose, so velocities are
        naturally in body/tool frame. Use this for TRF jogging.

        Uses velocity mode - Ruckig smoothly accelerates/decelerates
        to reach target velocity. Call with velocity=0 to stop.

        Args:
            axis: Axis index (0=X, 1=Y, 2=Z)
            velocity: Target velocity (m/s for linear, rad/s for rotation)
            is_rotation: True for rotation axes (RX, RY, RZ)
        """
        with self._lock:
            # Update target velocity in-place (zero allocation)
            self._target_velocity_arr.fill(0.0)
            if is_rotation:
                self._target_velocity_arr[3 + axis] = velocity
            else:
                self._target_velocity_arr[axis] = velocity

            self.inp.control_interface = ControlInterface.Velocity
            self.inp.target_velocity = self._target_velocity_arr
            self._target_acceleration_arr.fill(0.0)
            self.inp.target_acceleration = self._target_acceleration_arr

            self._apply_limits()
            self.active = True

    def set_jog_velocity_1dof_wrf(
        self,
        axis: int,
        velocity: float,
        is_rotation: bool,
    ) -> None:
        """
        Set 1-DOF jog velocity in world reference frame (WRF).

        Transforms the velocity from world frame to body frame (tangent space)
        before applying to Ruckig. Requires reference_pose to be set.

        Args:
            axis: Axis index (0=X, 1=Y, 2=Z)
            velocity: Target velocity (m/s for linear, rad/s for rotation)
            is_rotation: True for rotation axes (RX, RY, RZ)
        """
        with self._lock:
            if self.reference_pose is None:
                logger.warning(
                    "set_jog_velocity_1dof_wrf called without reference_pose"
                )
                return

            # Reuse pre-allocated buffer for world velocity
            self._world_vel_buf.fill(0.0)
            if is_rotation:
                self._world_vel_buf[3 + axis] = velocity
            else:
                self._world_vel_buf[axis] = velocity

            # Transform from world frame to body frame (tangent space)
            # Body velocity = R^T @ world velocity
            R = self.reference_pose[:3, :3]

            # JIT-compiled transform into target buffer (zero allocation)
            np.dot(R.T, self._world_vel_buf[:3], self._target_velocity_arr[:3])
            np.dot(R.T, self._world_vel_buf[3:], self._target_velocity_arr[3:])

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
        with self._lock:
            if not self.active or self.reference_pose is None:
                self._vel_np_buf.fill(0.0)
                return (
                    self.reference_pose
                    if self.reference_pose is not None
                    else np.eye(4, dtype=np.float64),
                    self._vel_np_buf,
                    True,
                )

            result, pos, vel = self._tick_ruckig()

            if result in _RUCKIG_ERRORS:
                self._vel_np_buf.fill(0.0)
                return (
                    self.reference_pose
                    if self.reference_pose is not None
                    else np.eye(4, dtype=np.float64),
                    self._vel_np_buf,
                    True,
                )

            # Convert tangent back to pose
            smoothed_pose = self._tangent_to_pose(pos)
            # Copy velocity into pre-allocated buffer
            self._vel_np_buf[:] = vel

            # Don't auto-deactivate in velocity mode - caller controls via set_jog_velocity(0)
            return smoothed_pose, self._vel_np_buf, result == Result.Finished

    def reset(self) -> None:
        """Reset executor state."""
        with self._lock:
            self._vel_scale = 1.0
            self._acc_scale = 1.0
            self.reference_pose = None
            self.active = False
            self._init_state()
