"""
Quintic polynomial primitives and multi-axis quintic trajectory.
"""
from typing import Dict, Optional, List

import numpy as np
from .motion_constraints import MotionConstraints


class QuinticPolynomial:
    """
    Single-axis quintic polynomial trajectory primitive.

    Provides C² continuous trajectories (continuous position, velocity, acceleration)
    with zero jerk at boundaries. This is the building block for S-curve profiles
    and advanced multi-segment trajectories.
    """

    def __init__(
        self,
        q0: float,
        qf: float,
        v0: float = 0,
        vf: float = 0,
        a0: float = 0,
        af: float = 0,
        T: float = 1.0,
    ):
        """
        Generate quintic polynomial trajectory.

        Args:
            q0: Initial position
            qf: Final position
            v0: Initial velocity (default 0)
            vf: Final velocity (default 0)
            a0: Initial acceleration (default 0)
            af: Final acceleration (default 0)
            T: Duration of trajectory (must be > 0)
        """
        if T <= 0:
            raise ValueError(f"Duration must be positive, got T={T}")

        self.T = T
        self.q0 = q0
        self.qf = qf

        # Store boundary conditions
        self.boundary_conditions = {"q0": q0, "qf": qf, "v0": v0, "vf": vf, "a0": a0, "af": af}

        # Solve for polynomial coefficients using analytical method
        self.coeffs = self._solve_coefficients_analytical(q0, qf, v0, vf, a0, af, T)

        # Pre-compute coefficient derivatives for faster evaluation
        self._prepare_derivative_coeffs()

    def _solve_coefficients_analytical(self, q0, qf, v0, vf, a0, af, T):
        """
        Analytical solution for quintic polynomial coefficients.

        Uses closed-form solution to avoid numerical issues with matrix inversion.
        Works in normalized time [0,1] then scales back to actual time.

        The quintic polynomial is: q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵

        Returns:
            numpy array of coefficients [a0, a1, a2, a3, a4, a5]
        """
        # Scale boundary conditions for normalized time τ = t/T
        v0_norm = v0 * T
        vf_norm = vf * T
        a0_norm = a0 * T**2
        af_norm = af * T**2

        # Coefficients in normalized time domain
        a0_ = q0
        a1_ = v0_norm
        a2_ = a0_norm / 2.0

        a3_ = 10 * (qf - q0) - 6 * v0_norm - 4 * vf_norm - (3 * a0_norm - af_norm) / 2.0
        a4_ = -15 * (qf - q0) + 8 * v0_norm + 7 * vf_norm + (3 * a0_norm - 2 * af_norm) / 2.0
        a5_ = 6 * (qf - q0) - 3 * (v0_norm + vf_norm) - (a0_norm - af_norm) / 2.0

        # Convert back to actual time domain
        coeffs = np.array([a0_, a1_ / T, a2_ / T**2, a3_ / T**3, a4_ / T**4, a5_ / T**5])
        return coeffs

    def _prepare_derivative_coeffs(self):
        """Pre-compute coefficients for velocity, acceleration, and jerk."""
        self.vel_coeffs = np.array(
            [self.coeffs[1], 2 * self.coeffs[2], 3 * self.coeffs[3], 4 * self.coeffs[4], 5 * self.coeffs[5]]
        )
        self.acc_coeffs = np.array([2 * self.coeffs[2], 6 * self.coeffs[3], 12 * self.coeffs[4], 20 * self.coeffs[5]])
        self.jerk_coeffs = np.array([6 * self.coeffs[3], 24 * self.coeffs[4], 60 * self.coeffs[5]])

    def position(self, t: float) -> float:
        """Evaluate position at time t using Horner's method."""
        if t < 0:
            return self.q0
        if t > self.T:
            return self.qf
        result = self.coeffs[5]
        for i in range(4, -1, -1):
            result = result * t + self.coeffs[i]
        return result

    def velocity(self, t: float) -> float:
        """Evaluate velocity at time t using Horner's method."""
        if t < 0:
            return self.boundary_conditions["v0"]
        if t > self.T:
            return self.boundary_conditions["vf"]
        result = self.vel_coeffs[4] if len(self.vel_coeffs) > 4 else 0
        for i in range(min(3, len(self.vel_coeffs) - 1), -1, -1):
            result = result * t + self.vel_coeffs[i]
        return result

    def acceleration(self, t: float) -> float:
        """Evaluate acceleration at time t using Horner's method."""
        if t < 0:
            return self.boundary_conditions["a0"]
        if t > self.T:
            return self.boundary_conditions["af"]
        result = self.acc_coeffs[3] if len(self.acc_coeffs) > 3 else 0
        for i in range(min(2, len(self.acc_coeffs) - 1), -1, -1):
            result = result * t + self.acc_coeffs[i]
        return result

    def jerk(self, t: float) -> float:
        """Evaluate jerk at time t using Horner's method."""
        if t < 0 or t > self.T:
            return 0
        result = self.jerk_coeffs[2] if len(self.jerk_coeffs) > 2 else 0
        for i in range(min(1, len(self.jerk_coeffs) - 1), -1, -1):
            result = result * t + self.jerk_coeffs[i]
        return result

    def evaluate(self, t: float, derivative: int = 0) -> float:
        """
        Unified evaluation function for any derivative order.

        Args:
            t: Time point to evaluate
            derivative: 0=position, 1=velocity, 2=acceleration, 3=jerk
        """
        if derivative == 0:
            return self.position(t)
        if derivative == 1:
            return self.velocity(t)
        if derivative == 2:
            return self.acceleration(t)
        if derivative == 3:
            return self.jerk(t)
        raise ValueError(f"Derivative order {derivative} not supported (max is 3)")

    def get_trajectory_points(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Generate trajectory points at specified time interval.

        Args:
            dt: Time step (default 0.01 for 100Hz)
        """
        time_points = np.arange(0, self.T + dt, dt)
        trajectory = {
            "time": time_points,
            "position": np.array([self.position(t) for t in time_points]),
            "velocity": np.array([self.velocity(t) for t in time_points]),
            "acceleration": np.array([self.acceleration(t) for t in time_points]),
            "jerk": np.array([self.jerk(t) for t in time_points]),
        }
        return trajectory

    def validate_continuity(self, tolerance: float = 1e-10) -> Dict[str, bool]:
        """
        Validate that boundary conditions are satisfied.
        """
        validation = {
            "q0": abs(self.position(0) - self.boundary_conditions["q0"]) < tolerance,
            "qf": abs(self.position(self.T) - self.boundary_conditions["qf"]) < tolerance,
            "v0": abs(self.velocity(0) - self.boundary_conditions["v0"]) < tolerance,
            "vf": abs(self.velocity(self.T) - self.boundary_conditions["vf"]) < tolerance,
            "a0": abs(self.acceleration(0) - self.boundary_conditions["a0"]) < tolerance,
            "af": abs(self.acceleration(self.T) - self.boundary_conditions["af"]) < tolerance,
        }
        return validation

    def validate_numerical_stability(self) -> Dict[str, object]:
        """
        Check for potential numerical stability issues.
        """
        stability: Dict[str, object] = {"is_stable": True, "warnings": [], "metrics": {}}

        # Check condition number (ratio of time to distance)
        distance = abs(self.qf - self.q0)
        if distance > 1e-6:
            time_distance_ratio = self.T / distance
            stability["metrics"]["time_distance_ratio"] = time_distance_ratio
            if time_distance_ratio > 100:
                stability["is_stable"] = False
                stability["warnings"].append(f"Poor conditioning: T/d ratio = {time_distance_ratio:.1f}")

        # Check coefficient magnitudes
        coeff_magnitudes = [abs(c) for c in self.coeffs]
        max_coeff = max(coeff_magnitudes)
        nz = [m for m in coeff_magnitudes if m > 1e-10]
        min_nonzero_coeff = min(nz) if nz else 0.0

        if min_nonzero_coeff > 0:
            coeff_ratio = max_coeff / min_nonzero_coeff
            stability["metrics"]["coefficient_ratio"] = coeff_ratio
            if coeff_ratio > 1e6:
                stability["warnings"].append(f"Large coefficient ratio: {coeff_ratio:.2e}")

        if self.T < 0.001:
            stability["warnings"].append(f"Very small duration T={self.T} may cause numerical issues")

        max_jerk = max(abs(self.jerk(t)) for t in np.linspace(0, self.T, 10))
        if max_jerk > 1e6:
            stability["warnings"].append(f"Very large jerk values: {max_jerk:.2e}")

        return stability


class MultiAxisQuinticTrajectory:
    """
    Multi-axis synchronized quintic trajectory generator.

    Ensures all axes complete their motion simultaneously using a
    time-scaling approach.
    """

    def __init__(
        self,
        q0: List[float],
        qf: List[float],
        v0: Optional[List[float]] = None,
        vf: Optional[List[float]] = None,
        a0: Optional[List[float]] = None,
        af: Optional[List[float]] = None,
        T: Optional[float] = None,
        constraints: Optional["MotionConstraints"] = None,
    ):
        """
        Generate synchronized quintic trajectories for multiple axes.

        Args:
            q0: Initial positions for each axis
            qf: Final positions for each axis
            v0: Initial velocities (default zeros)
            vf: Final velocities (default zeros)
            a0: Initial accelerations (default zeros)
            af: Final accelerations (default zeros)
            T: Duration (if None, calculated from constraints)
            constraints: Motion constraints for time calculation
        """
        self.num_axes = len(q0)

        # Default boundary conditions
        v0 = v0 if v0 is not None else [0] * self.num_axes
        vf = vf if vf is not None else [0] * self.num_axes
        a0 = a0 if a0 is not None else [0] * self.num_axes
        af = af if af is not None else [0] * self.num_axes

        # Calculate minimum time if not specified
        if T is None:
            T = self._calculate_minimum_time(q0, qf, v0, vf, constraints)

        self.T = T

        # Generate quintic polynomial for each axis
        self.axis_trajectories: List[QuinticPolynomial] = []
        for i in range(self.num_axes):
            quintic = QuinticPolynomial(q0[i], qf[i], v0[i], vf[i], a0[i], af[i], T)
            self.axis_trajectories.append(quintic)

    def _calculate_minimum_time(self, q0, qf, v0, vf, constraints: Optional["MotionConstraints"]) -> float:
        """
        Calculate minimum time based on velocity and acceleration constraints.
        """
        if constraints is None:
            # Default time based on distance
            max_distance = max(abs(qf[i] - q0[i]) for i in range(self.num_axes))
            return max(2.0, max_distance / 50.0)  # Assume 50 units/s default

        min_times: List[float] = []
        for i in range(self.num_axes):
            distance = abs(qf[i] - q0[i])

            # Time based on velocity limit
            if constraints.max_velocity and i < len(constraints.max_velocity):
                t_vel = distance / constraints.max_velocity[i]
                min_times.append(t_vel)

            # Time based on acceleration limit (triangular profile)
            if constraints.max_acceleration and i < len(constraints.max_acceleration):
                t_acc = 2 * np.sqrt(distance / constraints.max_acceleration[i])
                min_times.append(t_acc)

        # Use maximum time to ensure all constraints are satisfied
        return max(min_times) * 1.2 if min_times else 2.0

    def evaluate_all(self, t: float) -> Dict[str, List[float]]:
        """
        Evaluate all axes at time t.
        """
        result: Dict[str, List[float]] = {"position": [], "velocity": [], "acceleration": [], "jerk": []}
        for traj in self.axis_trajectories:
            result["position"].append(traj.position(t))
            result["velocity"].append(traj.velocity(t))
            result["acceleration"].append(traj.acceleration(t))
            result["jerk"].append(traj.jerk(t))
        return result

    def get_trajectory_points(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """Generate trajectory points for all axes."""
        time_points = np.arange(0, self.T + dt, dt)
        num_points = len(time_points)

        positions = np.zeros((num_points, self.num_axes))
        velocities = np.zeros((num_points, self.num_axes))
        accelerations = np.zeros((num_points, self.num_axes))
        jerks = np.zeros((num_points, self.num_axes))

        for i, t in enumerate(time_points):
            values = self.evaluate_all(t)
            positions[i] = values["position"]
            velocities[i] = values["velocity"]
            accelerations[i] = values["acceleration"]
            jerks[i] = values["jerk"]

        return {
            "time": time_points,
            "position": positions,
            "velocity": velocities,
            "acceleration": accelerations,
            "jerk": jerks,
        }
