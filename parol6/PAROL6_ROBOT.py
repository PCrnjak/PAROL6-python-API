"""PAROL6 robot kinematics, limits, and configuration."""

import atexit
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pinokin import Robot

from parol6.tools import get_tool_transform

logger = logging.getLogger(__name__)

# -----------------------------
# Typing aliases
# -----------------------------
Vec6f = NDArray[np.float64]
Vec6i = NDArray[np.int32]
Limits2f = NDArray[np.float64]  # shape (6,2)

# -----------------------------
# Kinematics and conversion constants
# -----------------------------
Microstep = 32
steps_per_revolution = 200

# Conversion constants
degree_per_step_constant: float = 360.0 / (Microstep * steps_per_revolution)
radian_per_step_constant: float = (2.0 * np.pi) / (Microstep * steps_per_revolution)
radian_per_sec_2_deg_per_sec_const: float = 360.0 / (2.0 * np.pi)
deg_per_sec_2_radian_per_sec_const: float = (2.0 * np.pi) / 360.0

# -----------------------------
# Joint limits
# -----------------------------
# Limits (deg) you get after homing and moving to extremes
_joint_limits_degree: Limits2f = np.array(
    [
        [-123.046875, 123.046875],
        [-145.0088, -3.375],
        [107.866, 287.8675],
        [-105.46975, 105.46975],
        [-90.0, 90.0],
        [0.0, 360.0],
    ],
    dtype=np.float64,
)

_joint_limits_radian: Limits2f = np.deg2rad(_joint_limits_degree)


# URDF path for pinokin Robot
_urdf_path = str(
    Path(__file__).resolve().parent / "urdf_model" / "urdf" / "PAROL6.urdf"
)

# Current robot instance (tool transform applied in-place)
robot: Robot = Robot(_urdf_path)


def apply_tool(
    tool_name: str,
    variant_key: str = "",
    tcp_offset_m: tuple[float, float, float] | None = None,
) -> None:
    """
    Apply tool transform to the robot model.

    Parameters
    ----------
    tool_name : str
        Name of the tool from the tool registry
    variant_key : str
        Optional variant key for the tool
    tcp_offset_m : tuple, optional
        Additional (x, y, z) offset in meters, composed in the tool's local frame.
    """
    T_tool = get_tool_transform(tool_name, variant_key=variant_key or None)

    if tcp_offset_m is not None and any(v != 0 for v in tcp_offset_m):
        T_offset = np.eye(4, dtype=np.float64)
        T_offset[0, 3] = tcp_offset_m[0]
        T_offset[1, 3] = tcp_offset_m[1]
        T_offset[2, 3] = tcp_offset_m[2]
        T_tool = T_tool @ T_offset

    label = f"'{tool_name}:{variant_key}'" if variant_key else f"'{tool_name}'"
    if not np.allclose(T_tool, np.eye(4)):
        robot.set_tool_transform(T_tool)
        logger.info(f"Applied tool {label} to robot model")
    else:
        robot.clear_tool_transform()
        logger.info(f"Applied tool {label} (identity)")


# Initialize with no tool
apply_tool("NONE")


@atexit.register
def _cleanup_robot() -> None:
    global robot
    del robot


# -----------------------------
# Additional raw parameter arrays
# -----------------------------
# Reduction ratio per joint
_joint_ratio: NDArray[np.float64] = np.array(
    [6.4, 20.0, 20.0 * (38.0 / 42.0), 4.0, 4.0, 10.0], dtype=np.float64
)

# Joint speeds (steps/s)
_joint_max_speed_hw: Vec6i = np.array(
    [15000, 25000, 32000, 10000, 10000, 27000], dtype=np.int32
)
_joint_min_speed: Vec6i = np.array([100, 100, 100, 100, 100, 100], dtype=np.int32)

# Effective max speeds with scaling applied
_joint_max_speed: Vec6i = _joint_max_speed_hw.copy()

# Jog speeds (steps/s) - 80% of scaled max for safety margin during jogging
_joint_max_jog_speed: Vec6i = (_joint_max_speed * 0.8).astype(np.int32)
_joint_min_jog_speed: Vec6i = np.array([100, 100, 100, 100, 100, 100], dtype=np.int32)

# Joint accelerations (steps/s^2) per joint
# Derived: a_max = v_max * 3 (reach max speed in ~0.33s)
_joint_max_acc: Vec6i = (_joint_max_speed * 3).astype(np.int32)

# Maximum jerk limits (steps/s^3) per joint
# Derived: j_max = a_max * 10 (reach max accel in ~0.1s)
_joint_max_jerk: Vec6i = (_joint_max_acc * 10).astype(np.int32)

# Compute joint angular velocities/accelerations in rad/s
_joint_speed_rad = (
    _joint_max_speed.astype(float) * radian_per_step_constant / _joint_ratio
)
_joint_acc_rad = _joint_max_acc.astype(float) * radian_per_step_constant / _joint_ratio
_joint_jerk_rad = (
    _joint_max_jerk.astype(float) * radian_per_step_constant / _joint_ratio
)


# Pre-computed Cartesian limits from Jacobian pseudoinverse workspace sampling.
# Derived from _compute_tcp_velocity_at_config() over 500/200/200 random configs
# with seeds 42/43/44, using median velocity and mean angular rates from wrist joints.
# Values are floored to reasonable precision to avoid false precision.
#
# Linear units: mm/s, mm/s^2, mm/s^3
# Angular units: deg/s, deg/s^2, deg/s^3
_cart_linear_velocity_max: float = 200
_cart_angular_velocity_max: float = 100
_cart_linear_acc_max: float = 550
_cart_angular_acc_max: float = 275
_cart_linear_jerk_max: float = 5500
_cart_angular_jerk_max: float = 2750

# Min values as 1% of max
_cart_linear_velocity_min: float = _cart_linear_velocity_max * 0.01
_cart_angular_velocity_min: float = _cart_angular_velocity_max * 0.01
_cart_linear_acc_min: float = _cart_linear_acc_max * 0.01
_cart_angular_acc_min: float = _cart_angular_acc_max * 0.01
_cart_linear_jerk_min: float = _cart_linear_jerk_max * 0.01
_cart_angular_jerk_min: float = _cart_angular_jerk_max * 0.01

# Jog limits (80% of max for safety margin)
_cart_linear_velocity_max_JOG: float = _cart_linear_velocity_max * 0.8
_cart_linear_velocity_min_JOG: float = _cart_linear_velocity_min


def log_derived_limits() -> None:
    """Log the derived Cartesian limits. Call at controller startup."""
    logger.info("=== Derived Kinematic Limits ===")
    logger.info("Joint velocity (rad/s): %s", np.round(_joint_speed_rad, 3))
    logger.info("Joint accel (rad/s²): %s", np.round(_joint_acc_rad, 2))
    logger.info("Joint jerk (rad/s³): %s", np.round(_joint_jerk_rad, 1))
    logger.info(
        "Cartesian linear velocity: %.1f mm/s (jog: %.1f mm/s)",
        _cart_linear_velocity_max,
        _cart_linear_velocity_max_JOG,
    )
    logger.info("Cartesian angular velocity: %.2f deg/s", _cart_angular_velocity_max)
    logger.info(
        "Cartesian linear accel: %.1f mm/s², angular: %.2f deg/s²",
        _cart_linear_acc_max,
        _cart_angular_acc_max,
    )
    logger.info(
        "Cartesian linear jerk: %.1f mm/s³, angular: %.2f deg/s³",
        _cart_linear_jerk_max,
        _cart_angular_jerk_max,
    )
    logger.info("================================")


# Standby positions
_standby_deg: Vec6f = np.array([90.0, -90.0, 180.0, 0.0, 0.0, 180.0], dtype=np.float64)


# -----------------------------
# Typed hierarchical API
# -----------------------------
@dataclass(frozen=True)
class Joint:
    """Minimal joint configuration - all values in native units (deg for position, steps/s for speed)."""

    limits_deg: Limits2f  # Position limits in degrees [6, 2]
    speed_max: Vec6i  # Max speed in steps/s
    speed_min: Vec6i  # Min speed in steps/s
    jog_speed_max: Vec6i  # Max jog speed in steps/s
    jog_speed_min: Vec6i  # Min jog speed in steps/s
    acc_max: Vec6i  # Max acceleration in steps/s²
    jerk_max: Vec6i  # Max jerk in steps/s³
    ratio: Vec6f  # Gear ratio per joint
    standby_deg: Vec6f  # Standby position in degrees


@dataclass(frozen=True)
class RangeF:
    min: float
    max: float


@dataclass(frozen=True)
class CartVel:
    linear: RangeF
    jog: RangeF
    angular: RangeF


@dataclass(frozen=True)
class CartAcc:
    linear: RangeF
    angular: RangeF


@dataclass(frozen=True)
class CartJerk:
    linear: RangeF
    angular: RangeF


@dataclass(frozen=True)
class Cart:
    vel: CartVel
    acc: CartAcc
    jerk: CartJerk


@dataclass(frozen=True)
class Conv:
    degree_per_step: float
    radian_per_step: float
    rad_sec_to_deg_sec: float
    deg_sec_to_rad_sec: float


joint: Final[Joint] = Joint(
    limits_deg=_joint_limits_degree,
    speed_max=_joint_max_speed,
    speed_min=_joint_min_speed,
    jog_speed_max=_joint_max_jog_speed,
    jog_speed_min=_joint_min_jog_speed,
    acc_max=_joint_max_acc,
    jerk_max=_joint_max_jerk,
    ratio=_joint_ratio,
    standby_deg=_standby_deg,
)

cart: Final[Cart] = Cart(
    vel=CartVel(
        linear=RangeF(min=_cart_linear_velocity_min, max=_cart_linear_velocity_max),
        jog=RangeF(
            min=_cart_linear_velocity_min_JOG, max=_cart_linear_velocity_max_JOG
        ),
        angular=RangeF(min=_cart_angular_velocity_min, max=_cart_angular_velocity_max),
    ),
    acc=CartAcc(
        linear=RangeF(min=_cart_linear_acc_min, max=_cart_linear_acc_max),
        angular=RangeF(min=_cart_angular_acc_min, max=_cart_angular_acc_max),
    ),
    jerk=CartJerk(
        linear=RangeF(min=_cart_linear_jerk_min, max=_cart_linear_jerk_max),
        angular=RangeF(min=_cart_angular_jerk_min, max=_cart_angular_jerk_max),
    ),
)

conv: Final[Conv] = Conv(
    degree_per_step=degree_per_step_constant,
    radian_per_step=radian_per_step_constant,
    rad_sec_to_deg_sec=radian_per_sec_2_deg_per_sec_const,
    deg_sec_to_rad_sec=deg_per_sec_2_radian_per_sec_const,
)


# -----------------------------
# CAN helpers and bitfield utils (used by transports/gripper)
# -----------------------------
def extract_from_can_id(can_id: int) -> tuple[int, int, int]:
    id2 = (can_id >> 7) & 0xF
    can_command = (can_id >> 1) & 0x3F
    error_bit = can_id & 0x1
    return id2, can_command, error_bit


def combine_2_can_id(id2: int, can_command: int, error_bit: int) -> int:
    can_id = 0
    can_id |= (id2 & 0xF) << 7
    can_id |= (can_command & 0x3F) << 1
    can_id |= error_bit & 0x1
    return can_id


def fuse_bitfield_2_bytearray(var_in: list[int] | tuple[int, ...]) -> bytes:
    number = 0
    for b in var_in:
        number = (2 * number) + int(b)
    return bytes([number])


def split_2_bitfield(var_in: int) -> list[int]:
    return [(var_in >> i) & 1 for i in range(7, -1, -1)]


if __name__ == "__main__":
    # Recalculate Cartesian limits from current joint parameters.
    # Run: python -m parol6.PAROL6_ROBOT
    #
    # Uses Jacobian pseudoinverse workspace sampling to derive achievable
    # TCP velocity/acceleration/jerk while maintaining tool orientation.
    # Copy the printed values into the pre-computed constants above.

    from parol6.config import steps_to_rad

    def _compute_tcp_velocity_at_config(
        q: NDArray, direction: int, v_max_joint: NDArray
    ) -> float | None:
        """Max TCP velocity in one Cartesian direction.

        For linear directions (0-2), rejects samples that cause orientation change.
        For angular directions (3-5), rejects samples that cause linear translation.
        """
        try:
            J = robot.jacob0(q)
            if np.linalg.cond(J) > 1e6:
                return None
            desired = np.zeros(6)
            desired[direction] = 1.0
            q_dot = np.linalg.pinv(J) @ desired
            if direction < 3:
                if np.linalg.norm(J[3:, :] @ q_dot) > 0.01:
                    return None
            else:
                if np.linalg.norm(J[:3, :] @ q_dot) > 0.01:
                    return None
            return float(np.min(v_max_joint / (np.abs(q_dot) + 1e-10)))
        except (np.linalg.LinAlgError, ValueError):
            return None

    _home_rad = np.deg2rad(_standby_deg)

    def _sample_limit(
        n_samples: int, seed: int, v_max: NDArray, spread_deg: float = 30.0
    ) -> tuple[float, float]:
        """Sample around home position and return (median_linear_m, median_angular_rad).

        Samples joint configurations from a Gaussian centered on home with
        std dev of ``spread_deg`` degrees, clamped to joint limits.
        """
        rng = np.random.default_rng(seed)
        spread_rad = np.deg2rad(spread_deg)
        lin_results = []
        ang_results = []
        for _ in range(n_samples):
            q = _home_rad + rng.normal(0, spread_rad, size=6)
            q = np.clip(q, _joint_limits_radian[:, 0], _joint_limits_radian[:, 1])
            for d in range(3):
                v = _compute_tcp_velocity_at_config(q, d, v_max)
                if v is not None and v > 0.001:
                    lin_results.append(v)
            for d in range(3, 6):
                v = _compute_tcp_velocity_at_config(q, d, v_max)
                if v is not None and v > 0.001:
                    ang_results.append(v)
        linear = float(np.median(lin_results)) if lin_results else 0.1
        angular = float(np.median(ang_results)) if ang_results else 0.1
        return linear, angular

    vel_lin, vel_ang = _sample_limit(500, 42, _joint_speed_rad)
    acc_lin, acc_ang = _sample_limit(200, 43, _joint_acc_rad)
    jerk_lin, jerk_ang = _sample_limit(200, 44, _joint_jerk_rad)

    print("=== Recalculated Cartesian Limits ===")
    print(f"_cart_linear_velocity_max: float = {vel_lin * 1000:.0f}")
    print(f"_cart_angular_velocity_max: float = {np.degrees(vel_ang):.0f}")
    print(f"_cart_linear_acc_max: float = {acc_lin * 1000:.0f}")
    print(f"_cart_angular_acc_max: float = {np.degrees(acc_ang):.0f}")
    print(f"_cart_linear_jerk_max: float = {jerk_lin * 1000:.0f}")
    print(f"_cart_angular_jerk_max: float = {np.degrees(jerk_ang):.0f}")

    print("\n=== Joint Info ===")
    j_step_rad = np.zeros(6, dtype=np.float64)
    steps_to_rad(np.array([1, 1, 1, 1, 1, 1], dtype=np.int32), j_step_rad)
    print("Smallest step (deg):", np.rad2deg(j_step_rad))
    print("Standby deg:", joint.standby_deg)
