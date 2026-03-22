"""
Central configuration for PAROL6 tunables and shared constants.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

TRACE: int = 5
logging.addLevelName(TRACE, "TRACE")

# Command queue limits
MAX_COMMAND_QUEUE_SIZE: int = 100
MAX_BLEND_LOOKAHEAD: int = int(os.getenv("PAROL6_MAX_BLEND_LOOKAHEAD", "100"))
MAX_POLL_COUNT: int = 25  # Max UDP messages to read per control tick

# Serial transport defaults
SERIAL_RX_RING_DEFAULT: int = 262144
# Add Logger.trace if missing
if not hasattr(logging.Logger, "trace"):

    def _trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    logging.Logger.trace = _trace  # type: ignore[attr-defined]
    logging.TRACE = TRACE  # type: ignore[attr-defined]

TRACE_ENABLED = str(os.getenv("PAROL_TRACE", "0")).lower() in ("1", "true", "yes", "on")

logger = logging.getLogger(__name__)

# Default control/sample rates (Hz)
CONTROL_RATE_HZ: float = float(os.getenv("PAROL6_CONTROL_RATE_HZ", "100"))

DEFAULT_ACCEL_PERCENT: float = 100.0

# Motion thresholds (mm)
NEAR_MM_TOL_MM: float = 2.0  # Proximity threshold for considering positions "near" (mm)
ENTRY_MM_TOL_MM: float = 5.0  # Entry trajectory threshold for smooth motion (mm)

# Trajectory path sampling (fixed samples for TOPP-RA input)
PATH_SAMPLES: int = int(os.getenv("PAROL6_PATH_SAMPLES", "50"))

# Centralized loop interval (seconds).
INTERVAL_S: float = max(1e-6, 1.0 / max(CONTROL_RATE_HZ, 1.0))

# Server/runtime defaults (overridable by env/CLI in headless commander)
SERVER_IP: str = "127.0.0.1"
SERVER_PORT: int = 5001
SERVER_STREAM_DEFAULT: bool = False
FAKE_SERIAL: bool = False
SERIAL_BAUD: int = 3_000_000
AUTO_HOME_DEFAULT: bool = True
LOG_LEVEL_DEFAULT: str = "INFO"

# COM port persistence file stored in user config directory by default (cross-platform).
_default_com_file = Path.home() / ".parol6" / "com_port.txt"
COM_PORT_FILE: str = os.getenv("PAROL6_COM_FILE", str(_default_com_file))

# Multicast/broadcast status configuration (all overridable via env)
# These defaults implement local-only multicast on loopback by default.
MCAST_GROUP: str = os.getenv("PAROL6_MCAST_GROUP", "239.255.0.101")
MCAST_PORT: int = int(os.getenv("PAROL6_MCAST_PORT", "50510"))
MCAST_TTL: int = int(os.getenv("PAROL6_MCAST_TTL", "1"))
MCAST_IF: str = os.getenv("PAROL6_MCAST_IF", "127.0.0.1")

# Transport selection for status updates. Default MULTICAST; set to UNICAST on CI if multicast is not available.
STATUS_TRANSPORT: str = (
    os.getenv("PAROL6_STATUS_TRANSPORT", "MULTICAST").strip().upper()
)
# Host to use for unicast fallback (defaults to loopback)
STATUS_UNICAST_HOST: str = os.getenv("PAROL6_STATUS_UNICAST_HOST", "127.0.0.1")

# Status update/broadcast rates
STATUS_RATE_HZ: float = float(os.getenv("PAROL6_STATUS_RATE_HZ", "50"))
STATUS_STALE_S: float = float(os.getenv("PAROL6_STATUS_STALE_S", "0.5"))

# Validate STATUS_RATE_HZ divides evenly into CONTROL_RATE_HZ for polling
if int(CONTROL_RATE_HZ) % int(STATUS_RATE_HZ) != 0:
    raise ValueError(
        f"STATUS_RATE_HZ ({STATUS_RATE_HZ}) must divide evenly into "
        f"CONTROL_RATE_HZ ({CONTROL_RATE_HZ})"
    )
STATUS_BROADCAST_INTERVAL: int = int(CONTROL_RATE_HZ) // int(STATUS_RATE_HZ)

# Max ticks to hold MOVE at trajectory endpoint waiting for Position_in to converge.
# At 100Hz control rate, 20 ticks = 200ms. If the robot hasn't reached the target
# by then, the segment completes anyway to avoid blocking the pipeline.
SETTLE_MAX_TICKS: int = int(os.getenv("PAROL6_SETTLE_MAX_TICKS", "20"))

# Loop timing tuning - busy threshold before deadline to switch from sleep to busy-wait
BUSY_THRESHOLD_MS: float = float(os.getenv("PAROL6_BUSY_THRESHOLD_MS", "1.0"))


# Ack/Tracking policy toggles
def _env_bool_optional(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


FORCE_ACK: bool | None = _env_bool_optional("PAROL6_FORCE_ACK")


def save_com_port(port: str) -> bool:
    """
    Save COM port to persistent file.

    Args:
        port: COM port string to save

    Returns:
        True if successful, False otherwise
    """
    try:
        com_port_path = Path(COM_PORT_FILE)
        com_port_path.parent.mkdir(parents=True, exist_ok=True)
        com_port_path.write_text(port.strip())
        logger.info(f"Saved COM port {port} to {COM_PORT_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save COM port: {e}")
        return False


def load_com_port() -> str | None:
    """
    Load saved COM port from file.

    Returns:
        COM port string if found, None otherwise
    """
    try:
        com_port_path = Path(COM_PORT_FILE)
        if com_port_path.exists():
            port = com_port_path.read_text().strip()
            if port:
                logger.info(f"Loaded COM port {port} from {COM_PORT_FILE}")
                return port
    except Exception as e:
        logger.error(f"Failed to load COM port: {e}")
    return None


def get_com_port_with_fallback() -> str:
    """
    Resolve COM port from environment or file.

    Priority:
      1) Environment variables: PAROL6_COM_PORT or PAROL6_SERIAL
      2) com_port.txt (if present and non-empty)

    Returns:
      Port string if available, otherwise an empty string "".
    """
    # 1) Environment variables
    env_port = os.getenv("PAROL6_COM_PORT") or os.getenv("PAROL6_SERIAL")
    if env_port and env_port.strip():
        port = env_port.strip()
        logger.info(f"Using COM port from environment: {port}")
        return port

    # 2) Persistence file
    saved_port = load_com_port()
    if saved_port:
        return saved_port

    return ""


import parol6.PAROL6_ROBOT as PAROL6_ROBOT  # noqa: E402 - must be after steps_to_rad() definition due to circular import

# Type alias for conversion function return types
IndexArg = Union[int, NDArray[np.int_], None]

# Import robot-specific constants
_degree_per_step = PAROL6_ROBOT.degree_per_step_constant
_radian_per_step = PAROL6_ROBOT.radian_per_step_constant
_joint_ratio = PAROL6_ROBOT.joint.ratio

# Standby/home position in degrees - pass-through from robot definition
STANDBY_ANGLES_DEG: list[float] = list(PAROL6_ROBOT.joint.standby_deg)
# Alias
HOME_ANGLES_DEG: list[float] = STANDBY_ANGLES_DEG


# JIT helper for rad_to_steps (needs wrapper because of thread-local scratch buffer)
@njit(cache=True)
def _rad_to_steps_jit(
    rad: NDArray[np.float64],
    out: NDArray[np.int32],
    scratch: NDArray[np.float64],
    radian_per_step_inv: float,
    joint_ratio: NDArray[np.float64],
) -> NDArray[np.int32]:
    np.multiply(rad, radian_per_step_inv, scratch)
    np.multiply(scratch, joint_ratio, scratch)
    np.rint(scratch, scratch)
    out[:] = scratch.astype(np.int32)
    return out


@njit(cache=True)
def deg_to_steps(deg: NDArray[np.float64], out: NDArray[np.int32]) -> NDArray[np.int32]:
    """Convert degrees to steps (gear ratio aware). Zero-allocation when out is provided."""
    for i in range(6):
        out[i] = np.int32(np.rint((deg[i] / _degree_per_step) * _joint_ratio[i]))
    return out


@njit(cache=True)
def deg_to_steps_scalar(deg: float, idx: int) -> np.int32:
    """Convert single degree value to steps."""
    return np.int32(np.rint((deg / _degree_per_step) * _joint_ratio[idx]))


@njit(cache=True)
def steps_to_deg(
    steps: NDArray[np.int32], out: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert steps to degrees (gear ratio aware). Zero-allocation."""
    np.multiply(steps, _degree_per_step, out)
    np.divide(out, _joint_ratio, out)
    return out


@njit(cache=True)
def steps_to_deg_scalar(steps: int, idx: int) -> np.float64:
    """Convert single steps value to degrees."""
    return np.float64((float(steps) * _degree_per_step) / _joint_ratio[idx])


# Thread-local scratch buffer for rad_to_steps intermediate calculation
_tls = threading.local()


def _get_scratch_f64() -> NDArray[np.float64]:
    """Get thread-local float64 scratch buffer (6 elements)."""
    buf = getattr(_tls, "scratch_f64", None)
    if buf is None:
        buf = np.zeros(6, dtype=np.float64)
        _tls.scratch_f64 = buf
    return buf


def rad_to_steps(
    rad: ArrayLike, out: NDArray[np.int32], idx: IndexArg = None
) -> NDArray[np.int32]:
    """Convert radians to steps. Zero-allocation (uses thread-local scratch)."""
    scratch = _get_scratch_f64()
    return _rad_to_steps_jit(
        np.asarray(rad, dtype=np.float64),
        out,
        scratch,
        1.0 / _radian_per_step,
        _joint_ratio,
    )


@njit(cache=True)
def rad_to_steps_scalar(rad: float, idx: int) -> np.int32:
    """Convert single radian value to steps."""
    return np.int32(np.rint((rad / _radian_per_step) * _joint_ratio[idx]))


@njit(cache=True)
def steps_to_rad(
    steps: NDArray[np.int32], out: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert steps to radians. Zero-allocation. JIT-compiled."""
    np.multiply(steps, _radian_per_step, out)
    np.divide(out, _joint_ratio, out)
    return out


@njit(cache=True)
def steps_to_rad_scalar(steps: int, idx: int) -> np.float64:
    """Convert single steps value to radians."""
    return np.float64((float(steps) * _radian_per_step) / _joint_ratio[idx])


@njit(cache=True)
def speed_steps_to_deg(
    sps: NDArray[np.int32], out: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert speed: steps/s to deg/s. Zero-allocation."""
    np.multiply(sps, _degree_per_step, out)
    np.divide(out, _joint_ratio, out)
    return out


@njit(cache=True)
def speed_steps_to_deg_scalar(sps: float, idx: int) -> np.float64:
    """Convert single speed value: steps/s to deg/s."""
    return np.float64((sps * _degree_per_step) / _joint_ratio[idx])


@njit(cache=True)
def speed_deg_to_steps(
    dps: NDArray[np.float64], out: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Convert speed: deg/s to steps/s. Zero-allocation."""
    for i in range(6):
        out[i] = np.int32((dps[i] / _degree_per_step) * _joint_ratio[i])
    return out


@njit(cache=True)
def speed_deg_to_steps_scalar(dps: float, idx: int) -> np.int32:
    """Convert single speed value: deg/s to steps/s."""
    return np.int32((dps / _degree_per_step) * _joint_ratio[idx])


@njit(cache=True)
def speed_steps_to_rad(
    sps: NDArray[np.int32], out: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convert speed: steps/s to rad/s. Zero-allocation."""
    np.multiply(sps, _radian_per_step, out)
    np.divide(out, _joint_ratio, out)
    return out


@njit(cache=True)
def speed_steps_to_rad_scalar(sps: float, idx: int) -> np.float64:
    """Convert single speed value: steps/s to rad/s."""
    return np.float64((sps * _radian_per_step) / _joint_ratio[idx])


@njit(cache=True)
def speed_rad_to_steps(
    rps: NDArray[np.float64], out: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Convert speed: rad/s to steps/s. Zero-allocation."""
    for i in range(6):
        out[i] = np.int32((rps[i] / _radian_per_step) * _joint_ratio[i])
    return out


@njit(cache=True)
def speed_rad_to_steps_scalar(rps: float, idx: int) -> np.int32:
    """Convert single speed value: rad/s to steps/s."""
    return np.int32((rps / _radian_per_step) * _joint_ratio[idx])


# -----------------------------------------------------------------------------
# Robot Limits - Unified SI-unit hierarchy
# -----------------------------------------------------------------------------
# All values in SI units: rad/s, rad/s², rad/s³ for joint; m/s, m/s², m/s³ for cart
#
# Usage:
#   LIMITS.joint.hard.velocity      → [6] joint velocity limits (rad/s)
#   LIMITS.joint.jog.velocity       → [6] jog velocity limits (rad/s)
#   LIMITS.joint.position.rad       → [6,2] position limits [min,max] (rad)
#   LIMITS.joint.position.rad[:, 0] → [6] min position limits (rad)
#   LIMITS.cart.hard.velocity.linear   → linear velocity limit (m/s)
#   LIMITS.cart.hard.velocity.angular  → angular velocity limit (rad/s)
#   LIMITS.cart.jog.velocity.linear    → jog linear velocity limit (m/s)


@dataclass(frozen=True, slots=True)
class Kinodynamic:
    """Joint kinodynamic limits (velocity, acceleration, jerk)."""

    # SI units for algorithms
    velocity: NDArray[np.float64]  # rad/s
    acceleration: NDArray[np.float64]  # rad/s²
    jerk: NDArray[np.float64]  # rad/s³
    # Step units for hardware/simulation
    velocity_steps: NDArray[np.int32]  # steps/s
    acceleration_steps: NDArray[np.int32]  # steps/s²
    jerk_steps: NDArray[np.int32]  # steps/s³


@dataclass(frozen=True, slots=True)
class JointPosition:
    """Joint position limits in various units."""

    deg: NDArray[np.float64]  # [6, 2] - [min, max] per joint
    rad: NDArray[np.float64]  # [6, 2]
    steps: NDArray[np.int32]  # [6, 2]


@dataclass(frozen=True, slots=True)
class JointLimits:
    """All joint limits."""

    hard: Kinodynamic  # Hardware limits
    jog: Kinodynamic  # Jog limits (reduced for safety)
    position: JointPosition


@dataclass(frozen=True, slots=True)
class LinearAngular:
    """Cartesian linear/angular component pair (SI units)."""

    linear: float  # m/s, m/s², or m/s³
    angular: float  # rad/s, rad/s², or rad/s³


@dataclass(frozen=True, slots=True)
class CartKinodynamic:
    """Cartesian kinodynamic limits with linear/angular components."""

    velocity: LinearAngular
    acceleration: LinearAngular
    jerk: LinearAngular


@dataclass(frozen=True, slots=True)
class CartLimits:
    """All Cartesian limits."""

    hard: CartKinodynamic
    jog: CartKinodynamic


@dataclass(frozen=True, slots=True)
class RobotLimits:
    """Unified robot limits namespace."""

    joint: JointLimits
    cart: CartLimits


def _build_kinodynamic(
    v_steps: ArrayLike, a_steps: ArrayLike, j_steps: ArrayLike
) -> Kinodynamic:
    """Build Kinodynamic from step-based limits, with both SI and step units."""
    v_steps_arr = np.asarray(v_steps, dtype=np.int32)
    a_steps_arr = np.asarray(a_steps, dtype=np.int32)
    j_steps_arr = np.asarray(j_steps, dtype=np.int32)
    v_rad = np.array(
        [float(speed_steps_to_rad_scalar(v_steps_arr[i], i)) for i in range(6)]
    )
    a_rad = np.array(
        [float(speed_steps_to_rad_scalar(a_steps_arr[i], i)) for i in range(6)]
    )
    j_rad = np.array(
        [float(speed_steps_to_rad_scalar(j_steps_arr[i], i)) for i in range(6)]
    )
    return Kinodynamic(
        velocity=v_rad,
        acceleration=a_rad,
        jerk=j_rad,
        velocity_steps=v_steps_arr,
        acceleration_steps=a_steps_arr,
        jerk_steps=j_steps_arr,
    )


def _build_joint_position(limits_deg: NDArray) -> JointPosition:
    """Build JointPosition from degree limits."""
    limits_rad = np.deg2rad(limits_deg)
    # Allocate once for module init (not hot path)
    tmp = np.zeros(6, dtype=np.int32)
    limits_steps = np.column_stack(
        [
            deg_to_steps(limits_deg[:, 0], tmp).copy(),
            deg_to_steps(limits_deg[:, 1], tmp).copy(),
        ]
    )
    return JointPosition(deg=limits_deg.copy(), rad=limits_rad, steps=limits_steps)


def _build_cart_kinodynamic(
    vel_lin_mm: float,
    vel_ang_deg: float,
    acc_lin_mm: float,
    acc_ang_deg: float,
    jerk_lin_mm: float,
    jerk_ang_deg: float,
) -> CartKinodynamic:
    """Build CartKinodynamic from mm/deg values, converting to SI."""
    return CartKinodynamic(
        velocity=LinearAngular(
            linear=vel_lin_mm / 1000.0, angular=np.radians(vel_ang_deg)
        ),
        acceleration=LinearAngular(
            linear=acc_lin_mm / 1000.0, angular=np.radians(acc_ang_deg)
        ),
        jerk=LinearAngular(
            linear=jerk_lin_mm / 1000.0, angular=np.radians(jerk_ang_deg)
        ),
    )


# Build the unified LIMITS structure
LIMITS: RobotLimits = RobotLimits(
    joint=JointLimits(
        hard=_build_kinodynamic(
            PAROL6_ROBOT.joint.speed_max,
            PAROL6_ROBOT.joint.acc_max,
            PAROL6_ROBOT.joint.jerk_max,
        ),
        jog=_build_kinodynamic(
            PAROL6_ROBOT.joint.jog_speed_max,
            PAROL6_ROBOT.joint.acc_max,  # Same acc for jog
            PAROL6_ROBOT.joint.jerk_max,  # Same jerk for jog
        ),
        position=_build_joint_position(PAROL6_ROBOT.joint.limits_deg),
    ),
    cart=CartLimits(
        hard=_build_cart_kinodynamic(
            PAROL6_ROBOT.cart.vel.linear.max,
            PAROL6_ROBOT.cart.vel.angular.max,
            PAROL6_ROBOT.cart.acc.linear.max,
            PAROL6_ROBOT.cart.acc.angular.max,
            PAROL6_ROBOT.cart.jerk.linear.max,
            PAROL6_ROBOT.cart.jerk.angular.max,
        ),
        jog=_build_cart_kinodynamic(
            PAROL6_ROBOT.cart.vel.jog.max,
            PAROL6_ROBOT.cart.vel.angular.max * 0.8,
            PAROL6_ROBOT.cart.acc.linear.max,
            PAROL6_ROBOT.cart.acc.angular.max,
            PAROL6_ROBOT.cart.jerk.linear.max,
            PAROL6_ROBOT.cart.jerk.angular.max,
        ),
    ),
)

# Validate limits at module load
if np.any(LIMITS.joint.hard.velocity <= 0) or np.any(
    LIMITS.joint.hard.acceleration <= 0
):
    raise ValueError("Joint limits must be positive. Check PAROL6_ROBOT config.")

# Jog min speeds - derived from control rate (1 step per tick minimum)
JOG_MIN_STEPS: int = int(CONTROL_RATE_HZ)  # steps/s
CART_LIN_JOG_MIN: float = CONTROL_RATE_HZ / 100  # mm/s (scales with control rate)
CART_ANG_JOG_MIN: float = 1.0  # deg/s

# Per-joint IK safety margins (radians) - [min_margin, max_margin] per joint
# Direction-aware: J3 backwards bend (max) is a trap, but inward (min) is safe
IK_SAFETY_MARGINS_RAD: NDArray[np.float64] = np.array(
    [
        [0.0, 0.0],  # J1 - base rotation, symmetric
        [0.00, 0.05],  # J2 - shoulder, symmetric
        [0.03, 0.8],  # J3 - elbow: min=inward (safe), max=backwards bend (TRAP)
        [0.0, 0.0],  # J4 - wrist, symmetric
        [0.0, 0.0],  # J5 - wrist, symmetric
        [0.03, 0.03],  # J6 - tool rotation, symmetric
    ],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
