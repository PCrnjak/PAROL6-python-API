# Clean, hierarchical, vectorized, and typed robot configuration and helpers
from dataclasses import dataclass
from math import pi
from typing import Final, Callable, Sequence, Union, Any
import logging
from numpy.typing import ArrayLike
import numpy as np
from numpy.typing import NDArray
import roboticstoolbox as rtb
from roboticstoolbox.tools.urdf import URDF
from roboticstoolbox import Link, ET
from pathlib import Path
from parol6.tools import get_tool_transform
from spatialmath import SE3

logger = logging.getLogger(__name__)

# -----------------------------
# Typing aliases
# -----------------------------
IndexArg = Union[int, NDArray[np.int_], None]

Vec6f = NDArray[np.float64]
Vec6i = NDArray[np.int32]
Limits2f = NDArray[np.float64]  # shape (6,2)

# -----------------------------
# Kinematics and conversion constants
# -----------------------------
Joint_num = 6
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

_joint_limits_radian: Limits2f = np.deg2rad(_joint_limits_degree).astype(np.float64)

# URDF-based robot model (frames/limits aligned with controller)
def _load_urdf() -> URDF:
    """Load and cache the URDF object for robot reconstruction."""
    base_path = Path(__file__).resolve().parent / "urdf_model"
    urdf_path = base_path / "urdf" / "PAROL6.urdf"
    urdf_string = urdf_path.read_text(encoding="utf-8")
    return URDF.loadstr(urdf_string, str(urdf_path), base_path=base_path)

# Cache the URDF object (parsed once, reused for robot reconstruction)
_cached_urdf = _load_urdf()

# Current robot instance (rebuilt when tool changes)
robot = None

def apply_tool(tool_name: str) -> None:
    """
    Rebuild the robot with the specified tool as an additional link.
    This ensures the tool transform is properly integrated into the kinematic chain
    and affects forward kinematics calculations.
    
    Parameters
    ----------
    tool_name : str
        Name of the tool from tools.TOOL_CONFIGS
    """
    global robot
    
    # Get tool transform
    T_tool = get_tool_transform(tool_name)
    
    # Get the base elinks from cached URDF
    base_links = list(_cached_urdf.elinks)
    
    # Create a tool link if there's a non-identity transform
    if tool_name != "NONE" and not np.allclose(T_tool, np.eye(4)):
        # Create an ELink for the tool
        # The tool is a fixed transform from the last joint
        tool_link = Link(
            ET.SE3(SE3(T_tool)),
            name=f"tool_{tool_name}",
            parent=base_links[-1]  # Attach to the last link
        )
        
        # Add tool link to the chain
        all_links = base_links + [tool_link]
        logger.info(f"Applied tool '{tool_name}' to robot model as link")
    else:
        all_links = base_links
        logger.info(f"Applied tool '{tool_name}' (no additional link needed)")
    
    # Create robot with the complete link chain
    robot = rtb.Robot(
        all_links,
        name=_cached_urdf.name,
    )

# Initialize with no tool
apply_tool("NONE")

# -----------------------------
# Additional raw parameter arrays
# -----------------------------
# Reduction ratio per joint
_joint_ratio: NDArray[np.float64] = np.array(
    [6.4, 20.0, 20.0 * (38.0 / 42.0), 4.0, 4.0, 10.0], dtype=np.float64
)

# Joint speeds (steps/s)
_joint_max_speed: Vec6i = np.array([6500, 18000, 20000, 20000, 22000, 22000], dtype=np.int32)
_joint_min_speed: Vec6i = np.array([100, 100, 100, 100, 100, 100], dtype=np.int32)

# Jog speeds (steps/s)
_joint_max_jog_speed: Vec6i = np.array([1500, 3000, 3600, 7000, 7000, 18000], dtype=np.int32)
_joint_min_jog_speed: Vec6i = np.array([100, 100, 100, 100, 100, 100], dtype=np.int32)

# Joint accelerations (rad/s^2) - scalar limits applied per joint
_joint_max_acc_rad: float = float(32000)
_joint_min_acc_rad: float = float(100)

# Maximum jerk limits (steps/s^3) per joint
_joint_max_jerk: Vec6i = np.array([1600, 1000, 1100, 3000, 3000, 2000], dtype=np.int32)

# Cartesian limits
_cart_linear_velocity_min_JOG: float = float(0.002)
_cart_linear_velocity_max_JOG: float = float(0.06)

_cart_linear_velocity_min: float = float(0.002)
_cart_linear_velocity_max: float = float(0.06)

_cart_linear_acc_min: float = float(0.002)
_cart_linear_acc_max: float = float(0.06)

_cart_angular_velocity_min: float = float(0.7)   # deg/s
_cart_angular_velocity_max: float = float(25.0)  # deg/s

# Standby positions
_standby_deg: Vec6f = np.array([90.0, -90.0, 180.0, 0.0, 0.0, 180.0], dtype=np.float64)
_standby_rad: Vec6f = np.deg2rad(_standby_deg).astype(np.float64)

# -----------------------------
# Vectorized helpers (ops)
# -----------------------------
def _apply_ratio(values: NDArray, idx: IndexArg) -> NDArray:
    """
    Apply per-joint gear ratio.
    If idx is None, broadcast ratio across last dimension (length 6).
    If idx is an int or ndarray of ints, select ratios accordingly.
    """
    if idx is None:
        return values * _joint_ratio
    idx_arr = np.asarray(idx)
    return values * _joint_ratio[idx_arr]


def deg_to_steps(deg: ArrayLike, idx: IndexArg = None) -> np.int32 | NDArray[np.int32]:
    """Degrees to steps (gear ratio aware). Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(deg):
        return np.int32((deg / degree_per_step_constant) * _joint_ratio[idx]) # type: ignore
    deg_arr = np.asarray(deg, dtype=np.float64)
    steps_f = _apply_ratio(deg_arr / degree_per_step_constant, idx)
    return steps_f.astype(np.int32, copy=False)


def steps_to_deg(steps: ArrayLike, idx: IndexArg = None) -> np.float64 | NDArray[np.float64]:
    """Steps to degrees (gear ratio aware). Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(steps):
        return np.float64((steps * degree_per_step_constant) / _joint_ratio[idx]) # type: ignore
    steps_arr = np.asarray(steps, dtype=np.float64)
    ratio = _joint_ratio if idx is None else _joint_ratio[np.asarray(idx)]
    return (steps_arr * degree_per_step_constant) / ratio


def rad_to_steps(rad: ArrayLike, idx: IndexArg = None) -> np.int32 | NDArray[np.int32]:
    """Radians to steps. Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(rad):
        return np.int32((rad / radian_per_step_constant) * _joint_ratio[idx]) # type: ignore
    rad_arr = np.asarray(rad, dtype=np.float64)
    deg_arr = np.rad2deg(rad_arr)
    return deg_to_steps(deg_arr, idx)


def steps_to_rad(steps: ArrayLike, idx: IndexArg = None) -> np.float64 | NDArray[np.float64]:
    """Steps to radians. Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(steps):
        return np.float64((steps * radian_per_step_constant) / _joint_ratio[idx]) # type: ignore
    deg_arr = steps_to_deg(steps, idx)
    return np.deg2rad(deg_arr)


def speed_steps_to_deg(sps: ArrayLike, idx: IndexArg = None) -> np.float64 | NDArray[np.float64]:
    """Speed: steps/s to deg/s. Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(sps):
        return np.float64((sps * degree_per_step_constant) / _joint_ratio[idx]) # type: ignore
    sps_arr = np.asarray(sps, dtype=np.float64)
    ratio = _joint_ratio if idx is None else _joint_ratio[np.asarray(idx)]
    return (sps_arr * degree_per_step_constant) / ratio


def speed_deg_to_steps(dps: ArrayLike, idx: IndexArg = None) -> np.int32 | NDArray[np.int32]:
    """Speed: deg/s to steps/s. Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(dps):
        return np.int32((dps / degree_per_step_constant) * _joint_ratio[idx]) # type: ignore
    dps_arr = np.asarray(dps, dtype=np.float64)
    stepsps = _apply_ratio(dps_arr / degree_per_step_constant, idx)
    return stepsps.astype(np.int32, copy=False)


def speed_steps_to_rad(sps: ArrayLike, idx: IndexArg = None) -> np.float64 | NDArray[np.float64]:
    """Speed: steps/s to rad/s. Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(sps):
        return np.float64((sps * radian_per_step_constant) / _joint_ratio[idx]) # type: ignore
    sps_arr = np.asarray(sps, dtype=np.float64)
    ratio = _joint_ratio if idx is None else _joint_ratio[np.asarray(idx)]
    return (sps_arr * radian_per_step_constant) / ratio


def speed_rad_to_steps(rps: ArrayLike, idx: IndexArg = None) -> np.int32 | NDArray[np.int32]:
    """Speed: rad/s to steps/s. Fast scalar path when idx is int."""
    if isinstance(idx, (int, np.integer)) and np.isscalar(rps):
        return np.int32((rps / radian_per_step_constant) * _joint_ratio[idx]) # type: ignore
    rps_arr = np.asarray(rps, dtype=np.float64)
    stepsps = _apply_ratio(rps_arr / radian_per_step_constant, idx)
    return stepsps.astype(np.int32, copy=False)


def clip_speed_to_joint_limits(sps: ArrayLike) -> NDArray[np.int32]:
    """Clip steps/s vector to per-joint limits (int32)."""
    sps_arr = np.asarray(sps, dtype=np.int32)
    return np.clip(sps_arr, -_joint_max_speed, _joint_max_speed).astype(np.int32, copy=False)


def clamp_steps_delta(
    prev_steps: ArrayLike, target_steps: ArrayLike, dt: float, safety: float = 1.2
) -> NDArray[np.int32]:
    """
    Clamp per-tick step change to max allowed based on joint.max_speed and dt.
    Returns int32 array.
    """
    prev_arr = np.asarray(prev_steps, dtype=np.int32)
    tgt_arr = np.asarray(target_steps, dtype=np.int32)
    step_diff = tgt_arr - prev_arr
    max_step_diff = (_joint_max_speed * dt * safety).astype(np.int32)
    sign = np.sign(step_diff).astype(np.int32)
    over = np.abs(step_diff) > max_step_diff
    clamped = tgt_arr.copy()
    clamped[over] = prev_arr[over] + sign[over] * max_step_diff[over]
    return clamped.astype(np.int32, copy=False)

# -----------------------------
# Limits (steps) derived from deg
# -----------------------------
_joint_limits_steps_list: list[list[int]] = []
for j in range(6):
    mn_deg, mx_deg = float(_joint_limits_degree[j, 0]), float(_joint_limits_degree[j, 1])
    mn_steps = int(deg_to_steps(mn_deg, idx=j))
    mx_steps = int(deg_to_steps(mx_deg, idx=j))
    _joint_limits_steps_list.append([mn_steps, mx_steps])
_joint_limits_steps: NDArray[np.int32] = np.array(_joint_limits_steps_list, dtype=np.int32)  # (6,2)

# -----------------------------
# Typed hierarchical API
# -----------------------------
@dataclass(frozen=True)
class JointLimits:
    deg: Limits2f
    rad: Limits2f
    steps: NDArray[np.int32]


@dataclass(frozen=True)
class JointJogSpeed:
    max: Vec6i
    min: Vec6i


@dataclass(frozen=True)
class JointSpeed:
    max: Vec6i
    min: Vec6i
    jog: JointJogSpeed


@dataclass(frozen=True)
class JointAcc:
    max_rad: float
    min_rad: float


@dataclass(frozen=True)
class JointJerk:
    max: Vec6i


@dataclass(frozen=True)
class Standby:
    deg: Vec6f
    rad: Vec6f


@dataclass(frozen=True)
class Joint:
    limits: JointLimits
    speed: JointSpeed
    acc: JointAcc
    jerk: JointJerk
    ratio: NDArray[np.float64]
    standby: Standby


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


@dataclass(frozen=True)
class Cart:
    vel: CartVel
    acc: CartAcc


@dataclass(frozen=True)
class Conv:
    degree_per_step: float
    radian_per_step: float
    rad_sec_to_deg_sec: float
    deg_sec_to_rad_sec: float


@dataclass(frozen=True)
class Ops:
    # Use Callable[..., T] to allow optional idx parameter without arity errors in type checkers
    deg_to_steps: Callable[..., np.int32 | NDArray[np.int32]]
    steps_to_deg: Callable[..., np.float64 | NDArray[np.float64]]
    rad_to_steps: Callable[..., np.int32 | NDArray[np.int32]]
    steps_to_rad: Callable[..., np.float64 | NDArray[np.float64]]
    speed_deg_to_steps: Callable[..., np.int32 | NDArray[np.int32]]
    speed_steps_to_deg: Callable[..., np.float64 | NDArray[np.float64]]
    speed_rad_to_steps: Callable[..., np.int32 | NDArray[np.int32]]
    speed_steps_to_rad: Callable[..., np.float64 | NDArray[np.float64]]
    clip_speed_to_joint_limits: Callable[[ArrayLike], NDArray[np.int32]]
    clamp_steps_delta: Callable[[ArrayLike, ArrayLike, float, float], NDArray[np.int32]]


joint: Final[Joint] = Joint(
    limits=JointLimits(
        deg=_joint_limits_degree,
        rad=_joint_limits_radian,
        steps=_joint_limits_steps,
    ),
    speed=JointSpeed(
        max=_joint_max_speed,
        min=_joint_min_speed,
        jog=JointJogSpeed(
            max=_joint_max_jog_speed,
            min=_joint_min_jog_speed,
        ),
    ),
    acc=JointAcc(
        max_rad=_joint_max_acc_rad,
        min_rad=_joint_min_acc_rad,
    ),
    jerk=JointJerk(
        max=_joint_max_jerk,
    ),
    ratio=_joint_ratio,
    standby=Standby(
        deg=_standby_deg,
        rad=_standby_rad,
    ),
)

cart: Final[Cart] = Cart(
    vel=CartVel(
        linear=RangeF(min=_cart_linear_velocity_min, max=_cart_linear_velocity_max),
        jog=RangeF(min=_cart_linear_velocity_min_JOG, max=_cart_linear_velocity_max_JOG),
        angular=RangeF(min=_cart_angular_velocity_min, max=_cart_angular_velocity_max),
    ),
    acc=CartAcc(
        linear=RangeF(min=_cart_linear_acc_min, max=_cart_linear_acc_max),
    ),
)

conv: Final[Conv] = Conv(
    degree_per_step=degree_per_step_constant,
    radian_per_step=radian_per_step_constant,
    rad_sec_to_deg_sec=radian_per_sec_2_deg_per_sec_const,
    deg_sec_to_rad_sec=deg_per_sec_2_radian_per_sec_const,
)

ops: Final[Ops] = Ops(
    deg_to_steps=deg_to_steps,
    steps_to_deg=steps_to_deg,
    rad_to_steps=rad_to_steps,
    steps_to_rad=steps_to_rad,
    speed_deg_to_steps=speed_deg_to_steps,
    speed_steps_to_deg=speed_steps_to_deg,
    speed_rad_to_steps=speed_rad_to_steps,
    speed_steps_to_rad=speed_steps_to_rad,
    clip_speed_to_joint_limits=clip_speed_to_joint_limits,
    clamp_steps_delta=clamp_steps_delta,
)

# -----------------------------
# Fast, vectorized limit checking with edge-triggered logging
# -----------------------------
_last_violation_mask = np.zeros(6, dtype=bool)
_last_any_violation = False
# TODO: confirm whether this is actually faster than the previous loop based approach
def check_limits(q: ArrayLike, target_q: ArrayLike | None = None, allow_recovery: bool = True, *, log: bool = True) -> bool:
    """
    Vectorized limits check in radians.
    - q: current joint angles in radians (array-like)
    - target_q: optional target joint angles in radians (array-like)
    - allow_recovery: allow movement that heads back toward valid range if currently violating
    - log: emit edge-triggered warning/info logs on violation state changes

    Returns True if move is allowed (within limits or valid recovery), False otherwise.
    """
    global _last_violation_mask, _last_any_violation

    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    mn = joint.limits.rad[:, 0]
    mx = joint.limits.rad[:, 1]

    below = q_arr < mn
    above = q_arr > mx
    cur_viol = below | above

    if target_q is None:
        ok_mask = ~cur_viol
        t_below = t_above = None
    else:
        t = np.asarray(target_q, dtype=np.float64).reshape(-1)
        t_below = t < mn
        t_above = t > mx
        t_viol = t_below | t_above
        if allow_recovery:
            rec_ok = (above & (t <= q_arr)) | (below & (t >= q_arr))
        else:
            rec_ok = np.zeros(6, dtype=bool)
        ok_mask = (~cur_viol & ~t_viol) | (cur_viol & rec_ok)

    all_ok = bool(np.all(ok_mask))

    if log:
        viol = ~ok_mask
        any_viol = bool(np.any(viol))

        # Edge-triggered violation logs
        if any_viol and (np.any(viol != _last_violation_mask) or not _last_any_violation):
            idxs = np.where(viol)[0]
            tokens = []
            for i in idxs:
                if cur_viol[i]:
                    tokens.append(f"J{i+1}:" + ("cur<min" if below[i] else "cur>max"))
                else:
                    # target violates
                    if t_below is not None and t_below[i]:
                        tokens.append(f"J{i+1}:target<min")
                    elif t_above is not None and t_above[i]:
                        tokens.append(f"J{i+1}:target>max")
                    else:
                        tokens.append(f"J{i+1}:violation")
            logger.warning("LIMIT VIOLATION: %s", " ".join(tokens))
        elif (not any_viol) and _last_any_violation:
            logger.info("Limits back in range")

        _last_violation_mask[:] = viol
        _last_any_violation = any_viol

    return all_ok


def check_limits_mask(q: ArrayLike, target_q: ArrayLike | None = None, allow_recovery: bool = True) -> NDArray[np.bool_]:
    """Return per-joint boolean mask (True if OK for that joint)."""
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    mn = joint.limits.rad[:, 0]
    mx = joint.limits.rad[:, 1]
    below = q_arr < mn
    above = q_arr > mx
    cur_viol = below | above

    if target_q is None:
        return ~cur_viol
    t = np.asarray(target_q, dtype=np.float64).reshape(-1)
    t_below = t < mn
    t_above = t > mx
    t_viol = t_below | t_above
    if allow_recovery:
        rec_ok = (above & (t <= q_arr)) | (below & (t >= q_arr))
    else:
        rec_ok = np.zeros(6, dtype=bool)
    ok_mask = (~cur_viol & ~t_viol) | (cur_viol & rec_ok)
    return ok_mask

# -----------------------------
# CAN helpers and bitfield utils (used by transports/gripper)
# -----------------------------
def extract_from_can_id(can_id: int):
    id2 = (can_id >> 7) & 0xF
    can_command = (can_id >> 1) & 0x3F
    error_bit = can_id & 0x1
    return id2, can_command, error_bit

def combine_2_can_id(id2: int, can_command: int, error_bit: int) -> int:
    can_id = 0
    can_id |= (id2 & 0xF) << 7
    can_id |= (can_command & 0x3F) << 1
    can_id |= (error_bit & 0x1)
    return can_id

def fuse_bitfield_2_bytearray(var_in):
    number = 0
    for b in var_in:
        number = (2 * number) + int(b)
    return bytes([number])

def split_2_bitfield(var_in: int):
    return [(var_in >> i) & 1 for i in range(7, -1, -1)]

if __name__ == "__main__":
    # Simple sanity prints
    j_step_rad = steps_to_rad(np.array([1, 1, 1, 1, 1, 1], dtype=np.int32))
    print("Smallest step (deg):", np.rad2deg(j_step_rad))
    print("Standby rad:", joint.standby.rad)
