import threading
import time

import numpy as np
from numpy.typing import ArrayLike

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.server.state import ControllerState, get_fkine_flat_mm, get_fkine_se3
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.utils.ik import AXIS_MAP, solve_ik
from spatialmath import SE3
from typing import Any
import math


class StatusCache:
    """
    Thread-safe cache of the aggregate STATUS payload components and formatted ASCII.

    Fields:
      - angles_deg: 6 floats
      - speeds: 6 ints (steps/sec)
      - io: 5 ints [in1,in2,out1,out2,estop]
      - gripper: >=6 ints [id,pos,spd,cur,status,obj]
      - pose: 16 floats (flattened transform)
      - last_update_s: wall clock time of last cache update
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Public snapshots (materialized only when they change)
        self.angles_deg: np.ndarray = np.zeros((6,), dtype=np.float64)
        self.speeds: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.io: np.ndarray = np.zeros((5,), dtype=np.uint8)
        self.gripper: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.pose: np.ndarray = np.zeros((16,), dtype=np.float64)

        self.last_update_s: float = 0.0  # last cache build (any section)
        self.last_serial_s: float = 0.0  # last time a fresh serial frame was observed
        self._last_tool_name: str = "NONE"  # Track tool changes

        # Cached ASCII fragments to reduce allocations
        self._angles_ascii: str = "0,0,0,0,0,0"
        self._speeds_ascii: str = "0,0,0,0,0,0"
        self._io_ascii: str = "0,0,0,0,0"
        self._gripper_ascii: str = "0,0,0,0,0,0"
        self._pose_ascii: str = ",".join("0" for _ in range(16))

        # Action tracking fields
        self._action_current: str = ""
        self._action_state: str = "IDLE"

        # Enablement arrays (12 ints each)
        self.joint_en = np.ones((12,), dtype=np.uint8)
        self.cart_en_wrf = np.ones((12,), dtype=np.uint8)
        self.cart_en_trf = np.ones((12,), dtype=np.uint8)
        self._joint_en_ascii: str = ",".join(str(int(v)) for v in self.joint_en)
        self._cart_en_wrf_ascii: str = ",".join(str(int(v)) for v in self.cart_en_wrf)
        self._cart_en_trf_ascii: str = ",".join(str(int(v)) for v in self.cart_en_trf)

        self._ascii_full: str = (
            f"STATUS|POSE={self._pose_ascii}"
            f"|ANGLES={self._angles_ascii}"
            f"|SPEEDS={self._speeds_ascii}"
            f"|IO={self._io_ascii}"
            f"|GRIPPER={self._gripper_ascii}"
            f"|ACTION_CURRENT={self._action_current}"
            f"|ACTION_STATE={self._action_state}"
            f"|JOINT_EN={self._joint_en_ascii}"
            f"|CART_EN_WRF={self._cart_en_wrf_ascii}"
            f"|CART_EN_TRF={self._cart_en_trf_ascii}"
        )

        # Change-detection caches to avoid expensive recomputation when inputs unchanged
        self._last_pos_in: np.ndarray = np.zeros((6,), dtype=np.int32)

    def _format_csv_from_list(self, vals: ArrayLike) -> str:
        # Using str() on each value preserves prior formatting semantics
        return ",".join(str(v) for v in vals)  # type: ignore

    def _compute_joint_enable(self, q_rad: np.ndarray, delta_rad: float = math.radians(0.2)) -> None:
        """Compute per-joint +/- enable bits based on joint limits and a small delta."""
        # Be robust to uninitialized robot in type-checked context
        robot: Any = getattr(PAROL6_ROBOT, "robot", None)
        if robot is None:
            self.joint_en[:] = 1
            return
        qlim = getattr(robot, "qlim", None)
        if qlim is None:
            self.joint_en[:] = 1
            return
        allow_plus = (q_rad + delta_rad) <= qlim[1, :]
        allow_minus = (q_rad - delta_rad) >= qlim[0, :]
        # Pack into [J1+,J1-,J2+,J2-,...,J6+,J6-]
        bits = []
        for i in range(6):
            bits.append(1 if allow_plus[i] else 0)
            bits.append(1 if allow_minus[i] else 0)
        self.joint_en[:] = np.asarray(bits, dtype=np.uint8)
        self._joint_en_ascii = self._format_csv_from_list(self.joint_en.tolist())

    def _compute_cart_enable(self, T: SE3, frame: str, q_rad: np.ndarray,
                             delta_mm: float = 0.5, delta_deg: float = 0.5) -> None:
        """Compute per-axis +/- enable bits for the given frame (WRF/TRF) via small-step IK."""
        bits = []
        # Build small delta transforms
        t_step_m = delta_mm / 1000.0
        r_step_rad = math.radians(delta_deg)
        for axis, (v_lin, v_rot) in AXIS_MAP.items():
            # Compose delta SE3 for this axis
            dT = SE3()
            # Translation
            dx = v_lin[0] * t_step_m
            dy = v_lin[1] * t_step_m
            dz = v_lin[2] * t_step_m
            if abs(dx) > 0 or abs(dy) > 0 or abs(dz) > 0:
                dT = dT * SE3(dx, dy, dz)
            # Rotation
            rx = v_rot[0] * r_step_rad
            ry = v_rot[1] * r_step_rad
            rz = v_rot[2] * r_step_rad
            if abs(rx) > 0:
                dT = dT * SE3.Rx(rx)
            if abs(ry) > 0:
                dT = dT * SE3.Ry(ry)
            if abs(rz) > 0:
                dT = dT * SE3.Rz(rz)

            # Apply in specified frame
            if frame == "WRF":
                T_target = dT * T
            else:  # TRF
                T_target = T * dT

            try:
                ik = solve_ik(
                    PAROL6_ROBOT.robot, T_target, q_rad, jogging=True, quiet_logging=True
                )
                bits.append(1 if ik.success else 0)
            except Exception:
                bits.append(0)

        arr = np.asarray(bits, dtype=np.uint8)
        if frame == "WRF":
            self.cart_en_wrf[:] = arr
            self._cart_en_wrf_ascii = self._format_csv_from_list(arr.tolist())
        else:
            self.cart_en_trf[:] = arr
            self._cart_en_trf_ascii = self._format_csv_from_list(arr.tolist())

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state with change gating:
          - Only recompute angles/pose when Position_in changes (and beyond optional deadband)
          - Only refresh IO/speeds/gripper when their inputs actually change
        """
        now = time.time()
        changed_any = False

        with self._lock:
            # Check if position or tool changed
            tool_changed = state.current_tool != self._last_tool_name
            pos_changed = self._last_pos_in is None or not np.array_equal(
                state.Position_in, self._last_pos_in
            )

            if pos_changed or tool_changed:
                if pos_changed:
                    np.copyto(self._last_pos_in, state.Position_in)
                if tool_changed:
                    self._last_tool_name = state.current_tool

                # Vectorized steps->deg
                self.angles_deg = np.asarray(
                    PAROL6_ROBOT.ops.steps_to_deg(state.Position_in)
                )  # float64, shape (6,)
                # Publish angles list and ASCII
                self._angles_ascii = self._format_csv_from_list(self.angles_deg)
                changed_any = True

                # Get cached fkine (automatically updates if needed)
                pose_flat_mm = get_fkine_flat_mm(state)  # Already in mm for translation
                np.copyto(self.pose, pose_flat_mm)
                self._pose_ascii = self._format_csv_from_list(self.pose)
                changed_any = True

                # Compute enablement arrays at 50 Hz when pose/angles change
                try:
                    q_rad = np.asarray(PAROL6_ROBOT.ops.steps_to_rad(state.Position_in), dtype=float)
                except Exception:
                    q_rad = np.zeros((6,), dtype=float)
                try:
                    T = get_fkine_se3(state)
                except Exception:
                    T = SE3()
                # JOINT_EN
                self._compute_joint_enable(q_rad)
                # CART_EN for both frames
                self._compute_cart_enable(T, "WRF", q_rad)
                self._compute_cart_enable(T, "TRF", q_rad)

            # 2) IO (first 5)
            if not np.array_equal(self.io, state.InOut_in[:5]):
                np.copyto(self.io, state.InOut_in[:5])
                self._io_ascii = self._format_csv_from_list(self.io)
                changed_any = True

            # 3) Speeds (steps/sec from Speed_in)
            if not np.array_equal(self.speeds, state.Speed_in):
                np.copyto(self.speeds, state.Speed_in)
                self._speeds_ascii = self._format_csv_from_list(self.speeds)
                changed_any = True

            # 4) Gripper
            if not np.array_equal(self.gripper, state.Gripper_data_in):
                np.copyto(self.gripper, state.Gripper_data_in)
                self._gripper_ascii = self._format_csv_from_list(self.gripper)
                changed_any = True

            # 5) Action tracking
            if (
                self._action_current != state.action_current
                or self._action_state != state.action_state
            ):
                self._action_current = state.action_current
                self._action_state = state.action_state
                changed_any = True

            # 6) Assemble full ASCII only if any section changed
            if changed_any:
                self._ascii_full = (
                    f"STATUS|POSE={self._pose_ascii}"
                    f"|ANGLES={self._angles_ascii}"
                    f"|SPEEDS={self._speeds_ascii}"
                    f"|IO={self._io_ascii}"
                    f"|GRIPPER={self._gripper_ascii}"
                    f"|ACTION_CURRENT={self._action_current}"
                    f"|ACTION_STATE={self._action_state}"
                    f"|JOINT_EN={self._joint_en_ascii}"
                    f"|CART_EN_WRF={self._cart_en_wrf_ascii}"
                    f"|CART_EN_TRF={self._cart_en_trf_ascii}"
                )
                self.last_update_s = now

    def to_ascii(self) -> str:
        """Return the full ASCII STATUS payload."""
        with self._lock:
            return self._ascii_full

    def mark_serial_observed(self) -> None:
        """Mark that a fresh serial frame was observed just now."""
        with self._lock:
            self.last_serial_s = time.time()

    def age_s(self) -> float:
        """Seconds since last fresh serial observation (used to gate broadcasting)."""
        with self._lock:
            if self.last_serial_s <= 0:
                return 1e9
            return time.time() - self.last_serial_s


# Module-level singleton
_status_cache: StatusCache | None = None


def get_cache() -> StatusCache:
    global _status_cache
    if _status_cache is None:
        _status_cache = StatusCache()
    return _status_cache
