import threading
import time
from typing import List, Optional
from numpy.typing import ArrayLike
import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.server.state import ControllerState


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

        # Cached ASCII fragments to reduce allocations
        self._angles_ascii: str = "0,0,0,0,0,0"
        self._speeds_ascii: str = "0,0,0,0,0,0"
        self._io_ascii: str = "0,0,0,0,0"
        self._gripper_ascii: str = "0,0,0,0,0,0"
        self._pose_ascii: str = ",".join("0" for _ in range(16))
        self._ascii_full: str = (
            f"STATUS|POSE={self._pose_ascii}"
            f"|ANGLES={self._angles_ascii}"
            f"|SPEEDS={self._speeds_ascii}"
            f"|IO={self._io_ascii}"
            f"|GRIPPER={self._gripper_ascii}"
        )

        # Change-detection caches to avoid expensive recomputation when inputs unchanged
        self._last_pos_in: np.ndarray = np.zeros((6,), dtype=np.int32)

    def _format_csv_from_list(self, vals: ArrayLike) -> str:
        # Using str() on each value preserves prior formatting semantics
        return ",".join(str(v) for v in vals) # type: ignore

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state with change gating:
          - Only recompute angles/pose when Position_in changes (and beyond optional deadband)
          - Only refresh IO/speeds/gripper when their inputs actually change
        """
        now = time.time()
        changed_any = False

        with self._lock:
            if self._last_pos_in is None or not np.array_equal(state.Position_in, self._last_pos_in): # Position changed
                np.copyto(self._last_pos_in, state.Position_in)
                # Vectorized steps->deg
                self.angles_deg = np.asarray(PAROL6_ROBOT.ops.steps_to_deg(state.Position_in))  # float64, shape (6,)
                # Publish angles list and ASCII
                self._angles_ascii = self._format_csv_from_list(self.angles_deg)
                changed_any = True

                # Vectorized steps->rad for FK
                q_current = PAROL6_ROBOT.ops.steps_to_rad(state.Position_in)  # float64, shape (6,)
                # robot.fkine expects joint vector in radians
                current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A  # 4x4
                pose_flat = current_pose_matrix.reshape(-1)  # 16
                self.pose = np.asarray(pose_flat, dtype=np.float64)
                # Convert translation from meters to mm for all consumers (indices 3, 7, 11)
                self.pose[3] *= 1000.0   # X translation
                self.pose[7] *= 1000.0   # Y translation
                self.pose[11] *= 1000.0  # Z translation
                self._pose_ascii = self._format_csv_from_list(self.pose)
                changed_any = True

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

            # 5) Assemble full ASCII only if any section changed
            if changed_any:
                self._ascii_full = (
                    f"STATUS|POSE={self._pose_ascii}"
                    f"|ANGLES={self._angles_ascii}"
                    f"|SPEEDS={self._speeds_ascii}"
                    f"|IO={self._io_ascii}"
                    f"|GRIPPER={self._gripper_ascii}"
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
_status_cache: Optional[StatusCache] = None


def get_cache() -> StatusCache:
    global _status_cache
    if _status_cache is None:
        _status_cache = StatusCache()
    return _status_cache
