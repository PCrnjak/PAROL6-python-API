from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np  # type: ignore

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.server.state import ControllerState


class StatusCache:
    """
    Thread-safe cache of the aggregate STATUS payload components and formatted ASCII.

    Fields:
      - angles_deg: 6 floats
      - io: 5 ints [in1,in2,out1,out2,estop]
      - gripper: >=6 ints [id,pos,spd,cur,status,obj]
      - pose: 16 floats (flattened transform)
      - last_update_s: monotonic time of last successful update
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.angles_deg: List[float] = [0.0] * 6
        self.io: List[int] = [0, 0, 0, 0, 0]
        self.gripper: List[int] = [0, 0, 0, 0, 0, 0]
        self.pose: List[float] = [0.0] * 16
        self.last_update_s: float = 0.0  # last cache build (any update)
        self.last_serial_s: float = 0.0  # last time a fresh serial frame was observed
        # Cached ASCII fragments to reduce allocations
        self._angles_ascii: str = "0,0,0,0,0,0"
        self._io_ascii: str = "0,0,0,0,0"
        self._gripper_ascii: str = "0,0,0,0,0,0"
        self._pose_ascii: str = ",".join("0" for _ in range(16))
        self._ascii_full: str = f"STATUS|POSE={self._pose_ascii}|ANGLES={self._angles_ascii}|IO={self._io_ascii}|GRIPPER={self._gripper_ascii}"
        # Change-detection caches to avoid expensive recomputation when inputs unchanged
        self._last_pos_in: np.ndarray | None = None
        self._last_io5: np.ndarray | None = None
        self._last_grip6: np.ndarray | None = None

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state with change gating:
          - Only recompute angles/pose when Position_in changes
          - Always refresh IO/gripper (cheap)
        """
        with self._lock:
            # Detect position changes (gate expensive FK/angle math)
            pos_in = np.asarray(state.Position_in, dtype=np.int32)
            pos_changed = False
            if self._last_pos_in is None or self._last_pos_in.shape != (6,):
                self._last_pos_in = pos_in.copy()
                pos_changed = True
            else:
                # np.array_equal is fast for small arrays
                if not np.array_equal(pos_in, self._last_pos_in):
                    self._last_pos_in[:] = pos_in
                    pos_changed = True

            if pos_changed:
                # Angles (deg) from steps
                angles_rad = [PAROL6_ROBOT.STEPS2RADS(int(p), i) for i, p in enumerate(pos_in)]
                self.angles_deg = list(np.rad2deg(angles_rad))
                self._angles_ascii = ",".join(str(a) for a in self.angles_deg)

                # Pose via FK
                q_current = np.array([PAROL6_ROBOT.STEPS2RADS(int(p), i) for i, p in enumerate(pos_in)])
                current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A
                pose_flat = current_pose_matrix.flatten().tolist()
                if len(pose_flat) == 16:
                    self.pose = [float(x) for x in pose_flat]
                    self._pose_ascii = ",".join(str(x) for x in self.pose)

            # IO (first 5)
            io5 = np.asarray(state.InOut_in[:5], dtype=np.uint8)
            self.io = io5.tolist()
            self._io_ascii = ",".join(str(int(x)) for x in io5)

            # Gripper (first 6)
            grip6 = np.asarray(state.Gripper_data_in[:6], dtype=np.int32)
            if grip6.shape[0] < 6:
                # Pad to 6 if shorter
                grip6 = np.pad(grip6, (0, 6 - grip6.shape[0]), mode="constant")
            self.gripper = grip6.tolist()
            self._gripper_ascii = ",".join(str(int(x)) for x in grip6)

            # Assemble full ASCII (cheap string concatenation)
            self._ascii_full = f"STATUS|POSE={self._pose_ascii}|ANGLES={self._angles_ascii}|IO={self._io_ascii}|GRIPPER={self._gripper_ascii}"
            self.last_update_s = time.time()

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
