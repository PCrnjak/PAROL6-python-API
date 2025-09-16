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

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state. Computes:
          - angles in degrees from Position_in (steps -> rad -> deg)
          - IO: state.InOut_in[:5]
          - gripper: state.Gripper_data_in (first 6 values if longer)
          - pose: via forward kinematics using current joint angles (rad)
        """
        with self._lock:
            # Angles (deg)
            angles_rad = [PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)]
            self.angles_deg = list(np.rad2deg(angles_rad))
            self._angles_ascii = ",".join(str(a) for a in self.angles_deg)

            # IO (first 5)
            self.io = list(state.InOut_in[:5])
            self._io_ascii = ",".join(str(x) for x in self.io)

            # Gripper (first 6)
            # Ensure at least 6 elements
            g = state.Gripper_data_in
            if len(g) < 6:
                g = (g + [0] * 6)[:6]
            else:
                g = g[:6]
            self.gripper = list(g)
            self._gripper_ascii = ",".join(str(x) for x in self.gripper)

            # Pose via FK
            q_current = np.array([PAROL6_ROBOT.STEPS2RADS(p, i) for i, p in enumerate(state.Position_in)])
            current_pose_matrix = PAROL6_ROBOT.robot.fkine(q_current).A
            pose_flat = current_pose_matrix.flatten().tolist()
            # Ensure 16 elements
            if len(pose_flat) == 16:
                self.pose = [float(x) for x in pose_flat]
                self._pose_ascii = ",".join(str(x) for x in self.pose)

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
