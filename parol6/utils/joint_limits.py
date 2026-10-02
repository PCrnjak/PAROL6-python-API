"""Joint travel as the arm reports it.

A joint parked on its limit reads back a hair past the limit's decimal:
the motor step it stands on rounds to the nearest step, outward as often
as not, and a float carrying the angle rounds again. Every check of a
joint against its travel allows half a motor step either side, so what
the arm reports can be commanded again; anything further out is refused.
"""

import numpy as np
from numpy.typing import NDArray

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import LIMITS

#: Half a motor step of each joint [rad].
HALF_STEP_RAD: NDArray[np.float64] = (
    0.5 * PAROL6_ROBOT.radian_per_step_constant / PAROL6_ROBOT.joint.ratio
)

#: Each joint's travel widened by half a motor step [rad].
TRAVEL_MIN_RAD: NDArray[np.float64] = LIMITS.joint.position.rad[:, 0] - HALF_STEP_RAD
TRAVEL_MAX_RAD: NDArray[np.float64] = LIMITS.joint.position.rad[:, 1] + HALF_STEP_RAD

# Plain floats: the degree check runs on every streamed servo target, where
# indexing a numpy table would box a scalar per comparison.
_MIN_RAD: tuple[float, ...] = tuple(float(v) for v in TRAVEL_MIN_RAD)
_MAX_RAD: tuple[float, ...] = tuple(float(v) for v in TRAVEL_MAX_RAD)
_MIN_DEG: tuple[float, ...] = tuple(float(v) for v in np.degrees(TRAVEL_MIN_RAD))
_MAX_DEG: tuple[float, ...] = tuple(float(v) for v in np.degrees(TRAVEL_MAX_RAD))


def joint_outside_travel_rad(q: "NDArray[np.float64] | list[float]") -> int:
    """Index of the first joint of ``q`` [rad] more than half a motor step
    past its travel, or -1 when every joint is inside it."""
    for i in range(6):
        if not (_MIN_RAD[i] <= q[i] <= _MAX_RAD[i]):
            return i
    return -1


def joint_outside_travel_deg(angles: "NDArray[np.float64] | list[float]") -> int:
    """Index of the first joint of ``angles`` [deg] more than half a motor
    step past its travel, or -1 when every joint is inside it."""
    for i in range(6):
        if not (_MIN_DEG[i] <= angles[i] <= _MAX_DEG[i]):
            return i
    return -1
