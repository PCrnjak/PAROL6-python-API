"""Shared self-collision pre-flight check used by motion commands.

`guard_joint_path(positions)` raises `MotionError(SYS_SELF_COLLISION)` if any
sampled configuration along the interpolated joint path would self-collide
(or world-collide given runtime obstacles attached to the singleton checker).
Disabled-by-config or unloaded-checker scenarios are no-ops.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import COLLISION_PATH_SAMPLES
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import MotionError


def guard_config(q: NDArray[np.float64]) -> None:
    """Raise MotionError if q is in collision. No-op if checker disabled."""
    checker = PAROL6_ROBOT.collision
    if checker is None:
        return
    q_arr = np.ascontiguousarray(q, dtype=np.float64)
    if checker.in_collision(q_arr):
        pairs = checker.colliding_pairs(q_arr)
        raise MotionError(
            make_error(
                ErrorCode.SYS_SELF_COLLISION,
                sample="target",
                total="1",
                pairs=str(pairs[:4]),
            )
        )


def guard_joint_path(positions: NDArray[np.float64]) -> None:
    """Raise MotionError if any sample in the path is in collision.

    `positions` is (N, nq) joint positions in radians. Endpoints are always
    included; up to COLLISION_PATH_SAMPLES interior samples are checked.
    """
    checker = PAROL6_ROBOT.collision
    if checker is None:
        return

    pos = np.ascontiguousarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n == 0:
        return

    # Subsample at COLLISION_PATH_SAMPLES interior points, always including
    # first and last rows.
    target = max(2, COLLISION_PATH_SAMPLES + 2)
    if n <= target:
        idx = np.arange(n)
    else:
        idx = np.unique(
            np.concatenate(
                [
                    np.linspace(0, n - 1, target).round().astype(int),
                    np.array([0, n - 1]),
                ]
            )
        )
    sub = np.ascontiguousarray(pos[idx], dtype=np.float64)
    hit = checker.check_path(sub)
    if hit >= 0:
        sample = int(idx[hit])
        pairs = checker.colliding_pairs(pos[sample])
        raise MotionError(
            make_error(
                ErrorCode.SYS_SELF_COLLISION,
                sample=str(sample),
                total=str(n),
                pairs=str(pairs[:4]),
            )
        )
