"""Shared self-collision pre-flight check used by motion commands.

`guard_joint_path(positions)` raises `TrajectoryPlanningError(SYS_SELF_COLLISION)`
if any sampled configuration along the interpolated joint path would self-collide
(or world-collide given runtime obstacles attached to the singleton checker).
It runs server-side during trajectory build, so it raises the planning-time
error type (like the other do_setup guards), not the client-side `MotionError`.
Disabled-by-config or unloaded-checker scenarios are no-ops.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import COLLISION_PATH_SAMPLES
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode
from parol6.utils.errors import TrajectoryPlanningError


def _format_pairs(pairs: list[tuple[str, str]]) -> str:
    """Render colliding (name, name) pairs as a human-readable string.

    Caps at 4 to keep error messages tractable when many pairs collide
    simultaneously (rare in practice; usually the first one is the
    actionable one anyway).
    """
    if not pairs:
        return "?"
    head = pairs[:4]
    rendered = ", ".join(f"{a} vs {b}" for a, b in head)
    if len(pairs) > 4:
        rendered += f" (+{len(pairs) - 4} more)"
    return rendered


def guard_joint_path(positions: NDArray[np.float64]) -> None:
    """Raise MotionError if any sample in the path is in collision.

    `positions` is (N, nq) joint positions in radians. Endpoints are always
    included; up to COLLISION_PATH_SAMPLES interior samples are checked.
    """
    checker = PAROL6_ROBOT.collision
    if checker is None:
        return

    # Load-bearing: normalize to a C-contiguous float64 array for the C++ bindings.
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n == 0:
        return

    # Subsample at COLLISION_PATH_SAMPLES interior points. np.linspace includes
    # both endpoints, and n > target guarantees spacing > 1 so the rounded
    # indices are strictly increasing; np.unique stays as cheap insurance.
    target = max(2, COLLISION_PATH_SAMPLES + 2)
    if n <= target:
        idx = None
        sub = pos  # already contiguous float64 — no copy needed
    else:
        idx = np.unique(np.linspace(0, n - 1, target).round().astype(int))
        sub = pos[idx]  # fancy indexing yields a fresh contiguous float64 array
    hit = checker.check_path(sub)
    if hit >= 0:
        sample = hit if idx is None else int(idx[hit])
        pairs = checker.colliding_pairs(pos[sample])
        raise TrajectoryPlanningError(
            make_error(
                ErrorCode.SYS_SELF_COLLISION,
                sample=str(sample),
                total=str(n),
                pairs=_format_pairs(pairs),
            )
        )
