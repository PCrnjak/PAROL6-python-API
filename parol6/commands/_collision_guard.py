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

# Penetration-depth tolerance (metres) for the start-in-collision escape check:
# a move whose min-distance drops by no more than this counts as "not deeper",
# absorbing numerical jitter in the signed-distance query.
_ESCAPE_TOL = 1e-4


def collision_blocked(checker, current_q, target_q) -> bool:
    """Whether streaming toward ``target_q`` must stop.

    Approaching: blocked when ``target_q`` collides. Already colliding (a
    keep-out placed over the arm, or a tool-attach overlap): blocked only when
    the motion goes *deeper* — escaping is allowed, mirroring
    :func:`guard_joint_path`'s start-in-collision rule.
    """
    if checker.in_collision(current_q):
        return bool(
            checker.min_distance(target_q)
            < checker.min_distance(current_q) - _ESCAPE_TOL
        )
    return bool(checker.in_collision(target_q))


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
    """Raise if the path drives into collision.

    `positions` is (N, nq) joint positions in radians. Endpoints are always
    included; up to COLLISION_PATH_SAMPLES interior samples are checked.

    Normally any collision along the path is rejected. The exception is when the
    path *starts* already in collision (checker enabled at a colliding pose, or a
    tool attach created an overlap): rejecting outright would trap the arm, so an
    *escaping* move is permitted — one that adds no new colliding pair and goes no
    deeper than the start. Driving into a new collision, or deeper into the
    current one, still raises.
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

    def _raise(sample: int, pairs: list[tuple[str, str]]) -> None:
        exc = TrajectoryPlanningError(
            make_error(
                ErrorCode.SYS_SELF_COLLISION,
                sample=str(sample),
                total=str(n),
                pairs=_format_pairs(pairs),
            )
        )
        # Structured channel for the viz, alongside the formatted error string.
        exc.colliding_pairs = list(pairs)
        raise exc

    hit = checker.check_path(sub)
    if hit < 0:
        return  # entire path clear — the common case
    if hit > 0:
        # Start is clear but the path drives into a collision — reject it.
        sample = hit if idx is None else int(idx[hit])
        _raise(sample, checker.colliding_pairs(pos[sample]))

    # hit == 0: already in collision at the start. Permit an escaping move — one
    # that introduces no new colliding pair and goes no deeper than the start —
    # so the arm isn't trapped. (A global min-distance trend alone can't tell an
    # improving start-collision from a new shallower one, so we check pairs too.)
    d0 = checker.min_distance(pos[0])
    start_pairs = set(checker.colliding_pairs(pos[0]))
    for j in range(1, sub.shape[0]):
        new_pairs = set(checker.colliding_pairs(sub[j])) - start_pairs
        deeper = checker.min_distance(sub[j]) < d0 - _ESCAPE_TOL
        if new_pairs or deeper:
            sample = j if idx is None else int(idx[j])
            pairs = (
                sorted(new_pairs) if new_pairs else checker.colliding_pairs(pos[sample])
            )
            _raise(sample, pairs)
