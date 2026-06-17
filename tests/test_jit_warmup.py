"""Regression tests: every parol6 numba function must be compiled at warmup.

The control loop runs at 100Hz with no headroom for mid-loop JIT compilation,
so ``warmup_jit()`` pre-compiles all ``@njit`` functions on startup. These tests
guard that contract so a forgotten warmup entry fails CI loudly (by name)
instead of silently cold-compiling in the control loop — which, on a slow
runner, stalled the first integration test's home past its timeout.

- ``test_all_parol6_njit_compiled_at_warmup`` — every ``@njit`` function in the
  parol6 package has a compiled specialization after ``warmup_jit()``. Catches a
  new ``@njit`` that nobody added to the warmup.
- ``test_no_recompile_during_motion`` — a representative motion compiles no new
  specializations. Catches a function warmed with the wrong argument *types*
  (numba keys one specialization per signature), which would recompile — and
  stall — on the first real call.
"""

from __future__ import annotations

import gc
import importlib
import pkgutil

import pytest
from numba.core.registry import CPUDispatcher

import parol6
from parol6.client.dry_run_client import DryRunRobotClient
from parol6.utils.warmup import warmup_jit

# Valid PAROL6 joint targets (deg), within limits — home/standby plus two
# reachable waypoints, enough to exercise move planning, IK and blending.
_HOME = [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
_W1 = [80.0, -80.0, 190.0, 10.0, 10.0, 190.0]
_W2 = [70.0, -70.0, 200.0, 20.0, 20.0, 200.0]


def _parol6_dispatchers() -> list[CPUDispatcher]:
    """Every numba ``CPUDispatcher`` defined in the parol6 package.

    Imports every parol6 submodule first so a function is not missed merely
    because nothing has imported its module yet.
    """
    for mod in pkgutil.walk_packages(parol6.__path__, "parol6."):
        try:
            importlib.import_module(mod.name)
        except Exception:  # optional/hardware-only modules may not import here
            continue
    seen: dict[int, CPUDispatcher] = {
        id(d): d
        for d in gc.get_objects()
        if isinstance(d, CPUDispatcher) and d.py_func.__module__.startswith("parol6.")
    }
    return list(seen.values())


def _name(d: CPUDispatcher) -> str:
    return f"{d.py_func.__module__}.{d.py_func.__name__}"


@pytest.mark.unit
def test_all_parol6_njit_compiled_at_warmup() -> None:
    """``warmup_jit()`` must compile every parol6 ``@njit`` function."""
    warmup_jit()
    unwarmed = sorted(_name(d) for d in _parol6_dispatchers() if not d.signatures)
    assert not unwarmed, (
        "These parol6 @njit functions are not compiled by warmup_jit(), so the "
        "control loop would cold-compile them mid-run. Add them to "
        "parol6/utils/warmup.py:\n  " + "\n  ".join(unwarmed)
    )


@pytest.mark.unit
def test_no_recompile_during_motion() -> None:
    """A representative motion must trigger no new JIT compilation.

    Exercises the planning/IK/trajectory hot path through the in-process dry-run
    simulator after warmup; any new specialization means warmup used argument
    types that differ from the real call.
    """
    warmup_jit()
    dispatchers = _parol6_dispatchers()
    before = {id(d): len(d.signatures) for d in dispatchers}

    client = DryRunRobotClient()
    client.move_j(_HOME, speed=0.5)
    client.move_j(_W1, speed=0.5, r=5.0)  # blended
    client.move_j(_W2, speed=0.5, r=5.0)  # blended
    client.move_j(_HOME, speed=0.5)
    client.wait_motion(timeout=10.0)

    recompiled = sorted(
        f"{_name(d)}: {before[id(d)]} -> {len(d.signatures)}"
        for d in dispatchers
        if len(d.signatures) > before[id(d)]
    )
    assert not recompiled, (
        "These parol6 @njit functions compiled a new specialization during "
        "motion — warmup_jit() warmed them with the wrong argument types:\n  "
        + "\n  ".join(recompiled)
    )
