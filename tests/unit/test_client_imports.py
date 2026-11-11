"""A robot program's client loads neither the planner nor numba's compiler."""

import json
import subprocess
import sys

import pytest

_PROBE = """
import json, sys
import parol6
parol6.RobotClient()
from numba.core.registry import CPUDispatcher
ran = sorted(
    f"{name}.{attr}"
    for name, module in list(sys.modules.items())
    if name.split(".")[0] in ("parol6", "pinokin")
    for attr, value in list(vars(module).items())
    if isinstance(value, CPUDispatcher) and value.signatures
)
print(json.dumps({"modules": sorted(sys.modules), "jitted": ran}))
"""


@pytest.mark.unit
def test_a_client_loads_no_planner_and_runs_nothing_jitted() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    ).stdout
    report = json.loads(out.splitlines()[-1])
    planner = [
        m
        for m in report["modules"]
        if m.split(".")[0] in ("matplotlib", "toppra")
        or m in ("scipy.interpolate", "scipy.optimize", "parol6.client.dry_run_client")
    ]
    assert not planner, planner
    assert not report["jitted"], report["jitted"]
