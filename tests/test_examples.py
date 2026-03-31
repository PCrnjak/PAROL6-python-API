"""Verify that all example scripts run successfully in the simulator.

These tests run each example as an isolated subprocess so they don't
conflict with the shared integration test server.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

EXAMPLES = sorted(
    p.name for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_")
)

ENV = {
    **os.environ,
    "PAROL6_FAKE_SERIAL": "1",
    "PAROL6_STATUS_RATE_HZ": "20",
}


@pytest.mark.integration
@pytest.mark.parametrize("script", EXAMPLES)
def test_example_runs(script):
    """Run each example as a subprocess and check it exits cleanly."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)],
        env=ENV,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
