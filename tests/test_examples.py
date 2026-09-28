"""Verify that all example scripts run successfully in the simulator.

These tests run each example as an isolated subprocess so they don't
conflict with the shared integration test server.
"""

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from parol6 import Robot

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

EXAMPLES = sorted(
    p.name for p in EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_")
)

# Written for a controller that is already running, as a Waldo Commander
# session runs one; the other examples start their own.
ATTACHED = {"demo_showcase.py", "precision.py"}

ENV = {
    **os.environ,
    "PAROL6_FAKE_SERIAL": "1",
    "PAROL6_STATUS_RATE_HZ": "20",
}

# What the controller logs for a command it failed or refused in its turn,
# and a client raising it: an example that exits 0 without having waited on
# the failure still reports it here.
FAILURE = re.compile(r"Command \d+ failed|Inline command failed|MotionError")


@pytest.mark.examples
@pytest.mark.timeout(300)
@pytest.mark.parametrize("script", EXAMPLES)
def test_example_runs(script, ports, monkeypatch, caplog):
    """Run each example as a subprocess and check it exits cleanly, with
    every command it sent carried out."""
    # Windows can reserve the default status port even with no listener.
    # The subprocess and its controller share the OS-probed test port.
    env = {**ENV, "PAROL6_MCAST_PORT": str(ports.mcast_port)}
    with contextlib.ExitStack() as stack:
        if script in ATTACHED:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
            # Its log reaches this process's logging, where caplog reads it.
            stack.enter_context(Robot(host="127.0.0.1", port=5001, normalize_logs=True))
        result = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    # An example that starts its own controller relays its log to stderr.
    output = "\n".join((result.stdout, result.stderr, caplog.text))
    failures = [line for line in output.splitlines() if FAILURE.search(line)]
    assert not failures, (
        f"{script} exited 0, but commands it sent failed:\n" + "\n".join(failures[:20])
    )
