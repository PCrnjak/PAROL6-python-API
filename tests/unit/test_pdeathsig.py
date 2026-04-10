"""Test that child processes exit when parent is killed (dirty shutdown).

Verifies that set_pdeathsig() causes multiprocessing.Process children to
receive SIGTERM when their parent is SIGKILL'd, preventing orphaned
resource_tracker processes.
"""

import os
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.unit
@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL not available on Windows")
def test_child_exits_on_parent_death(tmp_path):
    """Children calling set_pdeathsig() exit when parent is SIGKILL'd."""
    # Child target — must be a separate module for spawn to import
    (tmp_path / "_pdeathsig_child.py").write_text(
        "import os, signal, time\n"
        "def child_main(pid_file):\n"
        "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "    from parol6.server import set_pdeathsig\n"
        "    set_pdeathsig()\n"
        "    with open(pid_file, 'w') as f: f.write(str(os.getpid()))\n"
        "    while True: time.sleep(0.1)\n"
    )

    # Parent script — spawns child then sleeps forever
    pid_file = str(tmp_path / "child.pid")
    (tmp_path / "_pdeathsig_parent.py").write_text(
        "import multiprocessing, os, sys, time\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from _pdeathsig_child import child_main\n"
        "if __name__ == '__main__':\n"
        "    multiprocessing.set_start_method('spawn', force=True)\n"
        f"    p = multiprocessing.Process(target=child_main, args=({pid_file!r},), daemon=True)\n"
        "    p.start()\n"
        "    sys.stdout.write(str(os.getpid()) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    while True: time.sleep(60)\n"
    )

    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / "_pdeathsig_parent.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Read parent PID
    assert proc.stdout is not None
    line = proc.stdout.readline().decode().strip()
    assert line, (
        f"Parent failed to start. stderr: {proc.stderr.read().decode()[:500] if proc.stderr else ''}"
    )
    parent_pid = int(line)

    # Wait for child to write its PID
    deadline = time.monotonic() + 10.0
    child_pid = None
    while time.monotonic() < deadline:
        if os.path.exists(pid_file):
            content = open(pid_file).read().strip()
            if content:
                child_pid = int(content)
                break
        time.sleep(0.2)
    assert child_pid is not None, "Child never wrote its PID"

    # Both should be alive
    os.kill(parent_pid, 0)
    os.kill(child_pid, 0)

    # Kill parent (dirty shutdown)
    os.kill(parent_pid, signal.SIGKILL)
    proc.wait()

    # Child should exit within a few seconds
    deadline = time.monotonic() + 5.0
    child_alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except OSError:
            child_alive = False
            break
        time.sleep(0.1)

    if child_alive:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("Child process did not exit within 5s after parent was killed")

    # Verify no freshly-orphaned resource_tracker processes
    time.sleep(0.5)
    result = subprocess.run(
        ["pgrep", "-f", "resource_tracker"], capture_output=True, text=True
    )
    for pid_str in result.stdout.strip().splitlines():
        if not pid_str.strip():
            continue
        tracker_pid = int(pid_str.strip())
        try:
            with open(f"/proc/{tracker_pid}/status") as f:
                for status_line in f:
                    if status_line.startswith("PPid:"):
                        ppid = int(status_line.split()[1])
                        if ppid == 1:
                            age = time.time() - os.stat(f"/proc/{tracker_pid}").st_ctime
                            if age < 10:
                                os.kill(tracker_pid, signal.SIGKILL)
                                pytest.fail(
                                    f"Orphaned resource_tracker (PID={tracker_pid}) "
                                    f"found {age:.1f}s after test"
                                )
                        break
        except (FileNotFoundError, ProcessLookupError):
            pass
