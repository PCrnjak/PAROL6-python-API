"""Server management modules."""

import ctypes
import multiprocessing
import signal
import sys

# Use spawn method on all platforms to avoid fork issues with multi-threaded processes.
# This must be done before any multiprocessing is used. On Windows/macOS this is already
# the default, but on Linux it defaults to fork which causes warnings/deadlocks.
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")


def set_pdeathsig() -> None:
    """Arrange for SIGTERM when the parent process exits.

    Linux: instant kernel-level notification via prctl(PR_SET_PDEATHSIG).
    macOS/Windows: daemon thread polls parent liveness every second.
    """
    import os

    if sys.platform == "linux":
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        return

    import threading
    import time

    parent_pid = os.getppid()

    def _watch_parent() -> None:
        while True:
            time.sleep(1.0)
            if sys.platform == "win32":
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    break
            else:
                if os.getppid() != parent_pid:
                    break
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_watch_parent, daemon=True).start()
