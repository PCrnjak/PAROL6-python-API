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
    """Ask the kernel to send SIGTERM when the parent process exits (Linux only)."""
    if sys.platform != "linux":
        return
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
