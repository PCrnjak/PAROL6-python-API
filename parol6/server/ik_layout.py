"""Shared memory layout constants for the IK enablement pipeline.

Both ``status_cache`` (writer) and ``ik_worker`` (reader) must agree on
the byte offsets and sizes.  Keeping them in one place avoids drift.
"""

import sys
from multiprocessing.shared_memory import SharedMemory

# ── Input buffer (status_cache → ik_worker) ──────────────────────
IK_INPUT_Q_OFFSET = 0  # float64[6]  = 48 bytes
IK_INPUT_T_OFFSET = 48  # float64[16] = 128 bytes
IK_INPUT_TOOL_OFFSET = 176  # float64[16] = 128 bytes (4x4 tool transform)
IK_INPUT_SIZE = 304

# ── Output buffer (ik_worker → status_cache) ─────────────────────
IK_OUTPUT_JOINT_OFFSET = 0  # uint8[12] = 12 bytes
IK_OUTPUT_CART_WRF_OFFSET = 12  # uint8[12] = 12 bytes
IK_OUTPUT_CART_TRF_OFFSET = 24  # uint8[12] = 12 bytes
IK_OUTPUT_VERSION_OFFSET = 36  # uint64    = 8 bytes
IK_OUTPUT_SIZE = 44

# ── SharedMemory tracking ────────────────────────────────────────
# Python 3.13+ supports track=False to skip the resource_tracker daemon.
# On pre-3.13, we must manually unregister to prevent orphaned tracker processes.
SHM_EXTRA_KWARGS: dict = {"track": False} if sys.version_info >= (3, 13) else {}


def unregister_shm(shm: SharedMemory) -> None:
    """Unregister a SharedMemory segment from the resource_tracker (pre-3.13).

    On Python 3.13+ with track=False this is a no-op. On older versions, this
    prevents the resource_tracker daemon from lingering as an orphan process.
    Skipped on Windows where the resource_tracker uses _posixsubprocess.
    """
    if sys.version_info >= (3, 13) or sys.platform == "win32":
        return
    from multiprocessing.resource_tracker import unregister

    unregister("/" + shm.name, "shared_memory")
