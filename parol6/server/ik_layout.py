"""Shared memory layout constants for the IK enablement pipeline.

Both ``status_cache`` (writer) and ``ik_worker`` (reader) must agree on
the byte offsets and sizes.  Keeping them in one place avoids drift.
"""

# ── Input buffer (status_cache → ik_worker) ──────────────────────
IK_INPUT_Q_OFFSET = 0  # float64[6]  = 48 bytes
IK_INPUT_T_OFFSET = 48  # float64[16] = 128 bytes
IK_INPUT_SIZE = 176

# ── Output buffer (ik_worker → status_cache) ─────────────────────
IK_OUTPUT_JOINT_OFFSET = 0  # uint8[12] = 12 bytes
IK_OUTPUT_CART_WRF_OFFSET = 12  # uint8[12] = 12 bytes
IK_OUTPUT_CART_TRF_OFFSET = 24  # uint8[12] = 12 bytes
IK_OUTPUT_VERSION_OFFSET = 36  # uint64    = 8 bytes
IK_OUTPUT_SIZE = 44
