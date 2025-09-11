"""
Central configuration for PAROL6 tunables and shared constants.
"""

from __future__ import annotations

# IK / motion planning
# Iteration limit for jogging IK solves (kept conservative for speed while jogging)
JOG_IK_ILIMIT: int = 20

# Default control/sample rates (Hz)
CONTROL_RATE_HZ: float = 100.0

# Velocity/acceleration safety margins
VELOCITY_SAFETY_SCALE: float = 1.2  # e.g., clamp at 1.2x of budget
