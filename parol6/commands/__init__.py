"""
Commands package for PAROL6.

Important:
- Do NOT import individual command modules here to avoid circular imports.
- The controller uses dynamic discovery via CommandRegistry.discover_commands()
  which imports command modules at runtime to trigger @register_command decorators.
"""

# Re-export non-problematic IK helpers for convenience
from parol6.utils.ik import (
    CommandValue,
    normalize_angle,
    unwrap_angles,
    calculate_adaptive_tolerance,
    solve_ik_with_adaptive_tol_subdivision,
    quintic_scaling,
    AXIS_MAP,
)

__all__ = [
    "CommandValue",
    "normalize_angle",
    "unwrap_angles",
    "calculate_adaptive_tolerance",
    "solve_ik_with_adaptive_tol_subdivision",
    "quintic_scaling",
    "AXIS_MAP",
]
