"""
Custom exception types for PAROL6 command/control pipeline.
Keep this focused and non-redundant; prefer built-ins where appropriate.
"""

class IKError(RuntimeError):
    """Inverse kinematics failure (no solution, constraints violated, etc.)."""
    pass


class TrajectoryPlanningError(RuntimeError):
    """Trajectory generation/planning failure."""
    pass
