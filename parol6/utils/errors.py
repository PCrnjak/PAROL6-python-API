"""
Custom exception types for PAROL6 command/control pipeline.
Keep this focused and non-redundant; prefer built-ins where appropriate.
"""


class IKError(RuntimeError):
    """Inverse kinematics failure (no solution, constraints violated, etc.)."""

    def __init__(self, message: str):
        self.original_message = message
        super().__init__(f"IK ERROR: {message}")

    def __str__(self):
        return f"IK ERROR: {self.original_message}"


class TrajectoryPlanningError(RuntimeError):
    """Trajectory generation/planning failure."""

    def __init__(self, message: str):
        self.original_message = message
        super().__init__(f"Trajectory Planning Error: {message}")

    def __str__(self):
        return f"Trajectory Planning Error: {self.original_message}"
