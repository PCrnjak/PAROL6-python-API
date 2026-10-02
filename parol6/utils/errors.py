"""
Custom exception types for PAROL6 command/control pipeline.
Keep this focused and non-redundant; prefer built-ins where appropriate.
"""

from __future__ import annotations

from waldoctl.errors import RobotError


class IKError(RuntimeError):
    """Inverse kinematics failure (no solution, constraints violated, etc.)."""

    def __init__(self, robot_error: RobotError):
        self.robot_error = robot_error
        super().__init__(str(robot_error))


class TrajectoryPlanningError(RuntimeError):
    """Trajectory generation/planning failure."""

    def __init__(self, robot_error: RobotError):
        self.robot_error = robot_error
        # Structured self-collision pairs for the viz, set by the collision guard.
        self.colliding_pairs: list[tuple[str, str]] | None = None
        super().__init__(str(robot_error))


class MotionError(RobotError):
    """Pipeline planning/execution error detected via status broadcast or a
    completion: the runtime's :class:`RobotError`, raised as this client's
    own type so ``except RobotError`` reads it on every backend."""

    def __init__(self, robot_error: RobotError):
        self.robot_error = robot_error
        super().__init__(
            robot_error.command_index,
            robot_error.code,
            robot_error.title,
            robot_error.cause,
            robot_error.effect,
            robot_error.remedy,
        )

    def __reduce__(self) -> tuple:
        return (type(self), (self.robot_error,))
