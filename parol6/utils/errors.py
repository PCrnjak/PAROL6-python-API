"""
Custom exception types for PAROL6 command/control pipeline.
Keep this focused and non-redundant; prefer built-ins where appropriate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .error_catalog import RobotError


class IKError(RuntimeError):
    """Inverse kinematics failure (no solution, constraints violated, etc.)."""

    def __init__(self, robot_error: RobotError):
        self.robot_error = robot_error
        super().__init__(str(robot_error))


class TrajectoryPlanningError(RuntimeError):
    """Trajectory generation/planning failure."""

    def __init__(self, robot_error: RobotError):
        self.robot_error = robot_error
        super().__init__(str(robot_error))


class MotionError(RuntimeError):
    """Pipeline planning/execution error detected via status broadcast."""

    def __init__(self, robot_error: RobotError):
        self.robot_error = robot_error
        super().__init__(str(robot_error))

    @property
    def command_index(self) -> int:
        return self.robot_error.command_index
