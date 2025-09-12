"""
Base abstractions and helpers for command implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod

import parol6.PAROL6_ROBOT as PAROL6_ROBOT

@dataclass(eq=False)
class CommandBase(ABC):
    """
    Minimal reusable base for commands with shared lifecycle and safety helpers.
    """
    is_valid: bool = True
    is_finished: bool = False
    error_state: bool = False
    error_message: str = ""
 
    # Ensure command objects are usable as dict keys (e.g., in server command_id_map)
    def __hash__(self) -> int:
        # Identity-based hash is appropriate for ephemeral command instances
        return id(self)

    # ----- contract -----
    @abstractmethod
    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs) -> bool:
        """
        Execute one control-loop step. Return True when the command has finished.
        """
        raise NotImplementedError

    def prepare_for_execution(self, current_position_in) -> None:
        """
        Optional: prepare using current robot state (e.g., compute trajectory).
        Default is a no-op.
        """
        return
 
    # ----- lifecycle helpers -----

    def finish(self) -> None:
        """Mark command as finished."""
        self.is_finished = True

    def fail(self, message: str) -> None:
        """Mark command as invalid/failed with an error message."""
        self.is_valid = False
        self.error_state = True
        self.error_message = message
        self.is_finished = True

    # ----- safety / limits helpers -----

    @staticmethod
    def within_limits(joint_index: int, position_steps: int) -> bool:
        """
        Check if a joint position (in steps) is within configured limits.
        """
        min_limit, max_limit = PAROL6_ROBOT.Joint_limits_steps[joint_index]
        return min_limit <= position_steps <= max_limit

    @staticmethod
    def clamp_to_limits(joint_index: int, position_steps: int) -> int:
        """
        Clamp a joint position (in steps) to configured limits.
        """
        min_limit, max_limit = PAROL6_ROBOT.Joint_limits_steps[joint_index]
        return max(min(position_steps, max_limit), min_limit)

    @staticmethod
    def joint_dir_and_index(joint_selector: int) -> Tuple[int, int]:
        """
        Convert "jog selector" (0..5 positive, 6..11 negative for reverse) into
        (direction, joint_index) where direction is +1 or -1 and joint_index is 0..5.
        """
        direction = 1 if 0 <= joint_selector <= 5 else -1
        joint_index = joint_selector if direction == 1 else joint_selector - 6
        return direction, joint_index
