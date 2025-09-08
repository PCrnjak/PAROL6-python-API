"""
Utility Commands
Contains utility commands like Delay
"""

import logging
import time

logger = logging.getLogger(__name__)

class DelayCommand:
    """
    A non-blocking command that pauses execution for a specified duration.
    During the delay, it ensures the robot remains idle by sending the
    appropriate commands.
    """
    def __init__(self, duration):
        """
        Initializes and validates the Delay command.

        Args:
            duration (float): The delay time in seconds.
        """
        self.is_valid = False
        self.is_finished = False

        # --- 1. Parameter Validation ---
        if not isinstance(duration, (int, float)) or duration <= 0:
            logger.error(f"  -> VALIDATION FAILED: Delay duration must be a positive number, but got {duration}.")
            return

        logger.info(f"Initializing Delay for {duration} seconds...")
        
        self.duration = duration
        self.end_time = None  # Will be set in prepare_for_execution
        self.is_valid = True

    def prepare_for_execution(self, current_position_in):
        """Set the end time when the command actually starts."""
        self.end_time = time.time() + self.duration
        logger.info(f"  -> Delay starting for {self.duration} seconds...")

    def execute_step(self, Position_in, Homed_in, Speed_out, Command_out, **kwargs):
        """
        Checks if the delay duration has passed and keeps the robot idle.
        This method is called on every loop cycle (~0.01s).
        """
        if self.is_finished or not self.is_valid:
            return True

        # --- A. Keep the robot idle during the delay ---
        Command_out.value = 255  # Set command to idle
        Speed_out[:] = [0] * 6   # Set all speeds to zero

        # --- B. Check for completion ---
        if self.end_time and time.time() >= self.end_time:
            logger.info(f"Delay finished after {self.duration} seconds.")
            self.is_finished = True
        
        return self.is_finished