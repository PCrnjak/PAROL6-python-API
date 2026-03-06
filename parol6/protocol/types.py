"""
Type definitions for PAROL6 protocol.

Defines enums and dataclasses used across the public API.
"""

from typing import Literal


# Frame literals
Frame = Literal["WRF", "TRF"]

# Axis literals (unsigned — direction is encoded in signed speed)
Axis = Literal["X", "Y", "Z", "RX", "RY", "RZ"]
