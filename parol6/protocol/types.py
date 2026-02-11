"""
Type definitions for PAROL6 protocol.

Defines enums and dataclasses used across the public API.
"""

from dataclasses import dataclass
from typing import Literal


# Frame literals
Frame = Literal["WRF", "TRF"]

# Axis literals (unsigned — direction is encoded in signed speed)
Axis = Literal["X", "Y", "Z", "RX", "RY", "RZ"]


@dataclass(slots=True, frozen=True)
class PingResult:
    """Parsed PING response."""

    serial_connected: bool
    raw: str
