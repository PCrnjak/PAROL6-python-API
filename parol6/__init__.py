"""
PAROL6 Python Package

A unified library for controlling PAROL6 robot arms with async-first UDP client,
optional sync wrapper, and server management capabilities.

Key components:
- AsyncRobotClient: Async UDP client for robot operations
- RobotClient: Sync wrapper with automatic event loop handling
- ServerManager: Manages headless controller process lifecycle
- ensure_server: Convenience function to auto-start controller when needed
"""

from ._version import __version__
from .client.async_client import AsyncRobotClient
from .client.sync_client import RobotClient
from .client.manager import ServerManager, ensure_server
from . import PAROL6_ROBOT

__all__ = [
    "__version__",
    "AsyncRobotClient", 
    "RobotClient",
    "ServerManager",
    "ensure_server",
    "PAROL6_ROBOT"
]
