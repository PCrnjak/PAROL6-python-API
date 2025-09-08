"""
Sync wrapper for AsyncRobotClient using telethon-style syncify.

This module rewrites all public methods in the AsyncRobotClient
so they can run the loop on their own if it's not already running. This 
rewrite allows for quick scripts while maintaining async performance when
used in async contexts.
"""

import asyncio
import functools
import inspect
from typing import TYPE_CHECKING

from .async_client import AsyncRobotClient

if TYPE_CHECKING:
    pass


def _get_running_loop():
    """Get the currently running event loop or create one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def _syncify_wrap(cls, method_name):
    """Wrap an async method to work in both sync and async contexts."""
    method = getattr(cls, method_name)

    @functools.wraps(method)
    def syncified(self, *args, **kwargs):
        coro = method(self, *args, **kwargs)
        try:
            loop = asyncio.get_running_loop()
            # Loop is running, return the coroutine
            return coro
        except RuntimeError:
            # No running loop, create one and run to completion
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    # Save an accessible reference to the original method
    setattr(syncified, '__async_method__', method)
    setattr(cls, method_name, syncified)


def syncify(*classes):
    """
    Convert all async methods in the given classes into synchronous methods
    that return either the coroutine or the result based on whether 
    asyncio's event loop is running.
    """
    for cls in classes:
        for name in dir(cls):
            if not name.startswith('_') or name == '__call__':
                attr = getattr(cls, name)
                if inspect.iscoroutinefunction(attr):
                    _syncify_wrap(cls, name)


class RobotClient(AsyncRobotClient):
    """
    Synchronous robot client with automatic event loop handling.
    
    This class inherits from AsyncRobotClient and applies syncify
    to all async methods. When called:
    - If an event loop is running: returns the coroutine (async behavior)
    - If no event loop is running: runs the coroutine and returns the result
    
    This allows the same client to work seamlessly in both sync and async contexts:
    
    # Sync usage (no event loop)
    client = RobotClient()
    angles = client.get_angles()  # Automatically runs async method
    
    # Async usage (with event loop)
    client = RobotClient()
    angles = await client.get_angles()  # Returns coroutine
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Apply syncify to RobotClient
syncify(RobotClient)
