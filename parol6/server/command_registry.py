"""
Command registration system with decorator support.

This module provides a centralized registry for all commands, enabling
auto-discovery and registration through decorators. This eliminates the
need for manual command factory maintenance.
"""

from __future__ import annotations

import logging
import pkgutil
import time
from collections.abc import Callable
from importlib import import_module

from parol6.commands.base import CommandBase
from parol6.config import TRACE

logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    Singleton registry for command classes.

    Commands register themselves using the @register_command decorator.
    The registry supports auto-discovery of decorated commands and
    provides a centralized lookup mechanism.
    """

    _instance: CommandRegistry | None = None
    _commands: dict[str, type[CommandBase]] = {}
    _class_to_name: dict[type[CommandBase], str] = {}
    _discovered: bool = False

    def __new__(cls) -> CommandRegistry:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry (only runs once due to singleton)."""
        if not hasattr(self, "_initialized"):
            self._commands = {}
            self._class_to_name = {}
            self._discovered = False
            self._initialized = True

    def register(self, name: str, command_class: type[CommandBase]) -> None:
        """
        Register a command class with the given name.

        Args:
            name: The command name/identifier
            command_class: The command class to register

        Raises:
            ValueError: If a command with the same name is already registered
        """
        if name in self._commands:
            existing = self._commands[name]
            if existing != command_class:
                raise ValueError(
                    f"Command '{name}' is already registered with class {existing.__name__}. "
                    f"Cannot register with {command_class.__name__}"
                )
        else:
            self._commands[name] = command_class
            # Maintain reverse mapping for fast class -> name lookup
            self._class_to_name[command_class] = name
            logger.debug(f"Registered command '{name}' -> {command_class.__name__}")

    def get_command_class(self, name: str) -> type[CommandBase] | None:
        """
        Retrieve a command class by name.

        Args:
            name: The command name to look up

        Returns:
            The command class if found, None otherwise
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()

        return self._commands.get(name)

    def get_name_for_class(self, cls: type[CommandBase]) -> str | None:
        """
        Retrieve the registered command name for a given command class.
        Returns None if the class is not registered.
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()
        # Prefer explicit reverse map; fall back to class attribute set by decorator
        return self._class_to_name.get(cls) or getattr(cls, "_registered_name", None)

    def list_registered_commands(self) -> list[str]:
        """
        Return a list of all registered command names.

        Returns:
            List of command names (sorted)
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()

        return sorted(self._commands.keys())

    def discover_commands(self) -> None:
        """
        Auto-discover and register all decorated commands.

        This method imports all modules in the parol6.commands package
        to trigger the @register_command decorators.
        """
        if self._discovered:
            return

        logger.info("Discovering commands...")

        # Import parol6.commands package
        try:
            commands_package = import_module("parol6.commands")
        except ImportError as e:
            logger.error(f"Failed to import parol6.commands: {e}")
            return

        # Iterate through all modules in the commands package
        package_path = commands_package.__path__
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            if ispkg:
                continue  # Skip subpackages

            full_module_name = f"parol6.commands.{modname}"

            # Skip the base module
            if modname == "base":
                continue

            try:
                # Import the module (this triggers decorators)
                import_module(full_module_name)
                logger.debug(f"Imported command module: {full_module_name}")
            except ImportError as e:
                logger.warning(f"Failed to import {full_module_name}: {e}")
            except Exception as e:
                logger.error(f"Error loading {full_module_name}: {e}")

        self._discovered = True
        logger.info(
            f"Command discovery complete. {len(self._commands)} commands registered."
        )

    def create_command_from_parts(
        self, parts: list[str]
    ) -> tuple[CommandBase | None, str | None]:
        """
        Create a command instance from pre-split message parts.

        Args:
            parts: Pre-split message parts

        Returns:
            A tuple of (command, error_message):
            - (command, None) if successful
            - (None, None) if command name not registered
            - (None, error_message) if command is recognized but has invalid parameters
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()

        if not parts:
            logger.debug("Empty message parts")
            return None, None

        command_name = parts[0].upper()
        start_t = time.perf_counter()
        logger.log(TRACE, "match_start name=%s parts=%d", command_name, len(parts))

        # Direct O(1) lookup of command class
        command_class = self._commands.get(command_name)

        if command_class is None:
            logger.log(TRACE, "match_unknown name=%s", command_name)
            logger.debug(f"No command registered for: {command_name}")
            return None, None

        try:
            # Create instance and let it parse parameters
            command = command_class()
            can_handle, error = command.match(parts)  # Pass pre-split parts

            if can_handle:
                dur_ms = (time.perf_counter() - start_t) * 1000.0
                logger.log(TRACE, "match_ok name=%s dur_ms=%.2f", command_name, dur_ms)
                return command, None
            elif error:
                dur_ms = (time.perf_counter() - start_t) * 1000.0
                logger.log(
                    TRACE,
                    "match_error name=%s dur_ms=%.2f err=%s",
                    command_name,
                    dur_ms,
                    error,
                )
                logger.warning(f"Command '{command_name}' rejected: {error}")
                return None, error

        except Exception as e:
            dur_ms = (time.perf_counter() - start_t) * 1000.0
            logger.log(
                TRACE, "match_error name=%s dur_ms=%.2f exc=%s", command_name, dur_ms, e
            )
            logger.error(f"Error creating command '{command_name}': {e}")
            return None, str(e)

        return None, "Command validation failed"

    def clear(self) -> None:
        """
        Clear all registered commands.

        This is mainly useful for testing.
        """
        self._commands.clear()
        self._discovered = False
        logger.debug("Command registry cleared")


# Global registry instance
_registry = CommandRegistry()


def register_command(name: str) -> Callable[[type[CommandBase]], type[CommandBase]]:
    """
    Decorator to register a command class.

    Usage:
        @register_command("MoveJ")
        class MoveJointCommand(CommandBase):
            ...

    Args:
        name: The command name/identifier

    Returns:
        Decorator function that registers the class
    """

    def decorator(cls: type[CommandBase]) -> type[CommandBase]:
        # Verify it's a CommandBase subclass
        if not issubclass(cls, CommandBase):
            raise TypeError(f"Class {cls.__name__} must inherit from CommandBase")

        # Register with the global registry
        _registry.register(name, cls)

        # Add the command name as a class attribute for reference
        cls._registered_name = name

        return cls

    return decorator


# Module-level convenience functions that delegate to the registry singleton
get_command_class = _registry.get_command_class
list_registered_commands = _registry.list_registered_commands
discover_commands = _registry.discover_commands
clear_registry = _registry.clear
create_command_from_parts = _registry.create_command_from_parts
get_name_for_class = _registry.get_name_for_class
