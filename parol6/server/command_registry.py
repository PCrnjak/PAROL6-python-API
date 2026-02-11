"""
Command registration system with decorator support.

This module provides a centralized registry for all commands, enabling
auto-discovery and registration through decorators. This eliminates the
need for manual command factory maintenance.
"""

from __future__ import annotations

import logging
import pkgutil
from collections.abc import Callable
from enum import Enum
from importlib import import_module

import msgspec

from parol6.commands.base import CommandBase, QueryCommand, SystemCommand
from parol6.config import TRACE
from parol6.protocol.wire import (
    CmdType,
    Command,
    decode_command,
)


class CommandCategory(Enum):
    """Category of command, determining execution semantics."""

    SYSTEM = "system"  # Execute immediately, controller sends OK/error
    QUERY = "query"  # Execute immediately, controller sends query response
    MOTION = "motion"  # Queue for execution in control loop


logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    Singleton registry for command classes.

    Commands register themselves using the @register_command decorator.
    The registry supports auto-discovery of decorated commands and
    provides a centralized lookup mechanism.

    Supports single-pass decode via msgspec typed structs.
    """

    _instance: CommandRegistry | None = None
    _commands: dict[CmdType, type[CommandBase]]
    _class_to_type: dict[type[CommandBase], CmdType]
    _struct_to_command: dict[type[Command], type[CommandBase]]
    _class_to_category: dict[type[CommandBase], CommandCategory]
    _discovered: bool

    def __new__(cls) -> CommandRegistry:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry (only runs once due to singleton)."""
        if not hasattr(self, "_initialized"):
            self._commands = {}
            self._class_to_type = {}
            self._struct_to_command = {}
            self._class_to_category = {}
            self._discovered = False
            self._initialized = True

    @staticmethod
    def _determine_category(command_class: type[CommandBase]) -> CommandCategory:
        """Determine the category of a command class based on its inheritance."""
        if issubclass(command_class, SystemCommand):
            return CommandCategory.SYSTEM
        elif issubclass(command_class, QueryCommand):
            return CommandCategory.QUERY
        else:
            return CommandCategory.MOTION

    def register(self, cmd_type: CmdType, command_class: type[CommandBase]) -> None:
        """
        Register a command class with the given CmdType.

        Args:
            cmd_type: The command type enum value
            command_class: The command class to register

        Raises:
            ValueError: If a command with the same type is already registered
        """
        if cmd_type in self._commands:
            existing = self._commands[cmd_type]
            if existing != command_class:
                raise ValueError(
                    f"Command {cmd_type.name} is already registered with class {existing.__name__}. "
                    f"Cannot register with {command_class.__name__}"
                )
        else:
            self._commands[cmd_type] = command_class
            # Maintain reverse mapping for fast class -> type lookup
            self._class_to_type[command_class] = cmd_type

            # Determine and store category at registration time
            category = self._determine_category(command_class)
            self._class_to_category[command_class] = category

            # Also register by struct type if PARAMS_TYPE is set
            params_type = getattr(command_class, "PARAMS_TYPE", None)
            if params_type is not None:
                self._struct_to_command[params_type] = command_class

            logger.debug(
                f"Registered command {cmd_type.name} -> {command_class.__name__} ({category.value})"
            )

    def get_command_class(self, cmd_type: CmdType) -> type[CommandBase] | None:
        """
        Retrieve a command class by CmdType.

        Args:
            cmd_type: The command type to look up

        Returns:
            The command class if found, None otherwise
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()

        return self._commands.get(cmd_type)

    def get_type_for_class(self, cls: type[CommandBase]) -> CmdType | None:
        """
        Retrieve the registered CmdType for a given command class.
        Returns None if the class is not registered.
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()
        # Prefer explicit reverse map; fall back to class attribute set by decorator
        return self._class_to_type.get(cls) or getattr(cls, "_cmd_type", None)

    def list_registered_commands(self) -> list[CmdType]:
        """
        Return a list of all registered command types.

        Returns:
            List of CmdType values (sorted by int value)
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()

        return sorted(self._commands.keys(), key=lambda x: int(x))

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

    def get_command_for_struct(
        self, struct_type: type[Command]
    ) -> type[CommandBase] | None:
        """Return the command class registered for a given struct type.

        Unlike create_command_from_struct(), this does not instantiate the
        command or assign parameters — it just returns the class.
        """
        if not self._discovered:
            self.discover_commands()
        return self._struct_to_command.get(struct_type)

    def get_category(self, command_class: type[CommandBase]) -> CommandCategory:
        """Return the category for a registered command class."""
        if not self._discovered:
            self.discover_commands()
        return self._class_to_category.get(command_class, CommandCategory.MOTION)

    def create_command(
        self, data: bytes
    ) -> tuple[CommandBase | None, CommandCategory | None, str | None]:
        """
        Create a command instance from raw bytes via single-pass msgspec decode.

        Args:
            data: Raw msgpack-encoded command bytes

        Returns:
            A tuple of (command, category, error_message):
            - (command, category, None) if successful
            - (None, None, error_message) if decode fails or validation fails
        """
        try:
            cmd_struct = decode_command(data)
        except msgspec.ValidationError as e:
            logger.log(TRACE, "decode_error err=%s", e)
            return None, None, str(e)
        except Exception as e:
            logger.log(TRACE, "decode_error exc=%s", e)
            return None, None, f"Decode error: {e}"

        return self.create_command_from_struct(cmd_struct)

    def create_command_from_struct(
        self, cmd_struct: Command
    ) -> tuple[CommandBase | None, CommandCategory | None, str | None]:
        """
        Create a command instance from a pre-validated Command struct.

        Args:
            cmd_struct: Pre-validated Command struct (e.g., MovePoseCmd, HomeCmd)

        Returns:
            A tuple of (command, category, error_message):
            - (command, category, None) if successful
            - (None, None, error_message) if no handler found
        """
        if not self._discovered:
            self.discover_commands()

        struct_type = type(cmd_struct)
        command_class = self._struct_to_command.get(struct_type)
        if command_class is None:
            logger.log(TRACE, "no_handler struct=%s", struct_type.__name__)
            return None, None, f"No handler for {struct_type.__name__}"

        command = command_class(cmd_struct)
        category = self._class_to_category.get(command_class, CommandCategory.MOTION)

        return command, category, None

    def clear(self) -> None:
        """
        Clear all registered commands.

        This is mainly useful for testing.
        """
        self._commands.clear()
        self._class_to_type.clear()
        self._struct_to_command.clear()
        self._class_to_category.clear()
        self._discovered = False
        logger.debug("Command registry cleared")


# Global registry instance
_registry = CommandRegistry()


def register_command(
    cmd_type: CmdType,
) -> Callable[[type[CommandBase]], type[CommandBase]]:
    """
    Decorator to register a command class.

    Usage:
        @register_command(CmdType.MOVEJOINT)
        class MoveJCommand(CommandBase):
            ...

    Args:
        cmd_type: The command type enum value

    Returns:
        Decorator function that registers the class
    """

    def decorator(cls: type[CommandBase]) -> type[CommandBase]:
        # Verify it's a CommandBase subclass
        if not issubclass(cls, CommandBase):
            raise TypeError(f"Class {cls.__name__} must inherit from CommandBase")

        # Register with the global registry
        _registry.register(cmd_type, cls)

        # Add the command type as a class attribute for reference
        cls._cmd_type = cmd_type

        return cls

    return decorator


# Module-level convenience functions that delegate to the registry singleton
get_command_class = _registry.get_command_class
list_registered_commands = _registry.list_registered_commands
discover_commands = _registry.discover_commands
clear_registry = _registry.clear
create_command = _registry.create_command
create_command_from_struct = _registry.create_command_from_struct
get_type_for_class = _registry.get_type_for_class
get_command_for_struct = _registry.get_command_for_struct
