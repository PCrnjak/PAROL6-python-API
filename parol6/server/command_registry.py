"""
Command registration system with decorator support.

This module provides a centralized registry for all commands, enabling
auto-discovery and registration through decorators. This eliminates the
need for manual command factory maintenance.
"""

from __future__ import annotations

import logging
from typing import Dict, Type, Optional, List, Callable, Any
from importlib import import_module
import pkgutil

from parol6.commands.base import CommandBase

logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    Singleton registry for command classes.
    
    Commands register themselves using the @register_command decorator.
    The registry supports auto-discovery of decorated commands and
    provides a centralized lookup mechanism.
    """
    
    _instance: Optional[CommandRegistry] = None
    _commands: Dict[str, Type[CommandBase]] = {}
    _discovered: bool = False
    
    def __new__(cls) -> CommandRegistry:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the registry (only runs once due to singleton)."""
        if not hasattr(self, '_initialized'):
            self._commands = {}
            self._discovered = False
            self._initialized = True
    
    def register(self, name: str, command_class: Type[CommandBase]) -> None:
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
            logger.debug(f"Registered command '{name}' -> {command_class.__name__}")
    
    def get_command_class(self, name: str) -> Optional[Type[CommandBase]]:
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
    
    def list_registered_commands(self) -> List[str]:
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
            commands_package = import_module('parol6.commands')
        except ImportError as e:
            logger.error(f"Failed to import parol6.commands: {e}")
            return
        
        # Iterate through all modules in the commands package
        package_path = commands_package.__path__
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            if ispkg:
                continue  # Skip subpackages
            
            full_module_name = f'parol6.commands.{modname}'
            
            # Skip the base module
            if modname == 'base':
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
        logger.info(f"Command discovery complete. {len(self._commands)} commands registered.")
    
    def create_command_from_parts(self, parts: List[str]) -> Optional[CommandBase]:
        """
        Create a command instance from pre-split message parts.
        
        Args:
            parts: Pre-split message parts
            
        Returns:
            A command instance if a match is found, None otherwise
        """
        # Ensure commands are discovered
        if not self._discovered:
            self.discover_commands()
        
        if not parts:
            logger.debug("Empty message parts")
            return None
        
        command_name = parts[0].upper()
        
        # Direct O(1) lookup of command class
        command_class = self._commands.get(command_name)
        
        if command_class is None:
            logger.debug(f"No command registered for: {command_name}")
            return None
        
        try:
            # Create instance and let it parse parameters
            command = command_class()
            can_handle, error = command.match(parts)  # Pass pre-split parts
            
            if can_handle:
                return command
            elif error:
                logger.debug(f"Command '{command_name}' rejected: {error}")
                
        except Exception as e:
            logger.error(f"Error creating command '{command_name}': {e}")
        
        return None
    
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


def register_command(name: str) -> Callable[[Type[CommandBase]], Type[CommandBase]]:
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
    def decorator(cls: Type[CommandBase]) -> Type[CommandBase]:
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
