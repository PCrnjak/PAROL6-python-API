"""
CLI entry point for parol6-server command.

This module provides the command-line interface for starting the PAROL6 headless controller.
"""

from parol6.server.controller import main


def main_entry():
    """Entry point for the parol6-server command."""
    main()


if __name__ == "__main__":
    main_entry()
