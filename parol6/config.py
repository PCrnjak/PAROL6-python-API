"""
Central configuration for PAROL6 tunables and shared constants.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# IK / motion planning
# Iteration limit for jogging IK solves (kept conservative for speed while jogging)
JOG_IK_ILIMIT: int = 20

# Default control/sample rates (Hz)
CONTROL_RATE_HZ: float = 100.0

# Velocity/acceleration safety margins
VELOCITY_SAFETY_SCALE: float = 1.2  # e.g., clamp at 1.2x of budget

# Centralized loop interval (seconds).
INTERVAL_S: float = 0.01

# Server/runtime defaults (overridable by env/CLI in headless commander)
SERVER_IP: str = "127.0.0.1"
SERVER_PORT: int = 5001
SERVER_STREAM_DEFAULT: bool = False
FAKE_SERIAL: bool = False
SERIAL_BAUD: int = 3_000_000
AUTO_HOME_DEFAULT: bool = True
LOG_LEVEL_DEFAULT: str = "INFO"

# Command processing cooldown (milliseconds)
COMMAND_COOLDOWN_MS: int = 10

# COM port persistence file
COM_PORT_FILE: str = "com_port.txt"


def save_com_port(port: str) -> bool:
    """
    Save COM port to persistent file.
    
    Args:
        port: COM port string to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        com_port_path = Path(COM_PORT_FILE)
        com_port_path.write_text(port.strip())
        logger.info(f"Saved COM port {port} to {COM_PORT_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save COM port: {e}")
        return False


def load_com_port() -> Optional[str]:
    """
    Load saved COM port from file.
    
    Returns:
        COM port string if found, None otherwise
    """
    try:
        com_port_path = Path(COM_PORT_FILE)
        if com_port_path.exists():
            port = com_port_path.read_text().strip()
            if port:
                logger.info(f"Loaded COM port {port} from {COM_PORT_FILE}")
                return port
    except Exception as e:
        logger.error(f"Failed to load COM port: {e}")
    return None


def get_com_port_with_fallback() -> str:
    """
    Load COM port from file or prompt user for input.
    
    Returns:
        COM port string
    """
    # First try to load from file
    saved_port = load_com_port()
    if saved_port:
        # Prompt user to confirm or change
        print(f"Found saved COM port: {saved_port}")
        response = input("Press Enter to use this port, or type a new port: ").strip()
        if response:
            # User entered a new port
            port = response
            save_com_port(port)
        else:
            # User accepted saved port
            port = saved_port
    else:
        # No saved port, prompt for input
        import platform
        
        if platform.system() == "Windows":
            default_prompt = "Enter COM port (e.g., COM3): "
        else:
            default_prompt = "Enter COM port (e.g., /dev/ttyUSB0): "
        
        port = input(default_prompt).strip()
        if not port:
            # Use a default based on platform
            if platform.system() == "Windows":
                port = "COM3"
            else:
                port = "/dev/ttyUSB0"
            print(f"Using default port: {port}")
        
        # Save the port for next time
        save_com_port(port)
    
    return port
