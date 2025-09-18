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
CONTROL_RATE_HZ: float = float(os.getenv("PAROL6_CONTROL_RATE_HZ", "100"))

# Velocity/acceleration safety margins
VELOCITY_SAFETY_SCALE: float = 1.2  # e.g., clamp at 1.2x of budget

# Centralized loop interval (seconds).
INTERVAL_S: float = max(1e-6, 1.0 / max(CONTROL_RATE_HZ, 1.0))

# Server/runtime defaults (overridable by env/CLI in headless commander)
SERVER_IP: str = "127.0.0.1"
SERVER_PORT: int = 5001
SERVER_STREAM_DEFAULT: bool = False
FAKE_SERIAL: bool = False
SERIAL_BAUD: int = 3_000_000
AUTO_HOME_DEFAULT: bool = True
LOG_LEVEL_DEFAULT: str = "INFO"

# COM port persistence file (absolute path at repository root)
# This ensures persistence works regardless of current working directory.
COM_PORT_FILE: str = str((Path(__file__).resolve().parents[3] / "serial_port.txt"))

# Multicast/broadcast status configuration (all overridable via env)
# These defaults implement local-only multicast on loopback by default.
MCAST_GROUP: str = os.getenv("PAROL6_MCAST_GROUP", "239.255.0.101")
MCAST_PORT: int = int(os.getenv("PAROL6_MCAST_PORT", "50510"))
MCAST_TTL: int = int(os.getenv("PAROL6_MCAST_TTL", "1"))
MCAST_IF: str = os.getenv("PAROL6_MCAST_IF", "127.0.0.1")

# Status update/broadcast rates
STATUS_RATE_HZ: float = float(os.getenv("PAROL6_STATUS_RATE_HZ", "50"))
STATUS_STALE_S: float = float(os.getenv("PAROL6_STATUS_STALE_S", "0.2"))

# Ack/Tracking policy toggles
def _env_bool_optional(name: str) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None

FORCE_ACK: Optional[bool] = _env_bool_optional("PAROL6_FORCE_ACK")


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
    Resolve COM port from environment or file.

    Priority:
      1) Environment variables: PAROL6_COM_PORT or PAROL6_SERIAL
      2) com_port.txt (if present and non-empty)

    Returns:
      Port string if available, otherwise an empty string "".
    """
    # 1) Environment variables
    env_port = os.getenv("PAROL6_COM_PORT") or os.getenv("PAROL6_SERIAL")
    if env_port and env_port.strip():
        port = env_port.strip()
        logger.info(f"Using COM port from environment: {port}")
        return port

    # 2) Persistence file
    saved_port = load_com_port()
    if saved_port:
        return saved_port

    return ""
