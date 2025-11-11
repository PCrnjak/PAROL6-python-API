"""
Central configuration for PAROL6 tunables and shared constants.
"""

import logging
import os
from pathlib import Path

TRACE: int = 5
logging.addLevelName(TRACE, "TRACE")
# Add Logger.trace if missing
if not hasattr(logging.Logger, "trace"):

    def _trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    logging.Logger.trace = _trace  # type: ignore[attr-defined]
    logging.TRACE = TRACE  # type: ignore[attr-defined]

TRACE_ENABLED = str(os.getenv("PAROL_TRACE", "0")).lower() in ("1", "true", "yes", "on")

logger = logging.getLogger(__name__)

# Default control/sample rates (Hz)
CONTROL_RATE_HZ: float = float(os.getenv("PAROL6_CONTROL_RATE_HZ", "250"))

# Velocity/acceleration safety margins
VELOCITY_SAFETY_SCALE: float = 1.2  # e.g., clamp at 1.2x of budget

DEFAULT_ACCEL_PERCENT: float = 50.0

# Motion thresholds (mm)
NEAR_MM_TOL_MM: float = 2.0  # Proximity threshold for considering positions "near" (mm)
ENTRY_MM_TOL_MM: float = 5.0  # Entry trajectory threshold for smooth motion (mm)

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

# COM port persistence file stored in user config directory by default (cross-platform).
_default_com_file = Path.home() / ".parol6" / "com_port.txt"
COM_PORT_FILE: str = os.getenv("PAROL6_COM_FILE", str(_default_com_file))

# Multicast/broadcast status configuration (all overridable via env)
# These defaults implement local-only multicast on loopback by default.
MCAST_GROUP: str = os.getenv("PAROL6_MCAST_GROUP", "239.255.0.101")
MCAST_PORT: int = int(os.getenv("PAROL6_MCAST_PORT", "50510"))
MCAST_TTL: int = int(os.getenv("PAROL6_MCAST_TTL", "1"))
MCAST_IF: str = os.getenv("PAROL6_MCAST_IF", "127.0.0.1")

# Status update/broadcast rates
STATUS_RATE_HZ: float = float(os.getenv("PAROL6_STATUS_RATE_HZ", "50"))
STATUS_STALE_S: float = float(os.getenv("PAROL6_STATUS_STALE_S", "0.2"))


# Homing posture (degrees) for simulation/tests; can be overridden via env "PAROL6_HOME_ANGLES_DEG" (CSV)
def _parse_home_angles() -> list[float]:
    raw = os.getenv("PAROL6_HOME_ANGLES_DEG")
    if not raw:
        return [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
    try:
        parts = [p.strip() for p in raw.split(",")]
        vals = [float(p) for p in parts]
        # Ensure length 6
        if len(vals) != 6:
            return [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]
        return vals
    except Exception:
        return [90.0, -90.0, 180.0, 0.0, 0.0, 180.0]


HOME_ANGLES_DEG: list[float] = _parse_home_angles()


# Ack/Tracking policy toggles
def _env_bool_optional(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


FORCE_ACK: bool | None = _env_bool_optional("PAROL6_FORCE_ACK")


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
        com_port_path.parent.mkdir(parents=True, exist_ok=True)
        com_port_path.write_text(port.strip())
        logger.info(f"Saved COM port {port} to {COM_PORT_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save COM port: {e}")
        return False


def load_com_port() -> str | None:
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
