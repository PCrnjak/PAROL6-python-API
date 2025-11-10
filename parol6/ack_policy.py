import os
from typing import Callable, Optional


SYSTEM_COMMANDS: set[str] = {
    "STOP",
    "ENABLE",
    "DISABLE",
    "SET_PORT",
    "STREAM",
    "SIMULATOR",
}

QUERY_COMMANDS: set[str] = {
    "GET_POSE",
    "GET_ANGLES",
    "GET_IO",
    "GET_GRIPPER",
    "GET_SPEEDS",
    "GET_STATUS",
    "GET_GCODE_STATUS",
    "GET_LOOP_STATS",
    "GET_CURRENT_ACTION",
    "GET_QUEUE",
    "PING",
}


def is_localhost(host: str) -> bool:
    h = (host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1"}


class AckPolicy:
    """
    Centralized heuristic for deciding if a command requires an acknowledgment.

    Rules:
    - If force_ack is set, it overrides everything.
    - Safety-critical commands always require ack.
    - If running on localhost/loopback, default to no-ack (low drop risk).
    - If stream mode is ON, default to no-ack (high-rate streaming traffic).
    - Otherwise default to no-ack.
    """

    def __init__(
        self,
        get_stream_mode: Callable[[], bool],
        force_ack: Optional[bool] = None,
    ) -> None:
        self._get_stream_mode = get_stream_mode
        self._force_ack = force_ack

    @staticmethod
    def from_env(get_stream_mode: Callable[[], bool]) -> "AckPolicy":
        raw = os.getenv("PAROL6_FORCE_ACK", "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            force = True
        elif raw in {"0", "false", "no", "off"}:
            force = False
        else:
            force = None
        return AckPolicy(get_stream_mode=get_stream_mode, force_ack=force)

    def requires_ack(self, message: str) -> bool:
        # Forced override (e.g., diagnostics)
        if self._force_ack is not None:
            return bool(self._force_ack)

        name = (message or "").split("|", 1)[0].strip().upper()

        # System commands always require ACKs
        if name in SYSTEM_COMMANDS:
            return True

        # Query commands use request/response, not ACKs
        if name in QUERY_COMMANDS:
            return False

        # Motion and other commands: ACKs only when forced
        # Localhost and stream mode both favor no-ack by default
        return False
