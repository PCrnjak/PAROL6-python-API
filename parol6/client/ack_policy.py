from __future__ import annotations

import os
from typing import Callable, Optional


SAFETY_COMMANDS: set[str] = {
    "STOP",
    "ENABLE",
    "DISABLE",
    "CLEAR_ERROR",
    "HOME",
    "SET_PORT",
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
        host: str,
        get_stream_mode: Callable[[], bool],
        force_ack: Optional[bool] = None,
    ) -> None:
        self._host = host
        self._get_stream_mode = get_stream_mode
        self._force_ack = force_ack

    @staticmethod
    def from_env(host: str, get_stream_mode: Callable[[], bool]) -> "AckPolicy":
        raw = os.getenv("PAROL6_FORCE_ACK", "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            force = True
        elif raw in {"0", "false", "no", "off"}:
            force = False
        else:
            force = None
        return AckPolicy(host=host, get_stream_mode=get_stream_mode, force_ack=force)

    def requires_ack(self, message: str) -> bool:
        # Forced override (e.g., diagnostics)
        if self._force_ack is not None:
            return bool(self._force_ack)

        name = (message or "").split("|", 1)[0].strip().upper()

        # Always ack for safety-critical commands
        if name in SAFETY_COMMANDS:
            return True

        # Localhost defaults to no-ack
        if is_localhost(self._host):
            return False

        # Streaming mode defaults to no-ack
        if self._get_stream_mode():
            return False

        # Default: no-ack
        return False
