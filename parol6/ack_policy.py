import os

from parol6.protocol.wire import CmdType

# System command types (always require ACK)
SYSTEM_CMD_TYPES: set[CmdType] = {
    CmdType.RESUME,
    CmdType.HALT,
    CmdType.SET_PORT,
    CmdType.SIMULATOR,
    CmdType.SET_PROFILE,
    CmdType.RESET,
    CmdType.SET_IO,
}

# Query command types (use request/response, not ACK)
QUERY_CMD_TYPES: set[CmdType] = {
    CmdType.GET_POSE,
    CmdType.GET_ANGLES,
    CmdType.GET_IO,
    CmdType.GET_SPEEDS,
    CmdType.GET_STATUS,
    CmdType.GET_LOOP_STATS,
    CmdType.GET_CURRENT_ACTION,
    CmdType.GET_QUEUE,
    CmdType.GET_TOOL,
    CmdType.GET_TOOL_STATUS,
    CmdType.GET_PROFILE,
    CmdType.GET_ENABLEMENT,
    CmdType.GET_ERROR,
    CmdType.GET_TCP_SPEED,
    CmdType.PING,
}

# Streaming commands are fire-and-forget (no ACK needed)
FIRE_AND_FORGET: set[CmdType] = {
    CmdType.SERVOJ,
    CmdType.SERVOJ_POSE,
    CmdType.SERVOL,
    CmdType.JOGJ,
    CmdType.JOGL,
    CmdType.TELEPORT,
    CmdType.RESET_LOOP_STATS,
}

# Queued motion commands that return a command index in their ACK
QUEUED_CMD_TYPES: set[CmdType] = {
    CmdType.HOME,
    CmdType.MOVEJ,
    CmdType.MOVEJ_POSE,
    CmdType.MOVEL,
    CmdType.MOVEC,
    CmdType.MOVES,
    CmdType.MOVEP,
    CmdType.SET_TOOL,
    CmdType.DELAY,
    CmdType.CHECKPOINT,
    CmdType.TOOL_ACTION,
}


class AckPolicy:
    """
    Centralized heuristic for deciding if a command requires an acknowledgment.

    Rules:
    - If force_ack is set, it overrides everything.
    - System commands always require ack.
    - Query commands use request/response, not ACKs.
    - Streaming commands (servo/jog) are fire-and-forget.
    - Queued motion commands require ack (returns command index).

    When force_ack is not provided, the PAROL6_FORCE_ACK env var is checked.
    """

    def __init__(
        self,
        force_ack: bool | None = None,
    ) -> None:
        if force_ack is None:
            raw = os.getenv("PAROL6_FORCE_ACK", "").strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                force_ack = True
            elif raw in {"0", "false", "no", "off"}:
                force_ack = False
        self._force_ack = force_ack

    def requires_ack(self, cmd_type: CmdType) -> bool:
        """Check if a command type requires an ACK response."""
        # Forced override (e.g., diagnostics)
        if self._force_ack is not None:
            return bool(self._force_ack)

        # System commands always require ACKs
        if cmd_type in SYSTEM_CMD_TYPES:
            return True

        # Query commands use request/response, not ACKs
        if cmd_type in QUERY_CMD_TYPES:
            return False

        # Streaming commands are fire-and-forget
        if cmd_type in FIRE_AND_FORGET:
            return False

        # Queued motion commands require ACK (returns command index)
        if cmd_type in QUEUED_CMD_TYPES:
            return True

        # Default: no ACK
        return False
