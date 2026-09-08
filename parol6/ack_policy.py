import os

from parol6.protocol.wire import CmdType, QueryType

# System command types (always require ACK)
SYSTEM_CMD_TYPES: set[CmdType] = {
    CmdType.RESET,
    CmdType.ESTOP,
    CmdType.STOP,
    CmdType.CONNECT_HARDWARE,
    CmdType.SIMULATOR,
    CmdType.SELECT_PROFILE,
    CmdType.RESET_STATE,
    CmdType.WRITE_IO,
    CmdType.SET_SHAPES,
    CmdType.SET_STATUS_RATE,
    CmdType.SET_EXECUTION_SPEED,
    CmdType.PAUSE,
}

# Query command types (use request/response, not ACK)
QUERY_RESPONSE_TYPES: dict[CmdType, QueryType] = {
    CmdType.POSE: QueryType.POSE,
    CmdType.ANGLES: QueryType.ANGLES,
    CmdType.IO: QueryType.IO,
    CmdType.JOINT_SPEEDS: QueryType.SPEEDS,
    CmdType.STATUS: QueryType.STATUS,
    CmdType.LOOP_STATS: QueryType.LOOP_STATS,
    CmdType.ACTIVITY: QueryType.CURRENT_ACTION,
    CmdType.QUEUE: QueryType.QUEUE,
    CmdType.TOOLS: QueryType.TOOL,
    CmdType.TOOL_STATUS: QueryType.TOOL_STATUS,
    CmdType.PROFILE: QueryType.PROFILE,
    CmdType.REACHABLE: QueryType.ENABLEMENT,
    CmdType.ERROR: QueryType.ERROR,
    CmdType.TCP_SPEED: QueryType.TCP_SPEED,
    CmdType.PING: QueryType.PING,
    CmdType.IS_SIMULATOR: QueryType.IS_SIMULATOR,
    CmdType.TCP_OFFSET: QueryType.TCP_OFFSET,
    CmdType.TCP_TRANSFORM: QueryType.TCP_TRANSFORM,
    CmdType.SHAPES: QueryType.SHAPES,
    CmdType.STATUS_RATE: QueryType.STATUS_RATE,
    CmdType.EXECUTION_SPEED: QueryType.EXECUTION_SPEED,
    CmdType.COMMAND_COMPLETION: QueryType.COMMAND_COMPLETION,
}
QUERY_CMD_TYPES: set[CmdType] = set(QUERY_RESPONSE_TYPES)

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
    CmdType.SET_TCP_OFFSET,
    CmdType.SET_TCP_TRANSFORM,
    CmdType.HOME,
    CmdType.MOVEJ,
    CmdType.MOVEJ_POSE,
    CmdType.MOVEL,
    CmdType.MOVEC,
    CmdType.MOVES,
    CmdType.MOVEP,
    CmdType.SELECT_TOOL,
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
        # Forced override (e.g., diagnostics) takes precedence over everything
        if self._force_ack is not None:
            return bool(self._force_ack)

        if cmd_type in SYSTEM_CMD_TYPES:
            return True

        # Queries use request/response, not ACKs
        if cmd_type in QUERY_CMD_TYPES:
            return False

        # Streaming commands are fire-and-forget
        if cmd_type in FIRE_AND_FORGET:
            return False

        # Queued motion commands ACK to return the command index
        if cmd_type in QUEUED_CMD_TYPES:
            return True

        return False
