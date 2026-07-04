"""Workspace collision-world shapes — the SET_SHAPES command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from parol6.commands.base import ExecutionStatusCode, SystemCommand
from parol6.protocol.wire import CmdType, SetShapesCmd
from parol6.server.command_registry import register_command

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.SET_SHAPES)
class SetShapesCommand(SystemCommand[SetShapesCmd]):
    """Replace the workspace keep-out shapes on the collision checkers.

    A SystemCommand (like SELECT_PROFILE): safety configuration applies
    immediately at intake — never enabled-gated, never queued behind (or
    dropped with) motion. The controller mirrors the applied shapes to the
    planner subprocess via ``sync_shapes`` after this executes.
    """

    PARAMS_TYPE = SetShapesCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        state.set_shapes(self.p.shapes)
        self.finish()
        return ExecutionStatusCode.COMPLETED
