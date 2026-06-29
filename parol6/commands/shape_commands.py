"""Workspace collision-world shapes — the SET_SHAPES command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from parol6.commands.base import ExecutionStatusCode, MotionCommand
from parol6.protocol.wire import CmdType, SetShapesCmd
from parol6.server.command_registry import register_command

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.SET_SHAPES)
class SetShapesCommand(MotionCommand[SetShapesCmd]):
    """Replace the workspace keep-out shapes on the collision checkers.

    Routed through the planner (like SELECT_TOOL) so the planner subprocess
    updates its own checker; this controller-side step updates the live
    control-loop checker.
    """

    PARAMS_TYPE = SetShapesCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        state.set_shapes(self.p.shapes)
        self.finish()
        return ExecutionStatusCode.COMPLETED
