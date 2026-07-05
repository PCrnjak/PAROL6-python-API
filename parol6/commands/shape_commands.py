"""Workspace collision-world shapes — the SET_SHAPES command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from waldoctl import shape_from_wire

from parol6.commands.base import ExecutionStatusCode, SystemCommand
from parol6.protocol.wire import CmdType, SetShapesCmd
from parol6.server.command_registry import register_command

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.SET_SHAPES)
class SetShapesCommand(SystemCommand[SetShapesCmd]):
    """Replace the program-layer keep-out shapes on the collision checkers.

    A SystemCommand (like SELECT_PROFILE): safety configuration applies
    immediately at intake — never enabled-gated, never queued behind (or
    dropped with) motion. The controller mirrors the applied shapes to the
    planner subprocess via ``sync_shapes`` after this executes.

    This is the codec boundary: the wire form is rebuilt into waldoctl
    ``Shape`` objects here (running their construction-time validation), and
    everything downstream — state, checkers, subprocess syncs, readback —
    speaks ``Shape`` only.
    """

    PARAMS_TYPE = SetShapesCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        shapes = [
            shape_from_wire(w.kind, w.params, w.pose, w.collision, w.margin, w.name)
            for w in self.p.shapes
        ]
        state.set_shapes(shapes)
        self.finish()
        return ExecutionStatusCode.COMPLETED
