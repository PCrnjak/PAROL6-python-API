"""
Generic tool action command — dispatches to gripper commands by config type.
"""

import logging

from parol6.commands.base import ExecutionStatusCode, MotionCommand
from parol6.commands.gripper_commands import ElectricGripperCommand
from parol6.protocol.wire import CmdType, ToolActionCmd
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState
from parol6.tools import get_registry, tool_action_refusal
from parol6.utils.error_catalog import make_error
from parol6.utils.error_codes import ErrorCode

logger = logging.getLogger(__name__)


@register_command(CmdType.TOOL_ACTION)
class ToolActionCommand(MotionCommand[ToolActionCmd]):
    """Dispatch tool actions to the appropriate 100 Hz command engine."""

    PARAMS_TYPE = ToolActionCmd

    __slots__ = ("_delegate",)

    def __init__(self, p: ToolActionCmd) -> None:
        super().__init__(p)
        self._delegate: MotionCommand | None = None

    def do_setup(self, state: ControllerState) -> None:
        key = self.p.tool_key.strip().upper()
        action = self.p.action.strip().lower()
        params = self.p.params

        # Judged when the action's turn comes: the selection or calibration
        # it needs may be the command queued just ahead of it.
        refusal = tool_action_refusal(
            key,
            action,
            current_tool=state.current_tool,
            gripper_calibrated=state.gripper_calibrated,
        )
        if refusal is not None:
            self.fail(make_error(ErrorCode.COMM_VALIDATION_ERROR, detail=refusal))
            return

        cfg = get_registry().get(key)
        if cfg is None:
            raise ValueError(f"Unknown tool '{key}'")

        delegate = cfg.create_command(action, params)
        if delegate is None:
            raise ValueError(f"Tool '{key}' does not support actions")

        delegate.setup(state)
        self._delegate = delegate

    def halt(self, state: ControllerState) -> None:
        """Stop the tool where it is, keeping its grip, for whatever discards
        the action mid-way. Only an electric gripper has motion to halt; a
        pneumatic valve, once switched, strokes to its end."""
        if isinstance(self._delegate, ElectricGripperCommand):
            self._delegate.halt(state)

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        if self._delegate is None:
            self.fail(make_error(ErrorCode.MOTN_GRIPPER_UNKNOWN))
            return ExecutionStatusCode.FAILED
        result = self._delegate.tick(state)
        if self._delegate.is_finished:
            self.is_finished = True
            self.error_state = self._delegate.error_state
            self.robot_error = self._delegate.robot_error
        return result
