"""
Query commands that return immediate status information.
"""

from typing import TYPE_CHECKING

import numpy as np

from parol6 import config as cfg
from parol6.commands.base import QueryCommand
from parol6.protocol.wire import (
    CmdType,
    GetAnglesCmd,
    GetCurrentActionCmd,
    GetGripperCmd,
    GetIOCmd,
    GetLoopStatsCmd,
    GetPoseCmd,
    GetProfileCmd,
    GetQueueCmd,
    GetSpeedsCmd,
    GetStatusCmd,
    GetToolCmd,
    PingCmd,
    QueryType,
    pack_response,
)
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_flat_mm, get_fkine_se3
from parol6.server.status_cache import get_cache
from parol6.server.transports import is_simulation_mode
from parol6.tools import list_tools

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.GET_POSE)
class GetPoseCommand(QueryCommand[GetPoseCmd]):
    """Get current robot pose matrix in specified frame (WRF or TRF)."""

    PARAMS_TYPE = GetPoseCmd
    QUERY_TYPE = QueryType.POSE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        frame = self.p.frame or "WRF"
        if frame == "TRF":
            T = get_fkine_se3(state)
            T_inv = np.linalg.inv(T)
            T_inv[0:3, 3] *= 1000.0
            return pack_response(self.QUERY_TYPE, T_inv.reshape(-1))
        return pack_response(self.QUERY_TYPE, get_fkine_flat_mm(state))


@register_command(CmdType.GET_ANGLES)
class GetAnglesCommand(QueryCommand[GetAnglesCmd]):
    """Get current joint angles in degrees."""

    PARAMS_TYPE = GetAnglesCmd
    QUERY_TYPE = QueryType.ANGLES

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cfg.steps_to_rad(state.Position_in, self._q_rad_buf)
        return pack_response(self.QUERY_TYPE, np.rad2deg(self._q_rad_buf))


@register_command(CmdType.GET_IO)
class GetIOCommand(QueryCommand[GetIOCmd]):
    """Get current I/O status."""

    PARAMS_TYPE = GetIOCmd
    QUERY_TYPE = QueryType.IO

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(self.QUERY_TYPE, state.InOut_in[:5])


@register_command(CmdType.GET_GRIPPER)
class GetGripperCommand(QueryCommand[GetGripperCmd]):
    """Get current gripper status."""

    PARAMS_TYPE = GetGripperCmd
    QUERY_TYPE = QueryType.GRIPPER

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(self.QUERY_TYPE, state.Gripper_data_in)


@register_command(CmdType.GET_SPEEDS)
class GetSpeedsCommand(QueryCommand[GetSpeedsCmd]):
    """Get current joint speeds."""

    PARAMS_TYPE = GetSpeedsCmd
    QUERY_TYPE = QueryType.SPEEDS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(self.QUERY_TYPE, state.Speed_in)


@register_command(CmdType.GET_STATUS)
class GetStatusCommand(QueryCommand[GetStatusCmd]):
    """Get aggregated robot status (pose, angles, I/O, gripper) from cache."""

    PARAMS_TYPE = GetStatusCmd
    QUERY_TYPE = QueryType.STATUS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cache = get_cache()
        cache.update_from_state(state)
        return pack_response(
            self.QUERY_TYPE,
            [
                cache.pose,
                cache.angles_deg,
                cache.speeds,
                cache.io,
                cache.gripper,
            ],
        )


@register_command(CmdType.GET_LOOP_STATS)
class GetLoopStatsCommand(QueryCommand[GetLoopStatsCmd]):
    """Return control-loop metrics."""

    PARAMS_TYPE = GetLoopStatsCmd
    QUERY_TYPE = QueryType.LOOP_STATS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        target_hz = 1.0 / max(cfg.INTERVAL_S, 1e-9)
        mean_hz = (1.0 / state.mean_period_s) if state.mean_period_s > 0.0 else 0.0
        return pack_response(
            self.QUERY_TYPE,
            [
                target_hz,
                state.loop_count,
                state.overrun_count,
                state.mean_period_s,
                state.std_period_s,
                state.min_period_s,
                state.max_period_s,
                state.p95_period_s,
                state.p99_period_s,
                mean_hz,
            ],
        )


@register_command(CmdType.PING)
class PingCommand(QueryCommand[PingCmd]):
    """Respond to ping requests."""

    PARAMS_TYPE = PingCmd
    QUERY_TYPE = QueryType.PING

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        sim = is_simulation_mode()
        if sim:
            return pack_response(self.QUERY_TYPE, 0)
        return pack_response(
            self.QUERY_TYPE, 1 if get_cache().age_s() <= cfg.STATUS_STALE_S else 0
        )


@register_command(CmdType.GET_TOOL)
class GetToolCommand(QueryCommand[GetToolCmd]):
    """Get current tool configuration and available tools."""

    PARAMS_TYPE = GetToolCmd
    QUERY_TYPE = QueryType.TOOL

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(self.QUERY_TYPE, [state.current_tool, list_tools()])


@register_command(CmdType.GET_CURRENT_ACTION)
class GetCurrentActionCommand(QueryCommand[GetCurrentActionCmd]):
    """Get the current executing action/command and its state."""

    PARAMS_TYPE = GetCurrentActionCmd
    QUERY_TYPE = QueryType.CURRENT_ACTION

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(
            self.QUERY_TYPE,
            [state.action_current, state.action_state, state.action_next],
        )


@register_command(CmdType.GET_QUEUE)
class GetQueueCommand(QueryCommand[GetQueueCmd]):
    """Get the list of queued non-streamable commands."""

    PARAMS_TYPE = GetQueueCmd
    QUERY_TYPE = QueryType.QUEUE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(self.QUERY_TYPE, state.queue_nonstreamable)


@register_command(CmdType.GET_PROFILE)
class GetProfileCommand(QueryCommand[GetProfileCmd]):
    """Query the current motion profile."""

    PARAMS_TYPE = GetProfileCmd
    QUERY_TYPE = QueryType.PROFILE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(self.QUERY_TYPE, state.motion_profile)
