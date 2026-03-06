"""
Query commands that return immediate status information.
"""

from typing import TYPE_CHECKING

import numpy as np

from parol6 import config as cfg
from parol6.commands.base import QueryCommand
from parol6.protocol.wire import (
    AnglesResultStruct,
    CmdType,
    CurrentActionResultStruct,
    EnablementResultStruct,
    ErrorResultStruct,
    GetAnglesCmd,
    GetCurrentActionCmd,
    GetEnablementCmd,
    GetErrorCmd,
    GetIOCmd,
    GetLoopStatsCmd,
    GetPoseCmd,
    GetProfileCmd,
    GetQueueCmd,
    GetSpeedsCmd,
    GetStatusCmd,
    GetTcpSpeedCmd,
    GetToolCmd,
    GetToolStatusCmd,
    IOResultStruct,
    LoopStatsResultStruct,
    PingCmd,
    PingResultStruct,
    PoseResultStruct,
    ProfileResultStruct,
    QueryType,
    QueueResultStruct,
    SpeedsResultStruct,
    StatusResultStruct,
    TcpSpeedResultStruct,
    ToolResultStruct,
    ToolStatusResultStruct,
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
            return pack_response(PoseResultStruct(pose=T_inv.reshape(-1).tolist()))
        return pack_response(PoseResultStruct(pose=get_fkine_flat_mm(state).tolist()))


@register_command(CmdType.GET_ANGLES)
class GetAnglesCommand(QueryCommand[GetAnglesCmd]):
    """Get current joint angles in degrees."""

    PARAMS_TYPE = GetAnglesCmd
    QUERY_TYPE = QueryType.ANGLES

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cfg.steps_to_rad(state.Position_in, self._q_rad_buf)
        return pack_response(
            AnglesResultStruct(angles=np.rad2deg(self._q_rad_buf).tolist())
        )


@register_command(CmdType.GET_IO)
class GetIOCommand(QueryCommand[GetIOCmd]):
    """Get current I/O status."""

    PARAMS_TYPE = GetIOCmd
    QUERY_TYPE = QueryType.IO

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(IOResultStruct(io=state.InOut_in[:5].tolist()))


@register_command(CmdType.GET_SPEEDS)
class GetSpeedsCommand(QueryCommand[GetSpeedsCmd]):
    """Get current joint speeds."""

    PARAMS_TYPE = GetSpeedsCmd
    QUERY_TYPE = QueryType.SPEEDS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(SpeedsResultStruct(speeds=state.Speed_in.tolist()))


@register_command(CmdType.GET_STATUS)
class GetStatusCommand(QueryCommand[GetStatusCmd]):
    """Get aggregated robot status (pose, angles, I/O, tool_status) from cache."""

    PARAMS_TYPE = GetStatusCmd
    QUERY_TYPE = QueryType.STATUS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cache = get_cache()
        cache.update_from_state(state)
        ts = cache.tool_status
        return pack_response(
            StatusResultStruct(
                pose=cache.pose.tolist(),
                angles=cache.angles_deg.tolist(),
                speeds=cache.speeds_rad_s.tolist(),
                io=cache.io.tolist(),
                tool_status=[
                    ts.key,
                    ts.state,
                    ts.engaged,
                    ts.part_detected,
                    ts.fault_code,
                    list(ts.positions),
                    list(ts.channels),
                ],
            )
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
            LoopStatsResultStruct(
                target_hz=target_hz,
                loop_count=state.loop_count,
                overrun_count=state.overrun_count,
                mean_period_s=state.mean_period_s,
                std_period_s=state.std_period_s,
                min_period_s=state.min_period_s,
                max_period_s=state.max_period_s,
                p95_period_s=state.p95_period_s,
                p99_period_s=state.p99_period_s,
                mean_hz=mean_hz,
            )
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
            return pack_response(PingResultStruct(hardware_connected=0))
        hw = 1 if get_cache().age_s() <= cfg.STATUS_STALE_S else 0
        return pack_response(PingResultStruct(hardware_connected=hw))


@register_command(CmdType.GET_TOOL)
class GetToolCommand(QueryCommand[GetToolCmd]):
    """Get current tool configuration and available tools."""

    PARAMS_TYPE = GetToolCmd
    QUERY_TYPE = QueryType.TOOL

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(
            ToolResultStruct(tool=state.current_tool, available=list_tools())
        )


@register_command(CmdType.GET_TOOL_STATUS)
class GetToolStatusCommand(QueryCommand[GetToolStatusCmd]):
    """Get current tool status (key + DOF positions)."""

    PARAMS_TYPE = GetToolStatusCmd
    QUERY_TYPE = QueryType.TOOL_STATUS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cache = get_cache()
        cache.update_from_state(state)
        ts = cache.tool_status
        return pack_response(
            ToolStatusResultStruct(
                tool_key=ts.key,
                state=ts.state,
                engaged=ts.engaged,
                part_detected=ts.part_detected,
                fault_code=ts.fault_code,
                positions=list(ts.positions),
                channels=list(ts.channels),
            )
        )


@register_command(CmdType.GET_CURRENT_ACTION)
class GetCurrentActionCommand(QueryCommand[GetCurrentActionCmd]):
    """Get the current executing action/command and its state."""

    PARAMS_TYPE = GetCurrentActionCmd
    QUERY_TYPE = QueryType.CURRENT_ACTION

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(
            CurrentActionResultStruct(
                current=state.action_current,
                state=state.action_state.value,
                next=state.action_next,
                params=state.action_params,
            )
        )


@register_command(CmdType.GET_QUEUE)
class GetQueueCommand(QueryCommand[GetQueueCmd]):
    """Get the list of queued non-streamable commands."""

    PARAMS_TYPE = GetQueueCmd
    QUERY_TYPE = QueryType.QUEUE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(
            QueueResultStruct(
                queue=state.queue_nonstreamable,
                executing_index=state.executing_command_index,
                completed_index=state.completed_command_index,
                last_checkpoint=state.last_checkpoint,
                queued_duration=state.queued_duration,
            )
        )


@register_command(CmdType.GET_PROFILE)
class GetProfileCommand(QueryCommand[GetProfileCmd]):
    """Query the current motion profile."""

    PARAMS_TYPE = GetProfileCmd
    QUERY_TYPE = QueryType.PROFILE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(ProfileResultStruct(profile=state.motion_profile))


@register_command(CmdType.GET_ENABLEMENT)
class GetEnablementCommand(QueryCommand[GetEnablementCmd]):
    """Get joint and Cartesian enablement flags."""

    PARAMS_TYPE = GetEnablementCmd
    QUERY_TYPE = QueryType.ENABLEMENT

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cache = get_cache()
        cache.update_from_state(state)
        return pack_response(
            EnablementResultStruct(
                joint_en=cache.joint_en.tolist(),
                cart_en_wrf=cache.cart_en_wrf.tolist(),
                cart_en_trf=cache.cart_en_trf.tolist(),
            )
        )


@register_command(CmdType.GET_ERROR)
class GetErrorCommand(QueryCommand[GetErrorCmd]):
    """Get the current error state."""

    PARAMS_TYPE = GetErrorCmd
    QUERY_TYPE = QueryType.ERROR

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        error = state.error
        return pack_response(
            ErrorResultStruct(
                error=error.to_wire() if error is not None else None,
            )
        )


@register_command(CmdType.GET_TCP_SPEED)
class GetTcpSpeedCommand(QueryCommand[GetTcpSpeedCmd]):
    """Get current TCP linear speed in mm/s."""

    PARAMS_TYPE = GetTcpSpeedCmd
    QUERY_TYPE = QueryType.TCP_SPEED

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cache = get_cache()
        cache.update_from_state(state)
        return pack_response(TcpSpeedResultStruct(speed=cache.tcp_speed))
