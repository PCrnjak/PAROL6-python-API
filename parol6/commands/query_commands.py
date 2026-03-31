"""
Query commands that return immediate status information.
"""

from typing import TYPE_CHECKING

import numpy as np

from parol6 import config as cfg
from parol6.commands.base import QueryCommand
from parol6.protocol.wire import (
    ActivityCmd,
    AnglesCmd,
    AnglesResultStruct,
    CmdType,
    CurrentActionResultStruct,
    EnablementResultStruct,
    ErrorCmd,
    ErrorResultStruct,
    IOCmd,
    IOResultStruct,
    JointSpeedsCmd,
    LoopStatsCmd,
    LoopStatsResultStruct,
    PingCmd,
    PingResultStruct,
    PoseCmd,
    PoseResultStruct,
    ProfileCmd,
    ProfileResultStruct,
    QueryType,
    QueueCmd,
    QueueResultStruct,
    ReachableCmd,
    SimulatorStateCmd,
    SimulatorStateResultStruct,
    SpeedsResultStruct,
    StatusCmd,
    StatusResultStruct,
    TcpSpeedCmd,
    TcpSpeedResultStruct,
    ToolStatusCmd,
    ToolResultStruct,
    ToolStatusResultStruct,
    ToolsCmd,
    pack_response,
)
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_flat_mm, get_fkine_se3
from parol6.server.status_cache import get_cache
from parol6.tools import list_tools

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.POSE)
class PoseCommand(QueryCommand[PoseCmd]):
    """Get current robot pose matrix in specified frame (WRF or TRF)."""

    PARAMS_TYPE = PoseCmd
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


@register_command(CmdType.ANGLES)
class AnglesCommand(QueryCommand[AnglesCmd]):
    """Get current joint angles in degrees."""

    PARAMS_TYPE = AnglesCmd
    QUERY_TYPE = QueryType.ANGLES

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cfg.steps_to_rad(state.Position_in, self._q_rad_buf)
        return pack_response(
            AnglesResultStruct(angles=np.rad2deg(self._q_rad_buf).tolist())
        )


@register_command(CmdType.IO)
class IOCommand(QueryCommand[IOCmd]):
    """Get current I/O status."""

    PARAMS_TYPE = IOCmd
    QUERY_TYPE = QueryType.IO

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(IOResultStruct(io=state.InOut_in[:5].tolist()))


@register_command(CmdType.JOINT_SPEEDS)
class JointSpeedsCommand(QueryCommand[JointSpeedsCmd]):
    """Get current joint speeds."""

    PARAMS_TYPE = JointSpeedsCmd
    QUERY_TYPE = QueryType.SPEEDS

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(SpeedsResultStruct(speeds=state.Speed_in.tolist()))


@register_command(CmdType.STATUS)
class StatusCommand(QueryCommand[StatusCmd]):
    """Get aggregated robot status (pose, angles, I/O, tool_status) from cache."""

    PARAMS_TYPE = StatusCmd
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


@register_command(CmdType.LOOP_STATS)
class LoopStatsCommand(QueryCommand[LoopStatsCmd]):
    """Return control-loop metrics."""

    PARAMS_TYPE = LoopStatsCmd
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
        return pack_response(
            PingResultStruct(hardware_connected=int(state.hardware_connected))
        )


@register_command(CmdType.TOOLS)
class ToolsCommand(QueryCommand[ToolsCmd]):
    """Get current tool configuration and available tools."""

    PARAMS_TYPE = ToolsCmd
    QUERY_TYPE = QueryType.TOOL

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(
            ToolResultStruct(tool=state.current_tool, available=list_tools())
        )


@register_command(CmdType.TOOL_STATUS)
class ToolStatusCommand(QueryCommand[ToolStatusCmd]):
    """Get current tool status (key + DOF positions)."""

    PARAMS_TYPE = ToolStatusCmd
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


@register_command(CmdType.ACTIVITY)
class ActivityCommand(QueryCommand[ActivityCmd]):
    """Get the current executing action/command and its state."""

    PARAMS_TYPE = ActivityCmd
    QUERY_TYPE = QueryType.CURRENT_ACTION

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(
            CurrentActionResultStruct(
                current=state.action_current,
                state=state.action_state.name,
                next=state.action_next,
                params=state.action_params,
            )
        )


@register_command(CmdType.QUEUE)
class QueueCommand(QueryCommand[QueueCmd]):
    """Get the list of queued non-streamable commands."""

    PARAMS_TYPE = QueueCmd
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


@register_command(CmdType.PROFILE)
class ProfileCommand(QueryCommand[ProfileCmd]):
    """Query the current motion profile."""

    PARAMS_TYPE = ProfileCmd
    QUERY_TYPE = QueryType.PROFILE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        return pack_response(ProfileResultStruct(profile=state.motion_profile))


@register_command(CmdType.REACHABLE)
class ReachableCommand(QueryCommand[ReachableCmd]):
    """Get joint and Cartesian enablement flags."""

    PARAMS_TYPE = ReachableCmd
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


@register_command(CmdType.ERROR)
class ErrorCommand(QueryCommand[ErrorCmd]):
    """Get the current error state."""

    PARAMS_TYPE = ErrorCmd
    QUERY_TYPE = QueryType.ERROR

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        error = state.error
        return pack_response(
            ErrorResultStruct(
                error=error.to_wire() if error is not None else None,
            )
        )


@register_command(CmdType.TCP_SPEED)
class TcpSpeedCommand(QueryCommand[TcpSpeedCmd]):
    """Get current TCP linear speed in mm/s."""

    PARAMS_TYPE = TcpSpeedCmd
    QUERY_TYPE = QueryType.TCP_SPEED

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        cache = get_cache()
        cache.update_from_state(state)
        return pack_response(TcpSpeedResultStruct(speed=cache.tcp_speed))


@register_command(CmdType.SIMULATOR_STATE)
class SimulatorStateCommand(QueryCommand[SimulatorStateCmd]):
    """Query current simulator mode state."""

    PARAMS_TYPE = SimulatorStateCmd
    QUERY_TYPE = QueryType.SIMULATOR_STATE

    __slots__ = ()

    def compute(self, state: "ControllerState") -> bytes:
        from parol6.server.transports.transport_factory import is_simulation_mode

        return pack_response(SimulatorStateResultStruct(active=is_simulation_mode()))
