"""
Main controller for PAROL6 robot server.

Runs the fixed-rate control loop, dispatches UDP commands to the command
executor, and manages serial/simulator transport and status broadcasting.
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


from parol6.ack_policy import AckPolicy
from parol6.commands.base import (
    ExecutionStatusCode,
    MotionCommand,
    QueryCommand,
    SystemCommand,
)
from parol6.commands.system_commands import SetProfileCommand
from parol6.commands.utility_commands import ResetCommand
from parol6.server.command_executor import CommandExecutor, QueueFullError
from parol6.server.motion_planner import MotionPlanner, PlanCommand
from parol6.server.segment_player import SegmentPlayer
from parol6.protocol.wire import (
    CommandCode,
    pack_error,
    pack_ok,
    pack_ok_index,
    unpack_rx_frame_into,
)
from parol6.utils.error_catalog import RobotError, extract_robot_error, make_error
from parol6.utils.error_codes import ErrorCode
from parol6.server.command_registry import (
    CommandCategory,
    create_command,
    create_command_from_struct,
    discover_commands,
)
from parol6.server.state import ControllerState, StateManager
from waldoctl import ActionState
from parol6.server.status_broadcast import StatusBroadcaster
from parol6.server.async_logging import AsyncLogHandler
from parol6.server.loop_timer import (
    EventRateMetrics,
    GCTracker,
    LoopTimer,
    PhaseTimer,
    format_hz_summary,
)
from parol6.server.status_cache import close_cache, get_cache
from parol6.server.transport_manager import TransportManager
from parol6.server.transports.udp_transport import UDPTransport
from parol6.config import (
    TRACE,
    INTERVAL_S,
    MAX_POLL_COUNT,
    MCAST_GROUP,
    MCAST_PORT,
    MCAST_IF,
    MCAST_TTL,
    STATUS_RATE_HZ,
    STATUS_STALE_S,
    STATUS_BROADCAST_INTERVAL,
)

import psutil

logger = logging.getLogger("parol6.server.controller")


@dataclass
class ControllerConfig:
    """Configuration for the controller."""

    udp_host: str = "0.0.0.0"
    udp_port: int = 5001
    serial_port: str | None = None
    serial_baudrate: int = 3000000
    loop_interval: float = INTERVAL_S
    estop_recovery_delay: float = 1.0


class Controller:
    """
    Main controller that orchestrates all components of the PAROL6 server.

    This replaces the monolithic controller.py with a modular design:
    - State management via StateManager singleton
    - Transport abstraction for UDP and Serial
    - Command execution via CommandExecutor
    - Automatic command discovery and registration
    """

    def __init__(self, config: ControllerConfig):
        """
        Initialize the controller with configuration.

        Args:
            config: Configuration object for the controller
        """
        self.config = config
        self.running = False
        self.shutdown_event = threading.Event()
        self._initialized = False

        # Core components
        self.state_manager = StateManager()
        self.udp_transport: UDPTransport | None = None

        # E-stop state tracking (start as released to avoid false positive on first check)
        self.estop_active: bool = False

        # TX keepalive timeout
        self._tx_keepalive_s = float(os.getenv("PAROL6_TX_KEEPALIVE_S", "0.2"))

        # Status multicast broadcaster
        self._status_broadcaster: Any | None = None

        # Helper classes
        self._timer = LoopTimer(self.config.loop_interval)
        self._phase_timer = PhaseTimer(
            [
                "read",  # _read_from_firmware
                "poll_cmd",  # _poll_commands
                "status",  # _status_broadcaster.tick
                "estop",  # _handle_estop
                "execute",  # _execute_commands
                "write",  # _write_to_firmware
                "sim",  # tick_simulation
            ]
        )
        self._cmd_rate = EventRateMetrics()  # Command reception rate
        self._gc_tracker = GCTracker()  # GC frequency and duration tracking
        self._ack_policy = AckPolicy()
        self._async_log = AsyncLogHandler()
        self._transport_mgr = TransportManager(
            shutdown_event=self.shutdown_event,
            serial_port=self.config.serial_port,
            serial_baudrate=self.config.serial_baudrate,
        )
        self._executor = CommandExecutor(
            state_manager=self.state_manager,
        )

        # Motion pipeline: planner subprocess computes trajectories,
        # segment player consumes them in the control loop
        self._planner = MotionPlanner()
        self._segment_player = SegmentPlayer(self._planner)
        self._planner_needs_sync: bool = True  # first command always carries position

        # Initialize components on construction
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initialize all components during construction.

        Raises:
            RuntimeError: If critical components fail to initialize
        """
        try:
            # Discover and register all commands
            discover_commands()

            # Initialize UDP transport
            logger.info(
                f"Starting UDP server on {self.config.udp_host}:{self.config.udp_port}"
            )
            self.udp_transport = UDPTransport(
                self.config.udp_host, self.config.udp_port
            )
            if not self.udp_transport.create_socket():
                raise RuntimeError("Failed to create UDP socket")

            # Initialize robot state
            self.state_manager.reset_state()

            # Initialize serial transport via TransportManager
            self._transport_mgr.initialize()

            # Create status broadcaster
            try:
                logger.info(
                    f"StatusBroadcaster config: group={MCAST_GROUP} port={MCAST_PORT} ttl={MCAST_TTL} iface={MCAST_IF} rate_hz={STATUS_RATE_HZ} stale_s={STATUS_STALE_S}"
                )
                self._status_broadcaster = StatusBroadcaster(
                    state_mgr=self.state_manager,
                    group=MCAST_GROUP,
                    port=MCAST_PORT,
                    ttl=MCAST_TTL,
                    iface_ip=MCAST_IF,
                    rate_hz=STATUS_RATE_HZ,
                    stale_s=STATUS_STALE_S,
                )
                logger.info("StatusBroadcaster initialized")
            except Exception as e:
                logger.warning(f"Failed to create status broadcaster: {e}")

            self._initialized = True
            logger.info("Controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize controller: {e}")
            self._initialized = False
            raise RuntimeError(f"Controller initialization failed: {e}")

    def is_initialized(self) -> bool:
        """Check if controller is properly initialized."""
        return self._initialized

    def start(self):
        """Start the main control loop."""
        if self.running:
            logger.warning("Controller already running")
            return

        self._set_high_priority()
        self.running = True

        # Start async logging to move I/O off the control loop thread
        self._async_log.start()

        # Start motion planner subprocess
        self._planner.start()

        # Disable automatic GC — collections are deferred to slack time
        self._gc_tracker.take_control()

        # Start main control loop
        logger.info("Starting main control loop")
        self._timer.metrics.mark_started(time.perf_counter())
        self._main_control_loop()

    def stop(self):
        """Stop the controller and clean up resources."""
        if not self.running:
            return
        logger.info("Stopping controller...")
        self.running = False
        self.shutdown_event.set()

        # Stop motion planner subprocess
        try:
            self._planner.stop()
        except Exception:
            logger.warning("Error stopping motion planner", exc_info=True)

        # Stop IK worker subprocess
        try:
            close_cache()
        except Exception:
            logger.warning("Error stopping IK worker", exc_info=True)

        # Close status broadcaster
        try:
            if self._status_broadcaster:
                self._status_broadcaster.close()
        except Exception as e:
            logger.debug("Error closing status broadcaster: %s", e)

        # Clean up transports
        if self.udp_transport:
            self.udp_transport.close_socket()

        self._transport_mgr.disconnect()

        # Re-enable automatic GC and remove tracker callback
        self._gc_tracker.shutdown()

        # Stop async logging (flushes queued messages)
        self._async_log.stop()

        logger.info("Controller stopped")

    def _read_from_firmware(self, state: ControllerState) -> None:
        """Phase 1: Poll serial for data, unpack frames, handle auto-reconnect."""
        if self._transport_mgr.is_connected():
            # Poll serial for new data
            self._transport_mgr.poll_serial()
            try:
                mv, ver, ts = self._transport_mgr.get_latest_frame()
                if mv is not None and ver != self._transport_mgr._last_version:
                    ok = unpack_rx_frame_into(
                        mv,
                        pos_out=state.Position_in,
                        spd_out=state.Speed_in,
                        homed_out=state.Homed_in,
                        io_out=state.InOut_in,
                        temp_out=state.Temperature_error_in,
                        poserr_out=state.Position_error_in,
                        timing_out=state.Timing_data_in,
                        grip_out=state.Gripper_data_in,
                    )
                    if ok:
                        get_cache().mark_serial_observed()
                        if not self._transport_mgr.first_frame_received:
                            self._transport_mgr.first_frame_received = True
                            logger.info("First frame received from robot")
                        self._transport_mgr._last_version = ver
            except Exception as e:
                logger.warning(f"Error decoding latest serial frame: {e}")

        # Serial auto-reconnect when a port is known
        self._transport_mgr.auto_reconnect()

    def _handle_estop(self, state: ControllerState) -> None:
        """Phase 2: Handle E-stop activation and recovery."""
        if not (
            self._transport_mgr.is_connected()
            and self._transport_mgr.first_frame_received
        ):
            return

        if state.InOut_in[4] == 0:  # E-stop pressed
            if not self.estop_active:
                logger.warning("E-STOP activated")
                self.estop_active = True
                self._segment_player.cancel(state)
                self._planner_needs_sync = True
                self._planner.sync_tool(state.current_tool)
                if self._executor.active_command:
                    self._executor.cancel_active_command("E-Stop activated")
                self._executor.clear_queue("E-Stop activated")
                state.Command_out = CommandCode.DISABLE
                state.Speed_out.fill(0)
                state.error = make_error(ErrorCode.SYS_ESTOP_ACTIVE)
        elif state.InOut_in[4] == 1:  # E-stop released
            if self.estop_active:
                logger.info("E-STOP released - automatic recovery")
                self.estop_active = False
                state.enabled = True
                state.disabled_reason = ""
                state.Command_out = CommandCode.IDLE
                state.Speed_out.fill(0)
                if (
                    state.error is not None
                    and state.error.code == ErrorCode.SYS_ESTOP_ACTIVE
                ):
                    state.error = None

    def _execute_commands(self, state: ControllerState) -> None:
        """Phase 3: Execute active command."""
        # Segment player handles trajectory + inline commands from planner
        if self._segment_player.tick(state):
            return

        # Streaming command executor (jog/servo)
        if self._executor.active_command or self._executor.command_queue:
            self._executor.execute_active_command()
        else:
            state.Command_out = CommandCode.IDLE
            state.Speed_out.fill(0)
            state.Position_out[:] = state.Position_in

    def _write_to_firmware(self, state: ControllerState) -> None:
        """Phase 4: Write state to serial transport if changed."""
        ok = self._transport_mgr.write_frame(
            state.Position_out,
            state.Speed_out,
            state.Command_out.value,
            state.Affected_joint_out,
            state.InOut_out,
            state.Timeout_out,
            state.Gripper_data_out,
            keepalive_s=self._tx_keepalive_s,
        )
        if ok:
            # Auto-reset one-shot gripper modes after successful send
            if state.Gripper_data_out[4] in (1, 2):
                state.Gripper_data_out[4] = 0

    def _sync_timer_metrics(self, state: ControllerState) -> None:
        """Copy timing metrics from LoopTimer and PhaseTimer to controller state."""
        m = self._timer.metrics

        # Check if loop stats reset was requested
        if state.loop_stats_reset_pending:
            m.reset_stats(include_counters=True)
            state.loop_stats_reset_pending = False
            logger.debug("Loop stats reset completed")

        state.loop_count = m.loop_count
        state.overrun_count = m.overrun_count

        # Only copy rolling stats when they were updated (every stats_interval loops)
        if m.loop_count % self._timer._stats_interval == 0:
            state.mean_period_s = m.mean_period_s
            state.std_period_s = m.std_period_s
            state.min_period_s = m.min_period_s
            state.max_period_s = m.max_period_s
            state.p95_period_s = m.p95_period_s
            state.p99_period_s = m.p99_period_s

    def _log_periodic_status(self, state: ControllerState) -> None:
        """Log performance metrics every 3 seconds."""
        now = time.perf_counter()
        m = self._timer.metrics

        # Rate-limited overbudget warning (grace period handled in LoopMetrics)
        should_warn, pct = m.check_degraded(now, 0.25, 3.0)
        if should_warn:
            gc_dur = self._gc_tracker.recent_duration_ms()
            logger.warning(
                "loop overbudget by +%.0f%% (%s) gc=%.2fms",
                pct,
                format_hz_summary(m),
                gc_dur,
            )

        # Rate-limited debug log every 3s
        if not m.should_log(now, 3.0):
            return

        # Command rate from EventRateMetrics (decays to 0 when idle)
        cmd_hz = self._cmd_rate.rate_hz(now, max_age_s=6.0)
        gc_hz, gc_ms = self._gc_tracker.stats(now, max_age_s=6.0)

        logger.debug(
            "loop: %s cmd=%.1fHz ov=%d overshoot_p99=%.2fµs gc=%.1fHz/%.2fms",
            format_hz_summary(m),
            cmd_hz,
            state.overrun_count,
            m.p99_overshoot_s * 1_000_000,
            gc_hz,
            gc_ms,
        )

        # Log phase breakdown (p99 values to catch spikes)
        phases = self._phase_timer.phases
        logger.debug(
            "phases p99: read=%.2fms poll_cmd=%.2fms status=%.2fms estop=%.2fms exec=%.2fms write=%.2fms sim=%.2fms",
            phases["read"].p99_s * 1000,
            phases["poll_cmd"].p99_s * 1000,
            phases["status"].p99_s * 1000,
            phases["estop"].p99_s * 1000,
            phases["execute"].p99_s * 1000,
            phases["write"].p99_s * 1000,
            phases["sim"].p99_s * 1000,
        )

    def _main_control_loop(self):
        """Main control loop with phase-based structure and precise timing."""
        self._timer.start()
        pt = self._phase_timer
        tick_count = 0
        broadcast_interval = STATUS_BROADCAST_INTERVAL

        while self.running:
            try:
                state = self.state_manager.get_state()
                tick_count += 1

                with pt.phase("read"):
                    self._read_from_firmware(state)

                with pt.phase("poll_cmd"):
                    self._poll_commands(state)

                if tick_count % broadcast_interval == 0:
                    with pt.phase("status"):
                        if self._status_broadcaster:
                            self._status_broadcaster.tick()

                with pt.phase("estop"):
                    self._handle_estop(state)

                if not self.estop_active:
                    with pt.phase("execute"):
                        self._execute_commands(state)

                with pt.phase("write"):
                    self._write_to_firmware(state)

                with pt.phase("sim"):
                    self._transport_mgr.tick_simulation(state.current_tool)

                pt.tick()
                self._sync_timer_metrics(state)
                self._log_periodic_status(state)
                self._gc_tracker.collect_deferred(
                    self._timer.time_to_next_deadline(), tick_count
                )
                self._timer.wait_for_next_tick()

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in main control loop: {e}", exc_info=True)
                state.Command_out = CommandCode.IDLE
                state.Speed_out.fill(0)

    def _poll_commands(self, state: ControllerState) -> None:
        """Poll and process UDP commands (non-blocking)."""
        assert self.udp_transport is not None

        msgs = self.udp_transport.poll_receive_all(max_count=MAX_POLL_COUNT)
        for data, addr in msgs:
            self._process_command(data, addr, state)

    def _reply_error(self, addr: tuple[str, int], error: RobotError) -> None:
        """Send error response to client. Caller must ensure udp_transport is not None."""
        assert self.udp_transport is not None
        self.udp_transport.send(pack_error(error), addr)

    def _reply_ok(self, addr: tuple[str, int]) -> None:
        """Send OK response to client. Caller must ensure udp_transport is not None."""
        assert self.udp_transport is not None
        self.udp_transport.send(pack_ok(), addr)

    def _reply_ok_index(self, addr: tuple[str, int], index: int) -> None:
        """Send OK response with command index. Caller must ensure udp_transport is not None."""
        assert self.udp_transport is not None
        self.udp_transport.send(pack_ok_index(index), addr)

    def _process_command(
        self, data: bytes, addr: tuple[str, int], state: ControllerState
    ) -> None:
        """Process a single command from UDP.

        Args:
            data: Raw msgpack-encoded command bytes
            addr: Client address tuple (host, port)
            state: Controller state
        """
        self._cmd_rate.record(time.perf_counter())

        # Try stream fast-path first (avoids full command creation)
        result = self._executor.try_stream_fast_path(data, state)
        if result is True:
            return

        # If fast-path returned a decoded struct, reuse it; otherwise decode from bytes
        if result is not False:
            command, category, error = create_command_from_struct(result)
        else:
            command, category, error = create_command(data)

        if not command or category is None:
            if error:
                logger.warning(f"Command validation failed: {error}")
                self._reply_error(
                    addr, make_error(ErrorCode.COMM_VALIDATION_ERROR, detail=error)
                )
            else:
                logger.warning("Unknown command")
                self._reply_error(addr, make_error(ErrorCode.COMM_UNKNOWN_COMMAND))
            return

        cmd_name = type(command).__name__
        logger.log(TRACE, "cmd_received name=%s from=%s", cmd_name, addr)

        # Dispatch by category (determined at registration time, no isinstance needed)
        match category:
            case CommandCategory.QUERY:
                self._handle_query(command, state, addr)  # type: ignore[arg-type]
            case CommandCategory.SYSTEM:
                self._handle_system_command(command, state, addr)  # type: ignore[arg-type]
            case CommandCategory.MOTION:
                self._handle_motion_command(command, state, addr)  # type: ignore[arg-type]

    def _handle_motion_command(
        self, command: MotionCommand, state: ControllerState, addr: tuple[str, int]
    ) -> None:
        """Queue motion command for execution."""
        cmd_name = type(command).__name__

        cmd_type = command._cmd_type
        if not state.enabled:
            if cmd_type and self._ack_policy.requires_ack(cmd_type):
                reason = state.disabled_reason or "Controller disabled"
                self._reply_error(
                    addr, make_error(ErrorCode.SYS_CONTROLLER_DISABLED, detail=reason)
                )
            logger.warning(
                "Motion command rejected - controller disabled: %s", cmd_name
            )
            return

        # Streaming commands: cancel segment playback + existing streamable handling
        if getattr(command, "streamable", False):
            self._segment_player.cancel(state)
            self._planner_needs_sync = True
            if self.udp_transport:
                drained = self.udp_transport.drain_buffer()
                if drained > 0:
                    logger.log(TRACE, "udp_buffer_drained count=%d", drained)
            self._executor.cancel_active_streamable()
            removed = self._executor.clear_streamable_commands(
                "Streaming command prepare"
            )
            if removed:
                logger.log(TRACE, "queued_streamables_removed count=%d", removed)
            try:
                cmd_index = self._executor.queue_command(addr, command, None)
                logger.log(TRACE, "Command %s queued (index=%d)", cmd_name, cmd_index)
                if cmd_type and self._ack_policy.requires_ack(cmd_type):
                    self._reply_ok_index(addr, cmd_index)
            except QueueFullError:
                if cmd_type and self._ack_policy.requires_ack(cmd_type):
                    self._reply_error(addr, make_error(ErrorCode.COMM_QUEUE_FULL))
            return

        # Non-streaming commands → planner
        # Cancel active streaming command to avoid Position_in race
        if self._executor.cancel_active_streamable():
            self._planner_needs_sync = True

        # Clear error state from previous pipeline failure
        if state.error is not None:
            state.error = None
            state.action_state = ActionState.IDLE
            self._planner_needs_sync = True

        cmd_index = self._assign_command_index(state)
        position_in = state.Position_in.copy() if self._planner_needs_sync else None
        self._planner.submit(
            PlanCommand(
                command_index=cmd_index,
                params=command.p,
                position_in=position_in,
            )
        )
        self._planner_needs_sync = False
        logger.log(TRACE, "Command %s → planner (index=%d)", cmd_name, cmd_index)
        if cmd_type and self._ack_policy.requires_ack(cmd_type):
            self._reply_ok_index(addr, cmd_index)

    def _handle_query(
        self,
        command: QueryCommand,
        state: ControllerState,
        addr: tuple[str, int],
    ) -> None:
        """Execute query command and send response directly."""
        try:
            command.setup(state)
            response = command.compute(state)
            assert self.udp_transport is not None
            self.udp_transport.send(response, addr)
        except Exception as e:
            logger.error("Query error: %s", e)
            self._reply_error(
                addr, make_error(ErrorCode.COMM_DECODE_ERROR, detail=str(e))
            )

    def _handle_system_command(
        self,
        command: SystemCommand,
        state: ControllerState,
        addr: tuple[str, int],
    ) -> None:
        """Execute system command, apply side effects, and send reply."""
        try:
            command.setup(state)
            code = command.tick(state)

            # Reset: cancel motion pipeline so stale segments don't play
            if isinstance(command, ResetCommand):
                self._segment_player.cancel(state)
                self._planner_needs_sync = True
                self._executor.cancel_active_command("Reset")
                self._executor.clear_queue("Reset")

            # Infrastructure side effects (only 2-3 commands trigger these)
            if command._switch_simulator is not None:
                state.Command_out = CommandCode.IDLE
                state.Speed_out.fill(0)
                self._segment_player.cancel(state)
                self._planner_needs_sync = True
                self._executor.cancel_active_command("Simulator mode toggle")
                self._executor.clear_queue("Simulator mode toggle")
                success, error = self._transport_mgr.switch_simulator_mode(
                    command._switch_simulator, sync_state=state
                )
                if not success:
                    raise RuntimeError(error or "Simulator toggle failed")
            if command._switch_port is not None:
                self._transport_mgr.switch_to_port(command._switch_port)
            if command._sync_mock:
                self._transport_mgr.sync_mock_from_state(state)

            # Sync motion profile to planner (SetProfile is a SystemCommand)
            if isinstance(command, SetProfileCommand):
                self._planner.sync_profile(state.motion_profile)

            if code == ExecutionStatusCode.COMPLETED:
                self._reply_ok(addr)
            else:
                robot_error = command.robot_error or make_error(
                    ErrorCode.MOTN_TICK_FAILED, detail="System command failed"
                )
                self._reply_error(addr, robot_error)

        except Exception as e:
            logger.error("System command error: %s", e)
            self._reply_error(
                addr, extract_robot_error(e, ErrorCode.MOTN_SETUP_FAILED, detail=str(e))
            )

    def _assign_command_index(self, state: ControllerState) -> int:
        """Assign a monotonically increasing command index."""
        idx = state.next_command_index
        state.next_command_index += 1
        return idx

    def _set_high_priority(self) -> None:
        """Set highest non-privileged process priority and pin to CPU core."""
        try:
            p = psutil.Process()

            # Set priority
            if sys.platform == "win32":
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("Set process priority to HIGH_PRIORITY_CLASS")
            else:
                try:
                    p.nice(-10)
                    logger.info("Set process nice value to -10")
                except psutil.AccessDenied:
                    logger.debug("Cannot set negative nice value without privileges")

            # Pin to last CPU core (usually less contention from system tasks)
            if hasattr(p, "cpu_affinity"):
                try:
                    cpus = p.cpu_affinity()
                    if cpus and len(cpus) > 1:
                        target_core = cpus[-1]
                        p.cpu_affinity([target_core])
                        logger.info(f"Pinned process to CPU core {target_core}")
                except (AttributeError, NotImplementedError):
                    logger.debug("CPU affinity not supported on this platform")
                except psutil.AccessDenied:
                    logger.debug("Cannot set CPU affinity without privileges")

        except Exception as e:
            logger.warning(f"Failed to set process priority/affinity: {e}")
