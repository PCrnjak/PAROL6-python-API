"""
Main controller for PAROL6 robot server.
"""

import logging
import socket
import time
import threading
import argparse
import re
from typing import Optional, Dict, Any, List, Tuple, Deque, Union, Sequence, cast
from dataclasses import dataclass, field
from collections import deque

import numpy as np

from parol6.server.state import StateManager, ControllerState
from parol6.server.transports.udp_transport import UDPTransport
from parol6.server.transports.serial_transport import SerialTransport
from parol6.server.transports.mock_serial_transport import MockSerialTransport
from parol6.server.transports import create_and_connect_transport, is_simulation_mode
from parol6.server.command_registry import discover_commands, create_command_from_parts
from parol6.server.status_broadcast import start_status_services
from parol6.server.status_cache import get_cache
from parol6.protocol.wire import CommandCode, unpack_rx_frame_into
from parol6.gcode import GcodeInterpreter
from parol6.config import INTERVAL_S, save_com_port
from parol6.commands.base import CommandBase, ExecutionStatus, ExecutionStatusCode, QueryCommand, MotionCommand, SystemCommand

logger = logging.getLogger("parol6.server.controller")


@dataclass
class ExecutionContext:
    """Context passed to commands during execution."""
    udp_transport: Optional[UDPTransport]
    serial_transport: Optional[Union[SerialTransport, MockSerialTransport]]
    gcode_interpreter: Optional[GcodeInterpreter]
    addr: Optional[Tuple[str, int]]
    state: ControllerState


@dataclass
class QueuedCommand:
    """Represents a command in the queue with metadata."""
    command: CommandBase
    command_id: Optional[str] = None
    address: Optional[Tuple[str, int]] = None
    queued_time: float = field(default_factory=time.time)
    activated: bool = False
    initialized: bool = False


@dataclass
class ControllerConfig:
    """Configuration for the controller."""
    udp_host: str = '0.0.0.0'
    udp_port: int = 5001
    serial_port: Optional[str] = None
    serial_baudrate: int = 3000000
    loop_interval: float = INTERVAL_S
    estop_recovery_delay: float = 1.0
    auto_home: bool = False


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
        self.udp_transport: Optional[UDPTransport] = None
        self.serial_transport: Optional[Union[SerialTransport, MockSerialTransport]] = None
        
        # ACK management
        self.ack_socket = None
        try:
            self.ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            logger.error(f"Failed to create ACK socket: {e}")
        
        # Command queue and tracking (merged from CommandExecutor)
        self.command_queue: Deque[QueuedCommand] = deque(maxlen=100)
        self.active_command: Optional[QueuedCommand] = None
        
        # Command tracking
        self.current_command = None
        self.command_id_map: Dict[str, Any] = {}
        
        # E-stop recovery
        self.estop_active = None  # None = unknown, True = active, False = released
        self.first_frame_received = False  # Track if we've received data from robot
        self._serial_last_version = 0  # Version of last decoded serial frame
        
        # Thread for command processing
        self.command_thread = None
        
        # GCODE interpreter
        self.gcode_interpreter = GcodeInterpreter()
        
        # Stream mode
        self.stream_mode = False

        # Status services (updater + multicast broadcaster)
        self._status_updater: Optional[Any] = None
        self._status_broadcaster: Optional[Any] = None
        
        # Initialize components on construction
        self._initialize_components()
    
    def _send_ack(self, cmd_id: str, status: str, details: str, addr: Tuple[str, int]) -> None:
        """
        Send an acknowledgment message.
        
        Args:
            cmd_id: Command ID to acknowledge
            status: Status (QUEUED, EXECUTING, COMPLETED, FAILED, CANCELLED)
            details: Optional details message
            addr: Address tuple (host, port) to send to
        """
        if not cmd_id or not self.ack_socket:
            return
        
        # Debug log all outgoing ACKs
        logger.debug(f"ACK {status} cmd={cmd_id} details='{details}' addr={addr}")
        
        message = f"ACK|{cmd_id}|{status}|{details}".encode("utf-8")
        
        try:
            self.ack_socket.sendto(message, addr)
        except Exception as e:
            logger.error(f"Failed to send ACK to {addr[0]}:{addr[1]} - {e}")
    
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
            logger.info(f"Starting UDP server on {self.config.udp_host}:{self.config.udp_port}")
            self.udp_transport = UDPTransport(self.config.udp_host, self.config.udp_port)
            if not self.udp_transport.create_socket():
                raise RuntimeError("Failed to create UDP socket")
            
            # Initialize Serial transport using factory
            if self.config.serial_port or is_simulation_mode():
                self.serial_transport = create_and_connect_transport(
                    port=self.config.serial_port,
                    baudrate=self.config.serial_baudrate,
                    auto_find_port=True
                )
                
                if self.serial_transport:
                    if hasattr(self.serial_transport, 'port'):
                        logger.info(f"Connected to transport: {self.serial_transport.port}")
                    # Start reduced-copy serial reader thread if available
                    if hasattr(self.serial_transport, "start_reader"):
                        try:
                            self.serial_transport.start_reader(self.shutdown_event)
                            logger.info("Serial reader thread started")
                        except Exception as e:
                            logger.warning(f"Failed to start serial reader: {e}")
            else:
                logger.warning("No serial port configured. Waiting for SET_PORT via UDP or set --serial/PAROL6_COM_PORT/com_port.txt before connecting.")
            
            # Initialize robot state
            self.state_manager.reset_state()

            # Optionally queue auto-home per policy (default OFF)
            if self.config.auto_home:
                try:
                    home_cmd = create_command_from_parts(["HOME"])
                    if home_cmd:
                        # Queue without address/id for auto-home
                        self._queue_command(("127.0.0.1", 0), home_cmd, None)
                        logger.info("Auto-home queued")
                except Exception as e:
                    logger.warning(f"Failed to queue auto-home: {e}")
            
            # Start status updater and broadcaster (ASCII multicast at configured rate)
            try:
                self._status_updater, self._status_broadcaster = start_status_services(self.state_manager)
                logger.info("Status updater and broadcaster started")
            except Exception as e:
                logger.warning(f"Failed to start status services: {e}")

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
        
        self.running = True
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._command_processing_loop)
        self.command_thread.start()
        
        # Start main control loop
        logger.info("Starting main control loop")
        self._main_control_loop()
    
    def stop(self):
        """Stop the controller and clean up resources."""
        logger.info("Stopping controller...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for threads to finish
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=2.0)
        
        # Stop status services
        try:
            if self._status_broadcaster:
                self._status_broadcaster.stop()
            if self._status_updater:
                self._status_updater.stop()
        except Exception:
            pass

        # Clean up transports
        if self.udp_transport:
            self.udp_transport.close_socket()
        
        if self.serial_transport:
            self.serial_transport.disconnect()
        
        logger.info("Controller stopped")
    
    def _main_control_loop(self):
        """
        Main control loop that:
        1. Reads from firmware (serial)
        2. Handles E-stop and recovery
        3. Executes active command or fetches from GCODE
        4. Writes to firmware (serial)
        5. Maintains timing
        """
        
        tick = self.config.loop_interval
        next_t = time.perf_counter()
        prev_t = next_t  # for period measurement

        while self.running:
            try:
                loop_start = time.time()
                state = self.state_manager.get_state()
                
                # 1. Read from firmware
                if self.serial_transport and self.serial_transport.is_connected():
                    try:
                        mv, ver, ts = self.serial_transport.get_latest_frame_view()
                        if mv is not None and ver != self._serial_last_version:
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
                                if not self.first_frame_received:
                                    self.first_frame_received = True
                                    logger.debug("First frame received from robot")
                                self._serial_last_version = ver
                    except Exception as e:
                        logger.warning(f"Error decoding latest serial frame: {e}")
                
                # Serial auto-reconnect when a port is known
                if self.serial_transport and not self.serial_transport.is_connected():
                    if getattr(self.serial_transport, 'port', None):
                        self.serial_transport.auto_reconnect()

                # 2. Handle E-stop (only check when connected to robot and received first frame)
                if self.serial_transport and self.serial_transport.is_connected() and self.first_frame_received:
                    if state.InOut_in[4] == 0:  # E-stop pressed (0 = pressed, 1 = released)
                        if self.estop_active != True:  # Not already in E-stop state
                            logger.warning("E-STOP activated")
                            self.estop_active = True
                            # Cancel active command
                            if self.active_command:
                                self._cancel_active_command("E-Stop activated")
                            # Clear queue
                            self._clear_queue("E-Stop activated")
                            # Stop robot
                            state.Command_out = CommandCode.DISABLE
                            state.Speed_out.fill(0)
                    elif state.InOut_in[4] == 1:  # E-stop released (1 = released)
                        if self.estop_active == True:  # Was in E-stop state
                            # E-stop was released - automatic recovery
                            logger.info("E-STOP released - automatic recovery")
                            self.estop_active = False
                            # Re-enable immediately per policy
                            state.enabled = True
                            state.disabled_reason = ""
                            state.Command_out = CommandCode.IDLE
                            state.Speed_out.fill(0)
                
                # 3. Execute commands if not in E-stop (or E-stop state unknown)
                if self.estop_active != True:  # Execute if E-stop is False or None (unknown)
                    # Execute active command
                    if self.active_command or self.command_queue:
                        self._execute_active_command()
                    # Check for GCODE commands if program is running
                    elif self.gcode_interpreter.is_running:
                        self._fetch_gcode_commands()
                    else:
                        # No commands - idle
                        state.Command_out = CommandCode.IDLE
                        state.Speed_out.fill(0)
                        np.copyto(state.Position_out, state.Position_in)
                
                # 4. Write to firmware
                if self.serial_transport and self.serial_transport.is_connected():
                    # Optimized to pass arrays directly without creating lists
                    ok = self.serial_transport.write_frame(
                        cast(List[int], state.Position_out),
                        cast(List[int], state.Speed_out),
                        state.Command_out.value,
                        cast(List[int], state.Affected_joint_out),
                        cast(List[int], state.InOut_out),
                        state.Timeout_out,
                        cast(List[int], state.Gripper_data_out),
                    )
                    if ok:
                        # Auto-reset one-shot gripper modes after successful send
                        if state.Gripper_data_out[4] in (1, 2):
                            state.Gripper_data_out[4] = 0

                # 5. Maintain loop timing using deadline scheduling + update loop metrics
                now = time.perf_counter()
                # Update period metrics
                period = now - prev_t
                prev_t = now
                state.loop_count += 1
                state.last_period_s = float(period)
                if state.ema_period_s <= 0.0:
                    state.ema_period_s = float(period)
                else:
                    # EMA with alpha=0.1
                    state.ema_period_s = 0.1 * float(period) + 0.9 * float(state.ema_period_s)

                next_t += tick
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    # Overrun; catch up and log if severe
                    state.overrun_count += 1
                    next_t = time.perf_counter()
                    if -sleep > tick * 0.5:
                        logger.warning(f"Control loop overrun by {-sleep:.4f}s (target: {tick:.4f}s)")
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in main control loop: {e}", exc_info=True)
                # Continue running despite errors
    
    def _command_processing_loop(self):
        """
        Separate thread for processing incoming commands from UDP.
        """
        # Compile regex for command ID validation (8 hex chars)
        cmd_id_pattern = re.compile(r'^[0-9a-fA-F]{8}$')
        
        while self.running and self.udp_transport:
            try:
                # Check for new commands from UDP (blocking with short timeout)
                tup = self.udp_transport.receive_one()
                if tup is None:
                    continue
                message_str, addr = tup
                # Parse command ID and payload
                state = self.state_manager.get_state()
                parts = message_str.split('|', 1)
                cmd_id = parts[0] if (len(parts) > 1 and cmd_id_pattern.match(parts[0])) else None
                cmd_str = parts[1] if cmd_id else message_str
                # Parse command name
                cmd_parts = cmd_str.split('|')
                cmd_name = cmd_parts[0].upper()
                # Create command instance from message
                command = create_command_from_parts(cmd_parts)
                if not command:
                    logger.warning(f"Unknown command: {cmd_str}")
                    if cmd_id:
                        self._send_ack(cmd_id, "FAILED", "Unknown command", addr)
                    continue

                # Handle system commands (they can execute regardless of enable state)
                if isinstance(command, SystemCommand):
                    # System commands execute immediately without queueing
                    command.setup(state, udp_transport=self.udp_transport, addr=addr)
                    status = command.execute_step(state)
                    # Send ACK based on status
                    if cmd_id:
                        if status.code == ExecutionStatusCode.COMPLETED:
                            self._send_ack(cmd_id, "COMPLETED", status.message, addr)
                        elif status.code == ExecutionStatusCode.FAILED:
                            self._send_ack(cmd_id, "FAILED", status.message, addr)
                    continue

                # Check if controller is enabled for motion commands
                if isinstance(command, MotionCommand) and not state.enabled:
                    if cmd_id:
                        reason = state.disabled_reason or "Controller disabled"
                        self._send_ack(cmd_id, "FAILED", reason, addr)
                    logger.warning(f"Motion command rejected - controller disabled: {cmd_name}")
                    continue

                # Query commands execute immediately (bypass queue)
                if isinstance(command, QueryCommand):
                    # Execute query immediately with context
                    command.setup(
                        state,
                        udp_transport=self.udp_transport,
                        addr=addr,
                        gcode_interpreter=self.gcode_interpreter,
                    )
                    status = command.execute_step(state)
                    # Query commands typically send their own responses
                    if cmd_id and status.code == ExecutionStatusCode.FAILED:
                        self._send_ack(cmd_id, "FAILED", status.message, addr)
                    continue

                # Apply stream mode logic for streamable motion commands
                if self.stream_mode and isinstance(command, MotionCommand) and getattr(command, 'streamable', False):
                    # Cancel any active streamable command and replace it (suppress per-command ACK to reduce UDP chatter)
                    if self.active_command and isinstance(self.active_command.command, MotionCommand) and getattr(self.active_command.command, 'streamable', False):
                        self.active_command = None

                # Clear any queued streamable commands without per-command ACKs to reduce UDP chatter
                removed = 0
                for queued_cmd in list(self.command_queue):
                    if isinstance(queued_cmd.command, MotionCommand) and getattr(queued_cmd.command, 'streamable', False):
                        self.command_queue.remove(queued_cmd)
                        removed += 1
                if removed:
                    logger.debug(f"Stream mode: removed {removed} queued streamable command(s)")

                # Queue the command
                status = self._queue_command(addr, command, cmd_id)
                logger.debug(f"Command {cmd_name} queued with status: {status.code}")

                # Start execution if no active command
                if not self.active_command:
                    self._execute_active_command()
            except Exception as e:
                logger.error(f"Error in command processing: {e}", exc_info=True)
    
    
    def _queue_command(self, 
                      address: Optional[Tuple[str, int]],
                      command: CommandBase, 
                      command_id: Optional[str] = None
                      ) -> ExecutionStatus:
        """
        Add a command to the execution queue.
        
        Args:
            command: The command to queue
            command_id: Optional ID for tracking
            address: Optional (ip, port) for acknowledgments
            priority: Priority level (higher = executed first)
            
        Returns:
            ExecutionStatus indicating queue status
        """
        # Check if queue is full
        if len(self.command_queue) >= 100:  # max_queue_size
            logger.warning(f"Command queue full (max 100)")
            if command_id and address:
                self._send_ack(command_id, "FAILED", "Queue full", address)
            return ExecutionStatus.failed("Queue full")
        
        # Create queued command
        queued_cmd = QueuedCommand(
            command=command,
            command_id=command_id,
            address=address
        )
        
        self.command_queue.append(queued_cmd)
        
        # Send acknowledgment
        if command_id and address:
            queue_pos = len(self.command_queue)
            self._send_ack(command_id, "QUEUED", f"Position {queue_pos} in queue", address)
        
        logger.debug(f"Queued command: {type(command).__name__} (ID: {command_id})")
        
        return ExecutionStatus(
            code=ExecutionStatusCode.QUEUED,
            message=f"Command queued at position {len(self.command_queue)}"
        )
    
    def _execute_active_command(self) -> Optional[ExecutionStatus]:
        """
        Execute one step of the active command from the queue.
        
        Returns:
            ExecutionStatus of the execution, or None if no active command
        """
        # Check if we need to activate a new command from queue
        if self.active_command is None:
            if not self.command_queue:
                return None
            # Get next command from queue
            self.active_command = self.command_queue.popleft()
            # mark not yet activated
            self.active_command.activated = False
        
        # Execute active command step
        if self.active_command:
            ac = self.active_command
            try:
                state = self.state_manager.get_state()
                ac = self.active_command
                
                # Check if controller is enabled
                if state.enabled:
                    # Perform setup and EXECUTING ACK only once
                    if ac and not getattr(ac, "activated", False):
                        ac.command.setup(state)
                        
                        # Send executing acknowledgment once
                        if ac.command_id and ac.address:
                            self._send_ack(
                                ac.command_id,
                                "EXECUTING",
                                f"Starting {type(ac.command).__name__}",
                                ac.address
                            )
                        
                        ac.activated = True
                        logger.debug(f"Activated command: {type(ac.command).__name__} (id={ac.command_id})")

                else:
                    # Cancel command due to disabled controller
                    self._cancel_active_command("Controller disabled")
                    return ExecutionStatus(
                        code=ExecutionStatusCode.CANCELLED,
                        message="Controller disabled"
                    )
                
                # Execute command step
                status = ac.command.execute_step(state)

                # Enqueue any generated commands (e.g., from GCODE parsed in queued mode)
                if status.details and isinstance(status.details, dict) and 'enqueue' in status.details:
                    try:
                        for robot_cmd_str in status.details['enqueue']:

                            cmd_obj = create_command_from_parts(robot_cmd_str.split("|"))
                            if cmd_obj:
                                # Queue without address/id for generated commands
                                self._queue_command(("127.0.0.1", 0), cmd_obj, None)
                    except Exception as e:
                        logger.error(f"Error enqueuing generated commands: {e}")
                
                # Check if command is finished
                if status.code == ExecutionStatusCode.COMPLETED:
                    # Command completed successfully
                    name = type(ac.command).__name__
                    cid, addr = ac.command_id, ac.address
                    logger.debug(f"Command completed: {name} (id={cid}) at t={time.time():.6f}")
                    
                    # Send completion acknowledgment
                    if cid and addr:
                        self._send_ack(
                            cid,
                            "COMPLETED",
                            status.message,
                            addr
                        )
                    
                    # Record and clear
                    self.active_command = None
                    
                elif status.code == ExecutionStatusCode.FAILED:
                    # Command failed
                    name = type(ac.command).__name__
                    cid, addr = ac.command_id, ac.address
                    logger.debug(f"Command failed: {name} (id={cid}) - {status.message} at t={time.time():.6f}")
                    
                    # Send failure acknowledgment
                    if cid and addr:
                        self._send_ack(
                            cid,
                            "FAILED",
                            status.message,
                            addr
                        )
                    
                    self.active_command = None
                
                return status
                
            except Exception as e:
                logger.error(f"Command execution error: {e}")
                
                # Handle execution error - save command info before clearing
                cid = ac.command_id if ac else None
                addr = ac.address if ac else None
                
                if cid and addr:
                    self._send_ack(cid, "FAILED", f"Execution error: {str(e)}", addr)
                self.active_command = None
                
                return ExecutionStatus.failed(f"Execution error: {str(e)}", error=e)
        
        return None
    
    def _cancel_active_command(self, reason: str = "Cancelled by user") -> None:
        """
        Cancel the currently active command.
        
        Args:
            reason: Reason for cancellation
        """
        if not self.active_command:
            return
        
        logger.info(f"Cancelling active command: {type(self.active_command.command).__name__} - {reason}")
        
        # Send cancellation acknowledgment
        if self.active_command.command_id and self.active_command.address:
            self._send_ack(
                self.active_command.command_id,
                "CANCELLED",
                reason,
                self.active_command.address
            )
        
        # Record and clear
        self.active_command = None
    
    def _clear_queue(self, reason: str = "Queue cleared") -> List[Tuple[str, ExecutionStatus]]:
        """
        Clear all queued commands.
        
        Args:
            reason: Reason for clearing the queue
            
        Returns:
            List of (command_id, status) for cleared commands
        """
        cleared = []
        # TODO: don't send out an ack for every queued command, just one signalling queues been cleared
        while self.command_queue:
            queued_cmd = self.command_queue.popleft()
            
            # Send cancellation acknowledgment
            if queued_cmd.command_id and queued_cmd.address:
                self._send_ack(
                    queued_cmd.command_id,
                    "CANCELLED",
                    reason,
                    queued_cmd.address
                )
            
            # Record cleared command
            if queued_cmd.command_id:
                status = ExecutionStatus(
                    code=ExecutionStatusCode.CANCELLED,
                    message=reason
                )
                cleared.append((queued_cmd.command_id, status))
        
        logger.info(f"Cleared {len(cleared)} commands from queue: {reason}")
        
        return cleared
    
    
    
    def _fetch_gcode_commands(self):
        """
        Fetch next command from GCODE interpreter if program is running.
        Converts GCODE output to command objects and queues them.
        """
        if not self.gcode_interpreter.is_running:
            return
        
        try:
            # Get next command from GCODE program
            next_gcode_cmd = self.gcode_interpreter.get_next_command()
            if not next_gcode_cmd:
                return
            
            # Use command registry to create command object
            command_obj = create_command_from_parts(next_gcode_cmd.split("|"))
            
            if command_obj:
                # Queue without address/id for GCODE commands
                self._queue_command(("127.0.0.1", 0), command_obj, None)
                cmd_name = next_gcode_cmd.split('|')[0] if '|' in next_gcode_cmd else next_gcode_cmd
                logger.debug(f"Queued GCODE command: {cmd_name}")
            else:
                logger.warning(f"Unknown GCODE command generated: {next_gcode_cmd}")
                
        except Exception as e:
            logger.error(f"Error fetching GCODE commands: {e}")


def main():
    """Main entry point for the controller."""    
    # Parse arguments first to get logging level
    parser = argparse.ArgumentParser(description='PAROL6 Robot Controller')
    parser.add_argument('--host', default='0.0.0.0', help='UDP host address')
    parser.add_argument('--port', type=int, default=5001, help='UDP port')
    parser.add_argument('--serial', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=3000000, help='Serial baudrate')
    parser.add_argument('--auto-home', action='store_true',
                       help='Queue HOME on startup (default: off)')
    
    # Verbose logging options (from controller.py)
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Enable quiet logging (WARNING level)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set specific log level')
    args = parser.parse_args()
    
    # Determine log level
    if args.log_level:
        log_level = getattr(logging, args.log_level)
    elif args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    
    # Set up logging with determined level
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Create configuration
    config = ControllerConfig(
        udp_host=args.host,
        udp_port=args.port,
        serial_port=args.serial,
        serial_baudrate=args.baudrate,
        auto_home=bool(args.auto_home)
    )
    
    # Create and run controller
    try:
        controller = Controller(config)
        
        if controller.is_initialized():
            try:
                controller.start()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
            finally:
                controller.stop()
        else:
            logger.error("Controller not properly initialized")
            return 1
    except RuntimeError as e:
        logger.error(f"Failed to create controller: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
