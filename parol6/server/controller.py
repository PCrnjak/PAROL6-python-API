"""
Main controller for PAROL6 robot server.
"""

import logging
import os
import socket
import time
import threading
from typing import Optional, Dict, Any, List, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque

from parol6.server.state import StateManager, ControllerState
from parol6.server.transports.udp_transport import UDPTransport
from parol6.server.transports.serial_transport import SerialTransport
from parol6.server.command_registry import discover_commands, create_command
from parol6.protocol.wire import pack_tx_frame, unpack_rx_frame, CommandCode
from parol6.server.simulation import SimulationState, create_simulation_state, is_fake_serial_enabled, simulate_motion
from parol6.gcode import GcodeInterpreter
from parol6.config import INTERVAL_S, get_com_port_with_fallback, save_com_port, COMMAND_COOLDOWN_MS
from parol6.commands.base import CommandBase, ExecutionStatus, ExecutionStatusCode

logger = logging.getLogger("parol6.server.controller")


@dataclass
class ExecutionContext:
    """Context passed to commands during execution."""
    udp_transport: Optional[UDPTransport]
    serial_transport: Optional[SerialTransport]
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
    priority: int = 0  # Higher priority = executed first


@dataclass
class ControllerConfig:
    """Configuration for the controller."""
    udp_host: str = '0.0.0.0'
    udp_port: int = 5001
    serial_port: Optional[str] = None
    serial_baudrate: int = 3000000
    loop_interval: float = INTERVAL_S
    auto_estop_recovery: bool = True
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
        
        # Core components
        self.state_manager = StateManager()
        self.udp_transport = None
        self.serial_transport = None
        
        # Debug flags
        self.debug_loop = os.getenv("PAROL6_DEBUG_LOOP", "0").lower() in ("1", "true", "yes", "on")
        self.loop_counter = 0
        self.loop_times = deque(maxlen=50)  # Track last 50 loop times for averaging
        
        # ACK management (merged from AckManager)
        self.ack_port = self._get_ack_port()
        self.ack_socket = None
        try:
            self.ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            logger.error(f"Failed to create ACK socket: {e}")
        
        # Command queue and tracking (merged from CommandExecutor)
        self.command_queue: Deque[QueuedCommand] = deque(maxlen=100)
        self.active_command: Optional[QueuedCommand] = None
        self.command_history: Deque[Tuple[str, float, ExecutionStatusCode]] = deque(maxlen=50)
        
        # Command tracking
        self.current_command = None
        self.command_id_map: Dict[str, Any] = {}
        
        # Execution statistics
        self.total_executed = 0
        self.total_failed = 0
        self.total_cancelled = 0
        
        # E-stop recovery
        self.estop_active = None  # None = unknown, True = active, False = released
        self.estop_recovery_time = None
        self.first_frame_received = False  # Track if we've received data from robot
        
        # Thread for command processing
        self.command_thread = None
        
        # GCODE interpreter
        self.gcode_interpreter = GcodeInterpreter()
        
        # Stream mode
        self.stream_mode = False
        
        # Simulation mode
        self.simulation = None
        if is_fake_serial_enabled():
            self.simulation = create_simulation_state()
            logger.info("FAKE_SERIAL mode enabled - running in simulation")
    
    def _get_ack_port(self) -> int:
        """Get the acknowledgment port from environment or use default."""
        try:
            return int(os.getenv("PAROL6_ACK_PORT", "5002"))
        except Exception:
            return 5002
    
    def _send_ack(self, cmd_id: str, status: str, details: str = "", addr: Optional[Tuple[str, int]] = None) -> None:
        """
        Send an acknowledgment message.
        
        Args:
            cmd_id: Command ID to acknowledge
            status: Status (QUEUED, EXECUTING, COMPLETED, FAILED, CANCELLED)
            details: Optional details message
            addr: Optional address tuple (host, port) to send to
        """
        if not cmd_id or not self.ack_socket:
            return
        
        # Debug log all outgoing ACKs
        logger.debug(f"ACK {status} cmd={cmd_id} details='{details}' addr={addr}")
        
        message = f"ACK|{cmd_id}|{status}|{details}".encode("utf-8")
        
        try:
            # Send to original sender if address provided
            if addr and isinstance(addr, tuple) and len(addr) >= 1:
                try:
                    self.ack_socket.sendto(message, (addr[0], self.ack_port))
                except Exception as e:
                    logger.error(f"Failed to send ACK to {addr[0]}:{self.ack_port} - {e}")
            
            # Always mirror to localhost for local clients
            try:
                self.ack_socket.sendto(message, ("127.0.0.1", self.ack_port))
            except Exception:
                pass  # Best-effort
                
        except Exception as e:
            logger.warning(f"ACK send error: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize all components and establish connections.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Discover and register all commands
            discover_commands()
            
            # Initialize UDP transport
            logger.info(f"Starting UDP server on {self.config.udp_host}:{self.config.udp_port}")
            self.udp_transport = UDPTransport(self.config.udp_host, self.config.udp_port)
            if not self.udp_transport.create_socket():
                logger.error("Failed to create UDP socket")
                return False
            
            # Initialize Serial transport
            serial_port = self.config.serial_port
            port_from_file = False
            
            if not serial_port and not is_fake_serial_enabled():
                # No port specified and not in simulation - use persistence
                serial_port = get_com_port_with_fallback()
                # Check if port came from file (not from environment)
                env_port = os.getenv("PAROL6_COM_PORT") or os.getenv("PAROL6_SERIAL")
                port_from_file = serial_port and not env_port
                
            if serial_port:
                logger.info(f"Connecting to serial port {serial_port}")
                self.serial_transport = SerialTransport(
                    serial_port,
                    self.config.serial_baudrate
                )
                    
                if not self.serial_transport.connect():
                    logger.error("Failed to connect to serial port")
                    return False
                    
                # Only save if port was explicitly set (not loaded from file)
                if not port_from_file:
                    save_com_port(serial_port)
                
                # Update state with port info
                state = self.state_manager.get_state()
                state.com_port_str = serial_port
                state.com_port_cache = serial_port
            else:
                logger.warning("No serial port configured. Waiting for SET_PORT via UDP or set --serial/PAROL6_COM_PORT/com_port.txt before connecting.")
            
            # Initialize robot state
            self.state_manager.reset_state()

            # Optionally queue auto-home per policy (default OFF)
            if self.config.auto_home:
                try:
                    home_cmd = create_command("HOME")
                    if home_cmd:
                        self._queue_command(home_cmd)
                        logger.info("Auto-home queued")
                except Exception as e:
                    logger.warning(f"Failed to queue auto-home: {e}")
            
            logger.info("Controller initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize controller: {e}")
            return False
    
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
        last_time = time.time()
        
        while self.running:
            try:
                loop_start = time.time()
                state = self.state_manager.get_state()
                
                self.loop_counter += 1
                
                # 1. Read from firmware
                if self.serial_transport and self.serial_transport.is_connected():
                    frames = self.serial_transport.read_frames()
                    for frame in frames:
                        self._update_state_from_serial_frame(frame)
                elif self.simulation:
                    # Update simulation
                    self._process_firmware_data(None)
                
                # Serial auto-reconnect when a port is known
                if self.serial_transport and not self.serial_transport.is_connected():
                    if getattr(self.serial_transport, 'port', None):
                        try:
                            self.serial_transport.auto_reconnect()
                        except Exception as e:
                            logger.debug(f"Serial auto-reconnect attempt failed: {e}")

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
                            state.Speed_out[:] = [0] * 6
                    elif state.InOut_in[4] == 1:  # E-stop released (1 = released)
                        if self.estop_active == True:  # Was in E-stop state
                            # E-stop was released - automatic recovery
                            logger.info("E-STOP released - automatic recovery")
                            self.estop_active = False
                            # Re-enable immediately per policy (no keyboard flow)
                            state.enabled = True
                            state.disabled_reason = ""
                            state.Command_out = CommandCode.IDLE
                            state.Speed_out[:] = [0] * 6
                
                # 3. Execute commands if not in E-stop (or E-stop state unknown)
                if self.estop_active != True:  # Execute if E-stop is False or None (unknown)
                    # Execute active command
                    if self.active_command:
                        self._execute_active_command()
                    # Check for new command from queue
                    elif self.command_queue:
                        self._execute_active_command()
                    # Check for GCODE commands if program is running
                    elif self.gcode_interpreter.is_running:
                        self._fetch_gcode_commands()
                    else:
                        # No commands - idle
                        state.Command_out = CommandCode.IDLE
                        state.Speed_out[:] = [0] * 6
                        state.Position_out[:] = state.Position_in[:]
                
                # 4. Write to firmware
                if self.serial_transport and self.serial_transport.is_connected():
                    ok = self.serial_transport.write_frame(
                        list(state.Position_out),
                        list(state.Speed_out),
                        state.Command_out.value,
                        list(state.Affected_joint_out),
                        list(state.InOut_out),
                        state.Timeout_out,
                        list(state.Gripper_data_out)
                    )
                    if ok:
                        # Auto-reset one-shot gripper modes after successful send
                        if state.Gripper_data_out[4] in (1, 2):
                            state.Gripper_data_out[4] = 0
                
                # Attempt auto-recovery if scheduled
                self._handle_estop_recovery()

                # 5. Maintain loop timing
                elapsed = time.time() - loop_start
                if elapsed < self.config.loop_interval:
                    time.sleep(self.config.loop_interval - elapsed)
                elif elapsed > self.config.loop_interval * 1.5:
                    logger.warning(f"Control loop took {elapsed:.3f}s (target: {self.config.loop_interval:.3f}s)")
                
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
        while self.running:
            try:
                # Check for new commands from UDP
                messages = self.udp_transport.receive_messages()
                for msg in messages:
                    self._process_udp_command(msg.data, msg.address)
                    
            except Exception as e:
                logger.error(f"Error in command processing: {e}", exc_info=True)
    
    def _process_udp_command(self, message: str, addr):
        """
        Process a command received via UDP with proper lifecycle management.
        
        Args:
            message: The command message string
            addr: The sender's address
        """
        try:
            # Parse command ID and name first
            state = self.state_manager.get_state()
            parts = message.split('|', 1)
            cmd_id = None
            cmd_str = message
            if len(parts) > 1 and len(parts[0]) == 8 and all(c in "0123456789abcdef" for c in parts[0].lower()):
                cmd_id = parts[0]
                cmd_str = parts[1]
            # Parse command name
            cmd_parts = cmd_str.split('|')
            cmd_name = cmd_parts[0].upper()

            # Handle system commands immediately (no cooldown)
            if cmd_name in ["STOP", "ENABLE", "DISABLE", "CLEAR_ERROR", "SET_PORT", "STREAM"]:
                self._handle_system_command(cmd_name, cmd_parts, cmd_id, addr)
                return

            # Handle query commands immediately (no cooldown)
            if cmd_name in ["GET_POSE", "GET_ANGLES", "GET_IO", "GET_GRIPPER", "GET_SPEEDS", 
                            "GET_GCODE_STATUS", "GCODE_STOP", "GCODE_PAUSE", "GCODE_RESUME", 
                            "PING", "GET_SERVER_STATE", "GET_STATUS"]:
                self._handle_query_command(cmd_name, cmd_parts, cmd_id, addr)
                return

            # Apply cooldown only to streaming jog commands
            if state.cooldown_config.enabled and cmd_name in ["JOG", "CARTJOG", "MULTIJOG"]:
                current_time = time.time() * 1000  # Convert to milliseconds
                elapsed_ms = current_time - state.cooldown_config.last_processed_time
                if elapsed_ms < state.cooldown_config.cooldown_ms:
                    logger.debug(f"Command ignored due to cooldown ({elapsed_ms:.1f}ms < {state.cooldown_config.cooldown_ms}ms) for {cmd_name}")
                    return
                # Update last processed time
                state.cooldown_config.last_processed_time = current_time
            parts = message.split('|', 1)
            cmd_id = None
            cmd_str = message
            
            # Check if first part is a command ID (8-char hex)
            if len(parts) > 1 and len(parts[0]) == 8 and all(c in "0123456789abcdef" for c in parts[0].lower()):
                cmd_id = parts[0]
                cmd_str = parts[1]
            
            # Parse command name
            cmd_parts = cmd_str.split('|')
            cmd_name = cmd_parts[0].upper()
            
            # Handle system commands directly
            if cmd_name in ["STOP", "ENABLE", "DISABLE", "CLEAR_ERROR", "SET_PORT", "STREAM"]:
                self._handle_system_command(cmd_name, cmd_parts, cmd_id, addr)
                return
            
            # Handle query commands directly (GET_POSE, GET_ANGLES, etc)
            if cmd_name in ["GET_POSE", "GET_ANGLES", "GET_IO", "GET_GRIPPER", "GET_SPEEDS", 
                          "GET_GCODE_STATUS", "GCODE_STOP", "GCODE_PAUSE", "GCODE_RESUME", 
                          "PING", "GET_SERVER_STATE", "GET_STATUS"]:
                self._handle_query_command(cmd_name, cmd_parts, cmd_id, addr)
                return
            
            # Check if controller is enabled for motion commands
            if not state.enabled and cmd_name in ['MOVEPOSE','MOVEJOINT','MOVECART','JOG','MULTIJOG','CARTJOG',
                                                   'SMOOTH_CIRCLE','SMOOTH_ARC_CENTER','SMOOTH_ARC_PARAM',
                                                   'SMOOTH_SPLINE','SMOOTH_HELIX','SMOOTH_BLEND','HOME']:
                if cmd_id:
                    reason = state.disabled_reason or "Controller disabled"
                    self._send_ack(cmd_id, "FAILED", reason, addr)
                logger.warning(f"Motion command rejected - controller disabled: {cmd_name}")
                return
            
            # Create command instance from message
            command = create_command(cmd_str)
            if not command:
                logger.warning(f"Unknown command: {cmd_str}")
                if cmd_id:
                    self._send_ack(cmd_id, "FAILED", "Unknown command", addr)
                return
            
            # Apply stream mode logic for jog commands
            if self.stream_mode and cmd_name in ['JOG', 'CARTJOG', 'MULTIJOG']:
                # Cancel any active jog command and replace it
                if self.active_command and type(self.active_command.command).__name__ in ['JogCommand', 'CartesianJogCommand', 'MultiJogCommand']:
                    # Send cancellation for active command
                    if self.active_command.command_id:
                        self._send_ack(self.active_command.command_id, "CANCELLED", 
                                     "Replaced by new stream command", self.active_command.address)
                    self.active_command = None
                
                # Clear any queued jog commands
                for queued_cmd in list(self.command_queue):
                    if type(queued_cmd.command).__name__ in ['JogCommand', 'CartesianJogCommand', 'MultiJogCommand']:
                        self.command_queue.remove(queued_cmd)
                        if queued_cmd.command_id:
                            self._send_ack(queued_cmd.command_id, "CANCELLED",
                                         "Replaced by streaming jog", queued_cmd.address)
            
            # Queue the command
            status = self._queue_command(command, cmd_id, addr)
            logger.debug(f"Command {cmd_name} queued with status: {status.code}")
            
            # Start execution if no active command
            if not self.active_command:
                self._execute_active_command()
                
        except Exception as e:
            logger.error(f"Error processing UDP command: {e}", exc_info=True)
            if cmd_id:
                self._send_ack(cmd_id, "FAILED", str(e), addr)
    
    def _handle_system_command(self, command: str, parts: List[str], cmd_id: Optional[str], addr):
        """
        Handle system commands directly.
        
        Args:
            command: The command name (e.g., 'STOP', 'ENABLE')
            parts: Command parts split by |
            cmd_id: Optional command ID
            addr: Sender address
        """
        state = self.state_manager.get_state()
        
        if command == "STOP":
            logger.info("STOP command received")
            # Cancel active command
            if self.active_command:
                if self.active_command.command_id:
                    self._send_ack(self.active_command.command_id, "CANCELLED", 
                                 "Stopped by user", self.active_command.address)
                self.active_command = None
            
            # Clear queue
            self._clear_queue("Stopped by user")
            
            # Stop robot motion
            state.Command_out = CommandCode.IDLE
            state.Speed_out[:] = [0] * 6
            
            if cmd_id:
                self._send_ack(cmd_id, "COMPLETED", "Emergency stop executed", addr)
            
        elif command == "ENABLE":
            state.enabled = True
            state.disabled_reason = ""
            logger.info("Controller enabled")
            if cmd_id:
                self._send_ack(cmd_id, "COMPLETED", "Controller enabled", addr)
            
        elif command == "DISABLE":
            state.enabled = False
            state.disabled_reason = "Disabled by user"
            # Cancel active command
            if self.active_command:
                self._cancel_active_command("Disabled by user")
            # Clear queue
            self._clear_queue("Controller disabled")
            # Stop robot motion
            state.Command_out = CommandCode.IDLE
            state.Speed_out[:] = [0] * 6
            logger.info("Controller disabled")
            if cmd_id:
                self._send_ack(cmd_id, "COMPLETED", "Controller disabled", addr)
            
        elif command == "CLEAR_ERROR":
            state.soft_error = False
            logger.info("Errors cleared")
            if cmd_id:
                self._send_ack(cmd_id, "COMPLETED", "Errors cleared", addr)
            
        elif command == "SET_PORT" and len(parts) >= 2:
            new_port = parts[1].strip()
            if new_port:
                try:
                    # Disconnect any existing connection
                    if self.serial_transport:
                        try:
                            self.serial_transport.disconnect()
                        except Exception:
                            pass
                    # Create new transport and attempt connection immediately
                    self.serial_transport = SerialTransport(
                        new_port,
                        self.config.serial_baudrate
                    )
                    if self.serial_transport.connect():
                        save_com_port(new_port)
                        state.com_port_str = new_port
                        state.com_port_cache = new_port
                        if cmd_id:
                            self._send_ack(cmd_id, "COMPLETED", f"Port set to {new_port} and connected", addr)
                    else:
                        logger.error(f"Failed to connect to serial port {new_port}")
                        if cmd_id:
                            self._send_ack(cmd_id, "FAILED", f"Could not connect to {new_port}", addr)
                except Exception as e:
                    if cmd_id:
                        self._send_ack(cmd_id, "FAILED", f"Could not set port: {e}", addr)
                        
        elif command == "STREAM" and len(parts) >= 2:
            arg = parts[1].strip().upper()
            if arg == "ON":
                self.stream_mode = True
                state.stream_mode = True
                logger.info("Stream mode ON")
                if cmd_id:
                    self._send_ack(cmd_id, "COMPLETED", "Stream mode ON", addr)
            elif arg == "OFF":
                self.stream_mode = False
                state.stream_mode = False
                logger.info("Stream mode OFF")
                if cmd_id:
                    self._send_ack(cmd_id, "COMPLETED", "Stream mode OFF", addr)
            else:
                if cmd_id:
                    self._send_ack(cmd_id, "FAILED", "Expected ON or OFF", addr)
    
    def _handle_query_command(self, command: str, parts: List[str], cmd_id: Optional[str], addr):
        """
        Handle query commands directly without queueing.
        
        Args:
            command: The command name (e.g., 'GET_POSE', 'PING')
            parts: Command parts split by |
            cmd_id: Optional command ID
            addr: Sender address
        """
        state = self.state_manager.get_state()
        
        try:
            # Create command instance for query handling
            cmd_str = '|'.join(parts)
            command_obj = create_command(cmd_str)
            
            if command_obj:
                # Bind context and execute via new API
                command_obj.setup(
                    state,
                    udp_transport=self.udp_transport,
                    addr=addr,
                    gcode_interpreter=self.gcode_interpreter,
                )
                status = command_obj.execute_step(state)
                
                # Send acknowledgment
                if cmd_id:
                    if status.code == ExecutionStatusCode.COMPLETED:
                        self._send_ack(cmd_id, "COMPLETED", status.message or f"{command} done", addr)
                    else:
                        self._send_ack(cmd_id, "FAILED", status.message or "Query failed", addr)
            else:
                # Fallback for unimplemented query commands
                if command == "PING":
                    # Query commands now send via command itself; ack here if needed
                    if cmd_id:
                        self._send_ack(cmd_id, "COMPLETED", "PONG", addr)
                elif command == "GET_SERVER_STATE":
                    # Debug query for server state
                    active_type = type(self.active_command.command).__name__ if self.active_command else "None"
                    state_info = f"enabled={state.enabled};estop={self.estop_active};stream={self.stream_mode};queue_size={len(self.command_queue)};active={active_type};loop_counter={self.loop_counter}"
                    if cmd_id:
                        self._send_ack(cmd_id, "COMPLETED", state_info, addr)
                else:
                    logger.warning(f"Unhandled query command: {command}")
                    if cmd_id:
                        self._send_ack(cmd_id, "FAILED", "Query command not implemented", addr)
                        
        except Exception as e:
            logger.error(f"Error handling query command {command}: {e}", exc_info=True)
            if cmd_id:
                self._send_ack(cmd_id, "FAILED", str(e), addr)
    
    def _execute_immediate_command(self, command, cmd_id: Optional[str], addr):
        """
        Execute an immediate command without queueing using the new API.
        """
        state = self.state_manager.get_state()
        
        # Setup with context that may change between creation and execution
        command.setup(
            state,
            udp_transport=self.udp_transport,
            addr=addr,
            gcode_interpreter=self.gcode_interpreter
        )
        
        # Execute and map status to ACK
        status = command.execute_step(state)
        
        # If the command provided generated commands to enqueue (e.g., GCODE)
        if status.details and isinstance(status.details, dict) and 'enqueue' in status.details:
            try:
                for robot_cmd_str in status.details['enqueue']:
                    cmd_obj = create_command(robot_cmd_str)
                    if cmd_obj:
                        self._queue_command(cmd_obj)
            except Exception as e:
                logger.error(f"Error enqueuing generated commands: {e}")
        
        if cmd_id:
            self._send_ack(
                cmd_id,
                "COMPLETED" if status.code == ExecutionStatusCode.COMPLETED else "FAILED",
                status.message or "",
                addr
            )
    
    def _apply_stream_mode_logic(self, command):
        """
        Apply stream mode latest-wins logic for jog commands.
        
        Args:
            command: The new jog command
        """
        # Cancel any active jog command
        if self.current_command and type(self.current_command).__name__ in ['JogCommand', 'CartesianJogCommand', 'MultiJogCommand']:
            self.current_command = None
            logger.debug("Replaced active jog command (stream mode)")
    
    def _queue_command(self, 
                      command: CommandBase, 
                      command_id: Optional[str] = None,
                      address: Optional[Tuple[str, int]] = None,
                      priority: int = 0) -> ExecutionStatus:
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
            address=address,
            priority=priority
        )
        
        # Add to queue (sort by priority if needed)
        if priority > 0:
            # Insert based on priority
            inserted = False
            for i, existing in enumerate(self.command_queue):
                if priority > existing.priority:
                    # Use list conversion for insertion
                    queue_list = list(self.command_queue)
                    queue_list.insert(i, queued_cmd)
                    self.command_queue = deque(queue_list, maxlen=100)
                    inserted = True
                    break
            if not inserted:
                self.command_queue.append(queued_cmd)
        else:
            # Normal FIFO queueing
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
            
            # Setup the command
            try:
                state = self.state_manager.get_state()
                
                # Call setup lifecycle method
                self.active_command.command.setup(state)
                
                # Send executing acknowledgment
                if self.active_command.command_id and self.active_command.address:
                    self._send_ack(
                        self.active_command.command_id,
                        "EXECUTING",
                        f"Starting {type(self.active_command.command).__name__}",
                        self.active_command.address
                    )
                
                logger.debug(f"Activated command: {type(self.active_command.command).__name__} (id={self.active_command.command_id})")
                
            except Exception as e:
                logger.error(f"Command setup failed: {e}")
                
                # Handle setup failure
                if self.active_command.command_id and self.active_command.address:
                    self._send_ack(
                        self.active_command.command_id,
                        "FAILED",
                        f"Setup failed: {str(e)}",
                        self.active_command.address
                    )
                
                # Record failure and clear
                self._record_completion(ExecutionStatusCode.FAILED)
                self.active_command = None
                self.total_failed += 1
                
                return ExecutionStatus.failed(f"Setup failed: {str(e)}", error=e)
        
        # Execute active command step
        if self.active_command:
            try:
                state = self.state_manager.get_state()
                
                # Check if controller is enabled
                if not state.enabled:
                    # Cancel command due to disabled controller
                    self._cancel_active_command("Controller disabled")
                    return ExecutionStatus(
                        code=ExecutionStatusCode.CANCELLED,
                        message="Controller disabled"
                    )
                
                # Execute command step
                status = self.active_command.command.execute_step(state)

                # Enqueue any generated commands (e.g., from GCODE parsed in queued mode)
                if status.details and isinstance(status.details, dict) and 'enqueue' in status.details:
                    try:
                        for robot_cmd_str in status.details['enqueue']:
                            cmd_obj = create_command(robot_cmd_str)
                            if cmd_obj:
                                self._queue_command(cmd_obj)
                    except Exception as e:
                        logger.error(f"Error enqueuing generated commands: {e}")
                
                # Check if command is finished
                if status.code == ExecutionStatusCode.COMPLETED:
                    # Command completed successfully
                    logger.debug(f"Command completed: {type(self.active_command.command).__name__} (id={self.active_command.command_id}) at t={time.time():.6f}")
                    
                    # Send completion acknowledgment
                    if self.active_command.command_id and self.active_command.address:
                        self._send_ack(
                            self.active_command.command_id,
                            "COMPLETED",
                            status.message,
                            self.active_command.address
                        )
                    
                    # Record and clear
                    self._record_completion(ExecutionStatusCode.COMPLETED)
                    self.active_command = None
                    self.total_executed += 1
                    
                elif status.code == ExecutionStatusCode.FAILED:
                    # Command failed
                    logger.debug(f"Command failed: {type(self.active_command.command).__name__} (id={self.active_command.command_id}) - {status.message} at t={time.time():.6f}")
                    
                    # Send failure acknowledgment
                    if self.active_command.command_id and self.active_command.address:
                        self._send_ack(
                            self.active_command.command_id,
                            "FAILED",
                            status.message,
                            self.active_command.address
                        )
                    
                    # Record and clear
                    self._record_completion(ExecutionStatusCode.FAILED)
                    self.active_command = None
                    self.total_failed += 1
                
                return status
                
            except Exception as e:
                logger.error(f"Command execution error: {e}")
                
                # Handle execution error - save command info before clearing
                cmd_id = self.active_command.command_id if self.active_command else None
                addr = self.active_command.address if self.active_command else None
                
                if cmd_id and addr:
                    self._send_ack(cmd_id, "FAILED", f"Execution error: {str(e)}", addr)
                
                # Record and clear
                self._record_completion(ExecutionStatusCode.FAILED)
                self.active_command = None
                self.total_failed += 1
                
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
        self._record_completion(ExecutionStatusCode.CANCELLED)
        self.active_command = None
        self.total_cancelled += 1
    
    def _clear_queue(self, reason: str = "Queue cleared") -> List[Tuple[str, ExecutionStatus]]:
        """
        Clear all queued commands.
        
        Args:
            reason: Reason for clearing the queue
            
        Returns:
            List of (command_id, status) for cleared commands
        """
        cleared = []
        
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
            
            self.total_cancelled += 1
        
        logger.info(f"Cleared {len(cleared)} commands from queue: {reason}")
        
        return cleared
    
    def _record_completion(self, status: ExecutionStatusCode) -> None:
        """
        Record command completion in history.
        
        Args:
            status: Final status of the command
        """
        if self.active_command:
            self.command_history.append((
                self.active_command.command_id or "unknown",
                time.time(),
                status
            ))
    
    def _execute_command_step(self):
        """Execute one step of the current command."""
        if not self.current_command:
            return
        
        try:
            # Get current state
            state = self.state_manager.get_state()
            
            # Check if controller is enabled
            if not state.enabled:
                # Cancel command due to disabled controller
                logger.warning("Command cancelled - controller disabled")
                self.current_command = None
                return
            
            # Execute command step
            status = self.current_command.execute_step(state)
            
            # Check command status
            if status.code == ExecutionStatusCode.COMPLETED:
                # Command completed successfully
                logger.info(f"Command completed: {type(self.current_command).__name__}")
                
                # Call teardown
                self.current_command.teardown(state)
                
                # Send completion notification if tracked
                for cmd_id, (cmd, addr) in list(self.command_id_map.items()):
                    if cmd == self.current_command:
                        self._send_ack(cmd_id, "COMPLETED", "", addr)
                        del self.command_id_map[cmd_id]
                        break
                
                # Update statistics
                self.total_executed += 1
                
                # Clear current command
                self.current_command = None
                
            elif status.code == ExecutionStatusCode.FAILED:
                # Command failed
                logger.error(f"Command failed: {type(self.current_command).__name__} - {status.message}")
                
                # Handle error
                self.current_command.handle_error(status.error, state)
                self.current_command.teardown(state)
                
                # Send error notification if tracked
                for cmd_id, (cmd, addr) in list(self.command_id_map.items()):
                    if cmd == self.current_command:
                        self._send_ack(cmd_id, "FAILED", status.message, addr)
                        del self.command_id_map[cmd_id]
                        break
                
                # Update statistics
                self.total_failed += 1
                
                # Clear current command
                self.current_command = None
                
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            
            # Handle error
            state = self.state_manager.get_state()
            if hasattr(self.current_command, 'handle_error'):
                self.current_command.handle_error(e, state)
            if hasattr(self.current_command, 'teardown'):
                self.current_command.teardown(state)
            
            # Send error notification if tracked
            for cmd_id, (cmd, addr) in list(self.command_id_map.items()):
                if cmd == self.current_command:
                    self._send_ack(cmd_id, "FAILED", str(e), addr)
                    del self.command_id_map[cmd_id]
                    break
            
            self.total_failed += 1
            self.current_command = None
    
    def _update_state_from_serial_frame(self, frame) -> None:
        """
        Update controller state from a SerialFrame object.
        """
        try:
            state = self.state_manager.get_state()
            state.Position_in[:] = frame.position_in
            state.Speed_in[:] = frame.speed_in
            state.Homed_in[:] = frame.homed_in
            state.InOut_in[:] = frame.inout_in
            state.Temperature_error_in[:] = frame.temperature_error_in
            state.Position_error_in[:] = frame.position_error_in
            state.Gripper_data_in[:] = frame.gripper_data_in
            # timing_data_in and xtr_data available in frame if needed later
            
            # Mark that we've received the first frame
            if not self.first_frame_received:
                self.first_frame_received = True
                logger.debug("First frame received from robot")
        except Exception as e:
            logger.error(f"Error updating state from serial frame: {e}")

    def _process_firmware_data(self, data: bytes):
        """
        Process data received from firmware.
        
        Args:
            data: Raw bytes from firmware
        """
        try:
            # Use simulation if no serial
            if self.simulation and not self.serial_transport:
                state = self.state_manager.get_state()
                # Simulate motion
                simulate_motion(
                    self.simulation,
                    state.Command_out.value if hasattr(state.Command_out, 'value') else state.Command_out,
                    state.Position_out,
                    state.Speed_out
                )
                # Update state from simulation
                state.Position_in[:] = self.simulation.position_in
                state.Speed_in[:] = self.simulation.speed_in
                state.Homed_in[:] = self.simulation.homed_in
                state.InOut_in[:] = self.simulation.io_in
                return
            
            # Unpack firmware data
            unpacked = unpack_rx_frame(data)
            
            # Update state with firmware data
            state = self.state_manager.get_state()
            
            # Update input arrays
            state.Position_in[:] = unpacked.get('Position_in', [0] * 6)
            state.Speed_in[:] = unpacked.get('Speed_in', [0] * 6)
            state.Homed_in[:] = unpacked.get('Homed_in', [0] * 8)
            state.Temperature_error_in[:] = unpacked.get('Temperature_error_in', [0] * 8)
            state.Position_error_in[:] = unpacked.get('Position_error_in', [0] * 8)
            state.InOut_in[:] = unpacked.get('InOut_in', [0] * 8)
            state.Gripper_data_in[:] = unpacked.get('Gripper_data_in', [0] * 6)
            
            # Check for E-stop
            if unpacked.get('estop', False):
                if not self.estop_active:
                    logger.warning("E-STOP activated")
                    self.estop_active = True
                    self.estop_recovery_time = None
            else:
                if self.estop_active:
                    logger.info("E-STOP released")
                    self.estop_active = False
                    if self.config.auto_estop_recovery:
                        self.estop_recovery_time = time.time() + self.config.estop_recovery_delay
            
        except Exception as e:
            logger.error(f"Error processing firmware data: {e}")
    
    def _prepare_output_data(self) -> Optional[bytes]:
        """
        Prepare data to send to firmware.
        
        Returns:
            Packed bytes for firmware, or None if no data to send
        """
        try:
            state = self.state_manager.get_state()
            
            # Pack state for firmware
            data = pack_tx_frame(
                position_out=list(state.Position_out),
                speed_out=list(state.Speed_out),
                command_code=state.Command_out.value,
                affected_joint_out=list(state.Affected_joint_out),
                inout_out=list(state.InOut_out),
                timeout_out=state.Timeout_out,
                gripper_data_out=list(state.Gripper_data_out)
            )
            
            # Gripper mode auto-reset logic (from controller.py lines 336-341)
            # Reset gripper calibration/error clear modes after sending
            if state.Gripper_data_out[4] == 1 or state.Gripper_data_out[4] == 2:
                # Mode 1 = calibration, Mode 2 = error clear
                # Reset to 0 after packing to ensure one-shot behavior
                state.Gripper_data_out[4] = 0
                logger.debug("Auto-reset gripper mode to 0 after sending calibration/error clear")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing output data: {e}")
            return None
    
    def _handle_estop_recovery(self):
        """Handle automatic E-stop recovery."""
        if not self.estop_recovery_time:
            return
        
        if time.time() >= self.estop_recovery_time:
            logger.info("Attempting E-stop recovery...")
            
            # Clear E-stop state
            state = self.state_manager.get_state()
            state.Command_out.value = CommandCode.IDLE
            state.Speed_out[:] = [0] * 6
            
            # Clear recovery timer
            self.estop_recovery_time = None
            
            logger.info("E-stop recovery complete")
    
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
            command_obj = create_command(next_gcode_cmd)
            
            if command_obj:
                self._queue_command(command_obj)
                cmd_name = next_gcode_cmd.split('|')[0] if '|' in next_gcode_cmd else next_gcode_cmd
                logger.debug(f"Queued GCODE command: {cmd_name}")
            else:
                logger.warning(f"Unknown GCODE command generated: {next_gcode_cmd}")
                
        except Exception as e:
            logger.error(f"Error fetching GCODE commands: {e}")


def main():
    """Main entry point for the controller."""
    import argparse
    
    # Parse arguments first to get logging level
    parser = argparse.ArgumentParser(description='PAROL6 Robot Controller')
    parser.add_argument('--host', default='0.0.0.0', help='UDP host address')
    parser.add_argument('--port', type=int, default=5001, help='UDP port')
    parser.add_argument('--serial', help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baudrate', type=int, default=3000000, help='Serial baudrate')
    parser.add_argument('--no-auto-recovery', action='store_true', 
                       help='Disable automatic E-stop recovery')
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
        auto_estop_recovery=not args.no_auto_recovery,
        auto_home=bool(args.auto_home)
    )
    
    # Create and run controller
    controller = Controller(config)
    
    if controller.initialize():
        try:
            controller.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            controller.stop()
    else:
        logger.error("Failed to initialize controller")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
