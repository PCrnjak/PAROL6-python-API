"""
Mock serial transport for simulation and testing.

This module provides a complete serial port simulation that generates
realistic robot responses without requiring hardware. The simulation
operates at the wire protocol level, making it transparent to the
controller code.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from parol6.protocol.wire import pack_tx_frame, unpack_rx_frame, CommandCode
import parol6.PAROL6_ROBOT as PAROL6_ROBOT

logger = logging.getLogger(__name__)


@dataclass
class MockRobotState:
    """Internal state of the simulated robot."""
    # Joint positions (in steps)
    position_in: List[int] = field(default_factory=lambda: [0] * 6)
    # Joint speeds (in steps/sec)
    speed_in: List[int] = field(default_factory=lambda: [0] * 6)
    # Homed status per joint
    homed_in: List[int] = field(default_factory=lambda: [1] * 8)
    # I/O states
    io_in: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 1, 0, 0, 0])  # E-stop released
    # Error states
    temperature_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    position_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    # Gripper state
    gripper_data_in: List[int] = field(default_factory=lambda: [0] * 6)
    # Timing data
    timing_data_in: List[int] = field(default_factory=lambda: [0])
    
    # Simulation parameters
    update_rate: float = 0.01  # 100Hz update rate
    last_update: float = field(default_factory=time.time)
    homing_countdown: int = 0
    
    # Command state from controller
    command_out: int = CommandCode.IDLE
    position_out: List[int] = field(default_factory=lambda: [0] * 6)
    speed_out: List[int] = field(default_factory=lambda: [0] * 6)
    
    def __post_init__(self):
        """Initialize robot to standby position."""
        # Set initial positions to standby position for better IK
        for i in range(6):
            deg = PAROL6_ROBOT.Joints_standby_position_degree[i]
            steps = int(PAROL6_ROBOT.DEG2STEPS(deg, i))
            self.position_in[i] = steps
        
        # Ensure E-stop is not pressed (bit 4 = 1 means released)
        self.io_in[4] = 1


class MockSerialTransport:
    """
    Mock serial transport that simulates robot hardware responses.
    
    This class implements the exact same interface as SerialTransport,
    but generates simulated responses instead of communicating with
    real hardware. The simulation operates at the frame level, making
    it completely transparent to the controller.
    """
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 2000000, timeout: float = 0):
        """
        Initialize the mock serial transport.
        
        Args:
            port: Ignored (for interface compatibility)
            baudrate: Ignored (for interface compatibility)
            timeout: Ignored (for interface compatibility)
        """
        self.port = port or "MOCK_SERIAL"
        self.baudrate = baudrate
        self.timeout = timeout
        
        # Internal robot state
        self._state = MockRobotState()
        
        # Frame generation tracking
        self._frames_to_send: List[bytes] = []
        self._last_frame_time = time.time()
        self._frame_interval = 0.01  # 100Hz frame rate
        
        # Connection state
        self._connected = False
        
        # Statistics
        self._frames_sent = 0
        self._frames_received = 0
        
        logger.info("MockSerialTransport initialized - simulation mode active")
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Simulate serial port connection.
        
        Args:
            port: Optional port name (ignored)
            
        Returns:
            Always returns True for mock
        """
        if port:
            self.port = port
        
        self._connected = True
        self._state = MockRobotState()  # Reset state on connect
        logger.info(f"MockSerialTransport connected to simulated port: {self.port}")
        return True
    
    def disconnect(self) -> None:
        """Simulate serial port disconnection."""
        self._connected = False
        logger.info(f"MockSerialTransport disconnected from: {self.port}")
    
    def is_connected(self) -> bool:
        """
        Check if mock connection is active.
        
        Returns:
            Connection state
        """
        return self._connected
    
    def auto_reconnect(self) -> bool:
        """
        Mock auto-reconnect (always succeeds).
        
        Returns:
            True if not connected, False if already connected
        """
        if not self._connected:
            return self.connect(self.port)
        return False
    
    def write_frame(self,
                   position_out: List[int],
                   speed_out: List[int],
                   command_out: int,
                   affected_joint_out: List[int],
                   inout_out: List[int],
                   timeout_out: int,
                   gripper_data_out: List[int]) -> bool:
        """
        Process a command frame from the controller.
        
        Instead of writing to serial, this updates the internal
        simulation state.
        
        Args:
            position_out: Target positions
            speed_out: Speed commands
            command_out: Command code
            affected_joint_out: Affected joint flags
            inout_out: I/O commands
            timeout_out: Timeout value
            gripper_data_out: Gripper commands
            
        Returns:
            True if processed successfully
        """
        if not self._connected:
            return False
        
        # Update simulation state with command
        self._state.command_out = command_out
        self._state.position_out = list(position_out)
        self._state.speed_out = list(speed_out)
        
        # Track frame reception
        self._frames_received += 1
        
        # Simulate gripper state updates
        if gripper_data_out[4] == 1:  # Calibration mode
            # Simulate gripper calibration
            self._state.gripper_data_in[0] = gripper_data_out[5]  # Set device ID
            self._state.gripper_data_in[4] = 0x40  # Set calibrated bit
        elif gripper_data_out[4] == 2:  # Error clear mode
            # Clear gripper errors
            self._state.gripper_data_in[4] &= ~0x20  # Clear error bit
        
        # Update gripper position/speed/current if commanded
        if gripper_data_out[3] != 0:  # Gripper command active
            self._state.gripper_data_in[1] = gripper_data_out[0]  # Position
            self._state.gripper_data_in[2] = gripper_data_out[1]  # Speed
            self._state.gripper_data_in[3] = gripper_data_out[2]  # Current
        
        return True
    
    def read_frames(self) -> List[Any]:
        """
        Generate simulated response frames.
        
        This method simulates the robot's response by:
        1. Running the motion simulation
        2. Generating a response frame with current state
        3. Returning it as if received from serial
        
        Returns:
            List of frame objects (may be empty)
        """
        frames = []
        
        if not self._connected:
            return frames
        
        # Check if it's time to generate a frame
        now = time.time()
        if now - self._last_frame_time >= self._frame_interval:
            # Run simulation step
            dt = now - self._state.last_update
            self._simulate_motion(dt)
            self._state.last_update = now
            
            # Generate response frame
            frame = self._create_response_frame()
            if frame:
                frames.append(frame)
                self._frames_sent += 1
            
            self._last_frame_time = now
        
        return frames
    
    def _simulate_motion(self, dt: float) -> None:
        """
        Simulate one step of robot motion.
        
        Args:
            dt: Time delta since last update
        """
        state = self._state
        
        # Handle homing countdown
        if state.homing_countdown > 0:
            state.homing_countdown -= 1
            if state.homing_countdown == 0:
                # Homing complete - mark all joints as homed and move to zero
                for i in range(6):
                    state.homed_in[i] = 1
                    state.position_in[i] = 0
                    state.speed_in[i] = 0
        
        # Ensure E-stop stays released in simulation
        state.io_in[4] = 1
        
        # Simulate motion based on command type
        if state.command_out == CommandCode.HOME:
            # Start homing sequence
            if state.homing_countdown == 0:
                for i in range(6):
                    state.homed_in[i] = 0  # Mark as not homed
                # Schedule homing completion after ~0.2s
                state.homing_countdown = max(1, int(0.2 / dt))
            # Zero speeds during homing
            for i in range(6):
                state.speed_in[i] = 0
                
        elif state.command_out == CommandCode.JOG or state.command_out == 123:
            # Speed control mode
            for i in range(6):
                v = int(state.speed_out[i])
                # Apply speed limits
                max_v = int(PAROL6_ROBOT.Joint_max_speed[i])
                v = max(-max_v, min(max_v, v))
                
                # Integrate position
                new_pos = int(state.position_in[i] + v * dt)
                
                # Apply joint limits
                jmin, jmax = PAROL6_ROBOT.Joint_limits_steps[i]
                if new_pos < jmin:
                    new_pos = jmin
                    v = 0
                elif new_pos > jmax:
                    new_pos = jmax
                    v = 0
                
                state.speed_in[i] = v
                state.position_in[i] = new_pos
                
        elif state.command_out == CommandCode.MOVE or state.command_out == 156:
            # Position control mode
            for i in range(6):
                target = state.position_out[i]
                current = state.position_in[i]
                err = int(target - current)
                
                if err == 0:
                    state.speed_in[i] = 0
                    continue
                
                # Calculate step size based on max speed
                max_step = int(PAROL6_ROBOT.Joint_max_speed[i] * dt)
                if max_step < 1:
                    max_step = 1
                
                # Move toward target
                step = max(-max_step, min(max_step, err))
                new_pos = current + step
                
                # Apply joint limits
                jmin, jmax = PAROL6_ROBOT.Joint_limits_steps[i]
                if new_pos < jmin:
                    new_pos = jmin
                    step = 0
                elif new_pos > jmax:
                    new_pos = jmax
                    step = 0
                
                state.position_in[i] = int(new_pos)
                state.speed_in[i] = int(step / dt) if dt > 0 else 0
                
        else:
            # Idle or unknown command - hold position
            for i in range(6):
                state.speed_in[i] = 0
    
    def _create_response_frame(self) -> Optional[Any]:
        """
        Create a response frame from current simulation state.
        
        Returns:
            Frame object compatible with SerialTransport
        """
        # Create a frame-like object that matches SerialFrame structure
        from parol6.server.transports.serial_transport import SerialFrame
        
        frame = SerialFrame(
            position_in=list(self._state.position_in),
            speed_in=list(self._state.speed_in),
            homed_in=list(self._state.homed_in),
            inout_in=list(self._state.io_in),
            temperature_error_in=list(self._state.temperature_error_in),
            position_error_in=list(self._state.position_error_in),
            timing_data_in=list(self._state.timing_data_in),
            gripper_data_in=list(self._state.gripper_data_in),
            timestamp=time.time()
        )
        
        return frame
    
    def get_info(self) -> dict:
        """
        Get information about the mock transport.
        
        Returns:
            Dictionary with transport information
        """
        return {
            'port': self.port,
            'baudrate': self.baudrate,
            'connected': self._connected,
            'timeout': self.timeout,
            'mode': 'MOCK_SERIAL',
            'frames_sent': self._frames_sent,
            'frames_received': self._frames_received,
            'simulation_rate_hz': int(1.0 / self._frame_interval),
            'robot_state': {
                'homed': all(self._state.homed_in[i] == 1 for i in range(6)),
                'estop': self._state.io_in[4] == 0,
                'command': self._state.command_out
            }
        }


def create_mock_serial_transport() -> MockSerialTransport:
    """
    Factory function to create a mock serial transport.
    
    Returns:
        Configured MockSerialTransport instance
    """
    transport = MockSerialTransport()
    transport.connect()
    logger.info("Mock serial transport created and connected")
    return transport
