"""
Serial transport implementation for PAROL6 controller.

This module handles serial port communication, frame parsing, and
data exchange with the robot hardware.
"""

from __future__ import annotations

import serial
import struct
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

from parol6.protocol.wire import pack_tx_frame, unpack_rx_frame

logger = logging.getLogger(__name__)


@dataclass
class SerialFrame:
    """
    Represents a parsed serial frame from the robot.
    
    This contains all telemetry data received from the robot hardware.
    """
    position_in: List[int] = field(default_factory=lambda: [0] * 6)
    speed_in: List[int] = field(default_factory=lambda: [0] * 6)
    homed_in: List[int] = field(default_factory=lambda: [0] * 8)
    inout_in: List[int] = field(default_factory=lambda: [0] * 8)
    temperature_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    position_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    timing_data_in: List[int] = field(default_factory=lambda: [0])
    gripper_data_in: List[int] = field(default_factory=lambda: [0] * 6)
    xtr_data: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SerialParseState:
    """
    Internal state for parsing serial frames.
    
    This tracks the parsing state machine for incoming serial data.
    """
    start_cond1: int = 0
    start_cond2: int = 0
    start_cond3: int = 0
    good_start: int = 0
    data_len: int = 0
    data_buffer: List[bytes] = field(default_factory=lambda: [b""] * 255)
    data_counter: int = 0


class SerialTransport:
    """
    Manages serial port communication with the robot.
    
    This class handles:
    - Serial port connection and reconnection
    - Frame parsing and validation
    - Command transmission
    - Telemetry reception
    """
    
    # Frame delimiters
    START_BYTES = bytes([0xff, 0xff, 0xff])
    END_BYTES = bytes([0x01, 0x02])
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 2000000, timeout: float = 0):
        """
        Initialize the serial transport.
        
        Args:
            port: Serial port name (e.g., 'COM5', '/dev/ttyACM0')
            baudrate: Baud rate for serial communication
            timeout: Read timeout in seconds (0 for non-blocking)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.parse_state = SerialParseState()
        self.last_reconnect_attempt = 0.0
        self.reconnect_interval = 1.0  # seconds between reconnect attempts
        
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to the serial port.
        
        Args:
            port: Optional port override. If not provided, uses stored port.
            
        Returns:
            True if connection successful, False otherwise
        """
        if port:
            self.port = port
            
        if not self.port:
            logger.warning("No serial port specified")
            return False
            
        try:
            # Close existing connection if any
            if self.serial and self.serial.is_open:
                self.serial.close()
                
            # Create new connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            if self.serial.is_open:
                logger.info(f"Connected to serial port: {self.port}")
                # Save successful port for future use
                self._save_port(self.port)
                return True
            else:
                logger.error(f"Failed to open serial port: {self.port}")
                return False
                
        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}")
            self.serial = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to serial: {e}")
            self.serial = None
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the serial port."""
        if self.serial:
            try:
                if self.serial.is_open:
                    self.serial.close()
                logger.info(f"Disconnected from serial port: {self.port}")
            except Exception as e:
                logger.error(f"Error closing serial port: {e}")
            finally:
                self.serial = None
    
    def is_connected(self) -> bool:
        """
        Check if serial connection is active.
        
        Returns:
            True if connected and open, False otherwise
        """
        return self.serial is not None and self.serial.is_open
    
    def auto_reconnect(self) -> bool:
        """
        Attempt to reconnect to the serial port if disconnected.
        
        This implements a rate-limited reconnection attempt.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        now = time.time()
        
        # Rate limit reconnection attempts
        if now - self.last_reconnect_attempt < self.reconnect_interval:
            return False
            
        self.last_reconnect_attempt = now
        
        # Try to load port from file if not set
        if not self.port:
            self.port = self._load_port()
            
        if self.port:
            logger.info(f"Attempting to reconnect to {self.port}...")
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
        Write a command frame to the robot.
        
        Args:
            position_out: Target positions for joints
            speed_out: Speed commands for joints
            command_out: Command code
            affected_joint_out: Affected joint flags
            inout_out: I/O commands
            timeout_out: Timeout value
            gripper_data_out: Gripper commands
            
        Returns:
            True if write successful, False otherwise
        """
        if not self.is_connected():
            return False
            
        try:
            # Pack the frame using the wire protocol
            frame = pack_tx_frame(
                position_out,
                speed_out,
                command_out,
                affected_joint_out,
                inout_out,
                timeout_out,
                gripper_data_out
            )
            
            # Write to serial
            self.serial.write(frame)
            return True
            
        except serial.SerialException as e:
            logger.error(f"Serial write error: {e}")
            # Mark connection as lost
            self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Unexpected error writing frame: {e}")
            return False
    
    def read_frames(self) -> List[SerialFrame]:
        """
        Read and parse all available frames from the serial port.
        
        Returns:
            List of parsed SerialFrame objects (may be empty)
        """
        frames = []
        
        if not self.is_connected():
            return frames
            
        try:
            # Process all available bytes
            while self.serial.in_waiting > 0:
                input_byte = self.serial.read(1)
                
                frame = self._process_byte(input_byte)
                if frame:
                    frames.append(frame)
                    
        except serial.SerialException as e:
            logger.error(f"Serial read error: {e}")
            # Mark connection as lost
            self.disconnect()
        except Exception as e:
            logger.error(f"Unexpected error reading frames: {e}")
            
        return frames
    
    def _process_byte(self, input_byte: bytes) -> Optional[SerialFrame]:
        """
        Process a single byte through the frame parsing state machine.
        
        Args:
            input_byte: Single byte from serial stream
            
        Returns:
            Parsed SerialFrame if a complete frame was received, None otherwise
        """
        ps = self.parse_state  # Shorthand
        
        if ps.good_start != 1:
            # Looking for start sequence
            if ps.start_cond1 == 1 and ps.start_cond2 == 1 and ps.start_cond3 == 1:
                ps.good_start = 1
                ps.data_len = struct.unpack('B', input_byte)[0]
                
            elif input_byte == self.START_BYTES[2:3] and ps.start_cond2 == 1 and ps.start_cond1 == 1:
                ps.start_cond3 = 1
                
            elif ps.start_cond2 == 1 and ps.start_cond1 == 1:
                ps.start_cond1 = 0
                ps.start_cond2 = 0
                
            elif input_byte == self.START_BYTES[1:2] and ps.start_cond1 == 1:
                ps.start_cond2 = 1
                
            elif ps.start_cond1 == 1:
                ps.start_cond1 = 0
                
            elif input_byte == self.START_BYTES[0:1]:
                ps.start_cond1 = 1
                
        else:
            # Collecting frame data
            ps.data_buffer[ps.data_counter] = input_byte
            
            if ps.data_counter == ps.data_len - 1:
                # Frame complete - validate end sequence and parse
                if (ps.data_buffer[ps.data_len - 1] == self.END_BYTES[1:2] and 
                    ps.data_buffer[ps.data_len - 2] == self.END_BYTES[0:1]):
                    
                    # Extract payload (first 52 bytes)
                    payload = b"".join(ps.data_buffer[:52])
                    parsed = unpack_rx_frame(payload)
                    
                    if parsed is not None:
                        # Create SerialFrame from parsed data
                        frame = SerialFrame(
                            position_in=parsed["Position_in"],
                            speed_in=parsed["Speed_in"],
                            homed_in=parsed["Homed_in"],
                            inout_in=parsed["InOut_in"],
                            temperature_error_in=parsed["Temperature_error_in"],
                            position_error_in=parsed["Position_error_in"],
                            timing_data_in=parsed["Timing_data_in"],
                            gripper_data_in=parsed["Gripper_data_in"],
                            timestamp=time.time()
                        )
                        
                        # Reset parsing state
                        self._reset_parse_state()
                        return frame
                    else:
                        logger.warning("Failed to unpack RX frame")
                        
                # Reset parsing state (whether valid or not)
                self._reset_parse_state()
            else:
                ps.data_counter += 1
                
        return None
    
    def _reset_parse_state(self) -> None:
        """Reset the frame parsing state machine."""
        ps = self.parse_state
        ps.good_start = 0
        ps.start_cond1 = 0
        ps.start_cond2 = 0
        ps.start_cond3 = 0
        ps.data_len = 0
        ps.data_counter = 0
    
    def _save_port(self, port: str) -> None:
        """
        Save the serial port to a file for persistence.
        
        Args:
            port: Port name to save
        """
        try:
            with open("com_port.txt", "w") as f:
                f.write(port)
            logger.debug(f"Saved serial port to file: {port}")
        except Exception as e:
            logger.warning(f"Could not save port to file: {e}")
    
    def _load_port(self) -> Optional[str]:
        """
        Load the serial port from a file.
        
        Returns:
            Port name if found, None otherwise
        """
        try:
            with open("com_port.txt", "r") as f:
                port = f.read().strip()
                if port:
                    logger.debug(f"Loaded serial port from file: {port}")
                    return port
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not load port from file: {e}")
        
        return None
    
    def get_info(self) -> dict:
        """
        Get information about the current serial connection.
        
        Returns:
            Dictionary with connection information
        """
        info = {
            'port': self.port,
            'baudrate': self.baudrate,
            'connected': self.is_connected(),
            'timeout': self.timeout
        }
        
        if self.serial and self.serial.is_open:
            try:
                info['in_waiting'] = self.serial.in_waiting
                info['out_waiting'] = self.serial.out_waiting
            except:
                pass
                
        return info


def create_serial_transport(port: Optional[str] = None, 
                          baudrate: int = 2000000) -> SerialTransport:
    """
    Factory function to create and optionally connect a serial transport.
    
    Args:
        port: Optional serial port to connect to
        baudrate: Baud rate for communication
        
    Returns:
        SerialTransport instance (may or may not be connected)
    """
    transport = SerialTransport(port=port, baudrate=baudrate)
    
    # Try to connect if port provided
    if port:
        transport.connect(port)
    else:
        # Try to load and connect to saved port
        saved_port = transport._load_port()
        if saved_port:
            transport.connect(saved_port)
            
    return transport
