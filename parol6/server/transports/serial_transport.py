"""
Serial transport implementation for PAROL6 controller.

This module handles serial port communication, frame parsing, and
data exchange with the robot hardware.
"""

import serial
import logging
import time
import threading
from typing import Optional, List, Union, Sequence
import array

from parol6.protocol.wire import pack_tx_frame
from parol6.config import get_com_port_with_fallback

logger = logging.getLogger(__name__)

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
        self.last_reconnect_attempt = 0.0
        self.reconnect_interval = 1.0  # seconds between reconnect attempts

        # Reduced-copy latest-frame infrastructure (reader thread will publish here)
        self._scratch = bytearray(1024)
        self._scratch_mv = memoryview(self._scratch)
        self._stream = bytearray()
        self._frame_buf = bytearray(64)  # 52-byte payload + headroom
        self._frame_mv = memoryview(self._frame_buf)[:52]
        self._frame_version = 0
        self._frame_ts = 0.0
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_running = False
        
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
            
        if self.port:
            logger.info(f"Attempting to reconnect to {self.port}...")
            return self.connect(self.port)
        
        return False
    
    def write_frame(self, 
                   position_out: Union[List[int], array.array, Sequence[int]],
                   speed_out: Union[List[int], array.array, Sequence[int]],
                   command_out: int,
                   affected_joint_out: Union[List[int], array.array, Sequence[int]],
                   inout_out: Union[List[int], array.array, Sequence[int]],
                   timeout_out: int,
                   gripper_data_out: Union[List[int], array.array, Sequence[int]]) -> bool:
        """
        Write a command frame to the robot.
        
        Optimized to accept array-like objects directly without conversion.
        Supports lists, array.array, and other sequence types.
        
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
            ser = self.serial
            if ser is None:
                return False
            ser.write(frame)
            return True
            
        except serial.SerialException as e:
            logger.error(f"Serial write error: {e}")
            # Mark connection as lost
            self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Unexpected error writing frame: {e}")
            return False
    
    
    

    # ================================
    # Latest-frame API (reduced-copy)
    # ================================
    def start_reader(self, shutdown_event: threading.Event) -> threading.Thread:
        """
        Start a dedicated reader thread that parses incoming frames from the serial port
        and publishes the latest 52-byte payload into an internal stable buffer.

        Returns the started Thread object. If already running, returns the existing one.
        """
        if not self.is_connected():
            raise RuntimeError("SerialTransport.start_reader: serial port not connected")

        if self._reader_thread and self._reader_thread.is_alive():
            return self._reader_thread

        # Ensure a short timeout for responsive shutdown
        try:
            if self.serial:
                # Use a small blocking timeout to periodically check shutdown_event
                self.serial.timeout = 0.25
        except Exception:
            pass

        def _run() -> None:
            self._reader_running = True
            try:
                while not shutdown_event.is_set():
                    if not self.is_connected():
                        # Backoff a bit to avoid busy loop if disconnected
                        time.sleep(0.1)
                        continue
                    try:
                        # Read into preallocated scratch buffer
                        n = self.serial.readinto(self._scratch_mv) if self.serial else 0
                    except serial.SerialException as e:
                        logger.error(f"Serial reader error: {e}")
                        self.disconnect()
                        break
                    except Exception:
                        logger.exception("Serial reader unexpected exception")
                        break

                    if not n:
                        # Timeout or no data; loop to check shutdown_event
                        continue

                    # Append to rolling stream buffer and parse
                    self._stream.extend(self._scratch[:n])
                    self._parse_stream_for_frames()
            finally:
                self._reader_running = False

        t = threading.Thread(target=_run, name="SerialReader", daemon=True)
        self._reader_thread = t
        t.start()
        return t

    def _parse_stream_for_frames(self) -> None:
        """
        Parse as many complete frames as possible from the rolling stream buffer.

        Frame format:
          [0xFF,0xFF,0xFF] [LEN] [LEN bytes data ...]
        We expect the last two bytes of the LEN segment to be end markers (0x01,0x02)
        on real firmware. For robustness, we only copy the first 52 bytes of the LEN
        segment (if present) into the stable latest-frame buffer.
        """
        buf = self._stream
        START = self.START_BYTES
        END0, END1 = self.END_BYTES[0], self.END_BYTES[1]

        while True:
            # Find start sequence
            i = buf.find(START)
            if i == -1:
                # Keep up to last two bytes in case they begin a start sequence
                if len(buf) > 2:
                    del buf[:-2]
                break

            # Discard any leading noise before start
            if i > 0:
                del buf[:i]

            # Need at least header + length
            if len(buf) < 4:
                break

            length = buf[3]
            total_needed = 4 + length
            if len(buf) < total_needed:
                # Wait for more data
                break

            frame_seg = buf[4:4 + length]

            # Validate end markers if possible
            if length >= 2 and not (frame_seg[-2] == END0 and frame_seg[-1] == END1):
                # Bad frame; skip one byte to search for next start to be resilient
                del buf[:1]
                continue

            # Publish first 52 bytes if available
            if len(frame_seg) >= 52:
                self._frame_buf[:52] = frame_seg[:52]
                self._frame_version += 1
                self._frame_ts = time.time()

            # Consume this frame
            del buf[:total_needed]

    def get_latest_frame_view(self) -> tuple[Optional[memoryview], int, float]:
        """
        Return a tuple of (memoryview|None, version:int, timestamp:float).
        The memoryview points to a stable 52-byte buffer which is updated by the reader.
        """
        mv = self._frame_mv if self._frame_version > 0 else None
        return (mv, self._frame_version, self._frame_ts)
    
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
        saved_port = get_com_port_with_fallback()
        if saved_port:
            transport.connect(saved_port)
            
    return transport
