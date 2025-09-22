"""
Serial transport implementation for PAROL6 controller.

This module handles serial port communication, frame parsing, and
data exchange with the robot hardware.
"""

import serial
import logging
import time
import threading
from typing import Optional
import numpy as np
import os

from parol6.protocol.wire import pack_tx_frame_into
from parol6.config import get_com_port_with_fallback, INTERVAL_S

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
        self._scratch = bytearray(4096)
        self._scratch_mv = memoryview(self._scratch)
        # Fixed-size ring buffer for RX stream (drop-oldest on overflow)
        _cap = int(os.getenv("PAROL6_SERIAL_RX_RING_CAP", "262144"))
        self._ring = bytearray(_cap)
        self._r_cap = _cap
        self._r_head = 0
        self._r_tail = 0
        self._frame_buf = bytearray(64)  # 52-byte payload + headroom
        self._frame_mv = memoryview(self._frame_buf)[:52]
        self._frame_version = 0
        self._frame_ts = 0.0
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_running = False

        # Preallocated TX buffer (3 start + 1 len + 52 payload = 56 bytes)
        self._tx_buf = bytearray(56)
        self._tx_mv = memoryview(self._tx_buf)
        
        # Hz tracking for debug prints
        self._last_print_time = 0.0
        self._print_interval = 3.0  # seconds
        self._rx_msg_count = 0
        self._interval_msg_count = 0
        
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
                   position_out: np.ndarray,
                   speed_out: np.ndarray,
                   command_out: int,
                   affected_joint_out: np.ndarray,
                   inout_out: np.ndarray,
                   timeout_out: int,
                   gripper_data_out: np.ndarray) -> bool:
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
            # Write to serial using preallocated buffer and zero-alloc pack
            ser = self.serial
            if ser is None:
                return False

            pack_tx_frame_into(
                self._tx_mv,
                position_out,
                speed_out,
                command_out,
                affected_joint_out,
                inout_out,
                timeout_out,
                gripper_data_out
            )
            ser.write(self._tx_mv)
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
        if self.serial:
            # Small block so as not to busy loop, but can't use a larger timout
            # because serial will read until buffer is full or timeout.
            self.serial.timeout = INTERVAL_S / 2

        def _run() -> None:
            self._reader_running = True
            try:
                while not shutdown_event.is_set():
                    if not self.is_connected():
                        # Backoff a bit to avoid busy loop if disconnected
                        time.sleep(0.1)
                        continue
                    # Race-safe read: hold local ref and check is_open
                    ser = self.serial
                    if not ser or not getattr(ser, "is_open", False):
                        # Disconnected between iterations; back off briefly
                        time.sleep(0.1)
                        continue
                    try:
                        # Read into preallocated scratch buffer
                        n = ser.readinto(self._scratch_mv)
                    except serial.SerialException as e:
                        logger.error(f"Serial reader error: {e}")
                        self.disconnect()
                        break
                    except (OSError, TypeError, ValueError, AttributeError):
                        # fd likely closed during disconnect; stop quietly
                        logger.info("Serial reader stopping due to disconnect/closed FD", exc_info=False)
                        try:
                            self.disconnect()
                        except Exception:
                            pass
                        break
                    except Exception:
                        logger.exception("Serial reader unexpected exception")
                        break

                    if not n:
                        # Timeout or no data; loop to check shutdown_event
                        continue

                    # Append into ring buffer and parse
                    cap = self._r_cap
                    head = self._r_head
                    tail = self._r_tail
                    rb = self._ring
                    src = self._scratch
                    for i in range(n):
                        rb[head] = src[i]
                        head += 1
                        if head == cap:
                            head = 0
                        if head == tail:
                            tail += 1
                            if tail == cap:
                                tail = 0
                    self._r_head = head
                    self._r_tail = tail
                    self._parse_ring_for_frames()
            finally:
                self._reader_running = False

        t = threading.Thread(target=_run, name="SerialReader", daemon=True)
        self._reader_thread = t
        t.start()
        return t

    def _parse_ring_for_frames(self) -> None:
        """
        Parse as many complete frames as possible from the RX ring buffer in-place.

        Frame format:
          [0xFF,0xFF,0xFF] [LEN] [LEN bytes data ...]
        """
        START0, START1, START2 = 0xFF, 0xFF, 0xFF
        END0, END1 = self.END_BYTES[0], self.END_BYTES[1]
        cap = self._r_cap
        head = self._r_head
        tail = self._r_tail
        rb = self._ring

        def available(h: int, t: int) -> int:
            return (h - t + cap) % cap

        while available(head, tail) >= 4:
            # Find start sequence 0xFF 0xFF 0xFF by advancing tail
            found = False
            while available(head, tail) >= 3:
                b0 = rb[tail]
                b1 = rb[(tail + 1) % cap]
                b2 = rb[(tail + 2) % cap]
                if b0 == START0 and b1 == START1 and b2 == START2:
                    found = True
                    break
                tail = (tail + 1) % cap
            if not found or available(head, tail) < 4:
                break

            length = rb[(tail + 3) % cap]
            total_needed = 4 + length
            if available(head, tail) < total_needed:
                # Wait for more data
                break

            # Validate end markers if possible
            if length >= 2:
                e0 = rb[(tail + 4 + length - 2) % cap]
                e1 = rb[(tail + 4 + length - 1) % cap]
                if not (e0 == END0 and e1 == END1):
                    # Bad frame; skip one byte to resync
                    tail = (tail + 1) % cap
                    continue

            # Publish first 52 bytes if available
            payload_len = 52 if length >= 52 else length
            start = (tail + 4) % cap
            if start + payload_len <= cap:
                self._frame_buf[:payload_len] = rb[start:start + payload_len]
            else:
                first = cap - start
                self._frame_buf[:first] = rb[start:cap]
                self._frame_buf[first:payload_len] = rb[0:payload_len - first]

            if payload_len >= 52:
                self._frame_version += 1
                self._frame_ts = time.time()
                
                # Update Hz tracking if enabled
                self._update_hz_tracking()

            # Consume this frame
            tail = (tail + total_needed) % cap

        # Publish updated tail
        self._r_tail = tail

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
    
    def _update_hz_tracking(self) -> None:
        """
        Update EMA Hz tracking and print debug info periodically.
        
        This method calculates the instantaneous Hz based on time between messages,
        updates the EMA (Exponential Moving Average), tracks min/max values,
        and prints debug info every few seconds.
        """
        current_time = time.perf_counter()
        
        # Increment message counters
        self._rx_msg_count += 1
        self._interval_msg_count += 1
        
        # Check if it's time to print debug info
        if self._last_print_time == 0.0:
            self._last_print_time = current_time
        elif current_time - self._last_print_time >= self._print_interval:
            # Print debug information
            if self._interval_msg_count > 0:
                avg_hz = self._interval_msg_count / (current_time - self._last_print_time)
                logger.debug(f"Serial RX Stats - Avg Hz: {avg_hz:.2f} (Total: {self._rx_msg_count})"
                )
            else:
                logger.debug(f"Serial RX Stats - No messages received in last {self._print_interval:.1f}s")
            
            # Reset interval statistics
            self._last_print_time = current_time
            self._interval_msg_count = 0


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
