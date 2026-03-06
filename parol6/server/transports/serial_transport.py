"""
Serial transport implementation for PAROL6 controller.

This module handles serial port communication, frame parsing, and
data exchange with the robot hardware.
"""

import logging
import os
import time

import numba
import numpy as np
import serial

from parol6.config import SERIAL_RX_RING_DEFAULT
from parol6.protocol.wire import pack_tx_frame_into

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _append_to_ring_numba(
    rb: np.ndarray, src: np.ndarray, n: int, cap: int, head: int, tail: int
) -> tuple[int, int]:
    """JIT-compiled ring buffer append. Returns (new_head, new_tail)."""
    avail = (head - tail + cap) % cap
    free = cap - 1 - avail
    over = max(0, n - free)
    if over:
        tail = (tail + over) % cap

    first = min(n, cap - head)
    for i in range(first):
        rb[head + i] = src[i]
    remain = n - first
    for i in range(remain):
        rb[i] = src[first + i]
    head = (head + n) % cap

    return head, tail


@numba.njit(cache=True)
def _parse_frames_njit(
    rb: np.ndarray,
    head: int,
    tail: int,
    cap: int,
    frame_buf: np.ndarray,
) -> tuple[int, int, bool]:
    """
    JIT-compiled frame parser. Scans for 0xFF 0xFF 0xFF start sequence,
    validates end markers 0x01 0x02, extracts 52-byte payload.

    Returns:
        new_tail: Updated tail position
        frames_found: Number of complete frames found
        has_valid_frame: True if frame_buf contains valid data
    """
    START0, START1, START2 = 0xFF, 0xFF, 0xFF
    END0, END1 = 0x01, 0x02
    frames_found = 0
    has_valid = False

    while True:
        avail = (head - tail + cap) % cap
        if avail < 4:
            break

        # Find start sequence
        found = False
        while avail >= 3:
            if (
                rb[tail] == START0
                and rb[(tail + 1) % cap] == START1
                and rb[(tail + 2) % cap] == START2
            ):
                found = True
                break
            tail = (tail + 1) % cap
            avail = (head - tail + cap) % cap

        if not found or avail < 4:
            break

        length = rb[(tail + 3) % cap]
        total_needed = 4 + length
        if avail < total_needed:
            break

        # Validate end markers
        if length >= 2:
            e0 = rb[(tail + 4 + length - 2) % cap]
            e1 = rb[(tail + 4 + length - 1) % cap]
            if not (e0 == END0 and e1 == END1):
                tail = (tail + 1) % cap
                continue

        # Extract payload (first 52 bytes)
        payload_len = 52 if length >= 52 else length
        start = (tail + 4) % cap
        for i in range(payload_len):
            frame_buf[i] = rb[(start + i) % cap]

        if payload_len >= 52:
            frames_found += 1
            has_valid = True

        tail = (tail + total_needed) % cap

    return tail, frames_found, has_valid


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
    START_BYTES = bytes([0xFF, 0xFF, 0xFF])
    END_BYTES = bytes([0x01, 0x02])

    def __init__(
        self, port: str | None = None, baudrate: int = 2000000, timeout: float = 0
    ):
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
        self.serial: serial.Serial | None = None
        self.last_reconnect_attempt = 0.0
        self.reconnect_interval = 1.0  # seconds between reconnect attempts
        self._reconnect_failures = 0  # count consecutive failures to reduce log spam

        # Reduced-copy latest-frame infrastructure (poll_read will publish here)
        self._scratch = np.zeros(4096, dtype=np.uint8)
        self._scratch_mv = memoryview(self._scratch.data)
        # Fixed-size ring buffer for RX stream (drop-oldest on overflow)
        _cap = int(os.getenv("PAROL6_SERIAL_RX_RING_CAP", str(SERIAL_RX_RING_DEFAULT)))
        self._ring = np.zeros(_cap, dtype=np.uint8)
        self._r_cap = _cap
        self._r_head = 0
        self._r_tail = 0
        self._frame_buf = np.zeros(64, dtype=np.uint8)
        self._frame_mv = memoryview(self._frame_buf)[:52]  # type: ignore[invalid-argument-type]
        self._frame_version = 0
        self._frame_ts = 0.0

        # Preallocated TX buffer (3 start + 1 len + 52 payload = 56 bytes)
        self._tx_buf = bytearray(56)
        self._tx_mv = memoryview(self._tx_buf)

        # Hz tracking for debug prints
        self._last_print_time = 0.0
        self._print_interval = 3.0  # seconds
        self._rx_msg_count = 0
        self._interval_msg_count = 0

    def connect(self, port: str | None = None) -> bool:
        """
        Connect to the serial port.

        Args:
            port: Optional port override. If not provided, uses stored port.

        Returns:
            True if connection successful, False otherwise
        """
        # Reset failure counter on explicit connect call
        self._reconnect_failures = 0
        success = self._connect_internal(port, quiet=False)
        if not success:
            # Mark as failed so auto_reconnect logs at DEBUG level
            self._reconnect_failures = 1
        return success

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

        This implements a rate-limited reconnection attempt. After the first
        failure, subsequent reconnection attempts are logged at DEBUG level
        to reduce log spam.

        Returns:
            True if reconnection successful, False otherwise
        """
        now = time.time()

        # Rate limit reconnection attempts
        if now - self.last_reconnect_attempt < self.reconnect_interval:
            return False

        self.last_reconnect_attempt = now

        if self.port:
            # Log at INFO only on first attempt, DEBUG on subsequent
            log_level = logging.DEBUG if self._reconnect_failures > 0 else logging.INFO
            logger.log(log_level, f"Attempting to reconnect to {self.port}...")
            success = self._connect_internal(
                self.port, quiet=self._reconnect_failures > 0
            )
            if success:
                if self._reconnect_failures > 0:
                    logger.info(
                        f"Reconnected to {self.port} after {self._reconnect_failures} failed attempts"
                    )
                self._reconnect_failures = 0
            else:
                self._reconnect_failures += 1
            return success

        return False

    def _connect_internal(self, port: str | None, quiet: bool = False) -> bool:
        """
        Internal connection logic with optional quiet mode for reduced logging.

        Args:
            port: Port to connect to
            quiet: If True, log errors at DEBUG instead of ERROR

        Returns:
            True if connection successful, False otherwise
        """
        if port:
            self.port = port

        if not self.port:
            if not quiet:
                logger.warning("No serial port specified")
            return False

        try:
            # Close existing connection if any
            if self.serial and self.serial.is_open:
                self.serial.close()

            # Create new connection
            self.serial = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout
            )

            if self.serial.is_open:
                logger.info(f"Connected to serial port: {self.port}")
                return True
            else:
                log_level = logging.DEBUG if quiet else logging.ERROR
                logger.log(log_level, f"Failed to open serial port: {self.port}")
                return False

        except serial.SerialException as e:
            log_level = logging.DEBUG if quiet else logging.ERROR
            logger.log(log_level, f"Serial connection error: {e}")
            self.serial = None
            return False
        except Exception as e:
            log_level = logging.DEBUG if quiet else logging.ERROR
            logger.log(log_level, f"Unexpected error connecting to serial: {e}")
            self.serial = None
            return False

    def write_frame(
        self,
        position_out: np.ndarray,
        speed_out: np.ndarray,
        command_out: int,
        affected_joint_out: np.ndarray,
        inout_out: np.ndarray,
        timeout_out: int,
        gripper_data_out: np.ndarray,
    ) -> bool:
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
                gripper_data_out,
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
    def poll_read(self) -> bool:
        """Non-blocking read from serial. Returns True if data was read."""
        if not self.is_connected():
            return False
        ser = self.serial
        if not ser or not ser.is_open:
            return False

        try:
            iw = ser.in_waiting
            if iw == 0:
                return False

            k = min(iw, len(self._scratch))
            n = ser.readinto(self._scratch_mv[:k])
            if n is None or n <= 0:
                return False

            # Append to ring buffer and parse
            self._append_to_ring(n)
            self._parse_ring_for_frames()
            return True
        except serial.SerialException as e:
            logger.error(f"Serial poll error: {e}")
            self.disconnect()
            return False
        except (OSError, TypeError, ValueError, AttributeError) as e:
            logger.debug("Serial disconnect: %s", e)
            self.disconnect()
            return False

    def _append_to_ring(self, n: int) -> tuple[int, int]:
        """Append n bytes from _scratch to ring buffer. Returns (new_head, new_tail)."""
        head, tail = _append_to_ring_numba(
            self._ring, self._scratch, n, self._r_cap, self._r_head, self._r_tail
        )
        self._r_head = head
        self._r_tail = tail
        return head, tail

    def _parse_ring_for_frames(self) -> None:
        """
        Parse as many complete frames as possible from the RX ring buffer in-place.

        Frame format:
          [0xFF,0xFF,0xFF] [LEN] [LEN bytes data ...]
        """
        new_tail, frames_found, has_valid = _parse_frames_njit(
            self._ring, self._r_head, self._r_tail, self._r_cap, self._frame_buf
        )

        self._r_tail = new_tail
        if has_valid:
            self._frame_version += frames_found
            self._frame_ts = time.time()
            self._update_hz_tracking()

    def get_latest_frame_view(self) -> tuple[memoryview | None, int, float]:
        """
        Return a tuple of (memoryview|None, version:int, timestamp:float).
        The memoryview points to a stable 52-byte buffer which is updated by the reader.
        """
        mv = self._frame_mv if self._frame_version > 0 else None
        return (mv, self._frame_version, self._frame_ts)

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
                avg_hz = self._interval_msg_count / (
                    current_time - self._last_print_time
                )
                logger.debug(
                    f"Serial RX Stats - Avg Hz: {avg_hz:.2f} (Total: {self._rx_msg_count})"
                )
            else:
                logger.debug(
                    f"Serial RX Stats - No messages received in last {self._print_interval:.1f}s"
                )

            # Reset interval statistics
            self._last_print_time = current_time
            self._interval_msg_count = 0
