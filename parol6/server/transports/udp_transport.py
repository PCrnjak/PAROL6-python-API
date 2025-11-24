"""
UDP transport implementation for PAROL6 controller.

This module handles UDP socket communication, message parsing, and
response handling for the controller's network interface.
"""

from __future__ import annotations

import logging
import socket
import time

logger = logging.getLogger(__name__)


class UDPTransport:
    """
    Manages UDP socket communication for the controller.

    This class handles:
    - Socket creation and binding
    - Non-blocking message reception
    - Response sending
    - Connection management
    """

    def __init__(
        self, ip: str = "127.0.0.1", port: int = 5001, buffer_size: int = 1024
    ):
        """
        Initialize the UDP transport.

        Args:
            ip: IP address to bind to
            port: Port number to listen on
            buffer_size: Size of the receive buffer in bytes
        """
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.socket: socket.socket | None = None
        self._running = False
        self._rx = bytearray(self.buffer_size)
        self._rxv = memoryview(self._rx)

    def create_socket(self) -> bool:
        """
        Create and bind the UDP socket.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Use blocking mode with short timeout for responsive shutdown
            self.socket.setblocking(True)
            self.socket.settimeout(0.25)

            # Allow address/port reuse for fast restarts
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                # Not available on all platforms
                pass
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

            # Bind to address with small retry window to avoid transient EADDRINUSE
            _attempts = 3
            _delay_s = 0.1
            for _i in range(_attempts):
                try:
                    self.socket.bind((self.ip, self.port))
                    break
                except OSError as _e:
                    if _i == _attempts - 1:
                        raise
                    time.sleep(_delay_s)

            self._running = True
            logger.info(f"UDP socket created and bound to {self.ip}:{self.port}")
            return True

        except OSError as e:
            logger.error(f"Failed to create UDP socket: {e}")
            self.socket = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating UDP socket: {e}")
            self.socket = None
            return False

    def close_socket(self) -> None:
        """Close the UDP socket and clean up resources."""
        self._running = False

        if self.socket:
            try:
                self.socket.close()
                logger.info("UDP socket closed")
            except Exception as e:
                logger.error(f"Error closing UDP socket: {e}")
            finally:
                self.socket = None

    def receive_one(self) -> tuple[str, tuple[str, int]] | None:
        """
        Blocking receive of a single datagram using recvfrom_into with a short timeout.
        Returns (message_str, address) on success, or None on timeout/error.
        """
        if not self.socket or not self._running:
            return None
        try:
            nbytes, address = self.socket.recvfrom_into(self._rxv)
            if nbytes <= 0:
                return None
            try:
                # Decode ASCII payload and trim only CR/LF to avoid extra copies
                message_str = (
                    self._rxv[:nbytes]
                    .tobytes()
                    .decode("ascii", errors="ignore")
                    .rstrip("\r\n")
                )
            except Exception:
                logger.warning(f"Failed to decode UDP datagram from {address}")
                return None
            return (message_str, address)
        except TimeoutError:
            return None
        except OSError as e:
            logger.error(f"Socket error receiving UDP message: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in receive_one: {e}")
            return None

    def drain_buffer(self) -> int:
        """
        Drain all pending messages from the UDP receive buffer.

        This is useful in stream mode to discard stale commands when new ones arrive.
        Returns the number of messages drained.
        """
        if not self.socket or not self._running:
            return 0

        drained_count = 0
        original_timeout = None

        try:
            # Temporarily switch to non-blocking mode
            original_timeout = self.socket.gettimeout()
            self.socket.setblocking(False)

            # Read all pending messages until buffer is empty
            while True:
                try:
                    nbytes, _ = self.socket.recvfrom_into(self._rxv)
                    if nbytes > 0:
                        drained_count += 1
                except OSError:
                    # No more data available (expected)
                    break

            # Restore original timeout
            self.socket.settimeout(original_timeout)

        except Exception as e:
            logger.error(f"Error draining UDP buffer: {e}")
            # Try to restore timeout even if draining failed
            try:
                if original_timeout is not None:
                    self.socket.settimeout(original_timeout)
            except Exception as e2:
                logger.debug("Failed to restore UDP socket timeout: %s", e2)

        return drained_count

    def send_response(self, message: str, address: tuple[str, int]) -> bool:
        """
        Send a response message to a specific address.

        Args:
            message: The message string to send
            address: Tuple of (ip, port) to send to

        Returns:
            True if successful, False otherwise
        """
        if not self.socket or not self._running:
            logger.warning("Cannot send response - socket not available")
            return False

        try:
            # Encode and send the message
            data = message.encode("ascii")
            self.socket.sendto(data, address)
            return True

        except OSError as e:
            logger.error(f"Socket error sending UDP response: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending UDP response: {e}")
            return False

    def broadcast_message(self, message: str, addresses: list[tuple[str, int]]) -> int:
        """
        Broadcast a message to multiple addresses.

        Args:
            message: The message to broadcast
            addresses: List of (ip, port) tuples to send to

        Returns:
            Number of successful sends
        """
        success_count = 0

        for address in addresses:
            if self.send_response(message, address):
                success_count += 1

        return success_count

    def is_running(self) -> bool:
        """
        Check if the transport is running.

        Returns:
            True if running, False otherwise
        """
        return self._running and self.socket is not None

    def get_socket_info(self) -> dict:
        """
        Get information about the current socket.

        Returns:
            Dictionary with socket information
        """
        info = {
            "ip": self.ip,
            "port": self.port,
            "buffer_size": self.buffer_size,
            "running": self._running,
            "socket_open": self.socket is not None,
        }

        if self.socket:
            try:
                sockname = self.socket.getsockname()
                info["bound_address"] = sockname
            except Exception as e:
                logger.debug("Failed to get UDP sockname: %s", e)

        return info


def create_udp_transport(
    ip: str = "127.0.0.1", port: int = 5001
) -> UDPTransport | None:
    """
    Factory function to create and initialize a UDP transport.

    Args:
        ip: IP address to bind to
        port: Port number to listen on

    Returns:
        Initialized UDPTransport instance, or None if initialization failed
    """
    transport = UDPTransport(ip, port)

    if transport.create_socket():
        return transport
    else:
        return None
