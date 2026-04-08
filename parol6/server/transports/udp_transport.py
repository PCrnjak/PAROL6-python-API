"""
UDP transport implementation for PAROL6 controller.

This module handles UDP socket communication using binary msgpack protocol
for the controller's network interface.
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
        # Pre-allocated buffer for poll_receive_all (avoids list allocation per call)
        self._recv_all_buf: list[tuple[bytes, tuple[str, int]]] = []

    def create_socket(self) -> bool:
        """
        Create and bind the UDP socket.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Non-blocking mode for polling
            self.socket.setblocking(False)

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
            logger.debug(f"UDP socket created and bound to {self.ip}:{self.port}")
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
                logger.debug("UDP socket closed")
            except Exception as e:
                logger.debug(f"Error closing UDP socket: {e}")
            finally:
                self.socket = None

    def poll_receive(self) -> tuple[bytes, tuple[str, int]] | None:
        """Non-blocking receive. Returns raw bytes and address, or None if no data."""
        if not self.socket or not self._running:
            return None
        try:
            nbytes, address = self.socket.recvfrom_into(self._rxv)
            if nbytes <= 0:
                return None
            # Return raw bytes - let caller decode via msgspec
            return (bytes(self._rxv[:nbytes]), address)
        except BlockingIOError:
            return None
        except OSError as e:
            if e.errno in (11, 35):  # EAGAIN/EWOULDBLOCK
                return None
            if not self._running:
                # Socket was closed under us during shutdown — expected.
                return None
            logger.error(f"Socket error in poll_receive: {e}")
            return None

    def poll_receive_all(
        self, max_count: int = 10
    ) -> list[tuple[bytes, tuple[str, int]]]:
        """Non-blocking batch receive up to max_count. Reuses internal buffer."""
        # Returns the internal list directly; caller must consume before next call.
        self._recv_all_buf.clear()
        for _ in range(max_count):
            msg = self.poll_receive()
            if msg is None:
                break
            self._recv_all_buf.append(msg)
        return self._recv_all_buf

    def drain_buffer(self) -> int:
        """
        Drain all pending messages from the UDP receive buffer.

        This is useful in stream mode to discard stale commands when new ones arrive.
        Returns the number of messages drained.
        """
        if not self.socket or not self._running:
            return 0

        drained_count = 0
        try:
            # Socket is already non-blocking; read all pending messages
            while True:
                try:
                    nbytes, _ = self.socket.recvfrom_into(self._rxv)
                    if nbytes > 0:
                        drained_count += 1
                except (BlockingIOError, OSError):
                    # No more data available (expected)
                    break
        except Exception as e:
            logger.debug(f"Error draining UDP buffer: {e}")

        return drained_count

    def send(self, data: bytes, address: tuple[str, int]) -> bool:
        """
        Send raw bytes to a specific address.

        Args:
            data: Pre-packed msgpack bytes to send
            address: Tuple of (ip, port) to send to

        Returns:
            True if successful, False otherwise
        """
        if not self.socket or not self._running:
            logger.warning("Cannot send - socket not available")
            return False

        try:
            self.socket.sendto(data, address)
            return True

        except OSError as e:
            if not self._running:
                # Socket was closed under us during shutdown — expected.
                return False
            logger.error(f"Socket error sending: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending: {e}")
            return False

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
