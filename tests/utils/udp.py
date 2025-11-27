"""
UDP utilities for testing.

Provides helper functions for UDP communication during tests,
including acknowledgment listening and command sending utilities.
"""

import logging
import queue
import socket
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class UDPClient:
    """
    Simple UDP client for sending commands to the robot server during tests.
    """

    def __init__(self, server_ip: str = "127.0.0.1", server_port: int = 5001):
        """
        Initialize UDP client.

        Args:
            server_ip: IP address of the robot server
            server_port: Port of the robot server
        """
        self.server_ip = server_ip
        self.server_port = server_port

    def send_command(self, command: str, timeout: float = 2.0) -> str | None:
        """
        Send a command and wait for immediate response (for GET commands).

        Args:
            command: Command string to send
            timeout: Timeout for waiting for response

        Returns:
            Response string if received, None otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(timeout)

                # Send command
                sock.sendto(command.encode("utf-8"), (self.server_ip, self.server_port))
                logger.debug(f"Sent command: {command}")

                # Wait for response (for GET commands)
                try:
                    data, _ = sock.recvfrom(2048)
                    response = data.decode("utf-8")
                    logger.debug(f"Received response: {response}")
                    return response
                except TimeoutError:
                    logger.debug(f"No response received for command: {command}")
                    return None

        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            return None

    def send_command_no_response(self, command: str) -> bool:
        """
        Send a command without waiting for response (for motion commands).

        Args:
            command: Command string to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(command.encode("utf-8"), (self.server_ip, self.server_port))
                logger.debug(f"Sent command (no response): {command}")
                return True
        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            return False


class AckListener:
    """
    Listens for UDP acknowledgment messages during tests.

    Provides functionality to capture and wait for specific acknowledgments
    from the robot controller during command execution.
    """

    def __init__(self, listen_port: int = 5002):
        """
        Initialize acknowledgment listener.

        Args:
            listen_port: Port to listen on for acknowledgments
        """
        self.listen_port = listen_port
        self.socket: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.running = False

        # Storage for received acknowledgments
        self.ack_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.ack_history: list[dict[str, Any]] = []

    def start(self) -> bool:
        """
        Start listening for acknowledgments.

        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("AckListener already running")
            return True

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(("127.0.0.1", self.listen_port))
            self.socket.settimeout(0.1)  # Short timeout for non-blocking operation

            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()

            logger.debug(f"AckListener started on port {self.listen_port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start AckListener: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop listening for acknowledgments."""
        self.running = False

        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.socket:
            self.socket.close()
            self.socket = None

        logger.debug("AckListener stopped")

    def _listen_loop(self):
        """Main listening loop (runs in separate thread)."""
        while self.running and self.socket:
            try:
                data, addr = self.socket.recvfrom(1024)
                message = data.decode("utf-8")

                # Parse acknowledgment message: ACK|cmd_id|status|details
                parts = message.split("|", 3)
                if parts[0] == "ACK" and len(parts) >= 3:
                    ack_info = {
                        "cmd_id": parts[1],
                        "status": parts[2],
                        "details": parts[3] if len(parts) > 3 else "",
                        "timestamp": time.time(),
                        "sender": addr,
                    }

                    # Add to both queue and history
                    self.ack_queue.put(ack_info)
                    self.ack_history.append(ack_info)

                    logger.debug(f"Received ACK: {ack_info}")

            except TimeoutError:
                continue  # Normal timeout, keep listening
            except Exception as e:
                if self.running:  # Only log if we should still be running
                    logger.debug(f"Listen loop error: {e}")

    def wait_for_ack(
        self, cmd_id: str, timeout: float = 5.0, expected_status: str | None = None
    ) -> dict[str, Any] | None:
        """
        Wait for a specific acknowledgment.

        Args:
            cmd_id: Command ID to wait for
            timeout: Maximum time to wait
            expected_status: Specific status to wait for (optional)

        Returns:
            Acknowledgment info dict if received, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check queue with short timeout
                ack_info = self.ack_queue.get(timeout=0.1)

                if ack_info["cmd_id"] == cmd_id:
                    if expected_status is None or ack_info["status"] == expected_status:
                        return ack_info
                    else:
                        # Put back in queue if status doesn't match
                        self.ack_queue.put(ack_info)
                else:
                    # Put back in queue if cmd_id doesn't match
                    self.ack_queue.put(ack_info)

            except queue.Empty:
                continue

        logger.debug(
            f"Timeout waiting for ACK: cmd_id={cmd_id}, expected_status={expected_status}"
        )
        return None

    def get_all_acks_for_command(self, cmd_id: str) -> list[dict[str, Any]]:
        """
        Get all acknowledgments received for a specific command ID.

        Args:
            cmd_id: Command ID to search for

        Returns:
            List of acknowledgment info dicts
        """
        return [ack for ack in self.ack_history if ack["cmd_id"] == cmd_id]

    def get_recent_acks(self, count: int = 10) -> list[dict[str, Any]]:
        """
        Get the most recent acknowledgments.

        Args:
            count: Number of recent acknowledgments to return

        Returns:
            List of recent acknowledgment info dicts
        """
        return self.ack_history[-count:] if self.ack_history else []

    def clear_history(self):
        """Clear acknowledgment history and queue."""
        self.ack_history.clear()
        while not self.ack_queue.empty():
            try:
                self.ack_queue.get_nowait()
            except queue.Empty:
                break


def send_command_with_ack(
    command: str,
    server_ip: str = "127.0.0.1",
    server_port: int = 5001,
    ack_port: int = 5002,
    timeout: float = 5.0,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Send a command with acknowledgment tracking.

    This is a convenience function that sets up an acknowledgment listener,
    sends a command, and waits for the acknowledgment.

    Args:
        command: Command to send
        server_ip: Server IP address
        server_port: Server command port
        ack_port: Acknowledgment port
        timeout: Timeout for acknowledgment

    Returns:
        Tuple of (immediate_response, final_ack_info)
    """
    # Set up acknowledgment listener
    ack_listener = AckListener(ack_port)
    if not ack_listener.start():
        return None, None

    try:
        # Send command
        client = UDPClient(server_ip, server_port)
        response = client.send_command(command, timeout=1.0)

        # For tracked commands, the response might be empty and we need to wait for ACK
        # For immediate commands (GET_*), we get response right away
        if response and not response.startswith("ACK"):
            # Got immediate response, no ACK expected
            return response, None

        # Wait for acknowledgment (extract command ID if present)
        # This is a simplified version - in practice, you'd extract the actual command ID
        # from the command string if it contains one
        parts = command.split("|", 1)
        if (
            len(parts) > 1
            and len(parts[0]) == 8
            and parts[0].replace("-", "").isalnum()
        ):
            cmd_id = parts[0]
            ack_info = ack_listener.wait_for_ack(cmd_id, timeout)
            return response, ack_info
        else:
            # No command ID in command, can't track ACK
            return response, None

    finally:
        ack_listener.stop()


def create_tracked_command(base_command: str, cmd_id: str | None = None) -> str:
    """
    Create a command with tracking ID.

    Args:
        base_command: Base command string
        cmd_id: Command ID to use (generates one if None)

    Returns:
        Command string with tracking ID prepended
    """
    import uuid

    if cmd_id is None:
        cmd_id = str(uuid.uuid4())[:8]

    return f"{cmd_id}|{base_command}"


def parse_server_response(response: str) -> dict[str, Any]:
    """
    Parse a server response into a structured format.

    Args:
        response: Raw response string from server

    Returns:
        Dictionary with parsed response data
    """
    if not response:
        return {"type": "empty", "data": None}

    parts = response.split("|", 1)
    response_type = parts[0]

    parsed: dict[str, Any] = {"type": response_type, "raw": response}

    if len(parts) > 1:
        data = parts[1]

        if response_type in ["POSE", "ANGLES", "SPEEDS"]:
            # Numeric array data
            try:
                parsed["data"] = [float(x) for x in data.split(",")]
            except ValueError:
                parsed["data"] = data
        elif response_type in ["IO", "GRIPPER"]:
            # Integer array data
            try:
                parsed["data"] = [int(x) for x in data.split(",")]
            except ValueError:
                parsed["data"] = data
        elif response_type == "STATUS":
            # Complex status data
            parsed["data"] = {}
            for item in data.split("|"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    parsed["data"][key] = value
        else:
            parsed["data"] = data
    else:
        parsed["data"] = None

    return parsed
