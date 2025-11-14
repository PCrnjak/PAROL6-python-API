"""
Process management utilities for testing.

Provides classes and functions to manage the controller.py subprocess
during integration tests, including startup, readiness checks, and cleanup.
"""

import logging
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class HeadlessCommanderProc:
    """
    Manages a controller.py subprocess for integration testing.

    Handles starting, stopping, and checking readiness of the commander process
    with configurable ports and environment variables.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 5001,
        ack_port: int = 5002,
        env: dict[str, str] | None = None,
    ):
        """
        Initialize the process manager.

        Args:
            ip: IP address for the server to bind to
            port: UDP port for command reception
            ack_port: UDP port for acknowledgment sending
            env: Additional environment variables to set
        """
        self.ip = ip
        self.port = port
        self.ack_port = ack_port
        self.env = env or {}

        self.process: subprocess.Popen | None = None
        self._output_lines: list[str] = []
        self._error_lines: list[str] = []
        self._output_thread: threading.Thread | None = None
        self._error_thread: threading.Thread | None = None

    def start(self, log_level: str = "WARNING", timeout: float = 10.0) -> bool:
        """
        Start the headless commander process.

        Args:
            log_level: Logging level for the subprocess
            timeout: Maximum time to wait for process startup

        Returns:
            True if started successfully, False otherwise
        """
        if self.process and self.process.poll() is None:
            logger.warning("Process already running")
            return True

        # Prepare environment
        process_env = os.environ.copy()
        process_env.update(
            {
                "PAROL6_FAKE_SERIAL": "1",  # Enable fake serial simulation
                "PAROL6_NOAUTOHOME": "1",  # Disable auto-homing for tests
                "PAROL6_SERVER_IP": self.ip,
                "PAROL6_SERVER_PORT": str(self.port),
                "PAROL6_ACK_PORT": str(self.ack_port),
            }
        )
        process_env.update(self.env)

        # Find the controller.py script
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "controller.py"
        )

        if not os.path.exists(script_path):
            logger.error(f"controller.py not found at {script_path}")
            return False

        # Prepare command
        cmd = [sys.executable, script_path, "--log-level", log_level, "--no-auto-home"]

        try:
            logger.info(f"Starting headless commander: {' '.join(cmd)}")
            logger.debug(
                f"Environment: FAKE_SERIAL=1, NOAUTOHOME=1, IP={self.ip}, PORT={self.port}, ACK_PORT={self.ack_port}"
            )

            self.process = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            # Start output capture threads
            self._start_output_capture()

            # Wait for process to be ready
            if self.wait_ready(timeout):
                logger.info(f"Headless commander started successfully (PID: {self.process.pid})")
                return True
            else:
                logger.error("Process failed to become ready within timeout")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            return False

    def _start_output_capture(self):
        """Start threads to capture stdout and stderr."""
        if not self.process:
            return

        def capture_output(stream, output_list):
            try:
                for line in iter(stream.readline, ""):
                    if line:
                        line = line.rstrip("\n\r")
                        output_list.append(line)
                        # Also log to our test logger for debugging
                        logger.debug(f"PROC: {line}")
            except Exception as e:
                logger.debug(f"Output capture error: {e}")

        self._output_thread = threading.Thread(
            target=capture_output, args=(self.process.stdout, self._output_lines), daemon=True
        )
        self._error_thread = threading.Thread(
            target=capture_output, args=(self.process.stderr, self._error_lines), daemon=True
        )

        self._output_thread.start()
        self._error_thread.start()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """
        Wait for the process to be ready by sending PING commands.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if process responds to PING, False otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                # Process died
                logger.error("Process terminated during startup")
                return False

            # Try to ping the server
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(1.0)
                    sock.sendto(b"PING", (self.ip, self.port))

                    try:
                        data, _ = sock.recvfrom(1024)
                        if data.decode("utf-8").strip() == "PONG":
                            return True
                    except TimeoutError:
                        pass  # No response yet

            except Exception as e:
                logger.debug(f"Ping attempt failed: {e}")

            time.sleep(0.5)  # Wait before retry

        return False

    def stop(self) -> bool:
        """
        Stop the headless commander process.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.process:
            return True

        try:
            # Try graceful shutdown first
            logger.debug("Attempting graceful shutdown...")
            self.process.terminate()

            try:
                self.process.wait(timeout=5.0)
                logger.info(f"Process terminated gracefully (exit code: {self.process.returncode})")
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate gracefully, forcing shutdown...")
                self.process.kill()
                self.process.wait()
                logger.info(f"Process killed (exit code: {self.process.returncode})")

        except Exception as e:
            logger.error(f"Error stopping process: {e}")
            return False
        finally:
            self.process = None

        return True

    def is_running(self) -> bool:
        """Check if the process is currently running."""
        return self.process is not None and self.process.poll() is None

    def get_output_lines(self) -> list[str]:
        """Get captured stdout lines."""
        return self._output_lines.copy()

    def get_error_lines(self) -> list[str]:
        """Get captured stderr lines."""
        return self._error_lines.copy()

    def clear_output(self):
        """Clear captured output lines."""
        self._output_lines.clear()
        self._error_lines.clear()


def wait_for_completion(result_or_id: Any, timeout: float = 10.0) -> dict[str, Any]:
    """
    Unified waiting logic for acknowledgment-tracked results in tests.

    Handles both direct result dictionaries and command IDs that need polling.

    Args:
        result_or_id: Either a result dict with status info, or a command ID string
        timeout: Maximum time to wait for completion

    Returns:
        Dictionary with status information
    """
    if isinstance(result_or_id, dict):
        # Already a result dictionary
        return result_or_id

    # If it's a string, it might be a command ID - for now just return a placeholder
    # In a real implementation, this would poll the robot_api for status
    return {
        "status": "TIMEOUT",
        "details": "wait_for_completion not fully implemented for command IDs",
        "completed": True,
        "command_id": result_or_id,
    }


def find_available_ports(start_port: int = 5001, count: int = 2) -> list[int]:
    """
    Find available UDP ports starting from the given port.

    Args:
        start_port: Port to start searching from
        count: Number of consecutive ports needed

    Returns:
        List of available port numbers
    """
    available_ports: list[int] = []
    current_port = start_port

    while len(available_ports) < count:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", current_port))
                available_ports.append(current_port)
        except OSError:
            # Port in use, reset search if we were building a consecutive sequence
            available_ports.clear()

        current_port += 1

        # Prevent infinite loop
        if current_port > start_port + 1000:
            break

    return available_ports


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a UDP port is available.

    Args:
        port: Port number to check
        host: Host address to check

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False
