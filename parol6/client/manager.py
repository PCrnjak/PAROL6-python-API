"""
Server management for PAROL6 controller.

Provides lifecycle management and automatic spawning of the controller process.
"""

import asyncio
import contextlib
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path

# Precompiled regex patterns for server log normalization
_SIMPLE_FORMAT_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2})\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL|TRACE)\s+([A-Za-z0-9_.-]+):\s+(.*)$"
)


@dataclass
class ServerOptions:
    """Options for launching the controller."""

    com_port: str | None = None
    no_autohome: bool = True  # Set PAROL6_NOAUTOHOME=1 by default
    extra_env: dict | None = None


class ServerManager:
    """
    Manages the lifecycle of the PAROL6 controller.

    - Writes com_port.txt in the controller working directory to preselect the port.
    - Spawns the controller as a subprocess using sys.executable.
    - Provides stop and liveness checks.
    - Can be used with a custom controller script path or defaults to the package's bundled controller.
    """

    def __init__(self, controller_path: str | None = None, normalize_logs: bool = False) -> None:
        """
        Initialize the ServerManager.

        Args:
            controller_path: Optional path to controller script. If None, uses bundled controller.
            normalize_logs: If True, parse and normalize controller log output to avoid duplicate
                          timestamp/level/module info. Set to True when used from web GUI.
        """
        if controller_path:
            self.controller_path = Path(controller_path).resolve()
            if not self.controller_path.exists():
                raise FileNotFoundError(f"Controller script not found: {self.controller_path}")
        else:
            # Use the package's bundled commander
            self.controller_path = Path(__file__).parent / "controller.py"

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self.normalize_logs = normalize_logs

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc and self._proc.poll() is None else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start_controller(
        self,
        com_port: str | None = None,
        no_autohome: bool = True,
        extra_env: dict | None = None,
        server_host: str | None = None,
        server_port: int | None = None,
    ) -> None:
        """Start the controller if not already running."""
        if self.is_running():
            return
        # Working directory should be the PAROL6-python-API repo root to find dependencies
        # Use a more direct approach to find the package root
        cwd = Path(__file__).resolve().parents[2]  # parol6/server/manager.py -> repo root

        env = os.environ.copy()
        # Disable autohome unless explicitly overridden
        if no_autohome:
            env["PAROL6_NOAUTOHOME"] = "1"
        if extra_env:
            env.update(extra_env)
        # Explicitly bind controller (if provided)
        if server_host:
            env["PAROL6_CONTROLLER_IP"] = server_host
        if server_port is not None:
            env["PAROL6_CONTROLLER_PORT"] = str(server_port)
        # Ensure the subprocess can import the local 'parol6' package
        existing_py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing_py_path}" if existing_py_path else str(cwd)

        # Launch the controller as a module to ensure package imports resolve
        args = [sys.executable, "-u", "-m", "parol6.server.controller"]

        # Decide controller log level:
        # - If PAROL_TRACE is set in the environment, prefer TRACE for the child
        # - Otherwise, inherit the current root logger level name
        root_logger = logging.getLogger()
        root_level = root_logger.level

        parol_trace_flag = str(env.get("PAROL_TRACE", "0")).strip().lower()
        if parol_trace_flag in ("1", "true", "yes", "on"):
            level_name = "TRACE"
        else:
            level_name = logging.getLevelName(root_level)
            # Normalize custom/unnamed levels (e.g. "Level 5")
            if isinstance(level_name, str) and level_name.upper().startswith("LEVEL"):
                if root_level == 5:
                    level_name = "TRACE"
                else:
                    level_name = "INFO"

        args.append(f"--log-level={level_name}")
        if com_port:
            args.append(f"--serial={com_port}")

        try:
            self._proc = subprocess.Popen(
                args,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start controller: {e}") from e

        # Start background reader thread to stream server stdout/stderr to logging
        if self._proc and self._proc.stdout is not None:
            self._stop_reader.clear()
            self._reader_thread = threading.Thread(
                target=self._stream_output,
                args=(self._proc,),
                name="ServerOutputReader",
                daemon=True,
            )
            self._reader_thread.start()

    def _stream_output(self, proc: subprocess.Popen) -> None:
        """Read controller stdout and forward to logging."""
        try:
            assert proc.stdout is not None
            # Maintain last logger/level for multi-line tracebacks
            last_logger = "parol6.server"

            for raw_line in iter(proc.stdout.readline, ""):
                if self._stop_reader.is_set():
                    break
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                if self.normalize_logs:
                    # Normalize child log line and route to embedded module logger
                    level = logging.INFO
                    logger_name: str | None = None
                    msg = line

                    s = _SIMPLE_FORMAT_RE.match(line)
                    if s:
                        _, level_name, logger_name, actual_message = s.groups()
                        logger_name = (logger_name or "").strip()
                        msg = actual_message
                        level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
                    elif line.startswith("Traceback"):
                        # Traceback and continuation lines -> keep last context, escalate on Traceback
                        level = logging.ERROR

                    # Choose target logger
                    target_logger_name = logger_name or last_logger or "parol6.server"
                    target_logger = logging.getLogger(target_logger_name)
                    target_logger.log(level, msg)

                    # Update last context if we identified a module
                    if logger_name:
                        last_logger = logger_name
                else:
                    # No normalization - forward line as-is to root logger
                    print(line)
        except Exception as e:
            logging.warning("ServerManager: output reader stopped: %s", e)

    def stop_controller(self, timeout: float = 2.0) -> None:
        """Stop the controller process if running.

        This method attempts a graceful shutdown first using SIGTERM (or terminate() on Windows)
        and then escalates to a forced kill if the process does not exit within ``timeout``.
        After sending SIGKILL it will wait up to ``kill_timeout`` seconds for the process to
        actually exit before giving up.
        """
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=timeout)
        self._reader_thread = None
        if self._proc and self._proc.poll() is None:
            logging.debug("Stopping Controller...")
            try:
                self._proc.terminate()
                self._proc.wait(timeout=timeout)
            except Exception as e:
                logging.warning("stop_controller: terminate/wait failed: %s", e)

            # Step 2: escalate to forced kill if still running
            if self._proc and self._proc.poll() is None:
                logging.warning("Controller did not exit after SIGTERM within %.1fs, sending SIGKILL", timeout)
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=timeout)
                except Exception as e:
                    logging.warning("stop_controller: kill/wait failed: %s", e)
            # Clear reference
            self._proc = None

    async def await_ready(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 10.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """
        Wait until the controller responds to PING over UDP, asynchronously.

        Returns:
            True if the server becomes ready within timeout, else False.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        addr: Tuple[str, int] = (host, port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)

        try:
            while True:
                now = loop.time()
                if now >= deadline:
                    return False

                # Try a PING and wait up to poll_interval (or remaining time)
                recv_timeout = min(poll_interval, deadline - now)

                try:
                    # send PING
                    await loop.sock_sendto(sock, b"PING", addr)

                    # wait for PONG with a timeout
                    data, _ = await asyncio.wait_for(
                        loop.sock_recvfrom(sock, 256),
                        timeout=recv_timeout,
                    )
                    if data.decode("ascii", errors="ignore").startswith("PONG"):
                        return True
                except (asyncio.TimeoutError, OSError):
                    # Timeout or send/recv error -> just try again until deadline
                    pass

                # Optional extra delay to avoid tight loop; keep within deadline
                remain = deadline - loop.time()
                if remain <= 0:
                    return False
                await asyncio.sleep(min(poll_interval, remain))
        finally:
            sock.close()

def is_server_running(
    host: str = "127.0.0.1",
    port: int = 5001,
    timeout: float = 1.0,
) -> bool:
    """Return True if a PAROL6 controller responds to UDP PING at host:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(b"PING", (host, port))
            data, _ = sock.recvfrom(256)
            return data.decode("ascii", errors="ignore").startswith("PONG")
    except Exception:
        return False


def manage_server(
    host: str = "127.0.0.1",
    port: int = 5001,
    com_port: str | None = None,
    extra_env: dict | None = None,
    normalize_logs: bool = False,
) -> ServerManager:
    """Synchronously start a PAROL6 controller at host:port and block until ready.

    Fast-fails if a server is already running there.

    Returns a ServerManager that owns the spawned controller.
    """
    if is_server_running(host=host, port=port):
        raise RuntimeError(f"Server already running at {host}:{port}")

    logging.info(f"Server not responding at {host}:{port}, starting controller...")

    # Prepare environment for child controller bind tuple
    env_to_pass = dict(extra_env) if extra_env else {}
    env_to_pass["PAROL6_CONTROLLER_IP"] = host
    env_to_pass["PAROL6_CONTROLLER_PORT"] = str(port)

    manager = ServerManager(normalize_logs=normalize_logs)
    manager.start_controller(
        com_port=com_port,
        no_autohome=True,
        extra_env=env_to_pass,
        server_host=host,
        server_port=port,
    )

    # Block until PING responds or timeout
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            if is_server_running(host=host, port=port, timeout=0.2):
                logging.info(f"Successfully started server at {host}:{port}")
                return manager
        except Exception:
            pass
        time.sleep(0.05)

    logging.error("Server spawn failed or not responding after startup")
    manager.stop_controller()
    raise RuntimeError("Failed to start PAROL6 controller")


@contextlib.contextmanager
def managed_server(
    host: str = "127.0.0.1",
    port: int = 5001,
    com_port: str | None = None,
    extra_env: dict | None = None,
    normalize_logs: bool = False,
):
    """Synchronous context manager that ensures the controller is stopped on exit.

    Usage:
        with managed_server(host, port) as mgr:
            ...
    """
    mgr = manage_server(
        host=host,
        port=port,
        com_port=com_port,
        extra_env=extra_env,
        normalize_logs=normalize_logs,
    )
    try:
        yield mgr
    finally:
        mgr.stop_controller()
