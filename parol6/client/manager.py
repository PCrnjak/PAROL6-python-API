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
import subprocess
import sys
import threading
import time
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

    async def start_controller(
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

        level_name = logging.getLevelName(logging.getLogger().level)
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

    async def stop_controller(self, timeout: float = 5.0) -> None:
        """Stop the controller process if running."""
        if not self.is_running():
            self._proc = None
            return

        proc = self._proc
        assert proc is not None

        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
        except Exception:
            # Fall back to kill below
            pass

        # Wait for graceful exit
        t0 = time.time()
        while proc.poll() is None and (time.time() - t0) < timeout:
            await asyncio.sleep(0.1)

        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

        # Stop and join background reader thread
        with contextlib.suppress(Exception):
            self._stop_reader.set()
            if proc.stdout:
                proc.stdout.close()
        if self._reader_thread and self._reader_thread.is_alive():
            with contextlib.suppress(Exception):
                self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

        self._proc = None

    async def await_ready(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 10.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """
        Wait until the controller responds to PING.

        Returns:
            True if the server becomes ready within timeout, else False.
        """
        import socket as _socket
        import time as _time

        deadline = _time.time() + max(0.0, float(timeout))
        while _time.time() < deadline:
            # Try a simple PING
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as sock:
                    sock.settimeout(min(0.5, poll_interval))
                    sock.sendto(b"PING", (host, port))
                    data, _ = sock.recvfrom(256)
                    if data.decode("ascii").startswith("PONG"):
                        return True
            except Exception:
                pass

            await asyncio.sleep(poll_interval)

        return False


async def ensure_server(
    host: str = "127.0.0.1",
    port: int = 5001,
    manage: bool = False,
    com_port: str | None = None,
    extra_env: dict | None = None,
    normalize_logs: bool = False,
) -> ServerManager | None:
    """
    Ensure a PAROL6 server is running and accessible.

    Args:
        host: Server host to check/connect to
        port: Server port to check/connect to
        manage: If True, automatically spawn controller if ping fails
        com_port: COM port for spawned controller
        extra_env: Additional environment variables for spawned controller
        normalize_logs: If True, parse and normalize controller log output to avoid duplicate
                          timestamp/level/module info. Set to True when used from web GUI.

    Returns:
        ServerManager instance if manage=True and server was spawned, None otherwise

    Usage:
        # Just check if server is running
        await ensure_server()

        # Auto-spawn if not running
        mgr = await ensure_server(manage=True, com_port="/dev/ttyACM0")
    """
    # Test if server is already running
    try:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            sock.sendto(b"PING", (host, port))
            data, _ = sock.recvfrom(256)
            if data.decode("ascii").startswith("PONG"):
                logging.info(f"Server already running at {host}:{port}")
                return None
    except Exception:
        pass

    if not manage:
        logging.warning(f"Server not responding at {host}:{port} and manage=False")
        return None

    # Spawn controller
    logging.info(f"Server not responding at {host}:{port}, starting controller...")
    # Prepare environment for child controller bind tuple
    env_to_pass = dict(extra_env) if extra_env else {}
    env_to_pass["PAROL6_CONTROLLER_IP"] = host
    env_to_pass["PAROL6_CONTROLLER_PORT"] = str(port)
    manager = ServerManager(normalize_logs=normalize_logs)
    await manager.start_controller(
        com_port=com_port,
        no_autohome=True,
        extra_env=env_to_pass,
        server_host=host,
        server_port=port,
    )

    # Wait for readiness within a short window
    ok = await manager.await_ready(host=host, port=port, timeout=5.0)
    if ok:
        logging.info(f"Successfully started server at {host}:{port}")
        return manager

    logging.error("Server spawn failed or not responding after startup")
    await manager.stop_controller()
    return None
