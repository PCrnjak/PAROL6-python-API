"""
Server management for PAROL6 controller.

Provides lifecycle management and automatic spawning of the controller process.
"""

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Deque
from collections import deque


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

    def __init__(self, controller_path: str | None = None) -> None:
        if controller_path:
            self.controller_path = Path(controller_path).resolve()
            if not self.controller_path.exists():
                raise FileNotFoundError(
                    f"Controller script not found: {self.controller_path}"
                )
        else:
            # Use the package's bundled commander
            self.controller_path = Path(__file__).parent / "controller.py"
            
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._log_buffer: Deque[str] = deque(maxlen=5000)

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc and self._proc.poll() is None else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _write_com_port_hint(self, com_port: str) -> None:
        """
        The controller.py reads com_port.txt at startup.
        Write it unconditionally before launch for consistent behavior across OSes.
        """
        cwd = self.controller_path.parent
        hint = cwd / "com_port.txt"
        try:
            hint.write_text(com_port.strip() + "\n", encoding="utf-8")
        except Exception as e:
            # Non-fatal: controller can still prompt or auto-detect depending on OS
            logging.warning("ServerManager: failed to write %s: %s", hint, e)

    async def start_controller(
        self, 
        com_port: str | None = None, 
        no_autohome: bool = True,
        extra_env: dict | None = None
    ) -> None:
        """Start the controller if not already running."""
        if self.is_running():
            return
        # Working directory should be the PAROL6-python-API repo root to find dependencies
        # Use a more direct approach to find the package root
        cwd = Path(__file__).resolve().parents[2]  # parol6/server/manager.py -> repo root

        # Optional COM port preseed
        if com_port:
            self._write_com_port_hint(com_port)

        env = os.environ.copy()
        # Disable autohome unless explicitly overridden
        if no_autohome:
            env["PAROL6_NOAUTOHOME"] = "1"
        if extra_env:
            env.update(extra_env)
        # Ensure the subprocess can import the local 'parol6' package
        existing_py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing_py_path}" if existing_py_path else str(cwd)

        # Launch the controller as a module to ensure package imports resolve
        args = [sys.executable, "-u", "-m", "parol6.server.controller"]
        
        # Optionally forward current logging level if explicitly enabled
        if os.getenv("PAROL6_PASS_LOG_LEVEL", "").lower() in ("1", "true", "yes", "on"):
            level_name = logging.getLevelName(logging.getLogger().level)
            args.append(f"--log-level={level_name}")
            
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
            for raw_line in iter(proc.stdout.readline, ""):
                if self._stop_reader.is_set():
                    break
                line = raw_line.rstrip("\r\n")
                if line:
                    # Skip timestamp prefix if present (format: HH:MM:SS.mmm [LEVEL] message)
                    space_pos = line.find(" ")
                    if space_pos > 0 and space_pos < 15:  # Reasonable timestamp length
                        # Check if it looks like a timestamp
                        timestamp_part = line[:space_pos]
                        if ":" in timestamp_part:
                            line = line[space_pos + 1:].lstrip()
                    
                    # Preserve severity if prefixes with [LEVEL]
                    level = logging.INFO
                    msg = line

                    if line.startswith("[DEBUG]"):
                        level, msg = logging.DEBUG, line[7:].lstrip()
                    elif line.startswith("[INFO]"):
                        level, msg = logging.INFO, line[6:].lstrip()
                    elif line.startswith("[WARNING]"):
                        level, msg = logging.WARNING, line[9:].lstrip()
                    elif line.startswith("[ERROR]"):
                        level, msg = logging.ERROR, line[7:].lstrip()
                    elif line.startswith("[CRITICAL]"):
                        level, msg = logging.CRITICAL, line[10:].lstrip()

                    self._log_buffer.append(raw_line.rstrip("\r\n"))
                    logging.log(level, "[SERVER] %s", msg)
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

    def get_logs(self, tail: int = 200) -> list[str]:
        """Return the last N lines captured from the controller stdout."""
        return list(self._log_buffer)[-tail:]

    async def await_ready(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 10.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """
        Wait until the controller responds to PING or reports ready in SERVER_STATE.

        Returns:
            True if the server becomes ready within timeout, else False.
        """
        import socket as _socket
        import time as _time

        deadline = _time.time() + max(0.0, float(timeout))
        while _time.time() < deadline:
            # Try a simple PING first
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as sock:
                    sock.settimeout(min(0.5, poll_interval))
                    sock.sendto(b"PING", (host, port))
                    data, _ = sock.recvfrom(256)
                    if data.decode("utf-8").strip().upper() == "PONG":
                        return True
            except Exception:
                pass

            # Try GET_SERVER_STATE and check for {"ready": true}
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as sock:
                    sock.settimeout(min(0.5, poll_interval))
                    sock.sendto(b"GET_SERVER_STATE", (host, port))
                    data, _ = sock.recvfrom(4096)
                    resp = data.decode("utf-8")
                    try:
                        from ..protocol import wire as _wire  # local import to avoid cycles
                        parsed = _wire.parse_server_state(resp)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict) and bool(parsed.get("ready")):
                        return True
            except Exception:
                pass

            await asyncio.sleep(poll_interval)

        return False

    async def get_status(self, host: str = "127.0.0.1", port: int = 5001, timeout: float = 1.0) -> dict:
        """
        Query controller's server state over UDP and merge with process info.
        Returns a dict; if unreachable, returns minimal info.
        """
        status = {
            "running": self.is_running(),
            "pid": self.pid,
            "server": None,
        }
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(timeout)
                sock.sendto(b"GET_SERVER_STATE", (host, port))
                data, _ = sock.recvfrom(4096)
                resp = data.decode("utf-8")
                try:
                    from ..protocol import wire as _wire  # local import avoids top-level deps
                    parsed = _wire.parse_server_state(resp)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    status["server"] = parsed
        except Exception as e:
            # Keep minimal process info and include a hint for diagnostics
            status.setdefault("error", str(e))
        return status


async def ensure_server(
    host: str = "127.0.0.1", 
    port: int = 5001, 
    manage: bool = False, 
    com_port: str | None = None, 
    extra_env: dict | None = None
) -> Optional[ServerManager]:
    """
    Ensure a PAROL6 server is running and accessible.
    
    Args:
        host: Server host to check/connect to
        port: Server port to check/connect to  
        manage: If True, automatically spawn controller if ping fails
        com_port: COM port for spawned controller
        extra_env: Additional environment variables for spawned controller
        
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
            if data.decode('utf-8').strip().upper() == "PONG":
                logging.info(f"Server already running at {host}:{port}")
                return None
    except Exception:
        pass
    
    if not manage:
        logging.warning(f"Server not responding at {host}:{port} and manage=False")
        return None
    
    # Spawn controller
    logging.info(f"Server not responding at {host}:{port}, starting controller...")
    manager = ServerManager()
    await manager.start_controller(
        com_port=com_port, 
        no_autohome=True,
        extra_env=extra_env
    )

    # Wait for readiness within a short window
    ok = await manager.await_ready(host=host, port=port, timeout=5.0)
    if ok:
        logging.info(f"Successfully started server at {host}:{port}")
        return manager

    logging.error("Server spawn failed or not responding after startup")
    await manager.stop_controller()
    return None
