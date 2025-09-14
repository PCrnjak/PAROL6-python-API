"""
Host/port-aware command tracker for UDP ACKs.

Provides:
- CommandTracker: binds an ACK listener on ack_port and sends tracked commands to (host, port)
- Non-blocking and blocking send flows with typed-ish results compatible with existing callers
- Minimal logging and lifecycle controls (start/stop/is_active)
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Tuple, Union

from ..protocol.types import AckStatus, TrackingStatus


logger = logging.getLogger(__name__)

# Shared tracker registry (keyed by (host, port, ack_port))
_shared_lock = threading.Lock()
_SHARED_TRACKERS: Dict[Tuple[str, int, int], 'CommandTracker'] = {}


def get_shared_tracker(host: str, port: int, ack_port: int, sock_factory: Optional[Callable[[], socket.socket]] = None) -> 'CommandTracker':
    """Get or create a process-wide shared CommandTracker for the given endpoint."""
    key = (host, int(port), int(ack_port))
    with _shared_lock:
        tracker = _SHARED_TRACKERS.get(key)
        if tracker is None:
            tracker = CommandTracker(host=host, port=port, ack_port=ack_port, sock_factory=sock_factory)
            _SHARED_TRACKERS[key] = tracker
        return tracker


class CommandTracker:
    """
    Track commands by prepending a short UUID and listening for 'ACK|id|status|details?' datagrams.

    Notes:
    - ACK listener runs on a background thread bound to 'ack_port'.
    - 'send' will automatically call 'start' on first use.
    - For tests, a custom sock_factory can be provided to avoid real sockets.
    """

    def __init__(
        self,
        host: str,
        port: int,
        ack_port: int,
        sock_factory: Optional[Callable[[], socket.socket]] = None,
        history_ttl_s: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.ack_port = ack_port
        self._sock_factory = sock_factory
        self._history_ttl_s = history_ttl_s

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ack_sock: Optional[socket.socket] = None

        # id -> TrackingStatus + extra fields
        self._history: Dict[str, Dict] = {}

    # ---------- lifecycle ----------

    def start(self) -> None:
        """Start ACK listener thread if not already active."""
        with self._lock:
            if self._running:
                return
            try:
                self._ack_sock = self._create_socket()
                try:
                    self._ack_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except Exception:
                    pass
                try:
                    # May not be available on all platforms
                    self._ack_sock.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_REUSEPORT", socket.SO_REUSEADDR), 1)
                except Exception:
                    pass
                # Bind to all interfaces so server can reach us
                self._ack_sock.bind(("", int(self.ack_port)))
                self._ack_sock.settimeout(0.1)
            except OSError as e:
                logger.error("CommandTracker failed to bind ACK socket on %s: %s", self.ack_port, e)
                # Ensure cleanup
                if self._ack_sock:
                    try:
                        self._ack_sock.close()
                    finally:
                        self._ack_sock = None
                raise

            self._running = True
            self._thread = threading.Thread(target=self._listen_loop, name="parol6-ack-listener", daemon=True)
            self._thread.start()
            logger.debug("CommandTracker started on ack_port=%s", self.ack_port)

    def stop(self) -> None:
        """Stop ACK listener and cleanup resources."""
        with self._lock:
            self._running = False
            sock = self._ack_sock
            self._ack_sock = None
        # Join outside of lock to avoid deadlocks
        if self._thread:
            self._thread.join(timeout=0.5)
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        self._thread = None
        logger.debug("CommandTracker stopped")

    def is_active(self) -> bool:
        """Return True if the listener thread is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    # ---------- internals ----------

    def _create_socket(self) -> socket.socket:
        return self._sock_factory() if self._sock_factory else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _register_command(self, plain_message: str) -> Tuple[str, str]:
        """
        Register a new command in history and return (tracked_message, cmd_id).
        Tracked message format: '<id>|<plain_message>'.
        """
        cmd_id = str(uuid.uuid4())[:8]
        tracked = f"{cmd_id}|{plain_message}"
        with self._lock:
            self._history[cmd_id] = {
                "command": plain_message,
                "status": "SENT",
                "details": "",
                "completed": False,
                "ack_time": None,
                "sent_time": datetime.now(),
            }
        return tracked, cmd_id

    def _cleanup_old(self) -> None:
        """Remove history entries older than TTL to avoid unbounded growth."""
        cutoff = datetime.now() - timedelta(seconds=self._history_ttl_s)
        with self._lock:
            expired = [cid for cid, info in self._history.items() if info.get("sent_time") and info["sent_time"] < cutoff]
            for cid in expired:
                self._history.pop(cid, None)

    def _listen_loop(self) -> None:
        """Background ACK listener. Expects 'ACK|<id>|<status>|<details?>'."""
        assert self._ack_sock is not None
        sock = self._ack_sock
        while self._running:
            try:
                data, _ = sock.recvfrom(4096)
                msg = data.decode("utf-8", errors="replace")
                self._handle_ack_line(msg)
                self._cleanup_old()
            except socket.timeout:
                continue
            except OSError as e:
                # Socket likely closed during shutdown
                if self._running:
                    logger.debug("ACK listener socket error: %s", e)
            except Exception as e:
                logger.debug("ACK listener error: %s", e)

    def _handle_ack_line(self, message: str) -> None:
        parts = message.split("|", 3)
        if not parts or parts[0] != "ACK" or len(parts) < 3:
            return
        _, cmd_id, status = parts[0], parts[1], parts[2]
        details = parts[3] if len(parts) > 3 else ""
        self._handle_ack(cmd_id, status, details)

    def _handle_ack(self, cmd_id: str, status: str, details: str) -> None:
        """Update history for an ACK. Public for tests via _inject_ack."""
        with self._lock:
            entry = self._history.get(cmd_id)
            if not entry:
                return
            entry["status"] = status
            entry["details"] = details
            entry["ack_time"] = datetime.now()
            entry["completed"] = status in ("COMPLETED", "FAILED", "INVALID", "CANCELLED")

    # ---------- API ----------

    def send(
        self,
        message: str,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, TrackingStatus]:
        """
        Send a command to (host, port).

        Returns:
        - If wait_for_ack=True and non_blocking=True -> returns the command_id (str) immediately
        - If wait_for_ack=True and non_blocking=False -> returns a TrackingStatus dict (typed at runtime)
        - If wait_for_ack=False -> returns a success message string on best-effort fire-and-forget
        """
        if wait_for_ack:
            # Ensure listener is running
            if not self.is_active():
                self.start()
                # Allow a brief moment for the ACK socket to bind before sending
                time.sleep(0.05)

            tracked, cmd_id = self._register_command(message)
            try:
                sock = self._create_socket()
                with sock:
                    sock.sendto(tracked.encode("utf-8"), (self.host, int(self.port)))
            except Exception as e:
                logger.error("Failed to send tracked command: %s", e)
                # Mark as failed
                with self._lock:
                    entry = self._history.get(cmd_id)
                    if entry:
                        entry["status"] = "FAILED"
                        entry["details"] = str(e)
                        entry["completed"] = True
                        entry["ack_time"] = datetime.now()
                return cmd_id if non_blocking else self._build_status(cmd_id)

            if non_blocking:
                return cmd_id

            # Blocking wait: return on first ACK (QUEUED/EXECUTING/COMPLETED/FAILED/INVALID/CANCELLED)
            end = time.time() + max(0.0, float(timeout))
            while time.time() < end:
                status = self.check_status(cmd_id)
                if status:
                    st = status.get("status")
                    if st and st != "SENT":
                        return status  # type: ignore[return-value]
                time.sleep(0.01)

            # Timeout path
            with self._lock:
                entry = self._history.get(cmd_id)
                if entry:
                    if not entry.get("completed", False):
                        entry["status"] = "TIMEOUT"
                        entry["details"] = "No acknowledgment received"
                        entry["completed"] = True
                        entry["ack_time"] = datetime.now()
            return self._build_status(cmd_id)
        else:
            # Fire-and-forget path does not use tracker/ids
            try:
                sock = self._create_socket()
                with sock:
                    sock.sendto(message.encode("utf-8"), (self.host, int(self.port)))
                return f"Sent: {message}"
            except Exception as e:
                logger.error("Failed to send command: %s", e)
                return f"Error sending command: {e}"

    def check_status(self, command_id: str) -> Optional[TrackingStatus]:
        """Return TrackingStatus snapshot for the given id, or None."""
        with self._lock:
            entry = self._history.get(command_id)
            if not entry:
                return None
            return self._build_status_nolock(command_id, entry)

    def get_stats(self) -> Dict:
        """Return basic stats and health information."""
        with self._lock:
            return {
                "active": self.is_active(),
                "commands_tracked": len(self._history),
                "thread_alive": self._thread.is_alive() if self._thread else False,
                "ack_port": self.ack_port,
                "host": self.host,
                "port": self.port,
            }

    # ---------- helpers ----------

    def _build_status(self, command_id: str) -> TrackingStatus:
        with self._lock:
            entry = self._history.get(command_id, {})
            return self._build_status_nolock(command_id, entry)

    @staticmethod
    def _build_status_nolock(command_id: str, entry: Dict) -> TrackingStatus:
        return {
            "command_id": command_id,
            "status": entry.get("status", "SENT"),
            "details": entry.get("details", ""),
            "completed": bool(entry.get("completed", False)),
            "ack_time": entry.get("ack_time"),
        }  # type: ignore[return-value]

    # Test helper to simulate ACKs without sockets
    def _inject_ack(self, command_id: str, status: AckStatus = "COMPLETED", details: str = "") -> None:
        self._handle_ack(command_id, status, details)


class LazyCommandTracker:
    """
    Minimal lazy-initialized tracker used by tests and simple clients.

    Behavior:
    - Stores the listen_port passed in constructor
    - Defers any heavy initialization until first use
    - track_command(message) returns (tracked_message, cmd_id), where tracked_message is '<id>|<message>'

    Note: This class does not start sockets/threads; it only generates command IDs for upstream usage.
    For full ACK tracking, use CommandTracker directly.
    """

    def __init__(self, listen_port: int = 5002, host: str = "127.0.0.1", port: int = 5001) -> None:
        self.listen_port = int(listen_port)
        self.host = host
        self.port = int(port)
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            # Placeholder for future initialization (e.g., spawning a CommandTracker)
            self._initialized = True

    def track_command(self, message: str) -> Tuple[str, str]:
        """
        Return a tuple (tracked_message, cmd_id).

        tracked_message format: '<cmd_id>|<message>'
        """
        self._ensure_initialized()
        cmd_id = str(uuid.uuid4())[:8]
        return f"{cmd_id}|{message}", cmd_id
