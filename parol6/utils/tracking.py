"""
Lazy command tracking utilities for PAROL6.

Provides optional UDP acknowledgment tracking with zero overhead when not used.
The tracking system is only initialized when explicitly requested.
"""

import os
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union


def _get_env_int(name: str, default: int) -> int:
    """
    Safe environment variable parsing for integers.
    Returns default for unset or empty string values.
    """
    value = os.getenv(name)
    if not value:  # None or empty string
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {name}='{value}' is not a valid integer")


# Global configuration with environment variable overrides  
SERVER_IP = os.getenv("PAROL6_SERVER_IP", "127.0.0.1")
SERVER_PORT = _get_env_int("PAROL6_SERVER_PORT", 5001)
ACK_PORT = _get_env_int("PAROL6_ACK_PORT", 5002)

# Global tracker - starts as None (no resources)
_command_tracker = None
_tracker_lock = threading.Lock()


def reset_tracking():
    """
    Reset and cleanup the command tracker.
    Useful for tests and cleanup scenarios.
    """
    global _command_tracker, _tracker_lock
    
    with _tracker_lock:
        if _command_tracker:
            _command_tracker._cleanup()
            _command_tracker = None


class LazyCommandTracker:
    """
    Command tracker with lazy initialization.
    Resources are ONLY allocated when tracking is actually used.
    """
    
    def __init__(self, listen_port=None, history_size=100):
        # Use ACK_PORT constant if not specified
        if listen_port is None:
            listen_port = ACK_PORT
        self.listen_port = listen_port
        self.history_size = history_size
        self.command_history = {}
        self.lock = threading.Lock()
        
        # Lazy initialization flags
        self._initialized = False
        self._thread = None
        self._socket = None
        self._running = False
    
    def _lazy_init(self):
        """
        Initialize resources only when first tracking is requested.
        This is called ONLY when someone uses tracking features.
        """
        if self._initialized:
            return True
            
        try:
            print("[Tracker] First tracking request - initializing resources...")
            
            # Socket initialization
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind(('', self.listen_port))
            self._socket.settimeout(0.1)
            
            # Start thread
            self._running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            
            self._initialized = True
            print(f"[Tracker] Initialized on port {self.listen_port}")
            return True
            
        except Exception as e:
            print(f"[Tracker] Failed to initialize: {e}")
            self._cleanup()
            return False
    
    def _cleanup(self):
        """Clean up resources"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._socket:
            self._socket.close()
            self._socket = None
        self._initialized = False
    
    def _listen_loop(self):
        """Listener thread - only runs if tracking is used"""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(2048)
                message = data.decode('utf-8')
                
                parts = message.split('|', 3)
                if parts[0] == 'ACK' and len(parts) >= 3:
                    cmd_id = parts[1]
                    status = parts[2]
                    details = parts[3] if len(parts) > 3 else ""
                    
                    with self.lock:
                        if cmd_id in self.command_history:
                            self.command_history[cmd_id].update({
                                'status': status,
                                'details': details,
                                'ack_time': datetime.now(),
                                'completed': status in ['COMPLETED', 'FAILED', 'INVALID', 'CANCELLED']
                            })
                    
                    # Clean old entries (only if we have many)
                    if len(self.command_history) > self.history_size:
                        self._cleanup_old_entries()
                        
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    pass  # Silently continue
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory growth"""
        with self.lock:
            now = datetime.now()
            expired = [cmd_id for cmd_id, info in self.command_history.items()
                      if now - info['sent_time'] > timedelta(seconds=30)]
            for cmd_id in expired:
                del self.command_history[cmd_id]
    
    def track_command(self, command: str) -> Tuple[str, Optional[str]]:
        """
        Track a command - initializes tracker if needed.
        Returns (modified_command, cmd_id)
        """
        # Initialize on first use
        if not self._initialized:
            if not self._lazy_init():
                # Initialization failed - fall back to non-tracking
                return command, None
        
        # Generate ID and modify command
        cmd_id = str(uuid.uuid4())[:8]
        tracked_command = f"{cmd_id}|{command}"
        
        # Register in history
        with self.lock:
            self.command_history[cmd_id] = {
                'command': command,
                'sent_time': datetime.now(),
                'status': 'SENT',
                'details': '',
                'completed': False
            }
        
        return tracked_command, cmd_id
    
    def get_status(self, cmd_id: str) -> Optional[Dict]:
        """Get status if tracker is initialized"""
        if not self._initialized:
            return None
        with self.lock:
            return self.command_history.get(cmd_id, None)
    
    def wait_for_completion(self, cmd_id: str, timeout: float = 5.0) -> Dict:
        """Wait for completion if tracker is initialized"""
        if not self._initialized:
            return {'status': 'NO_TRACKING', 'details': 'Tracker not initialized', 'completed': True}
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(cmd_id)
            if status and status['completed']:
                return status
            time.sleep(0.01)
        
        return self.get_status(cmd_id) or {
            'status': 'TIMEOUT',
            'details': 'No acknowledgment received',
            'completed': True
        }
    
    def is_active(self) -> bool:
        """Check if tracker is initialized and running"""
        return self._initialized and self._running


def _get_tracker_if_needed() -> Optional[LazyCommandTracker]:
    """
    Get tracker ONLY if tracking is requested.
    This ensures zero overhead for non-tracking operations.
    """
    global _command_tracker, _tracker_lock
    
    # Check if tracking is explicitly disabled
    disable_tracking = os.getenv("PAROL6_DISABLE_TRACKING", "").lower() in ("1", "true", "yes", "on")
    if disable_tracking:
        return None
    
    # Fast path - tracker already exists
    if _command_tracker is not None:
        return _command_tracker
    
    # Slow path - create tracker (only happens once)
    with _tracker_lock:
        if _command_tracker is None:
            _command_tracker = LazyCommandTracker()
        return _command_tracker


def send_robot_command_tracked(command_string: str) -> Tuple[str, Optional[str]]:
    """
    Send with tracking - initializes tracker on first use.
    
    Resource impact:
    - First call: Starts tracker thread
    - Subsequent calls: Minimal overhead (UUID generation)
    """
    tracker = _get_tracker_if_needed()
    if tracker:
        tracked_cmd, cmd_id = tracker.track_command(command_string)
        if cmd_id:
            # Send tracked command
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(tracked_cmd.encode('utf-8'), (SERVER_IP, SERVER_PORT))
                return f"Command sent with tracking (ID: {cmd_id})", cmd_id
            except Exception as e:
                return f"Error: {e}", None
    
    # Fall back to non-tracked
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(command_string.encode('utf-8'), (SERVER_IP, SERVER_PORT))
        return f"Successfully sent command: '{command_string[:50]}...'", None
    except Exception as e:
        return f"Error sending command: {e}", None


def send_and_wait(
    command_string: str, 
    timeout: float = 2.0, 
    non_blocking: bool = False
) -> Union[Dict, str, None]:
    """
    Send and wait for acknowledgment OR return a command_id immediately.
    First use initializes tracker.
    """
    result, cmd_id = send_robot_command_tracked(command_string)
    
    if cmd_id:
        # If non_blocking is True, return the ID right away
        if non_blocking:
            return cmd_id
            
        # Otherwise, proceed with the original blocking logic
        tracker = _get_tracker_if_needed()
        if tracker:
            status_dict = tracker.wait_for_completion(cmd_id, timeout)
            # Add the command_id to the returned dictionary
            status_dict['command_id'] = cmd_id
            return status_dict
    
    # Fallback for tracking failures
    if non_blocking:
        return None
    else:
        return {'status': 'NO_TRACKING', 'details': result, 'completed': True, 'command_id': None}


def check_command_status(command_id: str) -> Optional[Dict]:
    """
    Check status - returns None if tracker not initialized.
    Does NOT initialize tracker (read-only).
    """
    if _command_tracker and _command_tracker.is_active():
        return _command_tracker.get_status(command_id)
    return None


def is_tracking_active() -> bool:
    """
    Check if tracking is active.
    Returns False if never used (zero overhead check).
    """
    return _command_tracker is not None and _command_tracker.is_active()


def get_tracking_stats() -> Dict:
    """
    Get resource usage statistics.
    """
    if _command_tracker and _command_tracker.is_active():
        with _command_tracker.lock:
            return {
                'active': True,
                'commands_tracked': len(_command_tracker.command_history),
                'memory_bytes': len(str(_command_tracker.command_history)),
                'thread_active': _command_tracker._thread.is_alive() if _command_tracker._thread else False
            }
    else:
        return {
            'active': False,
            'commands_tracked': 0,
            'memory_bytes': 0,
            'thread_active': False
        }
