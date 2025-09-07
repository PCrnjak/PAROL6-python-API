"""
Pytest configuration and shared fixtures for PAROL6 Python API tests.

Provides command line options, fixtures for port management, server process management,
environment configuration, and test utilities used across the test suite.
"""

import os
import sys
import pytest
import time
from typing import Generator, Dict, Optional
from dataclasses import dataclass
import logging

# Add the parent directory to Python path so we can import the API modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from tests.utils import HeadlessCommanderProc, find_available_ports
except ImportError:
    # Fallback for when utils not available
    HeadlessCommanderProc = None
    find_available_ports = None

logger = logging.getLogger(__name__)


@dataclass
class TestPorts:
    """Configuration for test server ports."""
    server_ip: str = "127.0.0.1"
    server_port: int = 5001  
    ack_port: int = 5002


# ============================================================================
# PYTEST COMMAND LINE OPTIONS
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options for the test suite."""
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Enable hardware tests that require actual robot hardware and human confirmation"
    )
    parser.addoption(
        "--server-ip",
        action="store",
        default="127.0.0.1",
        help="IP address for test server communication"
    )
    parser.addoption(
        "--server-port", 
        action="store",
        type=int,
        default=None,
        help="Port for robot server communication (auto-detected if not specified)"
    )
    parser.addoption(
        "--ack-port",
        action="store", 
        type=int,
        default=None,
        help="Port for acknowledgment communication (auto-detected if not specified)"
    )
    parser.addoption(
        "--keep-server-running",
        action="store_true",
        default=False,
        help="Keep the test server running between test sessions for debugging"
    )


# ============================================================================
# PORT MANAGEMENT FIXTURE
# ============================================================================

@pytest.fixture(scope="session")
def ports(request) -> TestPorts:
    """
    Provide test port configuration.
    
    Automatically finds available ports if not specified via command line.
    Ensures ports don't conflict with existing services.
    """
    server_ip = request.config.getoption("--server-ip")
    server_port = request.config.getoption("--server-port")
    ack_port = request.config.getoption("--ack-port")
    
    # Auto-detect available ports if not specified
    if server_port is None or ack_port is None:
        logger.info("Auto-detecting available ports...")
        available_ports = find_available_ports(start_port=5001, count=2)
        
        if len(available_ports) < 2:
            pytest.fail("Could not find 2 consecutive available ports for testing")
            
        if server_port is None:
            server_port = available_ports[0]
        if ack_port is None:
            ack_port = available_ports[1]
            
        logger.info(f"Using auto-detected ports: server={server_port}, ack={ack_port}")
    
    return TestPorts(
        server_ip=server_ip,
        server_port=server_port,
        ack_port=ack_port
    )


# ============================================================================
# ENVIRONMENT CONFIGURATION FIXTURE  
# ============================================================================

@pytest.fixture(scope="session")
def robot_api_env(ports: TestPorts) -> Generator[Dict[str, str], None, None]:
    """
    Configure environment variables for robot_api client to use test ports.
    
    Sets environment variables so that robot_api.py will connect to the test
    server instead of the default production server.
    """
    # Store original environment values
    original_env = {}
    env_vars = {
        "PAROL6_SERVER_IP": ports.server_ip,
        "PAROL6_SERVER_PORT": str(ports.server_port),
        "PAROL6_ACK_PORT": str(ports.ack_port),
    }
    
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
        
    logger.debug(f"Set test environment: {env_vars}")
    
    try:
        yield env_vars
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        logger.debug("Restored original environment")


# ============================================================================
# SERVER PROCESS FIXTURE
# ============================================================================

@pytest.fixture(scope="session")
def server_proc(request, ports: TestPorts, robot_api_env) -> Generator[HeadlessCommanderProc, None, None]:
    """
    Launch headless_commander.py server process for integration tests.
    
    Starts the server with FAKE_SERIAL mode and waits for readiness.
    Automatically cleans up the process when tests complete.
    """
    keep_running = request.config.getoption("--keep-server-running")
    
    # Create process manager
    proc = HeadlessCommanderProc(
        ip=ports.server_ip,
        port=ports.server_port,
        ack_port=ports.ack_port,
        env={
            "PAROL6_FAKE_SERIAL": "1",
            "PAROL6_NOAUTOHOME": "1",
        }
    )
    
    # Start the server process
    logger.info(f"Starting test server on {ports.server_ip}:{ports.server_port}")
    if not proc.start(log_level="WARNING", timeout=15.0):
        pytest.fail("Failed to start headless commander server for testing")
    
    try:
        # Wait a bit for full initialization
        time.sleep(1.0)
        yield proc
        
    finally:
        if not keep_running:
            logger.info("Stopping test server")
            proc.stop()
        else:
            logger.info("Leaving test server running (--keep-server-running)")


# ============================================================================
# HARDWARE TEST SUPPORT
# ============================================================================

@pytest.fixture
def human_prompt(request):
    """
    Provide human confirmation prompts for hardware tests.
    
    Automatically skips tests marked with @pytest.mark.hardware unless
    --run-hardware is specified. For enabled hardware tests, provides
    a utility function to prompt for human confirmation.
    """
    # Check if hardware tests are enabled
    run_hardware = request.config.getoption("--run-hardware")
    
    # Skip hardware tests if not enabled
    if request.node.get_closest_marker("hardware") and not run_hardware:
        pytest.skip("Hardware tests disabled. Use --run-hardware to enable.")
    
    def prompt_user(message: str, timeout: Optional[float] = None) -> bool:
        """
        Prompt user for confirmation during hardware tests.
        
        Args:
            message: Message to display to user
            timeout: Optional timeout in seconds
            
        Returns:
            True if user confirms, False otherwise
        """
        if not run_hardware:
            return False
            
        print(f"\n{'='*60}")
        print("HARDWARE TEST CONFIRMATION REQUIRED")
        print(f"{'='*60}")
        print(f"{message}")
        print(f"{'='*60}")
        
        try:
            if timeout:
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("User confirmation timeout")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            
            response = input("Continue? [y/N]: ").strip().lower()
            
            if timeout:
                signal.alarm(0)  # Cancel timeout
                
            return response in ['y', 'yes']
            
        except (KeyboardInterrupt, TimeoutError):
            print("\nUser confirmation cancelled or timed out")
            return False
        except Exception as e:
            print(f"\nError getting user confirmation: {e}")
            return False
    
    return prompt_user


# ============================================================================
# COMMON TEST UTILITIES
# ============================================================================

@pytest.fixture
def temp_env():
    """
    Provide temporary environment variable context manager.
    
    Useful for tests that need to modify environment variables temporarily.
    """
    class TempEnv:
        def __init__(self):
            self.original = {}
            
        def set(self, key: str, value: str):
            """Set an environment variable temporarily."""
            if key not in self.original:
                self.original[key] = os.environ.get(key)
            os.environ[key] = value
            
        def restore(self):
            """Restore all modified environment variables."""
            for key, original_value in self.original.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            self.original.clear()
    
    temp = TempEnv()
    try:
        yield temp
    finally:
        temp.restore()


@pytest.fixture
def mock_time(monkeypatch):
    """
    Provide controllable time mocking for tests that depend on timing.
    
    Useful for testing timeout behavior without actually waiting.
    """
    import time
    
    class MockTime:
        def __init__(self):
            self.current_time = time.time()
            
        def time(self):
            return self.current_time
            
        def advance(self, seconds: float):
            """Advance the mock time by the specified number of seconds."""
            self.current_time += seconds
            
        def sleep(self, seconds: float):
            """Mock sleep that just advances time."""
            self.advance(seconds)
    
    mock = MockTime()
    monkeypatch.setattr(time, 'time', mock.time)
    monkeypatch.setattr(time, 'sleep', mock.sleep)
    return mock


# ============================================================================
# PYTEST CONFIGURATION HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components in isolation"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions with FAKE_SERIAL"
    )
    config.addinivalue_line(
        "markers", "hardware: Hardware tests that require actual robot hardware and human confirmation"  
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (typically hardware or complex integration tests)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests that exercise complete workflows"
    )
    config.addinivalue_line(
        "markers", "gcode: Tests specifically for GCODE parsing and interpretation functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    # Skip hardware tests by default unless --run-hardware is specified
    if not config.getoption("--run-hardware"):
        skip_hardware = pytest.mark.skip(reason="Hardware tests disabled (use --run-hardware to enable)")
        for item in items:
            if item.get_closest_marker("hardware"):
                item.add_marker(skip_hardware)


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    logger.info("Starting PAROL6 Python API test session")
    
    # Print test configuration info
    config = session.config
    logger.info(f"Hardware tests: {'enabled' if config.getoption('--run-hardware') else 'disabled'}")
    logger.info(f"Server IP: {config.getoption('--server-ip')}")
    
    server_port = config.getoption('--server-port')
    ack_port = config.getoption('--ack-port')
    if server_port and ack_port:
        logger.info(f"Server ports: {server_port}/{ack_port}")
    else:
        logger.info("Server ports: auto-detect")


def wait_until_stopped(timeout=90.0, settle_window=1.0, poll_interval=0.2, 
                       speed_threshold=2.0, angle_threshold=0.5, show_progress=True):
    """
    Wait for robot to stop moving with responsive polling instead of blind sleeps.
    
    Args:
        timeout: Maximum time to wait in seconds
        settle_window: How long robot must be stable to be considered stopped
        poll_interval: How often to check status
        speed_threshold: Max joint speed to be considered stopped (steps/sec)
        angle_threshold: Max angle change to be considered stopped (degrees)
        show_progress: Print dots to show progress
        
    Returns:
        True if robot stopped, False if timeout
    """
    import time
    
    start_time = time.time()
    last_angles = None
    settle_start = None
    last_progress = 0
    
    if show_progress:
        print("Waiting for robot to stop...", end="", flush=True)
    
    while time.time() - start_time < timeout:
        # Try speed-based detection first (preferred)
        try:
            import robot_api
            speeds = robot_api.get_robot_joint_speeds()
            if speeds:
                max_speed = max(abs(s) for s in speeds)
                if max_speed < speed_threshold:
                    if settle_start is None:
                        settle_start = time.time()
                    elif time.time() - settle_start > settle_window:
                        if show_progress:
                            print(" stopped via speeds!")
                        return True
                else:
                    settle_start = None
            else:
                # Fallback to angle-based detection
                angles = robot_api.get_robot_joint_angles()
                if angles:
                    if last_angles is not None:
                        max_change = max(abs(a - b) for a, b in zip(angles, last_angles))
                        if max_change < angle_threshold:
                            if settle_start is None:
                                settle_start = time.time()
                            elif time.time() - settle_start > settle_window:
                                if show_progress:
                                    print(" stopped via angle delta!")
                                return True
                        else:
                            settle_start = None
                    last_angles = angles
        except Exception as e:
            if show_progress:
                print(f" error: {e}")
            pass
            
        # Show progress dots every few seconds
        if show_progress and int(time.time() - start_time) > last_progress:
            print(".", end="", flush=True)
            last_progress = int(time.time() - start_time)
            
        time.sleep(poll_interval)
    
    if show_progress:
        print(" timeout!")
    return False


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    logger.info(f"PAROL6 Python API test session finished with exit status: {exitstatus}")
