"""
Pytest configuration and shared fixtures for PAROL6 Python API tests.

Provides command line options, fixtures for port management, server process management,
environment configuration, and test utilities used across the test suite.
"""

import logging
import os
from collections.abc import Generator
from dataclasses import dataclass

import pytest
from parol6 import Robot


# Import utilities for port detection
def find_available_ports(start_port: int = 5001, count: int = 2) -> list[int]:
    """Simple fallback port finder if utils import fails."""
    import socket

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
        "--server-ip",
        action="store",
        default="127.0.0.1",
        help="IP address for test server communication",
    )
    parser.addoption(
        "--server-port",
        action="store",
        type=int,
        default=None,
        help="Port for robot server communication (auto-detected if not specified)",
    )
    parser.addoption(
        "--ack-port",
        action="store",
        type=int,
        default=None,
        help="Port for acknowledgment communication (auto-detected if not specified)",
    )
    parser.addoption(
        "--keep-server-running",
        action="store_true",
        default=False,
        help="Keep the test server running between test sessions for debugging",
    )
    parser.addoption(
        "--examples",
        action="store_true",
        default=False,
        help="Run example script tests (binds port 5001, can't run alongside server tests)",
    )


def pytest_collection_modifyitems(config, items):
    """With --examples: run only example tests. Without: skip them."""
    if config.getoption("--examples"):
        deselected = [item for item in items if "examples" not in item.keywords]
        items[:] = [item for item in items if "examples" in item.keywords]
        config.hook.pytest_deselected(items=deselected)
        return
    skip_examples = pytest.mark.skip(reason="needs --examples to run")
    for item in items:
        if "examples" in item.keywords:
            item.add_marker(skip_examples)


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

    return TestPorts(server_ip=server_ip, server_port=server_port, ack_port=ack_port)


# ============================================================================
# ENVIRONMENT CONFIGURATION FIXTURE
# ============================================================================


@pytest.fixture(scope="session")
def robot_api_env(ports: TestPorts) -> Generator[dict[str, str], None, None]:
    """
    Configure environment variables for robot_api client to use test ports.

    Sets environment variables so that robot_api.py will connect to the test
    server instead of the default production server.
    """
    # Store original environment values
    original_env = {}
    env_vars = {
        "PAROL6_CONTROLLER_IP": ports.server_ip,
        "PAROL6_CONTROLLER_PORT": str(ports.server_port),
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
def server_proc(request, ports: TestPorts, robot_api_env):
    """Launch parol6 server for integration tests.

    Starts the server with FAKE_SERIAL mode and waits for readiness.
    Automatically cleans up the server when tests complete.
    """
    keep_running = request.config.getoption("--keep-server-running")

    robot = Robot(host=ports.server_ip, port=ports.server_port, timeout=60.0)

    logger.info(f"Starting test server on {ports.server_ip}:{ports.server_port}")
    robot.start(
        extra_env={
            "PAROL6_FAKE_SERIAL": "1",
            "PAROL6_NOAUTOHOME": "1",
            "PAROL6_CONTROLLER_IP": ports.server_ip,
            "PAROL6_CONTROLLER_PORT": str(ports.server_port),
        },
    )

    try:
        yield robot
    finally:
        if not keep_running:
            logger.info("Stopping test server")
            robot.stop()
        else:
            logger.info("Leaving test server running (--keep-server-running)")


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
    monkeypatch.setattr(time, "time", mock.time)
    monkeypatch.setattr(time, "sleep", mock.sleep)
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
        "markers",
        "integration: Integration tests that test component interactions with FAKE_SERIAL",
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests that exercise complete workflows"
    )


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    logger.info("Starting PAROL6 Python API test session")

    # Print test configuration info
    config = session.config
    logger.info(f"Server IP: {config.getoption('--server-ip')}")

    server_port = config.getoption("--server-port")
    ack_port = config.getoption("--ack-port")
    if server_port and ack_port:
        logger.info(f"Server ports: {server_port}/{ack_port}")
    else:
        logger.info("Server ports: auto-detect")


# ============================================================================
# CLIENT FIXTURE
# ============================================================================


@pytest.fixture
def client(ports: TestPorts):
    """
    Provide a RobotClient configured for the test ports.

    This ensures all tests use the same connection configuration
    and connect to the auto-detected test server ports.
    """
    from parol6 import RobotClient

    return RobotClient(
        host=ports.server_ip,
        port=ports.server_port,
    )


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    logger.info(
        f"PAROL6 Python API test session finished with exit status: {exitstatus}"
    )
