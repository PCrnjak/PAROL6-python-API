"""
Unit tests for MockSerialTransport.

Tests the mock serial transport implementation to ensure:
1. Mock transport can be created and connected
2. Mock transport simulates robot responses correctly
3. Transport factory correctly selects mock when PAROL6_FAKE_SERIAL is set
4. Mock transport is compatible with SerialTransport interface
"""

import os
import threading
import time

import numpy as np
import parol6.PAROL6_ROBOT as PAROL6_ROBOT
import pytest
from parol6.config import HOME_ANGLES_DEG
from parol6.protocol.wire import CommandCode, unpack_rx_frame_into
from parol6.server.transports import create_transport, is_simulation_mode
from parol6.server.transports.mock_serial_transport import MockRobotState, MockSerialTransport


def _wait_for_latest_frame_and_decode(transport: MockSerialTransport, timeout_s: float = 0.5):
    """
    Helper: wait for a latest frame publication and decode into numpy arrays.
    Returns dict-like with arrays or None on timeout.
    """
    start = time.time()
    last_ver = -1
    # Ensure reader running
    shutdown = threading.Event()
    transport.start_reader(shutdown)

    pos = np.zeros(6, dtype=np.int32)
    spd = np.zeros(6, dtype=np.int32)
    homed = np.zeros(8, dtype=np.uint8)
    io_bits = np.zeros(8, dtype=np.uint8)
    temp = np.zeros(8, dtype=np.uint8)
    poserr = np.zeros(8, dtype=np.uint8)
    timing = np.zeros(1, dtype=np.int32)
    grip = np.zeros(6, dtype=np.int32)

    while time.time() - start < timeout_s:
        mv, ver, ts = transport.get_latest_frame_view()
        if mv is not None and ver != last_ver:
            ok = unpack_rx_frame_into(
                mv,
                pos_out=pos,
                spd_out=spd,
                homed_out=homed,
                io_out=io_bits,
                temp_out=temp,
                poserr_out=poserr,
                timing_out=timing,
                grip_out=grip,
            )
            if ok:
                return {
                    "pos": pos.copy(),
                    "spd": spd.copy(),
                    "homed": homed.copy(),
                    "io": io_bits.copy(),
                    "temp": temp.copy(),
                    "poserr": poserr.copy(),
                    "timing": timing.copy(),
                    "grip": grip.copy(),
                    "ver": ver,
                    "ts": ts,
                }
        time.sleep(0.005)
    return None


class TestMockSerialTransport:
    """Test MockSerialTransport functionality."""

    def test_create_and_connect(self):
        """Test that MockSerialTransport can be created and connected."""
        transport = MockSerialTransport()
        assert transport is not None
        assert not transport.is_connected()

        # Connect should always succeed for mock
        assert transport.connect()
        assert transport.is_connected()

        # Disconnect
        transport.disconnect()
        assert not transport.is_connected()

    def test_auto_reconnect(self):
        """Test auto-reconnect functionality."""
        transport = MockSerialTransport()

        # Auto-reconnect should succeed when not connected
        assert transport.auto_reconnect()
        assert transport.is_connected()

        # Auto-reconnect should return False when already connected
        assert not transport.auto_reconnect()

    def test_write_frame(self):
        """Test writing command frames."""
        transport = MockSerialTransport()
        transport.connect()

        # Prepare command data
        position_out = [0, 0, 0, 0, 0, 0]
        speed_out = [100, 100, 100, 100, 100, 100]
        command_out = CommandCode.JOG
        affected_joint = [1, 1, 1, 1, 1, 1, 0, 0]
        inout = [0, 0, 0, 0, 0, 0, 0, 0]
        timeout = 0
        gripper_data = [0, 0, 0, 0, 0, 0]

        # Write should succeed when connected
        success = transport.write_frame(
            position_out, speed_out, command_out, affected_joint, inout, timeout, gripper_data
        )
        assert success

        # Verify internal state updated
        assert transport._state.command_out == command_out
        assert np.array_equal(transport._state.position_out, position_out)
        assert np.array_equal(transport._state.speed_out, speed_out)

        # Disconnect and try again - should fail
        transport.disconnect()
        success = transport.write_frame(
            position_out, speed_out, command_out, affected_joint, inout, timeout, gripper_data
        )
        assert not success

    def test_read_frames(self):
        """
        Test reading response frames using latest-frame API (no legacy queues).
        """
        transport = MockSerialTransport()
        transport.connect()

        decoded = _wait_for_latest_frame_and_decode(transport, timeout_s=0.5)
        assert decoded is not None, "No frame published by mock transport"

        # Check data shapes
        assert decoded["pos"].shape == (6,)
        assert decoded["spd"].shape == (6,)
        assert decoded["homed"].shape == (8,)
        assert decoded["io"].shape == (8,)
        assert decoded["temp"].shape == (8,)
        assert decoded["poserr"].shape == (8,)
        assert decoded["timing"].shape == (1,)
        assert decoded["grip"].shape == (6,)

        # E-stop should be released in simulation (io bit 4)
        assert int(decoded["io"][4]) == 1

    def test_motion_simulation_jog(self):
        """Test JOG command simulation via latest-frame API."""
        transport = MockSerialTransport()
        transport.connect()

        # Baseline
        baseline = _wait_for_latest_frame_and_decode(transport, timeout_s=0.5)
        assert baseline is not None
        initial_pos = int(baseline["pos"][0])

        # Send JOG command to move joint 1
        speed_out = [1000, 0, 0, 0, 0, 0]
        assert transport.write_frame(
            [0] * 6, speed_out, CommandCode.JOG, [1] * 8, [0] * 8, 0, [0] * 6
        )

        # Wait for movement
        moved = None
        t0 = time.time()
        while time.time() - t0 < 0.5:
            decoded = _wait_for_latest_frame_and_decode(transport, timeout_s=0.1)
            if decoded is None:
                continue
            if int(decoded["pos"][0]) != initial_pos:
                moved = decoded
                break
        assert moved is not None, "Joint didn't move during JOG"

    def test_motion_simulation_move(self):
        """Test MOVE command simulation via latest-frame API."""
        transport = MockSerialTransport()
        transport.connect()

        target_pos = [5000, 0, 0, 0, 0, 0]
        assert transport.write_frame(
            target_pos, [0] * 6, CommandCode.MOVE, [1] * 8, [0] * 8, 0, [0] * 6
        )

        # Poll until position moves toward target or timeout
        final = None
        t0 = time.time()
        while time.time() - t0 < 1.0:
            decoded = _wait_for_latest_frame_and_decode(transport, timeout_s=0.1)
            if decoded is None:
                continue
            current_pos = int(decoded["pos"][0])
            if abs(current_pos - target_pos[0]) < 2000:
                final = decoded
                break
        assert final is not None, "Didn't move toward target sufficiently"

    def test_homing_simulation(self):
        """Test HOME command simulation via latest-frame API."""
        transport = MockSerialTransport()
        transport.connect()

        # Expected home positions (steps) derived from config HOME_ANGLES_DEG
        expected_steps = [
            int(PAROL6_ROBOT.ops.deg_to_steps(float(deg), i))
            for i, deg in enumerate(HOME_ANGLES_DEG)
        ]
        tol_steps = 500  # tolerance in steps

        # Send HOME command
        assert transport.write_frame(
            [0] * 6, [0] * 6, CommandCode.HOME, [1] * 8, [0] * 8, 0, [0] * 6
        )

        homing_started = False
        homing_completed = False
        t0 = time.time()
        last_homed_bits = None

        while time.time() - t0 < 1.0:
            decoded = _wait_for_latest_frame_and_decode(transport, timeout_s=0.1)
            if decoded is None:
                continue
            homed_bits = decoded["homed"].tolist()
            if not all(h == 1 for h in homed_bits):
                homing_started = True
            if homing_started and all(h == 1 for h in homed_bits):
                # Verify positions near configured home posture
                pos_list = decoded["pos"].tolist()
                if all(abs(int(pos_list[i]) - expected_steps[i]) < tol_steps for i in range(6)):
                    homing_completed = True
                    break
            last_homed_bits = homed_bits

        # Either already homed or homed sequence executed
        if not homing_started:
            assert last_homed_bits is not None
            assert all(h == 1 for h in last_homed_bits), "Robot should be homed"
        else:
            assert homing_completed, "Homing sequence started but did not complete"

    def test_gripper_simulation(self):
        """Test gripper command simulation."""
        transport = MockSerialTransport()
        transport.connect()

        # Test calibration mode
        gripper_data = [100, 150, 500, 0, 1, 42]  # mode=1 for calibration, id=42
        transport.write_frame([0] * 6, [0] * 6, CommandCode.IDLE, [0] * 8, [0] * 8, 0, gripper_data)

        # Check gripper state updated
        assert transport._state.gripper_data_in[0] == 42  # Device ID set
        assert transport._state.gripper_data_in[4] & 0x40 != 0  # Calibrated bit set

        # Test error clear mode
        gripper_data[4] = 2  # mode=2 for error clear
        transport.write_frame([0] * 6, [0] * 6, CommandCode.IDLE, [0] * 8, [0] * 8, 0, gripper_data)

        # Error bit should be cleared
        assert transport._state.gripper_data_in[4] & 0x20 == 0

    def test_get_info(self):
        """Test get_info method."""
        transport = MockSerialTransport(port="TEST_PORT", baudrate=115200)

        info = transport.get_info()
        assert info["port"] == "TEST_PORT"
        assert info["baudrate"] == 115200
        assert not info["connected"]
        assert info["mode"] == "MOCK_SERIAL"

        transport.connect()
        info = transport.get_info()
        assert info["connected"]
        assert "frames_sent" in info
        assert "frames_received" in info
        assert "simulation_rate_hz" in info
        assert "robot_state" in info


class TestTransportFactory:
    """Test transport factory with mock mode."""

    def test_simulation_mode_detection(self):
        """Test is_simulation_mode function."""
        # Should be False by default
        if "PAROL6_FAKE_SERIAL" in os.environ:
            del os.environ["PAROL6_FAKE_SERIAL"]
        assert not is_simulation_mode()

        # Test various true values
        for value in ["1", "true", "TRUE", "yes", "YES", "on", "ON"]:
            os.environ["PAROL6_FAKE_SERIAL"] = value
            assert is_simulation_mode()

        # Test false values
        for value in ["0", "false", "FALSE", "no", "NO", "off", "OFF", ""]:
            os.environ["PAROL6_FAKE_SERIAL"] = value
            assert not is_simulation_mode()

        # Clean up
        del os.environ["PAROL6_FAKE_SERIAL"]

    def test_create_transport_auto_detect(self):
        """Test transport factory auto-detection."""
        # Without FAKE_SERIAL, should create SerialTransport
        if "PAROL6_FAKE_SERIAL" in os.environ:
            del os.environ["PAROL6_FAKE_SERIAL"]

        from parol6.server.transports.serial_transport import SerialTransport

        transport = create_transport()
        assert isinstance(transport, SerialTransport)

        # With FAKE_SERIAL, should create MockSerialTransport
        os.environ["PAROL6_FAKE_SERIAL"] = "1"
        transport = create_transport()
        assert isinstance(transport, MockSerialTransport)

        # Clean up
        del os.environ["PAROL6_FAKE_SERIAL"]

    def test_create_transport_explicit(self):
        """Test explicit transport type selection."""
        # Explicit mock regardless of environment
        transport = create_transport(transport_type="mock")
        assert isinstance(transport, MockSerialTransport)

        # Explicit serial regardless of environment
        from parol6.server.transports.serial_transport import SerialTransport

        os.environ["PAROL6_FAKE_SERIAL"] = "1"
        transport = create_transport(transport_type="serial")
        assert isinstance(transport, SerialTransport)

        # Invalid type should raise
        with pytest.raises(ValueError):
            create_transport(transport_type="invalid")

        # Clean up
        if "PAROL6_FAKE_SERIAL" in os.environ:
            del os.environ["PAROL6_FAKE_SERIAL"]


class TestMockRobotState:
    """Test MockRobotState initialization."""

    def test_initial_state(self):
        """Test initial robot state."""
        state = MockRobotState()

        # Should start at standby position
        for i in range(6):
            deg = float(PAROL6_ROBOT.joint.standby.deg[i])
            steps = int(PAROL6_ROBOT.ops.deg_to_steps(deg, i))
            assert state.position_in[i] == steps

        # E-stop should be released
        assert state.io_in[4] == 1

        # No errors initially
        assert all(e == 0 for e in state.temperature_error_in)
        assert all(e == 0 for e in state.position_error_in)
