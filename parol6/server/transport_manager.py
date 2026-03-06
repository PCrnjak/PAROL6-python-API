"""Transport lifecycle management for serial and mock transports."""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from parol6.config import get_com_port_with_fallback
from parol6.server.transports import create_and_connect_transport, is_simulation_mode
from parol6.server.transports.mock_serial_transport import MockSerialTransport
from parol6.server.transports.serial_transport import SerialTransport

logger = logging.getLogger("parol6.server.transport_manager")


@njit(cache=True)
def _arrays_changed(
    pos: np.ndarray,
    pos_last: np.ndarray,
    spd: np.ndarray,
    spd_last: np.ndarray,
    aff: np.ndarray,
    aff_last: np.ndarray,
    io: np.ndarray,
    io_last: np.ndarray,
    grip: np.ndarray,
    grip_last: np.ndarray,
) -> bool:
    """Check if any TX array has changed. Returns True on first difference (early exit)."""
    for i in range(len(pos)):
        if pos[i] != pos_last[i]:
            return True
    for i in range(len(spd)):
        if spd[i] != spd_last[i]:
            return True
    for i in range(len(aff)):
        if aff[i] != aff_last[i]:
            return True
    for i in range(len(io)):
        if io[i] != io_last[i]:
            return True
    for i in range(len(grip)):
        if grip[i] != grip_last[i]:
            return True
    return False


@dataclass(slots=True)
class TxCoalesceState:
    """State for TX frame coalescing to avoid redundant writes.

    NOTE: Field types must match ControllerState output arrays and the dirty check
    in write_frame_coalesced(). If TX frame format changes, update both places.
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    spd: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    aff: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.uint8))
    io: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.uint8))
    grip: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    cmd: int = 0
    tout: int = 0
    last_sent: float = 0.0


class TransportManager:
    """Manages serial transport lifecycle (connect, disconnect, reconnect, switching).

    Encapsulates transport creation, connection management, and switching between
    real serial and simulator transports.
    """

    def __init__(
        self,
        shutdown_event: threading.Event,
        serial_port: str | None = None,
        serial_baudrate: int = 3000000,
    ):
        """Initialize the transport manager.

        Args:
            shutdown_event: Event to signal shutdown to transport threads.
            serial_port: Initial serial port (or None for auto-detect).
            serial_baudrate: Serial baud rate.
        """
        self._shutdown_event = shutdown_event
        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate

        self.transport: SerialTransport | MockSerialTransport | None = None
        self.first_frame_received = False
        self._last_version = 0
        self._last_tx = TxCoalesceState()

    def initialize(self) -> bool:
        """Create and connect initial transport.

        Returns:
            True if transport was created (may not be connected yet).
        """
        # Load persisted COM port if not provided
        try:
            if not self.serial_port:
                persisted = get_com_port_with_fallback()
                if persisted:
                    self.serial_port = persisted
                    logger.info("Using persisted serial port: %s", persisted)
        except Exception as e:
            logger.debug("Failed to load persisted COM port: %s", e)

        # Create transport
        if self.serial_port or is_simulation_mode():
            self.transport = create_and_connect_transport(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                auto_find_port=True,
            )

            if self.transport:
                if self.transport.is_connected():
                    logger.info("Connected to transport: %s", self.transport.port)
                else:
                    logger.warning(
                        "Serial transport not connected at startup (port=%s)",
                        self.serial_port,
                    )
                return True
        else:
            logger.warning(
                "No serial port configured. Waiting for SET_PORT via UDP or set --serial/PAROL6_COM_PORT/com_port.txt before connecting."
            )

        return False

    def is_connected(self) -> bool:
        """Check if transport is connected."""
        return self.transport is not None and self.transport.is_connected()

    def poll_serial(self) -> bool:
        """Poll serial for new data. Called each tick."""
        if not self.transport or not self.transport.is_connected():
            return False
        return self.transport.poll_read()

    def auto_reconnect(self) -> bool:
        """Attempt reconnection if disconnected.

        Returns:
            True if reconnection was successful.
        """
        if not self.transport or self.transport.is_connected():
            return False

        if self.transport.auto_reconnect():
            self.first_frame_received = False
            self._reset_tx_keepalive()
            logger.info("Serial reconnected")
            return True

        return False

    def switch_to_port(self, port: str) -> bool:
        """Switch to a new serial port (handles SET_PORT command).

        Args:
            port: New serial port path.

        Returns:
            True if switch was successful.
        """
        if self.transport is not None:
            try:
                self.transport.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting transport before switch: %s", e)

        self.serial_port = port

        try:
            self.transport = create_and_connect_transport(
                port=port,
                baudrate=self.serial_baudrate,
                auto_find_port=False,
            )
            if self.transport and self.transport.is_connected():
                self.first_frame_received = False
                self._reset_tx_keepalive()
                logger.info("Serial switched to port %s", port)
                return True
        except Exception as e:
            logger.warning("Failed to (re)connect serial on SET_PORT: %s", e)

        return False

    def switch_simulator_mode(
        self, enable: bool, sync_state: Any | None = None
    ) -> tuple[bool, str | None]:
        """Switch between real serial and simulator transport.

        Args:
            enable: True to enable simulator, False for real serial.
            sync_state: Optional ControllerState to sync simulator to.

        Returns:
            Tuple of (success, error_message).
        """
        mode_str = "on" if enable else "off"

        # Skip if already in the desired mode
        already_simulator = isinstance(self.transport, MockSerialTransport)
        if enable == already_simulator and self.transport is not None:
            logger.debug("Already in simulator mode=%s, skipping switch", mode_str)
            return True, None

        try:
            # Update env to drive transport_factory.is_simulation_mode()
            os.environ["PAROL6_FAKE_SERIAL"] = "1" if enable else "0"

            # Disconnect existing transport
            if self.transport:
                try:
                    self.transport.disconnect()
                except Exception as e:
                    logger.debug("Transport disconnect: %s", e)

            # Recreate transport according to new mode
            self.transport = create_and_connect_transport(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                auto_find_port=True,
            )

            # If enabling simulator, sync to last known controller state
            if (
                enable
                and sync_state is not None
                and isinstance(self.transport, MockSerialTransport)
            ):
                try:
                    self.transport.sync_from_controller_state(sync_state)
                except Exception as e:
                    logger.warning(
                        "Failed to sync simulator from controller state: %s", e
                    )

            if self.transport:
                self.first_frame_received = False
                self._reset_tx_keepalive()
                logger.info("Simulator mode toggled to %s", "on" if enable else "off")

                # Wait for first frame with timeout
                if not self._wait_for_first_frame(timeout=0.5):
                    logger.warning(
                        "Timeout waiting for first frame after SIMULATOR toggle"
                    )

                return True, None

            return False, "Failed to create transport"

        except Exception as e:
            logger.warning("Failed to (re)configure transport on SIMULATOR: %s", e)
            return False, f"Transport switch failed: {e}"

    def get_latest_frame(self) -> tuple[memoryview | None, int, float]:
        """Get latest frame from transport.

        Returns:
            Tuple of (memoryview, version, timestamp) or (None, 0, 0.0) if unavailable.
        """
        if not self.transport or not self.transport.is_connected():
            return None, 0, 0.0

        try:
            return self.transport.get_latest_frame_view()
        except Exception as e:
            logger.warning("Error getting latest frame: %s", e)
            return None, 0, 0.0

    def write_frame(
        self,
        position_out: np.ndarray,
        speed_out: np.ndarray,
        command_value: int,
        affected_joint_out: np.ndarray,
        inout_out: np.ndarray,
        timeout_out: int,
        gripper_data_out: np.ndarray,
        keepalive_s: float = 0.2,
    ) -> bool:
        """Write frame to transport with coalescing.

        Only writes if state has changed or keepalive timeout reached.

        Args:
            position_out: Position output array.
            speed_out: Speed output array.
            command_value: Command code value.
            affected_joint_out: Affected joint array.
            inout_out: I/O output array.
            timeout_out: Timeout value.
            gripper_data_out: Gripper data array.
            keepalive_s: Keepalive timeout in seconds.

        Returns:
            True if frame was written successfully.
        """
        if not self.transport or not self.transport.is_connected():
            return False

        # Check if state has changed or keepalive needed
        now = time.perf_counter()
        dirty = (
            (command_value != self._last_tx.cmd)
            or (timeout_out != self._last_tx.tout)
            or _arrays_changed(
                position_out,
                self._last_tx.pos,
                speed_out,
                self._last_tx.spd,
                affected_joint_out,
                self._last_tx.aff,
                inout_out,
                self._last_tx.io,
                gripper_data_out,
                self._last_tx.grip,
            )
        )

        if not dirty and (now - self._last_tx.last_sent < keepalive_s):
            return True  # No write needed

        # Write frame
        try:
            ok = self.transport.write_frame(
                position_out,
                speed_out,
                command_value,
                affected_joint_out,
                inout_out,
                timeout_out,
                gripper_data_out,
            )
            if ok:
                # Update last TX snapshot
                self._last_tx.cmd = command_value
                self._last_tx.pos[:] = position_out
                self._last_tx.spd[:] = speed_out
                self._last_tx.aff[:] = affected_joint_out
                self._last_tx.io[:] = inout_out
                self._last_tx.tout = timeout_out
                self._last_tx.grip[:] = gripper_data_out
                self._last_tx.last_sent = now
            return ok
        except Exception as e:
            logger.warning("Error writing frame: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect transport."""
        if self.transport:
            try:
                self.transport.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting transport: %s", e)
            self.transport = None

    def sync_mock_from_state(self, state: Any) -> None:
        """Sync mock transport from controller state after RESET.

        Args:
            state: ControllerState to sync from.
        """
        if isinstance(self.transport, MockSerialTransport):
            self.transport.sync_from_controller_state(state)
            # Skip stale frames
            _, ver, _ = self.transport.get_latest_frame_view()
            self._last_version = ver

    def tick_simulation(self, tool_name: str = "NONE") -> None:
        """Tick mock transport simulation if using MockSerialTransport.

        Called by controller each loop iteration for lockstep simulation.
        No-op for real serial transport.
        """
        if isinstance(self.transport, MockSerialTransport):
            self.transport.tick_simulation(tool_name)

    def _reset_tx_keepalive(self) -> None:
        """Reset TX keepalive to force prompt write."""
        self._last_tx.last_sent = 0.0

    def _wait_for_first_frame(self, timeout: float = 0.5) -> bool:
        """Wait for first frame with timeout.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if frame was received.
        """
        if not self.transport:
            return False

        wait_start = time.perf_counter()
        while time.perf_counter() - wait_start < timeout:
            mv, ver, _ = self.transport.get_latest_frame_view()
            if mv is not None and ver > 0:
                self.first_frame_received = True
                logger.info(
                    "First frame received (%.3fs)", time.perf_counter() - wait_start
                )
                return True
            time.sleep(0.01)

        return False
