"""
System control commands that can execute regardless of controller enable state.

These commands control the overall state of the robot controller (resume/halt, etc.)
and can execute even when the controller is disabled.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from parol6.commands.base import ExecutionStatusCode, SystemCommand
from parol6.config import save_com_port
from parol6.protocol.wire import (
    CmdType,
    HaltCmd,
    ResumeCmd,
    SetIOCmd,
    SetPortCmd,
    SetProfileCmd,
    SimulatorCmd,
)
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command(CmdType.RESUME)
class ResumeCommand(SystemCommand[ResumeCmd]):
    """Re-enable the robot controller, allowing motion commands."""

    PARAMS_TYPE = ResumeCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Execute resume - set controller to enabled state."""
        logger.info("RESUME command executed")
        state.enabled = True
        state.disabled_reason = ""
        state.Command_out = CommandCode.ENABLE

        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.HALT)
class HaltCommand(SystemCommand[HaltCmd]):
    """Halt the robot — stop all motion and disable."""

    PARAMS_TYPE = HaltCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Execute halt - zero speeds and set controller to disabled state."""
        logger.info("HALT command executed")
        state.Speed_out.fill(0)
        state.enabled = False
        state.disabled_reason = "User requested halt"
        state.Command_out = CommandCode.DISABLE

        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.SET_IO)
class SetIOCommand(SystemCommand[SetIOCmd]):
    """Set a digital I/O port state."""

    PARAMS_TYPE = SetIOCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Execute set port - update I/O port state."""
        logger.info(f"SET_IO: Setting port {self.p.port_index} to {self.p.value}")

        state.InOut_out[self.p.port_index] = self.p.value

        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.SET_PORT)
class SetSerialPortCommand(SystemCommand[SetPortCmd]):
    """Set the serial COM port used by the controller."""

    PARAMS_TYPE = SetPortCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Persist the serial port selection and signal controller to reconnect."""
        ok = save_com_port(self.p.port_str)
        if not ok:
            self.fail("Failed to save COM port")
            return ExecutionStatusCode.FAILED

        self._switch_port = self.p.port_str
        self.finish()
        return ExecutionStatusCode.COMPLETED


@register_command(CmdType.SIMULATOR)
class SimulatorCommand(SystemCommand[SimulatorCmd]):
    """Toggle simulator (fake serial) mode on/off."""

    PARAMS_TYPE = SimulatorCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Execute simulator toggle by setting env var and signaling reconfiguration."""
        os.environ["PAROL6_FAKE_SERIAL"] = "1" if self.p.on else "0"
        logger.info(f"SIMULATOR command executed: {'ON' if self.p.on else 'OFF'}")

        self._switch_simulator = self.p.on
        self.finish()
        return ExecutionStatusCode.COMPLETED


# Valid motion profile types
VALID_PROFILES = frozenset(("TOPPRA", "RUCKIG", "QUINTIC", "TRAPEZOID", "LINEAR"))


@register_command(CmdType.SET_PROFILE)
class SetProfileCommand(SystemCommand[SetProfileCmd]):
    """
    Set the motion profile for all moves.

    Format: [CmdType.SET_PROFILE, profile_type]

    Profile Types:
        TOPPRA    - Time-optimal path parameterization (default)
        RUCKIG    - Time-optimal jerk-limited (point-to-point only, joint moves only)
        QUINTIC   - C² smooth polynomial trajectories
        TRAPEZOID - Linear segments with parabolic blends
        LINEAR    - Direct interpolation (no smoothing)

    Note: RUCKIG is point-to-point and cannot follow Cartesian paths.
    Cartesian moves will use TOPPRA when RUCKIG is set.
    """

    PARAMS_TYPE = SetProfileCmd

    __slots__ = ()

    def do_setup(self, state: ControllerState) -> None:
        """Validate profile name against VALID_PROFILES."""
        profile = self.p.profile.upper()
        if profile not in VALID_PROFILES:
            valid_list = ", ".join(sorted(VALID_PROFILES))
            raise ValueError(
                f"Invalid profile '{self.p.profile}'. Valid profiles: {valid_list}"
            )

    def execute_step(self, state: ControllerState) -> ExecutionStatusCode:
        """Execute profile change."""
        profile = self.p.profile.upper()

        old_profile = state.motion_profile
        state.motion_profile = profile
        logger.info(
            f"SETPROFILE: Changed motion profile from {old_profile} to {profile}"
        )

        self.finish()
        return ExecutionStatusCode.COMPLETED
