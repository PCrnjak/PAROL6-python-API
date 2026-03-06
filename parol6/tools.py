"""
Typed tool configuration and registry for the PAROL6 robot.

Each tool type has a frozen config dataclass that holds physical description,
valid actions, and a ``populate_status()`` method the controller uses to fill
the 50 Hz ``ToolStatus`` broadcast from hardware state.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from pinokin import se3_from_trans, se3_mul, se3_rx
from waldoctl import (
    LinearMotion,
    MeshRole,
    MeshSpec,
    PartMotion,
    ToolState,
    ToolVariant,
)

if TYPE_CHECKING:
    from waldoctl import ToolStatus

    from parol6.commands.base import MotionCommand
    from parol6.commands.gripper_commands import (
        ElectricGripperCommand,
        PneumaticGripperCommand,
    )
    from parol6.server.state import ControllerState
    from parol6.server.transports.mock_serial_transport import MockRobotState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool simulator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolSimulator(Protocol):
    """Protocol for tool-type-specific simulation logic.

    Each tool config that needs simulation creates a simulator instance via
    ``create_simulator()``. The simulator's ``resolve_params()`` is called
    once on tool change, and ``tick()`` is called every simulation step.
    """

    def resolve_params(self, cfg: ToolConfig) -> None:
        """Compute simulation parameters from the tool config."""
        ...

    def tick(self, state: MockRobotState, dt: float) -> None:
        """Advance the tool simulation by *dt* seconds."""
        ...


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolConfig:
    """Immutable configuration for one tool type."""

    name: str
    description: str
    transform: np.ndarray  # 4x4 homogeneous transform (flange → TCP)
    meshes: tuple[MeshSpec, ...] = ()
    motions: tuple[PartMotion, ...] = ()
    variants: tuple[ToolVariant, ...] = ()

    def populate_status(self, hw: ControllerState, out: ToolStatus) -> None:
        """Fill *out* from hardware state. Override in subclasses."""

    def create_command(self, action: str, params: list) -> MotionCommand | None:
        """Create a command engine for this tool action. Returns None if not supported."""
        return None

    def create_simulator(self) -> ToolSimulator | None:
        """Create a simulator for this tool type. Returns None if no simulation needed."""
        return None


# ---------------------------------------------------------------------------
# Gripper configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PneumaticGripperConfig(ToolConfig):
    """Configuration for pneumatic grippers controlled via digital I/O."""

    io_port: int = 1
    valid_actions: tuple[str, ...] = ("open", "close", "move")

    def populate_status(self, hw: ControllerState, out: ToolStatus) -> None:
        port_idx = 2 if self.io_port == 1 else 3
        # Simulator writes ramped position into Gripper_data_in[1] (0-255).
        # Real hardware has no position feedback — use valve output directly.
        # Convention: 0.0 = open, 1.0 = closed. Pneumatic simulator ramps
        # toward 255 for open, so we invert.
        pos_byte = hw.Gripper_data_in[1]
        if pos_byte > 0 or hw.InOut_out[port_idx] == 0:
            out.positions = (1.0 - float(pos_byte) / 255.0,)
        else:
            out.positions = (1.0 - float(hw.InOut_out[port_idx]),)
        out.part_detected = bool(hw.InOut_in[port_idx])
        out.engaged = bool(hw.InOut_out[port_idx])
        out.state = ToolState.IDLE

    def create_command(self, action: str, params: list) -> PneumaticGripperCommand:
        from parol6.commands.gripper_commands import PneumaticGripperCommand

        if action not in self.valid_actions:
            raise ValueError(f"Invalid action '{action}' for pneumatic gripper")
        if action == "move":
            position = float(params[0]) if params and len(params) > 0 else 0.0
            action = "open" if position < 0.5 else "close"
        return PneumaticGripperCommand.from_tool_action(
            action=action, port=self.io_port
        )

    def create_simulator(self) -> PneumaticToolSimulator:
        return PneumaticToolSimulator()


@dataclass(frozen=True)
class ElectricGripperConfig(ToolConfig):
    """Configuration for electric grippers controlled via the serial gripper bus."""

    position_range: tuple[float, float] = (0.0, 1.0)
    speed_range: tuple[float, float] = (0.0, 1.0)
    current_range: tuple[int, int] = (100, 1000)
    valid_actions: tuple[str, ...] = ("move", "calibrate")

    # Motor controller / mechanical properties
    encoder_cpr: int = 16_384  # encoder counts per revolution
    gear_pd_mm: float = 12.0  # rack-and-pinion gear pitch diameter (mm)
    firmware_speed_range_tps: tuple[int, int] = (
        40,
        80_000,
    )  # CAN byte 0..255 → ticks/s
    motor_kt: float = 0.0  # motor torque constant (Nm/A); 0 = force estimation disabled

    def populate_status(self, hw: ControllerState, out: ToolStatus) -> None:
        current_ma = float(hw.Gripper_data_in[3])
        out.positions = (float(hw.Gripper_data_in[1]) / 255.0,)
        if self.motor_kt > 0 and self.gear_pd_mm > 0:
            torque_nm = self.motor_kt * (current_ma / 1000.0)
            gear_radius_m = self.gear_pd_mm / 2000.0
            force_n = torque_nm / gear_radius_m
        else:
            force_n = 0.0
        out.channels = (force_n, current_ma)
        out.part_detected = bool(hw.Gripper_data_in[5])
        out.engaged = bool(hw.Gripper_data_in[2])  # speed > 0
        out.state = ToolState.IDLE

    def create_command(self, action: str, params: list) -> ElectricGripperCommand:
        from parol6.commands.gripper_commands import ElectricGripperCommand

        if action not in self.valid_actions:
            raise ValueError(f"Invalid action '{action}' for electric gripper")
        position = float(params[0]) if len(params) > 0 else 0.0
        speed = float(params[1]) if len(params) > 1 else 0.5
        current = int(params[2]) if len(params) > 2 else 500
        return ElectricGripperCommand.from_tool_action(
            action=action, position=position, speed=speed, current=current
        )

    def create_simulator(self) -> ElectricGripperSimulator:
        return ElectricGripperSimulator()


# ---------------------------------------------------------------------------
# Tool simulators
# ---------------------------------------------------------------------------


class PneumaticToolSimulator:
    """Simulates binary-activation tool ramp (pneumatic grippers, vacuum, etc.).

    Reads the commanded I/O output to determine whether the tool is
    engaged, then ramps the tool position toward the target at the
    physical speed derived from the tool's LinearMotion descriptor.
    Writes the ramped position byte into ``gripper_data_in[1]`` for
    ``populate_status()`` to read.
    """

    __slots__ = ("_io_port", "_ramp_speed")

    def __init__(self) -> None:
        self._io_port: int = -1
        self._ramp_speed: float = 0.0

    def resolve_params(self, cfg: ToolConfig) -> None:
        self._io_port = -1
        self._ramp_speed = 0.0

        if not isinstance(cfg, PneumaticGripperConfig):
            return

        # Find first LinearMotion with estimated speed
        for m in cfg.motions:
            if isinstance(m, LinearMotion) and m.estimated_speed_m_s:
                # Normalized speed: fraction of full travel per second
                self._ramp_speed = m.estimated_speed_m_s / m.travel_m
                break

        if self._ramp_speed > 0:
            # Map io_port to InOut_out index (port 1 -> index 2, port 2 -> index 3)
            self._io_port = cfg.io_port + 1

    def tick(self, state: MockRobotState, dt: float) -> None:
        if self._io_port < 0:
            return

        # Read commanded I/O output to determine target (0=closed, 1=open)
        io_val = float(state.io_out[self._io_port])
        target = 1.0 if io_val > 0 else 0.0
        if target != state.tool_ramp_target:
            state.tool_ramp_target = target

        # Ramp toward target
        error = state.tool_ramp_target - state.tool_ramp_current
        if abs(error) < 1e-6:
            return
        step = self._ramp_speed * dt
        if abs(error) <= step:
            state.tool_ramp_current = state.tool_ramp_target
        elif error > 0:
            state.tool_ramp_current += step
        else:
            state.tool_ramp_current -= step

        # Write ramped position as byte into gripper_data_in[1] (same slot electric uses)
        state.gripper_data_in[1] = int(state.tool_ramp_current * 255.0 + 0.5)

        # Update part detection input when ramp completes
        if abs(state.tool_ramp_current - state.tool_ramp_target) < 1e-6:
            det_idx = self._io_port
            state.io_in[det_idx] = 1 if state.tool_ramp_target < 0.5 else 0


class ElectricGripperSimulator:
    """Simulates electric gripper position ramp via the @njit ramp function.

    Resolves tick_range, min/max speed from the tool config's mechanical
    parameters and LinearMotion descriptor, then delegates per-tick
    simulation to the ``_simulate_gripper_ramp_jit`` numba function.
    """

    __slots__ = ("_tick_range", "_min_speed", "_max_speed")

    def __init__(self) -> None:
        self._tick_range: float = 0.0
        self._min_speed: float = 0.0
        self._max_speed: float = 0.0

    def resolve_params(self, cfg: ToolConfig) -> None:
        if not isinstance(cfg, ElectricGripperConfig):
            return
        # Find jaw travel from default motion
        for m in cfg.motions:
            if isinstance(m, LinearMotion):
                travel_mm = m.travel_m * 1000.0
                self._tick_range = (
                    travel_mm / (math.pi * cfg.gear_pd_mm)
                ) * cfg.encoder_cpr
                break
        min_tps, max_tps = cfg.firmware_speed_range_tps
        self._min_speed = float(min_tps)
        self._max_speed = float(max_tps)

    def tick(self, state: MockRobotState, dt: float) -> None:
        from parol6.server.transports.mock_serial_transport import (
            _simulate_gripper_ramp_jit,
        )

        state.gripper_pos_f = _simulate_gripper_ramp_jit(
            state.gripper_ramp,
            state.gripper_data_in,
            state.gripper_pos_f,
            dt,
            self._tick_range,
            self._min_speed,
            self._max_speed,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, ToolConfig] = {}


def register_tool(key: str, config: ToolConfig) -> None:
    """Register a tool configuration by key (e.g. ``"PNEUMATIC"``)."""
    _TOOL_REGISTRY[key] = config


def get_registry() -> dict[str, ToolConfig]:
    """Return the tool registry (read-only view not enforced — callers cooperate)."""
    return _TOOL_REGISTRY


def list_tools() -> list[str]:
    """Get list of available tool keys."""
    return list(_TOOL_REGISTRY.keys())


def get_tool_transform(
    tool_name: str,
    variant_key: str | None = None,
) -> np.ndarray:
    """Get the 4x4 transformation matrix for a tool or variant.

    When *variant_key* is given and the matching variant has a
    ``tcp_origin``, returns a transform built from the variant's TCP
    instead of the tool-level transform.

    Raises ValueError if *tool_name* is not registered.
    """
    cfg = _TOOL_REGISTRY.get(tool_name)
    if cfg is None:
        raise ValueError(f"Unknown tool '{tool_name}'. Available: {list_tools()}")
    if variant_key is not None:
        for v in cfg.variants:
            if v.key == variant_key and v.tcp_origin is not None:
                return _make_tcp_transform(*v.tcp_origin)
        logger.warning("Variant '%s' not found for tool '%s'", variant_key, tool_name)
    return cfg.transform


# ---------------------------------------------------------------------------
# Built-in PAROL6 tools — registered at import time
# ---------------------------------------------------------------------------


def _make_tcp_transform(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> np.ndarray:
    """TCP transform for a tool mounted on the flange.

    Translates to (x, y, z) in the flange frame, then rotates 180deg
    around X so the TCP Z-axis points in the tool working direction.
    """
    trans = np.zeros((4, 4), dtype=np.float64)
    rot = np.zeros((4, 4), dtype=np.float64)
    out = np.zeros((4, 4), dtype=np.float64)
    se3_from_trans(x, y, z, trans)
    se3_rx(np.pi, rot)
    se3_mul(trans, rot, out)
    return out


# All PAROL6 tools share the same TCP orientation (180deg Rx).
_TCP_RPY = (math.pi, 0.0, 0.0)


register_tool(
    "NONE",
    ToolConfig(
        name="No Tool",
        description="Bare flange - no tool attached",
        transform=np.eye(4, dtype=np.float64),
    ),
)


# ---------------------------------------------------------------------------
# Pneumatic gripper — vertical & horizontal mounting variants
# ---------------------------------------------------------------------------

_PNEUMATIC_VERTICAL_MESHES = (
    MeshSpec(file="pneumatic_gripper_vertical_body.stl", role=MeshRole.BODY),
    MeshSpec(file="pneumatic_gripper_vertical_right_jaw.stl", role=MeshRole.JAW),
    MeshSpec(file="pneumatic_gripper_vertical_left_jaw.stl", role=MeshRole.JAW),
)
_PNEUMATIC_VERTICAL_MOTION = (
    LinearMotion(
        role=MeshRole.JAW,
        axis=(0.0, 1.0, 0.0),
        travel_m=0.0035,
        symmetric=True,
        estimated_speed_m_s=0.023,
        estimated_accel_m_s2=2.0,
    ),
)

_PNEUMATIC_HORIZONTAL_MESHES = (
    MeshSpec(file="pneumatic_gripper_horizontal_body.stl", role=MeshRole.BODY),
    MeshSpec(file="pneumatic_gripper_horizontal_right_jaw.stl", role=MeshRole.JAW),
    MeshSpec(file="pneumatic_gripper_horizontal_left_jaw.stl", role=MeshRole.JAW),
)
_PNEUMATIC_HORIZONTAL_MOTION = (
    LinearMotion(
        role=MeshRole.JAW,
        axis=(1.0, 0.0, 0.0),
        travel_m=0.01045,
        symmetric=True,
        estimated_speed_m_s=0.07,
        estimated_accel_m_s2=2.0,
    ),
)

register_tool(
    "PNEUMATIC",
    PneumaticGripperConfig(
        name="Pneumatic Gripper",
        description="Pneumatic gripper assembly (vertical/horizontal mounting)",
        transform=_make_tcp_transform(x=-0.055, z=-0.027),
        meshes=_PNEUMATIC_VERTICAL_MESHES,
        motions=_PNEUMATIC_VERTICAL_MOTION,
        variants=(
            ToolVariant(
                key="vertical",
                display_name="Vertical",
                meshes=_PNEUMATIC_VERTICAL_MESHES,
                motions=_PNEUMATIC_VERTICAL_MOTION,
                tcp_origin=(-0.055, 0.0, -0.027),
                tcp_rpy=_TCP_RPY,
            ),
            ToolVariant(
                key="horizontal",
                display_name="Horizontal",
                meshes=_PNEUMATIC_HORIZONTAL_MESHES,
                motions=_PNEUMATIC_HORIZONTAL_MOTION,
                tcp_origin=(0.0, 0.0, -0.082),
                tcp_rpy=_TCP_RPY,
            ),
        ),
        io_port=1,
    ),
)


# ---------------------------------------------------------------------------
# SSG-48 electric gripper — finger & pinch jaw variants
# ---------------------------------------------------------------------------

_SSG48_JAW_MOTION = (
    LinearMotion(
        role=MeshRole.JAW, axis=(0.0, 1.0, 0.0), travel_m=0.024, symmetric=True
    ),
)

_SSG48_FINGER_MESHES = (
    MeshSpec(file="ssg48_body.stl", role=MeshRole.BODY),
    MeshSpec(file="ssg48_finger_right.stl", role=MeshRole.JAW),
    MeshSpec(file="ssg48_finger_left.stl", role=MeshRole.JAW),
)

_SSG48_PINCH_MESHES = (
    MeshSpec(file="ssg48_body.stl", role=MeshRole.BODY),
    MeshSpec(file="ssg48_pinch_right.stl", role=MeshRole.JAW),
    MeshSpec(file="ssg48_pinch_left.stl", role=MeshRole.JAW),
)

register_tool(
    "SSG-48",
    ElectricGripperConfig(
        name="SSG-48 Electric Gripper",
        description="SSG-48 adaptive electric gripper (Spectral micro BLDC)",
        transform=_make_tcp_transform(z=-0.105),
        meshes=_SSG48_FINGER_MESHES,
        motions=_SSG48_JAW_MOTION,
        variants=(
            ToolVariant(
                key="finger",
                display_name="Finger",
                meshes=_SSG48_FINGER_MESHES,
                motions=_SSG48_JAW_MOTION,
                tcp_origin=(0.0, 0.0, -0.105),
                tcp_rpy=_TCP_RPY,
            ),
            ToolVariant(
                key="pinch",
                display_name="Pinch",
                meshes=_SSG48_PINCH_MESHES,
                motions=_SSG48_JAW_MOTION,
                tcp_origin=(0.0, 0.0, -0.105),
                tcp_rpy=_TCP_RPY,
            ),
        ),
        position_range=(0.0, 1.0),
        speed_range=(0.0, 1.0),
        current_range=(100, 1000),
    ),
)


# ---------------------------------------------------------------------------
# MSG AI stepper gripper — 100mm, 150mm, 200mm rail variants
# ---------------------------------------------------------------------------

_MSG_100_JAW_MOTION = (
    LinearMotion(
        role=MeshRole.JAW, axis=(0.0, 1.0, 0.0), travel_m=0.0267, symmetric=True
    ),
)
_MSG_150_JAW_MOTION = (
    LinearMotion(
        role=MeshRole.JAW, axis=(0.0, 1.0, 0.0), travel_m=0.0514, symmetric=True
    ),
)
_MSG_200_JAW_MOTION = (
    LinearMotion(
        role=MeshRole.JAW, axis=(0.0, 1.0, 0.0), travel_m=0.0767, symmetric=True
    ),
)

_MSG_100_MESHES = (
    MeshSpec(file="msg_ai_100_body.stl", role=MeshRole.BODY),
    MeshSpec(file="msg_ai_100_right_jaw.stl", role=MeshRole.JAW),
    MeshSpec(file="msg_ai_100_left_jaw.stl", role=MeshRole.JAW),
)

_MSG_150_MESHES = (
    MeshSpec(file="msg_ai_150_body.stl", role=MeshRole.BODY),
    MeshSpec(file="msg_ai_150_right_jaw.stl", role=MeshRole.JAW),
    MeshSpec(file="msg_ai_150_left_jaw.stl", role=MeshRole.JAW),
)

_MSG_200_MESHES = (
    MeshSpec(file="msg_ai_200_body.stl", role=MeshRole.BODY),
    MeshSpec(file="msg_ai_200_right_jaw.stl", role=MeshRole.JAW),
    MeshSpec(file="msg_ai_200_left_jaw.stl", role=MeshRole.JAW),
)

register_tool(
    "MSG",
    ElectricGripperConfig(
        name="MSG AI Stepper Gripper",
        description="MSG compliant AI stepper gripper (StepFOC)",
        transform=_make_tcp_transform(x=-0.029, z=-0.103),
        meshes=_MSG_100_MESHES,
        motions=_MSG_100_JAW_MOTION,
        variants=(
            ToolVariant(
                key="100mm",
                display_name="100mm Rail",
                meshes=_MSG_100_MESHES,
                motions=_MSG_100_JAW_MOTION,
                tcp_origin=(-0.029, 0.0, -0.103),
                tcp_rpy=_TCP_RPY,
            ),
            ToolVariant(
                key="150mm",
                display_name="150mm Rail",
                meshes=_MSG_150_MESHES,
                motions=_MSG_150_JAW_MOTION,
                tcp_origin=(-0.029, 0.0, -0.103),
                tcp_rpy=_TCP_RPY,
            ),
            ToolVariant(
                key="200mm",
                display_name="200mm Rail",
                meshes=_MSG_200_MESHES,
                motions=_MSG_200_JAW_MOTION,
                tcp_origin=(-0.029, 0.0, -0.103),
                tcp_rpy=_TCP_RPY,
            ),
        ),
        position_range=(0.0, 1.0),
        speed_range=(0.0, 1.0),
        current_range=(100, 1000),
        gear_pd_mm=16.67,  # 32P 21T gear: PD = 21/32" = 16.67mm
        firmware_speed_range_tps=(500, 60_000),  # StepFOC velocity range
    ),
)


# ---------------------------------------------------------------------------
# Vacuum gripper — pneumatic valve control, no jaws
# ---------------------------------------------------------------------------

register_tool(
    "VACUUM",
    PneumaticGripperConfig(
        name="Vacuum Gripper",
        description="Vacuum gripper (pneumatic valve I/O)",
        transform=_make_tcp_transform(z=-0.037),
        meshes=(MeshSpec(file="vacuum_gripper_body.stl", role=MeshRole.BODY),),
        motions=(),
        io_port=1,
    ),
)
