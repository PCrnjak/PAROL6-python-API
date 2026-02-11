"""Unified PAROL6 robot — lifecycle, configuration, kinematics, and factories.

This class directly satisfies the web commander's ``Robot`` Protocol.
All parol6-specific details (subprocess management, pinokin, IK solver, etc.)
are encapsulated here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pinokin import Robot as PinokinRobot
from pinokin import se3_from_rpy, so3_rpy

from parol6.client.async_client import AsyncRobotClient
from parol6.client.dry_run_client import DryRunRobotClient
from parol6.client.sync_client import RobotClient as SyncRobotClient
from parol6.config import HOME_ANGLES_DEG, LIMITS
from parol6.motion.trajectory import ProfileType
from parol6.protocol.wire import CmdType, MsgType, decode, encode
from parol6.tools import TOOL_CONFIGS
from parol6.utils.ik import check_limits, solve_ik

logger = logging.getLogger(__name__)

# Precompiled regex for server log normalization
_SIMPLE_FORMAT_RE = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2})\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL|TRACE)\s+([A-Za-z0-9_.-]+):\s+(.*)$"
)


# ===========================================================================
# Server lifecycle (private)
# ===========================================================================


def _is_server_running(
    host: str = "127.0.0.1",
    port: int = 5001,
    timeout: float = 1.0,
) -> bool:
    """Return True if a PAROL6 controller responds to UDP PING at host:port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            ping_msg = encode((CmdType.PING,))
            sock.sendto(ping_msg, (host, port))
            data, _ = sock.recvfrom(1024)
            resp = decode(data)
            return (
                isinstance(resp, (list, tuple))
                and len(resp) >= 1
                and resp[0] == MsgType.RESPONSE
            )
    except (OSError, socket.timeout):
        return False


class _ServerManager:
    """Manages the lifecycle of the PAROL6 controller subprocess."""

    def __init__(self, normalize_logs: bool = False) -> None:
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self.normalize_logs = normalize_logs

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc and self._proc.poll() is None else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start_controller(
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

        # repo root: parol6/robot.py -> parents[1]
        cwd = Path(__file__).resolve().parents[1]

        env = os.environ.copy()
        if no_autohome:
            env["PAROL6_NOAUTOHOME"] = "1"
        if extra_env:
            env.update(extra_env)
        if server_host:
            env["PAROL6_CONTROLLER_IP"] = server_host
        if server_port is not None:
            env["PAROL6_CONTROLLER_PORT"] = str(server_port)

        existing_py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{cwd}{os.pathsep}{existing_py_path}" if existing_py_path else str(cwd)
        )

        args = [sys.executable, "-u", "-m", "parol6.server.cli"]

        root_logger = logging.getLogger()
        root_level = root_logger.level

        parol_trace_flag = str(env.get("PAROL_TRACE", "0")).strip().lower()
        if parol_trace_flag in ("1", "true", "yes", "on"):
            level_name = "TRACE"
        else:
            level_name = logging.getLevelName(root_level)
            if isinstance(level_name, str) and level_name.upper().startswith("LEVEL"):
                if root_level == 5:
                    level_name = "TRACE"
                else:
                    level_name = "INFO"

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
                bufsize=1,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start controller: {e}") from e

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
            last_logger = "parol6.server"

            for raw_line in iter(proc.stdout.readline, ""):
                if self._stop_reader.is_set():
                    break
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                if self.normalize_logs:
                    level = logging.INFO
                    logger_name: str | None = None
                    msg = line

                    s = _SIMPLE_FORMAT_RE.match(line)
                    if s:
                        _, level_str, logger_name, actual_message = s.groups()
                        logger_name = (logger_name or "").strip()
                        msg = actual_message
                        level = getattr(
                            logging, (level_str or "INFO").upper(), logging.INFO
                        )
                    elif line.startswith("Traceback"):
                        level = logging.ERROR

                    target_logger_name = logger_name or last_logger or "parol6.server"
                    target_logger = logging.getLogger(target_logger_name)
                    target_logger.log(level, msg)

                    if logger_name:
                        last_logger = logger_name
                else:
                    print(line)
        except Exception as e:
            logging.warning("_ServerManager: output reader stopped: %s", e)

    def stop_controller(self, timeout: float = 2.0) -> None:
        """Stop the controller process if running."""
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=timeout)
        self._reader_thread = None
        if self._proc and self._proc.poll() is None:
            logging.debug("Stopping Controller...")
            try:
                self._proc.terminate()
                self._proc.wait(timeout=timeout)
            except Exception as e:
                logging.warning("stop_controller: terminate/wait failed: %s", e)

            if self._proc and self._proc.poll() is None:
                logging.warning(
                    "Controller did not exit after SIGTERM within %.1fs, sending SIGKILL",
                    timeout,
                )
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=timeout)
                except Exception as e:
                    logging.warning("stop_controller: kill/wait failed: %s", e)
            self._proc = None

    def await_ready(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 10.0,
        poll_interval: float = 0.2,
    ) -> bool:
        """Block until the controller responds to PING over UDP."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if _is_server_running(host, port, timeout=min(0.5, poll_interval)):
                return True
            remain = deadline - time.monotonic()
            if remain <= 0:
                return False
            time.sleep(min(poll_interval, remain))
        return False


# ===========================================================================
# Concrete joint / tool dataclass implementations
# ===========================================================================


@dataclass(frozen=True, slots=True)
class _PositionLimits:
    deg: NDArray[np.float64]
    rad: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _KinodynamicLimits:
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    jerk: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class _JointLimits:
    position: _PositionLimits
    hard: _KinodynamicLimits
    jog: _KinodynamicLimits


@dataclass(frozen=True, slots=True)
class _HomePosition:
    deg: NDArray[np.float64]
    rad: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _JointsSpec:
    count: int
    names: tuple[str, ...]
    limits: _JointLimits
    home: _HomePosition


@dataclass(frozen=True, slots=True)
class _ToolData:
    """Concrete ToolSpec for PAROL6 tools."""

    key: str
    display_name: str
    description: str
    tool_type: Any  # ToolType enum — imported lazily to avoid circular deps
    tcp_origin: tuple[float, float, float]
    tcp_rpy: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _PneumaticGripperData(_ToolData):
    """Concrete PneumaticGripperTool."""

    gripper_type: Any = None  # GripperType.PNEUMATIC
    io_port: int = 1


class _ToolsCollection:
    """Concrete ToolsSpec for PAROL6."""

    def __init__(self, tools: tuple[_ToolData, ...]) -> None:
        self._tools = tools
        self._by_key = {t.key: t for t in tools}

    @property
    def available(self) -> tuple[_ToolData, ...]:
        return self._tools

    @property
    def default(self) -> _ToolData:
        return self._by_key.get("NONE", self._tools[0])

    def __getitem__(self, key: str) -> _ToolData:
        return self._by_key[key]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._by_key
        # Cross-enum comparison by .value for structural typing compatibility
        if hasattr(item, "value"):
            return any(t.tool_type.value == item.value for t in self._tools)
        return False

    def by_type(self, tool_type: object) -> tuple[_ToolData, ...]:
        if hasattr(tool_type, "value"):
            return tuple(t for t in self._tools if t.tool_type.value == tool_type.value)
        return tuple(t for t in self._tools if t.tool_type == tool_type)


# ===========================================================================
# Helper builders
# ===========================================================================


def _build_joints() -> _JointsSpec:
    """Build JointsSpec from parol6 LIMITS and HOME_ANGLES_DEG."""
    home_deg = np.array(HOME_ANGLES_DEG, dtype=np.float64)
    return _JointsSpec(
        count=6,
        names=("J1", "J2", "J3", "J4", "J5", "J6"),
        limits=_JointLimits(
            position=_PositionLimits(
                deg=LIMITS.joint.position.deg,
                rad=LIMITS.joint.position.rad,
            ),
            hard=_KinodynamicLimits(
                velocity=LIMITS.joint.hard.velocity,
                acceleration=LIMITS.joint.hard.acceleration,
                jerk=LIMITS.joint.hard.jerk,
            ),
            jog=_KinodynamicLimits(
                velocity=LIMITS.joint.jog.velocity,
                acceleration=LIMITS.joint.jog.acceleration,
                jerk=LIMITS.joint.jog.jerk,
            ),
        ),
        home=_HomePosition(
            deg=home_deg,
            rad=np.deg2rad(home_deg),
        ),
    )


def _decompose_transform(
    T: NDArray[np.float64],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Extract (origin_m, rpy_rad) from a 4x4 homogeneous transform."""
    origin = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
    rpy_buf = np.zeros(3, dtype=np.float64)
    so3_rpy(T[:3, :3], rpy_buf)
    rpy = (float(rpy_buf[0]), float(rpy_buf[1]), float(rpy_buf[2]))
    return origin, rpy


def _build_tools() -> _ToolsCollection:
    """Build typed tool specs from parol6 TOOL_CONFIGS."""
    # Import enums here to avoid circular imports at module level.
    # The web commander defines these; parol6 just uses matching values.
    from enum import Enum

    class _ToolType(Enum):
        NONE = "none"
        GRIPPER = "gripper"

    class _GripperType(Enum):
        PNEUMATIC = "pneumatic"
        ELECTRIC = "electric"
        PARALLEL = "parallel"

    tools: list[_ToolData] = []
    for key, cfg in TOOL_CONFIGS.items():
        transform = cfg.get("transform", np.eye(4, dtype=np.float64))
        origin, rpy = _decompose_transform(transform)
        name_str = cfg.get("name", key)
        desc = cfg.get("description", "")

        if key == "NONE":
            tools.append(
                _ToolData(
                    key=key,
                    display_name=name_str,
                    description=desc,
                    tool_type=_ToolType.NONE,
                    tcp_origin=origin,
                    tcp_rpy=rpy,
                )
            )
        elif key == "PNEUMATIC":
            tools.append(
                _PneumaticGripperData(
                    key=key,
                    display_name=name_str,
                    description=desc,
                    tool_type=_ToolType.GRIPPER,
                    tcp_origin=origin,
                    tcp_rpy=rpy,
                    gripper_type=_GripperType.PNEUMATIC,
                    io_port=1,
                )
            )
        else:
            # Default: treat unknown tools as NONE type
            tools.append(
                _ToolData(
                    key=key,
                    display_name=name_str,
                    description=desc,
                    tool_type=_ToolType.NONE,
                    tcp_origin=origin,
                    tcp_rpy=rpy,
                )
            )

    return _ToolsCollection(tuple(tools))


def _resolve_urdf_path() -> str:
    urdf_res = pkg_files("parol6") / "urdf_model" / "urdf" / "PAROL6.urdf"
    return str(Path(str(urdf_res)).resolve())


def _resolve_mesh_dir() -> str:
    urdf = Path(_resolve_urdf_path())
    return str(urdf.parent.parent)


# ===========================================================================
# IK result type (parol6-native, structurally satisfies the Protocol)
# ===========================================================================


@dataclass
class Parol6IKResult:
    """IK result — structurally compatible with the web commander's IKResult Protocol."""

    q: NDArray[np.float64]  # radians
    success: bool
    violations: str | None = None
    iterations: int = 0
    residual: float = 0.0


# ===========================================================================
# Robot class
# ===========================================================================


class Robot:
    """Unified PAROL6 robot — satisfies the web commander's Robot Protocol.

    Combines identity, configuration, FK/IK kinematics, controller lifecycle,
    and client factories. Supports both sync and async context managers::

        # Sync
        with Robot() as robot:
            client = robot.create_sync_client()

        # Async
        async with Robot() as robot:
            client = robot.create_client()
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 10.0,
        normalize_logs: bool = False,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._manager = _ServerManager(normalize_logs=normalize_logs)

        # Build configuration eagerly
        self._joints = _build_joints()
        self._tools = _build_tools()
        self._urdf_path = _resolve_urdf_path()
        self._mesh_dir = _resolve_mesh_dir()
        self._motion_profiles = tuple(p.value.upper() for p in ProfileType)

        # Initialize pinokin for FK/IK
        self._pinokin = PinokinRobot(self._urdf_path)

        # Pre-allocated buffers for FK/IK
        self._q_buf = np.zeros(self._pinokin.nq, dtype=np.float64)
        self._T_buf = np.asfortranarray(np.zeros((4, 4), dtype=np.float64))
        self._rpy_buf = np.zeros(3, dtype=np.float64)
        self._fk_out = np.zeros(6, dtype=np.float64)
        self._T_target_buf = np.zeros((4, 4), dtype=np.float64)

    # -- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "PAROL6"

    # -- Structured sub-objects ---------------------------------------------

    @property
    def joints(self) -> _JointsSpec:
        return self._joints

    @property
    def tools(self) -> _ToolsCollection:
        return self._tools

    # -- Unit preferences ---------------------------------------------------

    @property
    def position_unit(self) -> Literal["mm", "m"]:
        return "mm"

    # -- Capability flags ---------------------------------------------------

    @property
    def has_force_torque(self) -> bool:
        return False

    @property
    def has_freedrive(self) -> bool:
        return False

    @property
    def digital_outputs(self) -> int:
        return 2

    @property
    def digital_inputs(self) -> int:
        return 2

    # -- Visualization ------------------------------------------------------

    @property
    def urdf_path(self) -> str:
        return self._urdf_path

    @property
    def mesh_dir(self) -> str:
        return self._mesh_dir

    @property
    def joint_index_mapping(self) -> tuple[int, ...]:
        return (0, 1, 2, 3, 4, 5)

    # -- Motion configuration -----------------------------------------------

    @property
    def motion_profiles(self) -> tuple[str, ...]:
        return self._motion_profiles

    @property
    def cartesian_frames(self) -> tuple[str, ...]:
        return ("WRF", "TRF")

    # -- Backend injection --------------------------------------------------

    @property
    def backend_package(self) -> str:
        return "parol6"

    @property
    def sync_client_class(self) -> type:
        return SyncRobotClient

    @property
    def async_client_class(self) -> type:
        return AsyncRobotClient

    # -- Kinematics ---------------------------------------------------------

    def _load_q_buf(self, q_rad: NDArray[np.float64]) -> None:
        """Copy joint radians into the padded pinokin q buffer."""
        n = min(len(q_rad), self._pinokin.nq)
        self._q_buf[:n] = q_rad[:n]
        self._q_buf[n:] = 0.0

    def fk(self, q_rad: NDArray[np.float64]) -> NDArray[np.float64]:
        self._load_q_buf(q_rad)
        self._pinokin.fkine_into(self._q_buf, self._T_buf)
        so3_rpy(self._T_buf[:3, :3], self._rpy_buf)
        self._fk_out[0] = self._T_buf[0, 3]
        self._fk_out[1] = self._T_buf[1, 3]
        self._fk_out[2] = self._T_buf[2, 3]
        self._fk_out[3] = self._rpy_buf[0]
        self._fk_out[4] = self._rpy_buf[1]
        self._fk_out[5] = self._rpy_buf[2]
        return self._fk_out

    def ik(
        self, pose: NDArray[np.float64], q_seed_rad: NDArray[np.float64]
    ) -> Parol6IKResult:
        se3_from_rpy(
            pose[0],
            pose[1],
            pose[2],
            pose[3],
            pose[4],
            pose[5],
            self._T_target_buf,
        )
        result = solve_ik(
            robot=self._pinokin,
            target_pose=self._T_target_buf,
            current_q=q_seed_rad,
            quiet_logging=True,
        )
        return Parol6IKResult(
            q=result.q.copy(),
            success=result.success,
            violations=result.violations,
            iterations=result.iterations,
            residual=result.residual,
        )

    def check_limits(self, q_rad: NDArray[np.float64]) -> bool:
        return check_limits(q_rad, log=False)

    def fk_batch(self, joint_path_rad: NDArray[np.float64]) -> NDArray[np.float64]:
        transforms = self._pinokin.batch_fk(joint_path_rad)
        n = len(transforms)
        result = np.empty((n, 6), dtype=np.float64)
        rpy = self._rpy_buf
        for i, T in enumerate(transforms):
            result[i, 0] = T[0, 3]
            result[i, 1] = T[1, 3]
            result[i, 2] = T[2, 3]
            so3_rpy(T[:3, :3], rpy)
            result[i, 3] = rpy[0]
            result[i, 4] = rpy[1]
            result[i, 5] = rpy[2]
        return result

    def ik_batch(
        self,
        poses: NDArray[np.float64],
        q_start_rad: NDArray[np.float64],
    ) -> list[Parol6IKResult]:
        results: list[Parol6IKResult] = []
        q_current = q_start_rad.copy()
        for i in range(poses.shape[0]):
            p = poses[i]
            se3_from_rpy(p[0], p[1], p[2], p[3], p[4], p[5], self._T_target_buf)
            result = solve_ik(
                robot=self._pinokin,
                target_pose=self._T_target_buf,
                current_q=q_current,
                quiet_logging=True,
            )
            ik_result = Parol6IKResult(
                q=result.q.copy(),
                success=result.success,
                violations=result.violations,
                iterations=result.iterations,
                residual=result.residual,
            )
            results.append(ik_result)
            if result.success:
                q_current[:] = result.q
        return results

    # -- Lifecycle ----------------------------------------------------------

    def start(self, **kwargs: Any) -> None:
        """Start the controller subprocess and block until ready.

        Keyword args override constructor defaults:
            host, port, timeout, com_port, extra_env
        """
        host: str = kwargs.get("host", self._host)
        port: int = kwargs.get("port", self._port)
        timeout: float = kwargs.get("timeout", self._timeout)
        com_port: str | None = kwargs.get("com_port")
        extra_env: dict[str, str] | None = kwargs.get("extra_env")

        if _is_server_running(host, port):
            raise RuntimeError(f"Server already running at {host}:{port}")

        self._manager.start_controller(
            com_port=com_port,
            server_host=host,
            server_port=port,
            extra_env=extra_env,
        )

        if not self._manager.await_ready(host=host, port=port, timeout=timeout):
            self._manager.stop_controller()
            raise RuntimeError("Controller failed to become ready")

    def stop(self) -> None:
        """Stop the controller subprocess."""
        self._manager.stop_controller()

    def is_available(self, **kwargs: Any) -> bool:
        """Check if a controller is reachable via UDP PING."""
        host: str = kwargs.get("host", self._host)
        port: int = kwargs.get("port", self._port)
        return _is_server_running(host=host, port=port)

    # -- Context managers ---------------------------------------------------

    def __enter__(self) -> Robot:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    async def __aenter__(self) -> Robot:
        await asyncio.to_thread(self.start)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.stop()

    # -- Factories ----------------------------------------------------------

    def create_client(self, **kwargs: Any) -> AsyncRobotClient:
        host: str = kwargs.get("host", self._host)
        port: int = kwargs.get("port", self._port)
        timeout: float = kwargs.get("timeout", 5.0)
        return AsyncRobotClient(host=host, port=port, timeout=timeout)

    def create_sync_client(self, **kwargs: Any) -> SyncRobotClient:
        host: str = kwargs.get("host", self._host)
        port: int = kwargs.get("port", self._port)
        timeout: float = kwargs.get("timeout", 5.0)
        return SyncRobotClient(host=host, port=port, timeout=timeout)

    def create_dry_run_client(self, **kwargs: Any) -> DryRunRobotClient:
        initial_joints_deg: list[float] | None = kwargs.get("initial_joints_deg")
        return DryRunRobotClient(initial_joints_deg=initial_joints_deg)
