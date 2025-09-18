"""
Synchronous facade for AsyncRobotClient.

- In sync code: use RobotClient and call methods directly.
- In async code (event loop running): use AsyncRobotClient and `await` the methods.
"""

import asyncio
import threading
import atexit
from typing import TypeVar, Union, Optional, List, Literal, Dict, Coroutine, Any

from .async_client import AsyncRobotClient
from ..protocol.types import Frame, Axis

T = TypeVar("T")


# Persistent background event loop for sync wrapper
_SYNC_LOOP: asyncio.AbstractEventLoop | None = None
_SYNC_THREAD: threading.Thread | None = None
_SYNC_LOOP_READY = threading.Event()


def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    _SYNC_LOOP_READY.set()
    loop.run_forever()


def _stop_sync_loop() -> None:
    global _SYNC_LOOP, _SYNC_THREAD
    if _SYNC_LOOP is not None:
        try:
            _SYNC_LOOP.call_soon_threadsafe(_SYNC_LOOP.stop)
        except Exception:
            pass
        _SYNC_LOOP = None
        _SYNC_THREAD = None


def _ensure_sync_loop() -> None:
    """Start a persistent background event loop if not started yet."""
    global _SYNC_LOOP, _SYNC_THREAD
    if _SYNC_LOOP is None:
        _SYNC_LOOP = asyncio.new_event_loop()
        _SYNC_THREAD = threading.Thread(
            target=_loop_worker, args=(_SYNC_LOOP,), name="parol6-sync-loop", daemon=True
        )
        _SYNC_THREAD.start()
        _SYNC_LOOP_READY.wait(timeout=1.0)
        atexit.register(_stop_sync_loop)


def _run(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine to completion using a persistent background event loop.
    If a loop is already running in this thread, raise to avoid deadlocks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread -> submit to persistent loop
        _ensure_sync_loop()
        assert _SYNC_LOOP is not None
        fut = asyncio.run_coroutine_threadsafe(coro, _SYNC_LOOP)
        return fut.result()
    # A loop is running in this thread; blocking would be unsafe.
    raise RuntimeError(
        "RobotClient was used while an event loop is running.\n"
        "Use AsyncRobotClient and `await` the method instead."
    )


class RobotClient:
    """
    Synchronous wrapper around AsyncRobotClient.
    All methods return concrete results (never coroutines).
    """

    # ---------- lifecycle ----------

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 2.0,
        retries: int = 1,
    ) -> None:
        self._inner = AsyncRobotClient(
            host=host, port=port, timeout=timeout, retries=retries
        )

    @property
    def async_client(self) -> AsyncRobotClient:
        """Access the underlying async client if you need it."""
        return self._inner

    # Expose common configuration attributes
    @property
    def host(self) -> str:
        return self._inner.host

    @property
    def port(self) -> int:
        return self._inner.port

    # ---------- motion / control ----------

    def ping(self) -> bool:
        return _run(self._inner.ping())

    def home(self) -> bool:
        return _run(self._inner.home())

    def stop(self) -> bool:
        return _run(self._inner.stop())

    def enable(self) -> bool:
        return _run(self._inner.enable())

    def disable(self) -> bool:
        return _run(self._inner.disable())

    def clear_error(self) -> bool:
        return _run(self._inner.clear_error())

    def stream_on(self) -> bool:
        return _run(self._inner.stream_on())

    def stream_off(self) -> bool:
        return _run(self._inner.stream_off())

    def set_com_port(self, port_str: str) -> bool:
        return _run(self._inner.set_com_port(port_str))

    # ---------- status / queries ----------

    def get_angles(self) -> List[float] | None:
        return _run(self._inner.get_angles())

    def get_io(self) -> List[int] | None:
        return _run(self._inner.get_io())

    def get_gripper_status(self) -> List[int] | None:
        return _run(self._inner.get_gripper_status())

    def get_speeds(self) -> List[float] | None:
        return _run(self._inner.get_speeds())

    def get_pose(self) -> List[float] | None:
        return _run(self._inner.get_pose())

    def get_gripper(self) -> List[int] | None:
        return _run(self._inner.get_gripper())

    def get_status(self) -> dict | None:
        return _run(self._inner.get_status())

    def get_loop_stats(self) -> dict | None:
        return _run(self._inner.get_loop_stats())

    # ---------- helper methods ----------

    def get_pose_rpy(self) -> List[float] | None:
        return _run(self._inner.get_pose_rpy())

    def get_pose_xyz(self) -> List[float] | None:
        return _run(self._inner.get_pose_xyz())

    def is_estop_pressed(self) -> bool:
        return _run(self._inner.is_estop_pressed())

    def is_robot_stopped(self, threshold_speed: float = 2.0) -> bool:
        return _run(self._inner.is_robot_stopped(threshold_speed))

    def wait_until_stopped(
        self,
        timeout: float = 90.0,
        settle_window: float = 1.0,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5
    ) -> bool:
        return _run(
            self._inner.wait_until_stopped(
                timeout=timeout,
                settle_window=settle_window,
                speed_threshold=speed_threshold,
                angle_threshold=angle_threshold,
            )
        )

    # ---------- responsive waits / raw send ----------

    def wait_for_server_ready(self, timeout: float = 5.0, interval: float = 0.05) -> bool:
        """Poll ping() until server responds or timeout."""
        return _run(self._inner.wait_for_server_ready(timeout=timeout, interval=interval))

    def wait_for_status(self, predicate, timeout: float = 5.0) -> bool:
        """
        Wait until a multicast status satisfies predicate(status) within timeout.
        Note: predicate is executed in the client's event loop thread.
        """
        return _run(self._inner.wait_for_status(predicate, timeout=timeout))

    def send_raw(self, message: str, await_reply: bool = False, timeout: float = 2.0) -> bool | str | None:
        """
        Send a raw UDP message; optionally await a single reply and return its text.
        Returns True on fire-and-forget send, str on reply, or None on timeout/error when awaiting.
        """
        return _run(self._inner.send_raw(message, await_reply=await_reply, timeout=timeout))

    # ---------- extended controls / motion ----------

    def move_joints(
        self,
        joint_angles: List[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        return _run(
            self._inner.move_joints(
                joint_angles,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
            )
        )

    def move_pose(
        self,
        pose: List[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        return _run(
            self._inner.move_pose(
                pose,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
            )
        )

    def move_cartesian(
        self,
        pose: List[float],
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        return _run(
            self._inner.move_cartesian(
                pose,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
            )
        )

    def move_cartesian_rel_trf(
        self,
        deltas: List[float],
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
    ) -> bool:
        return _run(
            self._inner.move_cartesian_rel_trf(
                deltas,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
            )
        )

    def jog_joint(
        self,
        joint_index: int,
        speed_percentage: int,
        duration: float | None = None,
        distance_deg: float | None = None,
    ) -> bool:
        return _run(
            self._inner.jog_joint(
                joint_index,
                speed_percentage,
                duration,
                distance_deg,
            )
        )

    def jog_cartesian(
        self,
        frame: Frame,
        axis: Axis,
        speed_percentage: int,
        duration: float,
    ) -> bool:
        return _run(
            self._inner.jog_cartesian(
                frame, axis, speed_percentage, duration
            )
        )

    def jog_multiple(
        self,
        joints: List[int],
        speeds: List[float],
        duration: float,
    ) -> bool:
        return _run(self._inner.jog_multiple(joints, speeds, duration))

    def set_io(self, index: int, value: int) -> bool:
        """Set digital I/O bit (0..7) to 0 or 1."""
        return _run(self._inner.set_io(index, value))

    def delay(self, seconds: float) -> bool:
        """Insert a non-blocking delay in the motion queue."""
        return _run(self._inner.delay(seconds))

    # ---------- IO / gripper ----------

    def control_pneumatic_gripper(
        self,
        action: str,
        port: int,
    ) -> bool:
        return _run(self._inner.control_pneumatic_gripper(action, port))

    def control_electric_gripper(
        self,
        action: str,
        position: int | None = 255,
        speed: int | None = 150,
        current: int | None = 500,
    ) -> bool:
        return _run(
            self._inner.control_electric_gripper(
                action, position, speed, current
            )
        )

    # ---------- GCODE ----------

    def execute_gcode(
        self,
        gcode_line: str,
    ) -> bool:
        return _run(self._inner.execute_gcode(gcode_line))

    def execute_gcode_program(
        self,
        program_lines: List[str],
    ) -> bool:
        return _run(self._inner.execute_gcode_program(program_lines))

    def load_gcode_file(
        self,
        filepath: str,
    ) -> bool:
        return _run(self._inner.load_gcode_file(filepath))

    def get_gcode_status(self) -> dict | None:
        return _run(self._inner.get_gcode_status())

    def pause_gcode_program(self) -> bool:
        return _run(self._inner.pause_gcode_program())

    def resume_gcode_program(self) -> bool:
        return _run(self._inner.resume_gcode_program())

    def stop_gcode_program(self) -> bool:
        return _run(self._inner.stop_gcode_program())

    # ---------- smooth motion ----------

    def smooth_circle(
        self,
        center: List[float],
        radius: float,
        plane: Literal["XY", "XZ", "YZ"] = "XY",
        frame: Literal["WRF", "TRF"] = "WRF",
        center_mode: Literal["ABSOLUTE", "TOOL", "RELATIVE"] = "ABSOLUTE",
        entry_mode: Literal["AUTO", "TANGENT", "DIRECT", "NONE"] = "NONE",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> bool:
        return _run(
            self._inner.smooth_circle(
                center=center,
                radius=radius,
                plane=plane,
                frame=frame,
                center_mode=center_mode,
                entry_mode=entry_mode,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                clockwise=clockwise,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
            )
        )

    def smooth_arc_center(
        self,
        end_pose: List[float],
        center: List[float],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> bool:
        return _run(
            self._inner.smooth_arc_center(
                end_pose=end_pose,
                center=center,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                clockwise=clockwise,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
            )
        )

    def smooth_arc_param(
        self,
        end_pose: List[float],
        radius: float,
        arc_angle: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
        clockwise: bool = False,
    ) -> bool:
        return _run(
            self._inner.smooth_arc_param(
                end_pose=end_pose,
                radius=radius,
                arc_angle=arc_angle,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                clockwise=clockwise,
            )
        )

    def smooth_spline(
        self,
        waypoints: List[List[float]],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
    ) -> bool:
        return _run(
            self._inner.smooth_spline(
                waypoints=waypoints,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
            )
        )

    def smooth_helix(
        self,
        center: List[float],
        radius: float,
        pitch: float,
        height: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: Optional[float] = None,
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
        clockwise: bool = False,
    ) -> bool:
        return _run(
            self._inner.smooth_helix(
                center=center,
                radius=radius,
                pitch=pitch,
                height=height,
                frame=frame,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                clockwise=clockwise,
            )
        )

    def smooth_blend(
        self,
        segments: List[Dict],
        blend_time: float = 0.5,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: Optional[List[float]] = None,
        duration: Optional[float] = None,
        speed_percentage: Optional[float] = None,
    ) -> bool:
        return _run(
            self._inner.smooth_blend(
                segments=segments,
                blend_time=blend_time,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
            )
        )

    def smooth_waypoints(
        self,
        waypoints: List[List[float]],
        blend_radii: Literal["AUTO"] | List[float] = "AUTO",
        blend_mode: Literal["parabolic", "circular", "none"] = "parabolic",
        via_modes: Optional[List[str]] = None,
        max_velocity: float = 100.0,
        max_acceleration: float = 500.0,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "quintic",
        duration: Optional[float] = None,
    ) -> bool:
        return _run(
            self._inner.smooth_waypoints(
                waypoints=waypoints,
                blend_radii=blend_radii,
                blend_mode=blend_mode,
                via_modes=via_modes,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                frame=frame,
                trajectory_type=trajectory_type,
                duration=duration,
            )
        )

    # ---------- work coordinate helpers ----------

    def set_work_coordinate_offset(
        self,
        coordinate_system: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> bool:
        return _run(
            self._inner.set_work_coordinate_offset(
                coordinate_system=coordinate_system,
                x=x,
                y=y,
                z=z,
            )
        )

    def zero_work_coordinates(
        self,
        coordinate_system: str = "G54",
    ) -> bool:
        return _run(
            self._inner.zero_work_coordinates(
                coordinate_system=coordinate_system,
            )
        )
