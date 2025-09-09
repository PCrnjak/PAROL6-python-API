"""
Synchronous facade for AsyncRobotClient.

- In sync code: use RobotClient and call methods directly.
- In async code (event loop running): this class raises to prevent blocking;
  use AsyncRobotClient instead and `await` the methods.

"""

import asyncio
from typing import Awaitable, TypeVar, Union, Optional, List, Literal, Dict

from .async_client import AsyncRobotClient
from ..protocol.types import Frame, Axis  # keep your existing imports

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine to completion when no event loop is running.
    If a loop is already running, raise to avoid deadlocks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -> safe to run
        return asyncio.run(coro)
    # A loop is running; blocking would be unsafe.
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
        ack_port: int = 5002,
    ) -> None:
        self._inner = AsyncRobotClient(
            host=host, port=port, timeout=timeout, retries=retries, ack_port=ack_port
        )

    @property
    def async_client(self) -> AsyncRobotClient:
        """Access the underlying async client if you need it."""
        return self._inner

    # ---------- motion / control ----------

    def ping(self) -> bool:
        return _run(self._inner.ping())

    def home(self, wait_for_ack: bool = False, timeout: float = 30.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.home(wait_for_ack, timeout, non_blocking))

    def stop(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.stop(wait_for_ack, timeout, non_blocking))

    def enable(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.enable(wait_for_ack, timeout, non_blocking))

    def disable(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.disable(wait_for_ack, timeout, non_blocking))

    def clear_error(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.clear_error(wait_for_ack, timeout, non_blocking))

    def stream_on(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.stream_on(wait_for_ack, timeout, non_blocking))

    def stream_off(self, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.stream_off(wait_for_ack, timeout, non_blocking))

    def set_com_port(self, port_str: str, wait_for_ack: bool = False, timeout: float = 2.0, non_blocking: bool = False) -> Union[str, dict]:
        return _run(self._inner.set_com_port(port_str, wait_for_ack, timeout, non_blocking))

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
        poll_interval: float = 0.2,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5,
        show_progress: bool = False,
    ) -> bool:
        return _run(
            self._inner.wait_until_stopped(
                timeout=timeout,
                settle_window=settle_window,
                poll_interval=poll_interval,
                speed_threshold=speed_threshold,
                angle_threshold=angle_threshold,
                show_progress=show_progress,
            )
        )

    # ---------- extended controls / motion ----------

    def move_joints(
        self,
        joint_angles: List[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.move_joints(
                joint_angles,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
                wait_for_ack,
                timeout,
                non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.move_pose(
                pose,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
                wait_for_ack,
                timeout,
                non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.move_cartesian(
                pose,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
                wait_for_ack,
                timeout,
                non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.move_cartesian_rel_trf(
                deltas,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
                wait_for_ack,
                timeout,
                non_blocking,
            )
        )

    def jog_joint(
        self,
        joint_index: int,
        speed_percentage: int,
        duration: float | None = None,
        distance_deg: float | None = None,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.jog_joint(
                joint_index,
                speed_percentage,
                duration,
                distance_deg,
                wait_for_ack,
                timeout,
                non_blocking,
            )
        )

    def jog_cartesian(
        self,
        frame: Frame,
        axis: Axis,
        speed_percentage: int,
        duration: float,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.jog_cartesian(
                frame, axis, speed_percentage, duration, wait_for_ack, timeout, non_blocking
            )
        )

    def jog_multiple(
        self,
        joints: List[int],
        speeds: List[float],
        duration: float,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.jog_multiple(joints, speeds, duration, wait_for_ack, timeout, non_blocking))

    # ---------- IO / gripper ----------

    def control_pneumatic_gripper(
        self,
        action: str,
        port: int,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.control_pneumatic_gripper(action, port, wait_for_ack, timeout, non_blocking))

    def control_electric_gripper(
        self,
        action: str,
        position: int | None = 255,
        speed: int | None = 150,
        current: int | None = 500,
        wait_for_ack: bool = False,
        timeout: float = 2.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.control_electric_gripper(
                action, position, speed, current, wait_for_ack, timeout, non_blocking
            )
        )

    # ---------- GCODE ----------

    def execute_gcode(
        self,
        gcode_line: str,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.execute_gcode(gcode_line, wait_for_ack, timeout, non_blocking))

    def execute_gcode_program(
        self,
        program_lines: List[str],
        wait_for_ack: bool = False,
        timeout: float = 30.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.execute_gcode_program(program_lines, wait_for_ack, timeout, non_blocking))

    def load_gcode_file(
        self,
        filepath: str,
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.load_gcode_file(filepath, wait_for_ack, timeout, non_blocking))

    def get_gcode_status(self) -> dict | None:
        return _run(self._inner.get_gcode_status())

    def pause_gcode_program(
        self,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.pause_gcode_program(wait_for_ack, timeout, non_blocking))

    def resume_gcode_program(
        self,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.resume_gcode_program(wait_for_ack, timeout, non_blocking))

    def stop_gcode_program(
        self,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(self._inner.stop_gcode_program(wait_for_ack, timeout, non_blocking))

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
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
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
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
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
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.smooth_spline(
                waypoints=waypoints,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 10.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
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
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
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
        wait_for_ack: bool = False,
        timeout: float = 15.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.smooth_blend(
                segments=segments,
                blend_time=blend_time,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
            )
        )

    # ---------- work coordinate helpers ----------

    def set_work_coordinate_offset(
        self,
        coordinate_system: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.set_work_coordinate_offset(
                coordinate_system=coordinate_system,
                x=x,
                y=y,
                z=z,
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
            )
        )

    def zero_work_coordinates(
        self,
        coordinate_system: str = "G54",
        wait_for_ack: bool = False,
        timeout: float = 5.0,
        non_blocking: bool = False,
    ) -> Union[str, dict]:
        return _run(
            self._inner.zero_work_coordinates(
                coordinate_system=coordinate_system,
                wait_for_ack=wait_for_ack,
                timeout=timeout,
                non_blocking=non_blocking,
            )
        )