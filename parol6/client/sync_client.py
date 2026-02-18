"""
Synchronous facade for AsyncRobotClient.

- In sync code: use RobotClient and call methods directly.
- In async code (event loop running): use AsyncRobotClient and `await` the methods.
"""

import asyncio
import atexit
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, overload

from ..protocol.types import Axis, Frame, PingResult
from ..protocol.wire import (
    CurrentActionResultStruct,
    LoopStatsResultStruct,
    StatusBuffer,
    StatusResultStruct,
    ToolResultStruct,
)
from .async_client import AsyncRobotClient

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
    if _SYNC_LOOP is None:
        return

    loop = _SYNC_LOOP

    async def _shutdown():
        # Cancel all pending tasks
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        # Let cancelled tasks finalize
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    try:
        asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        if _SYNC_THREAD is not None:
            _SYNC_THREAD.join(timeout=2.0)
    except (RuntimeError, asyncio.InvalidStateError):
        # Loop may already be stopped or thread may not be joinable
        pass

    _SYNC_LOOP = None
    _SYNC_THREAD = None


def _ensure_sync_loop() -> None:
    """Start a persistent background event loop if not started yet."""
    global _SYNC_LOOP, _SYNC_THREAD
    if _SYNC_LOOP is None:
        _SYNC_LOOP = asyncio.new_event_loop()
        _SYNC_THREAD = threading.Thread(
            target=_loop_worker,
            args=(_SYNC_LOOP,),
            name="parol6-sync-loop",
            daemon=True,
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

    Can be used as a context manager to ensure proper cleanup:

        with RobotClient() as client:
            client.resume()
            ...
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

    def close(self) -> None:
        """Close underlying AsyncRobotClient and release resources."""
        _run(self._inner.close())

    def __enter__(self) -> "RobotClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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

    def home(self, wait: bool = False, **wait_kwargs) -> int:
        """Home the robot to its home position.

        Returns the command index (≥ 0) on success, -1 on failure.

        Args:
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().
        """
        return _run(self._inner.home(wait=wait, **wait_kwargs))

    def resume(self) -> int:
        """Re-enable the robot controller, allowing motion commands.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.resume())

    def halt(self) -> int:
        """Halt the robot — stop all motion and disable.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.halt())

    def simulator_on(self) -> int:
        """Enable simulator mode (no physical robot hardware required).

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.simulator_on())

    def simulator_off(self) -> int:
        """Disable simulator mode, switching to real hardware.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.simulator_off())

    def set_serial_port(self, port_str: str) -> int:
        """Set the serial port for robot hardware communication.

        Args:
            port_str: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3').

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.set_serial_port(port_str))

    def reset(self) -> int:
        """Reset controller state to initial values."""
        return _run(self._inner.reset())

    # ---------- status / queries ----------
    def ping(self) -> PingResult | None:
        """Ping the controller to check connectivity.

        Returns:
            PingResult with serial_connected status, or None on timeout.
        """
        return _run(self._inner.ping())

    def get_angles(self) -> list[float] | None:
        """Get current joint angles in degrees.

        Returns:
            List of 6 joint angles [J1-J6] in degrees, or None on timeout.
        """
        return _run(self._inner.get_angles())

    def get_io(self) -> list[int] | None:
        """Get digital I/O status.

        Returns:
            List of 5 integers [in1, in2, out1, out2, estop], or None on timeout.
        """
        return _run(self._inner.get_io())

    def get_gripper_status(self) -> list[int] | None:
        """Get electric gripper status.

        Returns:
            List of integers [id, pos, speed, current, status, obj_detected],
            or None on timeout.
        """
        return _run(self._inner.get_gripper_status())

    def get_speeds(self) -> list[float] | None:
        """Get current joint speeds in steps per second.

        Returns:
            List of 6 joint speeds [J1-J6] in steps/sec, or None on timeout.
        """
        return _run(self._inner.get_speeds())

    def get_pose(self) -> list[float] | None:
        """Get current robot pose as a 4x4 transformation matrix.

        Returns:
            16-element flattened transformation matrix (row-major) with
            translation in mm, or None on timeout.
        """
        return _run(self._inner.get_pose())

    def get_gripper(self) -> list[int] | None:
        """Alias for get_gripper_status()."""
        return _run(self._inner.get_gripper())

    def get_status(self) -> StatusResultStruct | None:
        """Get aggregate robot status.

        Returns:
            StatusResultStruct with pose, angles, speeds, io, gripper, or None on timeout.
        """
        return _run(self._inner.get_status())

    def get_loop_stats(self) -> LoopStatsResultStruct | None:
        """Get control loop runtime statistics.

        Returns:
            LoopStatsResultStruct with loop timing metrics, or None on timeout.
        """
        return _run(self._inner.get_loop_stats())

    def reset_loop_stats(self) -> int:
        """Reset control-loop min/max metrics and overrun count."""
        return _run(self._inner.reset_loop_stats())

    def get_tool(self) -> ToolResultStruct | None:
        """
        Get the current tool configuration and available tools.

        Returns:
            ToolResultStruct with tool (current) and available (list), or None.
        """
        return _run(self._inner.get_tool())

    def set_tool(self, tool_name: str) -> int:
        """
        Set the current end-effector tool configuration.

        Args:
            tool_name: Name of the tool ('NONE', 'PNEUMATIC', 'ELECTRIC')

        Returns:
            True if successful
        """
        return _run(self._inner.set_tool(tool_name))

    def set_profile(self, profile: str) -> int:
        """
        Set the motion profile for all moves.

        Args:
            profile: Motion profile type ('TOPPRA', 'RUCKIG', 'QUINTIC', 'TRAPEZOID', 'LINEAR')
                Note: RUCKIG is point-to-point only; Cartesian moves will use TOPPRA.

        Returns:
            True if successful
        """
        return _run(self._inner.set_profile(profile))

    def get_profile(self) -> str | None:
        """
        Get the current motion profile.

        Returns:
            Current motion profile, or None on timeout.
        """
        return _run(self._inner.get_profile())

    def get_current_action(self) -> CurrentActionResultStruct | None:
        """
        Get the current executing action/command and its state.

        Returns:
            Struct with current action name, state, and next action.
        """
        return _run(self._inner.get_current_action())

    def get_queue(self) -> list[str] | None:
        """
        Get the list of queued non-streamable commands.

        Returns:
            List of queued command names.
        """
        return _run(self._inner.get_queue())

    # ---------- helper methods ----------

    def get_pose_rpy(self) -> list[float] | None:
        """Get robot pose as [x, y, z, rx, ry, rz] in mm and degrees.

        Returns:
            List of 6 floats [x, y, z, rx, ry, rz], or None on error.
        """
        return _run(self._inner.get_pose_rpy())

    def get_pose_xyz(self) -> list[float] | None:
        """Get robot position as [x, y, z] in mm.

        Returns:
            List of 3 floats [x, y, z], or None on error.
        """
        return _run(self._inner.get_pose_xyz())

    def is_estop_pressed(self) -> bool:
        """Check if E-stop is pressed.

        Returns:
            True if E-stop is pressed, False otherwise.
        """
        return _run(self._inner.is_estop_pressed())

    def is_robot_stopped(self, threshold_speed: float = 2.0) -> bool:
        """Check if robot has stopped moving.

        Args:
            threshold_speed: Speed threshold in steps/sec.

        Returns:
            True if all joints below threshold.
        """
        return _run(self._inner.is_robot_stopped(threshold_speed))

    def wait_motion_complete(
        self,
        timeout: float = 10.0,
        settle_window: float = 0.25,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5,
        motion_start_timeout: float = 1.0,
    ) -> bool:
        """Wait for robot to stop moving.

        Args:
            timeout: Maximum time to wait in seconds.
            settle_window: How long robot must be stable.
            speed_threshold: Max joint speed to be considered stopped.
            angle_threshold: Max angle change to be considered stopped.
            motion_start_timeout: Max time to wait for motion to start.

        Returns:
            True if robot stopped, False if timeout.
        """
        return _run(
            self._inner.wait_motion_complete(
                timeout=timeout,
                settle_window=settle_window,
                speed_threshold=speed_threshold,
                angle_threshold=angle_threshold,
                motion_start_timeout=motion_start_timeout,
            )
        )

    # ---------- responsive waits / raw send ----------

    def wait_ready(self, timeout: float = 5.0, interval: float = 0.05) -> bool:
        """Poll ping() until server responds or timeout."""
        return _run(self._inner.wait_ready(timeout=timeout, interval=interval))

    def wait_for_status(
        self, predicate: Callable[[StatusBuffer], bool], timeout: float = 5.0
    ) -> bool:
        """
        Wait until a multicast status satisfies predicate(status) within timeout.
        Note: predicate is executed in the client's event loop thread.
        """
        return _run(self._inner.wait_for_status(predicate, timeout=timeout))

    def wait_command_complete(self, command_index: int, timeout: float = 10.0) -> bool:
        """Wait until a specific command index has been completed.

        Args:
            command_index: The command index to wait for (returned by motion commands).
            timeout: Maximum time to wait in seconds.

        Returns:
            True if the command completed within timeout, False otherwise.

        Raises:
            MotionError: If the pipeline errored at or before command_index.
        """
        return _run(self._inner.wait_command_complete(command_index, timeout=timeout))

    # ---------- motion ----------

    @overload
    def moveJ(
        self,
        target: list[float],
        *,
        duration: float = ...,
        speed: float = ...,
        accel: float = ...,
        r: float = ...,
        rel: bool = ...,
        wait: bool = ...,
        **wait_kwargs,
    ) -> int: ...

    @overload
    def moveJ(
        self,
        target: list[float] | None = ...,
        *,
        pose: list[float],
        duration: float = ...,
        speed: float = ...,
        accel: float = ...,
        r: float = ...,
        wait: bool = ...,
        **wait_kwargs,
    ) -> int: ...

    def moveJ(
        self,
        target: list[float] | None = None,
        *,
        pose: list[float] | None = None,
        duration: float = 0.0,
        speed: float = 0.0,
        accel: float = 1.0,
        r: float = 0.0,
        rel: bool = False,
        wait: bool = True,
        **wait_kwargs,
    ) -> int:
        if pose is not None:
            return _run(
                self._inner.moveJ(
                    target or [],
                    pose=pose,
                    duration=duration,
                    speed=speed,
                    accel=accel,
                    r=r,
                    wait=wait,
                    **wait_kwargs,
                )
            )
        if target is None:
            raise ValueError("moveJ requires target or pose")
        return _run(
            self._inner.moveJ(
                target,
                duration=duration,
                speed=speed,
                accel=accel,
                r=r,
                rel=rel,
                wait=wait,
                **wait_kwargs,
            )
        )

    def moveL(
        self,
        pose: list[float],
        *,
        frame: Frame = "WRF",
        duration: float = 0.0,
        speed: float = 0.0,
        accel: float = 1.0,
        r: float = 0.0,
        rel: bool = False,
        wait: bool = True,
        **wait_kwargs,
    ) -> int:
        return _run(
            self._inner.moveL(
                pose,
                frame=frame,
                duration=duration,
                speed=speed,
                accel=accel,
                r=r,
                rel=rel,
                wait=wait,
                **wait_kwargs,
            )
        )

    def moveC(
        self,
        via: list[float],
        end: list[float],
        *,
        frame: Frame = "WRF",
        duration: float | None = None,
        speed: float | None = None,
        accel: float = 1.0,
        r: float = 0.0,
        wait: bool = True,
        **wait_kwargs,
    ) -> int:
        return _run(
            self._inner.moveC(
                via,
                end,
                frame=frame,
                duration=duration,
                speed=speed,
                accel=accel,
                r=r,
                wait=wait,
                **wait_kwargs,
            )
        )

    def moveS(
        self,
        waypoints: list[list[float]],
        *,
        frame: Frame = "WRF",
        duration: float | None = None,
        speed: float | None = None,
        accel: float = 1.0,
        wait: bool = True,
        **wait_kwargs,
    ) -> int:
        return _run(
            self._inner.moveS(
                waypoints,
                frame=frame,
                duration=duration,
                speed=speed,
                accel=accel,
                wait=wait,
                **wait_kwargs,
            )
        )

    def moveP(
        self,
        waypoints: list[list[float]],
        *,
        frame: Frame = "WRF",
        duration: float | None = None,
        speed: float | None = None,
        accel: float = 1.0,
        wait: bool = True,
        **wait_kwargs,
    ) -> int:
        return _run(
            self._inner.moveP(
                waypoints,
                frame=frame,
                duration=duration,
                speed=speed,
                accel=accel,
                wait=wait,
                **wait_kwargs,
            )
        )

    @overload
    def servoJ(
        self,
        target: list[float],
        *,
        speed: float = ...,
        accel: float = ...,
    ) -> int: ...

    @overload
    def servoJ(
        self,
        target: list[float] | None = ...,
        *,
        pose: list[float],
        speed: float = ...,
        accel: float = ...,
    ) -> int: ...

    def servoJ(
        self,
        target: list[float] | None = None,
        *,
        pose: list[float] | None = None,
        speed: float = 1.0,
        accel: float = 1.0,
    ) -> int:
        if pose is not None:
            return _run(
                self._inner.servoJ(target or [], pose=pose, speed=speed, accel=accel)
            )
        if target is None:
            raise ValueError("servoJ requires target or pose")
        return _run(self._inner.servoJ(target, speed=speed, accel=accel))

    def servoL(
        self,
        pose: list[float],
        *,
        speed: float = 1.0,
        accel: float = 1.0,
    ) -> int:
        return _run(self._inner.servoL(pose, speed=speed, accel=accel))

    @overload
    def jogJ(
        self,
        joint: int,
        speed: float,
        duration: float = ...,
        *,
        accel: float = ...,
    ) -> int: ...

    @overload
    def jogJ(
        self,
        *,
        joints: list[int],
        speeds: list[float],
        duration: float = ...,
        accel: float = ...,
    ) -> int: ...

    def jogJ(
        self,
        joint: int | None = None,
        speed: float = 0.0,
        duration: float = 0.1,
        *,
        joints: list[int] | None = None,
        speeds: list[float] | None = None,
        accel: float = 1.0,
    ) -> int:
        if joints is not None and speeds is not None:
            return _run(
                self._inner.jogJ(
                    joints=joints, speeds=speeds, duration=duration, accel=accel
                )
            )
        if joint is not None:
            return _run(self._inner.jogJ(joint, speed, duration, accel=accel))
        raise ValueError("jogJ requires either joint or joints/speeds")

    @overload
    def jogL(
        self,
        frame: Frame,
        axis: Axis,
        speed: float,
        duration: float = ...,
        *,
        accel: float = ...,
    ) -> int: ...

    @overload
    def jogL(
        self,
        frame: Frame,
        *,
        axes: list[Axis],
        speeds_list: list[float],
        duration: float = ...,
        accel: float = ...,
    ) -> int: ...

    def jogL(
        self,
        frame: Frame,
        axis: Axis | None = None,
        speed: float = 0.0,
        duration: float = 0.1,
        *,
        axes: list[Axis] | None = None,
        speeds_list: list[float] | None = None,
        accel: float = 1.0,
    ) -> int:
        if axes is not None and speeds_list is not None:
            return _run(
                self._inner.jogL(
                    frame,
                    axes=axes,
                    speeds_list=speeds_list,
                    duration=duration,
                    accel=accel,
                )
            )
        if axis is not None:
            return _run(self._inner.jogL(frame, axis, speed, duration, accel=accel))
        raise ValueError("jogL requires either axis or axes/speeds_list")

    def checkpoint(self, label: str) -> int:
        return _run(self._inner.checkpoint(label))

    def wait_for_command(self, index: int, timeout: float = 30.0) -> bool:
        return _run(self._inner.wait_for_command(index, timeout=timeout))

    def wait_for_checkpoint(self, label: str, timeout: float = 30.0) -> bool:
        return _run(self._inner.wait_for_checkpoint(label, timeout=timeout))

    def set_io(self, index: int, value: int) -> int:
        """Set digital I/O bit (0..7) to 0 or 1."""
        return _run(self._inner.set_io(index, value))

    def delay(self, seconds: float) -> int:
        """Insert a non-blocking delay in the motion queue."""
        return _run(self._inner.delay(seconds))

    # ---------- IO / gripper ----------

    def control_pneumatic_gripper(
        self,
        action: str,
        port: int,
        wait: bool = False,
        **wait_kwargs,
    ) -> int:
        """Control pneumatic gripper via digital outputs.

        Args:
            action: 'open' or 'close'.
            port: Port number (1 or 2).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.control_pneumatic_gripper(
                action, port, wait=wait, **wait_kwargs
            )
        )

    def control_electric_gripper(
        self,
        action: str,
        position: float = 0.0,
        speed: float = 0.5,
        current: int = 500,
        wait: bool = False,
        **wait_kwargs,
    ) -> int:
        """Control electric gripper.

        Args:
            action: 'move' or 'calibrate'.
            position: 0.0-1.0 (0=open, 1=closed).
            speed: 0.0-1.0 fraction of max speed.
            current: Current limit in mA (100-1000).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.control_electric_gripper(
                action, position, speed, current, wait=wait, **wait_kwargs
            )
        )
