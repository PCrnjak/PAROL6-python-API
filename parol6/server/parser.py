"""
Command parsing utilities and registry for PAROL6 server.
"""

from typing import Callable, Any, Optional, Tuple, Dict, List
import logging
import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands import (
    # Basic
    HomeCommand, DelayCommand,
    # Cartesian / Joint
    MovePoseCommand, MoveCartCommand, MoveJointCommand, CartesianJogCommand,
    # JOG
    JogCommand, MultiJogCommand,
    # Gripper
    GripperCommand,
    # Smooth motion
    SmoothCircleCommand, SmoothArcCenterCommand, SmoothArcParamCommand,
    SmoothHelixCommand, SmoothSplineCommand, SmoothBlendCommand, SmoothWaypointsCommand,
)

logger = logging.getLogger(__name__)

# Factory signature: takes list of string parts (split by '|'), returns command instance
CommandFactory = Callable[[List[str]], Any]

# Registry for command factories
_REGISTRY: Dict[str, CommandFactory] = {}


def register_command(name: str, factory: CommandFactory) -> None:
    """Register a command factory for a given command name."""
    _REGISTRY[name.upper()] = factory


def get_registry() -> Dict[str, CommandFactory]:
    """Access the current registry (read-only)."""
    return dict(_REGISTRY)


def parse_command_with_id(message: str) -> Tuple[Optional[str], str]:
    """
    Extract optional command ID and return remaining command string.

    Input format:
      [cmd_id|]COMMAND|params...

    Returns:
      (cmd_id or None, command_string_without_id)

    Notes:
      - Cleans up known logging artifacts like '...): ' or 'ID:' prefixes
      - Distinguishes real command names (ALL CAPS) from 8-char lowercase/hex IDs
    """
    # Clean up any logging artifacts
    if "ID:" in message or "):" in message:
        # Extract the actual command after these artifacts
        if "):" in message:
            try:
                message = message[message.rindex("):") + 2 :].strip()
            except ValueError:
                message = message.strip()
        elif "ID:" in message:
            try:
                message = message[message.index("ID:") + 3 :].strip()
            except ValueError:
                message = message.strip()
            # Remove any trailing parentheses or colons
            message = message.lstrip("):").strip()

    parts = message.split("|", 1)

    # Check if first part looks like a valid command ID (8 chars, alphanumeric)
    # IMPORTANT: Command IDs from uuid.uuid4()[:8] will contain lowercase letters/numbers
    # Actual commands are all uppercase, so exclude all-uppercase strings
    if (
        len(parts) > 1
        and len(parts[0]) == 8
        and parts[0].replace("-", "").isalnum()
        and not parts[0].isupper()  # Prevent "MOVEPOSE" from being treated as an ID
    ):
        return parts[0], parts[1]
    else:
        return None, message


def parse_with_registry(message: str) -> Tuple[Optional[str], Any | None, Optional[str]]:
    """
    Parse a message into a command instance using the registry.

    Returns:
      (cmd_id, command_instance or None, error_message or None)
    """
    cmd_id, cmd_str = parse_command_with_id(message)
    parts = cmd_str.split("|")
    if not parts:
        return cmd_id, None, "Empty command"

    name = parts[0].upper()
    factory = _REGISTRY.get(name)
    if factory is None:
        return cmd_id, None, f"No factory registered for command '{name}'"

    try:
        cmd_obj = factory(parts)
        return cmd_id, cmd_obj, None
    except Exception as e:
        logger.exception("Factory error while parsing '%s'", name)
        return cmd_id, None, f"Factory error for '{name}': {e}"


# ----------------------------------------
# Smooth motion parsing (ported from server)
# ----------------------------------------

def _calc_duration_from_speed(trajectory_length: float, speed_percentage: float) -> float:
    """Map speed percentage (0-100) to mm/s using robot caps and compute duration."""
    min_speed = PAROL6_ROBOT.Cartesian_linear_velocity_min * 1000  # mm/s
    max_speed = PAROL6_ROBOT.Cartesian_linear_velocity_max * 1000  # mm/s
    speed_mm_s = float(np.interp(speed_percentage, [0, 100], [min_speed, max_speed]))
    if speed_mm_s > 0:
        return float(trajectory_length / speed_mm_s)
    return 5.0


def _parse_start_pose(text: str) -> list[float] | None:
    """Parse start pose; 'CURRENT'/'NONE' returns None."""
    if text in ("CURRENT", "NONE"):
        return None
    try:
        return [float(v) for v in text.split(",")]
    except Exception:
        logger.warning("Invalid start pose format: %s", text)
        return None


def _factory_smooth(parts: List[str]) -> Any | None:
    """
    Factory for all SMOOTH_* commands. Returns a Smooth* command instance or None.
    """
    cmd = parts[0].upper()

    try:
        if cmd == 'SMOOTH_CIRCLE':
            # SMOOTH_CIRCLE|center|radius|plane|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk]|[center_mode]|[entry_mode]
            center = [float(x) for x in parts[1].split(',')]
            radius = float(parts[2])
            plane = parts[3]
            frame = parts[4]
            start_pose = _parse_start_pose(parts[5])
            timing_type = parts[6]
            timing_value = float(parts[7])
            clockwise = parts[8] == '1'
            trajectory_type = parts[9] if len(parts) > 9 else 'cubic'
            jerk_limit = None
            if trajectory_type == 's_curve' and len(parts) > 10 and parts[10] != 'DEFAULT':
                try:
                    jerk_limit = float(parts[10])
                except ValueError:
                    jerk_limit = None
            center_mode = parts[11] if len(parts) > 11 else 'ABSOLUTE'
            entry_mode = parts[12] if len(parts) > 12 else 'NONE'

            if timing_type == 'DURATION':
                duration = timing_value
            else:
                path_length = 2 * np.pi * radius
                duration = _calc_duration_from_speed(path_length, timing_value)

            return SmoothCircleCommand(center, radius, plane, duration, clockwise, frame, start_pose,
                                       trajectory_type, jerk_limit, center_mode, entry_mode)

        elif cmd == 'SMOOTH_ARC_CENTER':
            # SMOOTH_ARC_CENTER|end_pose|center|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk]
            end_pose = [float(x) for x in parts[1].split(',')]
            center = [float(x) for x in parts[2].split(',')]
            frame = parts[3]
            start_pose = _parse_start_pose(parts[4])
            timing_type = parts[5]
            timing_value = float(parts[6])
            clockwise = parts[7] == '1'
            trajectory_type = parts[8] if len(parts) > 8 else 'cubic'
            jerk_limit = None
            if trajectory_type == 's_curve' and len(parts) > 9 and parts[9] != 'DEFAULT':
                try:
                    jerk_limit = float(parts[9])
                except ValueError:
                    jerk_limit = None

            if timing_type == 'DURATION':
                duration = timing_value
            else:
                radius_estimate = float(np.linalg.norm(np.array(center) - np.array(end_pose[:3])))
                estimated_angle = np.pi / 2
                arc_length = radius_estimate * estimated_angle
                duration = _calc_duration_from_speed(arc_length, timing_value)

            return SmoothArcCenterCommand(end_pose, center, duration, clockwise, frame, start_pose,
                                          trajectory_type, jerk_limit)

        elif cmd == 'SMOOTH_ARC_PARAM':
            # SMOOTH_ARC_PARAM|end_pose|radius|angle|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk]
            end_pose = [float(x) for x in parts[1].split(',')]
            radius = float(parts[2])
            arc_angle = float(parts[3])
            frame = parts[4]
            start_pose = _parse_start_pose(parts[5])
            timing_type = parts[6]
            timing_value = float(parts[7])
            clockwise = parts[8] == '1'
            trajectory_type = parts[9] if len(parts) > 9 else 'cubic'
            jerk_limit = None
            if len(parts) > 10 and parts[10] != 'DEFAULT':
                try:
                    jerk_limit = float(parts[10])
                except ValueError:
                    jerk_limit = None

            if timing_type == 'DURATION':
                duration = timing_value
            else:
                arc_length = radius * float(np.deg2rad(arc_angle))
                duration = _calc_duration_from_speed(arc_length, timing_value)

            return SmoothArcParamCommand(end_pose, radius, arc_angle, duration, clockwise, frame, start_pose,
                                         trajectory_type, jerk_limit)

        elif cmd == 'SMOOTH_SPLINE':
            # SMOOTH_SPLINE|num_waypoints|frame|start_pose|timing_type|timing_value|trajectory_type|[jerk]|w1|w2|...
            num_waypoints = int(parts[1])
            frame = parts[2]
            start_pose = _parse_start_pose(parts[3])
            timing_type = parts[4]
            timing_value = float(parts[5])
            idx = 6
            trajectory_type = 'cubic'
            jerk_limit = None
            if idx < len(parts) and parts[idx] in ['cubic', 'quintic', 's_curve']:
                trajectory_type = parts[idx]
                idx += 1
                if trajectory_type == 's_curve' and idx < len(parts) and parts[idx] != 'DEFAULT':
                    try:
                        jerk_limit = float(parts[idx])
                    except ValueError:
                        pass
                    idx += 1
                elif trajectory_type == 's_curve':
                    idx += 1

            waypoints: List[List[float]] = []
            for _ in range(num_waypoints):
                if idx >= len(parts):
                    break
                wp_vals = [float(v) for v in parts[idx].split(',')]
                waypoints.append(wp_vals)
                idx += 1

            if timing_type == 'DURATION':
                duration = timing_value
            else:
                total_dist = 0.0
                for i in range(1, len(waypoints)):
                    total_dist += float(np.linalg.norm(np.array(waypoints[i][:3]) - np.array(waypoints[i-1][:3])))
                duration = _calc_duration_from_speed(total_dist, timing_value)

            return SmoothSplineCommand(waypoints, duration, frame, start_pose, trajectory_type, jerk_limit)

        elif cmd == 'SMOOTH_HELIX':
            # SMOOTH_HELIX|center|radius|pitch|height|frame|start_pose|timing_type|timing_value|clockwise|trajectory_type|[jerk]
            center = [float(x) for x in parts[1].split(',')]
            radius = float(parts[2])
            pitch = float(parts[3])
            height = float(parts[4])
            frame = parts[5]
            start_pose = _parse_start_pose(parts[6])
            timing_type = parts[7]
            timing_value = float(parts[8])
            clockwise = parts[9] == '1'
            trajectory_type = parts[10] if len(parts) > 10 else 'cubic'
            jerk_limit = None
            if trajectory_type == 's_curve' and len(parts) > 11 and parts[11] != 'DEFAULT':
                try:
                    jerk_limit = float(parts[11])
                except ValueError:
                    jerk_limit = None

            if timing_type == 'DURATION':
                duration = timing_value
            else:
                num_revs = height / pitch if pitch > 0 else 1
                horizontal_len = 2 * np.pi * radius * num_revs
                helix_len = float(np.sqrt(horizontal_len**2 + height**2))
                duration = _calc_duration_from_speed(helix_len, timing_value)

            return SmoothHelixCommand(center, radius, pitch, height, duration, clockwise, frame, start_pose,
                                      trajectory_type, jerk_limit)

        elif cmd == 'SMOOTH_BLEND':
            # SMOOTH_BLEND|num_segments|blend_time|frame|start_pose|timing_type|timing_value|segment1||segment2||...
            num_segments = int(parts[1])
            blend_time = float(parts[2])
            frame = parts[3]
            start_pose = _parse_start_pose(parts[4])
            timing_type = parts[5]

            if timing_type == 'DEFAULT':
                # Use segment durations as-is
                segments_start_idx = 6
                overall_duration = None
                overall_speed = None
            else:
                timing_value = float(parts[6])
                if timing_type == 'DURATION':
                    overall_duration = timing_value
                    overall_speed = None
                else:
                    overall_speed = timing_value
                    overall_duration = None
                segments_start_idx = 7

            segments_data = '|'.join(parts[segments_start_idx:])
            segment_strs = segments_data.split('||')

            segment_definitions: List[dict] = []
            total_original_duration = 0.0
            total_estimated_length = 0.0

            for seg_str in segment_strs:
                if not seg_str:
                    continue
                seg_parts = seg_str.split('|')
                seg_type = seg_parts[0]

                if seg_type == 'LINE':
                    end = [float(x) for x in seg_parts[1].split(',')]
                    segment_duration = float(seg_parts[2])
                    total_original_duration += segment_duration
                    total_estimated_length += 100.0  # conservative estimate
                    segment_definitions.append({
                        'type': 'LINE',
                        'end': end,
                        'duration': segment_duration,
                        'original_duration': segment_duration
                    })

                elif seg_type == 'CIRCLE':
                    center = [float(x) for x in seg_parts[1].split(',')]
                    radius = float(seg_parts[2])
                    plane = seg_parts[3]
                    segment_duration = float(seg_parts[4])
                    clockwise = seg_parts[5] == '1'
                    total_original_duration += segment_duration
                    total_estimated_length += float(2 * np.pi * radius)
                    segment_definitions.append({
                        'type': 'CIRCLE',
                        'center': center,
                        'radius': radius,
                        'plane': plane,
                        'duration': segment_duration,
                        'original_duration': segment_duration,
                        'clockwise': clockwise
                    })

                elif seg_type == 'ARC':
                    end = [float(x) for x in seg_parts[1].split(',')]
                    center = [float(x) for x in seg_parts[2].split(',')]
                    segment_duration = float(seg_parts[3])
                    clockwise = seg_parts[4] == '1'
                    total_original_duration += segment_duration
                    estimated_radius = 50.0
                    estimated_arc_angle = float(np.pi / 2)
                    total_estimated_length += estimated_radius * estimated_arc_angle
                    segment_definitions.append({
                        'type': 'ARC',
                        'end': end,
                        'center': center,
                        'duration': segment_duration,
                        'original_duration': segment_duration,
                        'clockwise': clockwise
                    })

                elif seg_type == 'SPLINE':
                    wp_list: List[List[float]] = []
                    for wp_str in seg_parts[2].split(';'):
                        if wp_str:
                            wp_list.append([float(v) for v in wp_str.split(',')])
                    segment_duration = float(seg_parts[3])
                    total_original_duration += segment_duration
                    est_len = 0.0
                    for i in range(1, len(wp_list)):
                        est_len += float(np.linalg.norm(np.array(wp_list[i][:3]) - np.array(wp_list[i-1][:3])))
                    total_estimated_length += est_len
                    segment_definitions.append({
                        'type': 'SPLINE',
                        'waypoints': wp_list,
                        'duration': segment_duration,
                        'original_duration': segment_duration
                    })

            # Adjust segment durations if overall timing is specified
            if overall_duration is not None and total_original_duration > 0:
                scale = float(overall_duration / total_original_duration)
                for seg in segment_definitions:
                    seg['duration'] = seg['original_duration'] * scale
                logger.info("Scaled blend segments to total duration: %.2fs", overall_duration)
            elif overall_speed is not None:
                overall_duration = _calc_duration_from_speed(total_estimated_length, float(overall_speed))
                if total_original_duration > 0:
                    scale = float(overall_duration / total_original_duration)
                    for seg in segment_definitions:
                        seg['duration'] = seg['original_duration'] * scale
                logger.info("Calculated blend duration from speed: %.2fs", overall_duration)
            else:
                logger.info("Using original blend segment durations (total: %.2fs)", total_original_duration)

            trajectory_type = 'cubic'
            jerk_limit = None
            return SmoothBlendCommand(segment_definitions, blend_time, frame, start_pose, trajectory_type, jerk_limit)

        elif cmd == 'SMOOTH_WAYPOINTS':
            # SMOOTH_WAYPOINTS|num_waypoints|blend_radii|blend_mode|via_modes|max_vel|max_acc|traj_type|frame|duration|waypoints...
            num_waypoints = int(parts[1])
            blend_radii_str = parts[2]
            blend_mode = parts[3]
            via_modes_str = parts[4]
            max_vel_str = parts[5]
            max_acc_str = parts[6]
            trajectory_type = parts[7]
            frame = parts[8]
            duration_str = parts[9]

            if blend_radii_str == 'auto':
                blend_radii: Any = 'auto'
            else:
                blend_radii = [float(r) for r in blend_radii_str.split(',')]

            via_modes = via_modes_str.split(',')

            max_velocity = None if max_vel_str == 'default' else float(max_vel_str)
            max_acceleration = None if max_acc_str == 'default' else float(max_acc_str)
            duration = None if duration_str == 'auto' else float(duration_str)

            waypoints: List[List[float]] = []
            for wp_str in parts[10:]:
                if wp_str:
                    waypoints.append([float(v) for v in wp_str.split(',')])

            return SmoothWaypointsCommand(
                waypoints, blend_radii, blend_mode, via_modes,
                max_velocity, max_acceleration, trajectory_type,
                frame, duration
            )

    except Exception as e:
        logger.error("Error parsing smooth motion command: %s", e)
        logger.debug("Command parts: %s", parts, exc_info=True)
        return None

    logger.warning("Unknown smooth motion command type: %s", cmd)
    return None


# ----------------------------------------
# Standard command factories
# ----------------------------------------

def _factory_movepose(parts: List[str]) -> Any:
    # MOVEPOSE|x|y|z|rx|ry|rz|DUR|SPD
    pose_vals = [float(p) for p in parts[1:7]]
    duration = None if parts[7].upper() == 'NONE' else float(parts[7])
    speed = None if parts[8].upper() == 'NONE' else float(parts[8])
    return MovePoseCommand(pose=pose_vals, duration=duration, velocity_percent=speed)


def _factory_movejoint(parts: List[str]) -> Any:
    # MOVEJOINT|j1|j2|j3|j4|j5|j6|DUR|SPD
    joint_vals = [float(p) for p in parts[1:7]]
    duration = None if parts[7].upper() == 'NONE' else float(parts[7])
    speed = None if parts[8].upper() == 'NONE' else float(parts[8])
    return MoveJointCommand(target_angles=joint_vals, duration=duration, velocity_percent=speed)


def _factory_movecart(parts: List[str]) -> Any:
    # MOVECART|x|y|z|rx|ry|rz|DUR|SPD
    pose_vals = [float(p) for p in parts[1:7]]
    duration = None if parts[7].upper() == 'NONE' else float(parts[7])
    speed = None if parts[8].upper() == 'NONE' else float(parts[8])
    return MoveCartCommand(pose=pose_vals, duration=duration, velocity_percent=speed)


def _factory_delay(parts: List[str]) -> Any:
    # DELAY|seconds
    duration = float(parts[1])
    return DelayCommand(duration=duration)


def _factory_home(parts: List[str]) -> Any:
    # HOME
    return HomeCommand()


def _factory_cartjog(parts: List[str]) -> Any:
    # CARTJOG|FRAME|AXIS|speed_pct|duration
    frame = parts[1].upper()
    axis = parts[2]
    speed = float(parts[3])
    duration = float(parts[4])
    return CartesianJogCommand(frame=frame, axis=axis, speed_percentage=speed, duration=duration)


def _factory_jog(parts: List[str]) -> Any:
    # JOG|joint|speed_pct|DUR|DIST
    joint_idx = int(parts[1])
    speed = float(parts[2])
    duration = None if parts[3].upper() == 'NONE' else float(parts[3])
    distance = None if parts[4].upper() == 'NONE' else float(parts[4])
    return JogCommand(joint=joint_idx, speed_percentage=speed, duration=duration, distance_deg=distance)


def _factory_multijog(parts: List[str]) -> Any:
    # MULTIJOG|j1,j2,...|v1,v2,...|duration
    joint_indices = [int(j) for j in parts[1].split(',')]
    speeds = [float(s) for s in parts[2].split(',')]
    duration = float(parts[3])
    return MultiJogCommand(joints=joint_indices, speed_percentages=speeds, duration=duration)


def _factory_pneumatic_gripper(parts: List[str]) -> Any:
    # PNEUMATICGRIPPER|action|port
    action, port = parts[1].lower(), int(parts[2])
    return GripperCommand(gripper_type='pneumatic', action=action, output_port=port)


def _factory_electric_gripper(parts: List[str]) -> Any:
    # ELECTRICGRIPPER|action|pos|spd|cur
    action_token = parts[1].upper()
    action = None if action_token in ('NONE', 'MOVE') else parts[1].lower()
    pos, spd, curr = int(parts[2]), int(parts[3]), int(parts[4])
    return GripperCommand(gripper_type='electric', action=action, position=pos, speed=spd, current=curr)


def register_default_factories() -> None:
    """Register default command factories."""
    # Standard motion / utility
    register_command('MOVEPOSE', _factory_movepose)
    register_command('MOVEJOINT', _factory_movejoint)
    register_command('MOVECART', _factory_movecart)
    register_command('DELAY', _factory_delay)
    register_command('HOME', _factory_home)
    register_command('CARTJOG', _factory_cartjog)
    register_command('JOG', _factory_jog)
    register_command('MULTIJOG', _factory_multijog)
    register_command('PNEUMATICGRIPPER', _factory_pneumatic_gripper)
    register_command('ELECTRICGRIPPER', _factory_electric_gripper)

    # Smooth motion
    for name in (
        'SMOOTH_CIRCLE',
        'SMOOTH_ARC_CENTER',
        'SMOOTH_ARC_PARAM',
        'SMOOTH_SPLINE',
        'SMOOTH_HELIX',
        'SMOOTH_BLEND',
        'SMOOTH_WAYPOINTS',
    ):
        register_command(name, _factory_smooth)
