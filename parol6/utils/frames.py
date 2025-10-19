"""
Shared TRF/WRF transformation utilities used by commands.

These helpers convert tool/frame-relative inputs (TRF) into world reference frame (WRF)
using the current tool pose derived from the robot's forward kinematics.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Dict, Any, cast

import numpy as np
from numpy.typing import NDArray
from spatialmath import SE3

import parol6.PAROL6_ROBOT as PAROL6_ROBOT

logger = logging.getLogger(__name__)

# Constants for TRF plane normal vectors
PLANE_NORMALS_TRF: Dict[str, NDArray] = {
    "XY": np.array([0.0, 0.0, 1.0]),  # Tool's Z-axis
    "XZ": np.array([0.0, 1.0, 0.0]),  # Tool's Y-axis
    "YZ": np.array([1.0, 0.0, 0.0]),  # Tool's X-axis
}


def point_trf_to_wrf_mm(point_mm: Sequence[float], tool_pose: SE3) -> NDArray:
    """Convert 3D point from TRF to WRF coordinates (both in mm)."""
    point_trf = SE3(point_mm[0] / 1000.0, point_mm[1] / 1000.0, point_mm[2] / 1000.0)
    point_wrf = cast(SE3, tool_pose * point_trf)
    return point_wrf.t * 1000.0


def pose6_trf_to_wrf(pose6_mm_deg: Sequence[float], tool_pose: SE3) -> NDArray:
    """Convert 6D pose [x,y,z,rx,ry,rz] from TRF to WRF (mm, degrees)."""
    pose_trf = SE3(pose6_mm_deg[0] / 1000.0, pose6_mm_deg[1] / 1000.0, pose6_mm_deg[2] / 1000.0) * SE3.RPY(  # type: ignore[arg-type]
        pose6_mm_deg[3:], unit="deg", order="xyz"
    )
    pose_wrf = cast(SE3, tool_pose * pose_trf)
    return np.concatenate([pose_wrf.t * 1000.0, pose_wrf.rpy(unit="deg", order="xyz")])


def se3_to_pose6_mm_deg(T: SE3) -> NDArray:
    """Convert SE3 transform to 6D pose [x,y,z,rx,ry,rz] (mm, degrees)."""
    return np.concatenate([T.t * 1000.0, T.rpy(unit="deg", order="xyz")])


def transform_center_trf_to_wrf(params: Dict[str, Any], tool_pose: SE3, transformed: Dict[str, Any]) -> None:
    """Transform 'center' parameter from TRF (mm) to WRF (mm) using tool_pose."""
    center_trf = SE3(params["center"][0] / 1000.0, params["center"][1] / 1000.0, params["center"][2] / 1000.0)
    center_wrf = cast(SE3, tool_pose * center_trf)
    transformed["center"] = center_wrf.t * 1000.0


def transform_start_pose_if_needed(
    start_pose: Optional[Sequence[float]], frame: str, current_position_in: np.ndarray
) -> Optional[NDArray]:
    """Transform start_pose from TRF to WRF if needed."""
    if frame == "TRF" and start_pose:
        current_q = PAROL6_ROBOT.ops.steps_to_rad(current_position_in)
        tool_pose = PAROL6_ROBOT.robot.fkine(current_q)
        return pose6_trf_to_wrf(start_pose, tool_pose)
    return np.asarray(start_pose, dtype=float) if start_pose is not None else None


def transform_command_params_to_wrf(
    command_type: str, params: Dict[str, Any], frame: str, current_position_in: np.ndarray
) -> Dict[str, Any]:
    """
    Transform command parameters from TRF to WRF.
    Handles position, orientation, and directional vectors correctly.
    """
    if frame == "WRF":
        return params

    # Get current tool pose
    current_q = PAROL6_ROBOT.ops.steps_to_rad(current_position_in)
    tool_pose = PAROL6_ROBOT.robot.fkine(current_q)

    transformed = params.copy()

    # SMOOTH_CIRCLE - Transform center and plane normal
    if command_type == "SMOOTH_CIRCLE":
        if "center" in params:
            transform_center_trf_to_wrf(params, tool_pose, transformed)

        if "plane" in params:
            normal_trf = PLANE_NORMALS_TRF[params["plane"]]
            normal_wrf = tool_pose.R @ normal_trf
            transformed["normal_vector"] = normal_wrf
            logger.info(f"  -> TRF circle plane {params['plane']} transformed to WRF")

    # SMOOTH_ARC_CENTER - Transform center, end_pose, and implied plane
    elif command_type == "SMOOTH_ARC_CENTER":
        if "center" in params:
            transform_center_trf_to_wrf(params, tool_pose, transformed)

        if "end_pose" in params:
            transformed["end_pose"] = pose6_trf_to_wrf(params["end_pose"], tool_pose)

        if "plane" in params:
            normal_trf = PLANE_NORMALS_TRF[params["plane"]]
            normal_wrf = tool_pose.R @ normal_trf
            transformed["normal_vector"] = normal_wrf

    # SMOOTH_ARC_PARAM - Transform end_pose and arc plane
    elif command_type == "SMOOTH_ARC_PARAM":
        if "end_pose" in params:
            transformed["end_pose"] = pose6_trf_to_wrf(params["end_pose"], tool_pose)

        # For parametric arc, the plane is usually XY of the tool
        if "plane" not in params:
            params["plane"] = "XY"  # Default to XY plane

        normal_trf = PLANE_NORMALS_TRF[params.get("plane", "XY")]
        normal_wrf = tool_pose.R @ normal_trf
        transformed["normal_vector"] = normal_wrf

    # SMOOTH_HELIX - Transform center and helix axis
    elif command_type == "SMOOTH_HELIX":
        if "center" in params:
            transform_center_trf_to_wrf(params, tool_pose, transformed)

        # Transform helix axis (default is Z-axis of tool)
        axis_trf = np.array([0.0, 0.0, 1.0])  # Tool's Z-axis
        axis_wrf = tool_pose.R @ axis_trf
        transformed["helix_axis"] = axis_wrf

        # Transform up vector (default is Y-axis of tool)
        up_trf = np.array([0.0, 1.0, 0.0])  # Tool's Y-axis
        up_wrf = tool_pose.R @ up_trf
        transformed["up_vector"] = up_wrf

    # SMOOTH_SPLINE - Transform waypoints
    elif command_type == "SMOOTH_SPLINE":
        if "waypoints" in params:
            transformed_waypoints = []
            for wp in params["waypoints"]:
                transformed_waypoints.append(pose6_trf_to_wrf(wp, tool_pose))
            transformed["waypoints"] = transformed_waypoints

    # SMOOTH_BLEND - Transform all segment definitions
    elif command_type == "SMOOTH_BLEND":
        if "segments" in params:
            transformed_segments = []
            for seg in params["segments"]:
                seg_transformed = seg.copy()

                # Transform based on segment type
                if seg["type"] == "LINE":
                    if "end" in seg:
                        seg_transformed["end"] = pose6_trf_to_wrf(seg["end"], tool_pose)

                elif seg["type"] == "ARC":
                    if "end" in seg:
                        seg_transformed["end"] = pose6_trf_to_wrf(seg["end"], tool_pose)

                    if "center" in seg:
                        # Create a temporary params dict to use the helper
                        seg_params = {"center": seg["center"]}
                        transform_center_trf_to_wrf(seg_params, tool_pose, seg_transformed)

                    # Transform plane normal if specified
                    if "plane" in seg:
                        normal_trf = PLANE_NORMALS_TRF[seg["plane"]]
                        normal_wrf = tool_pose.R @ normal_trf
                        seg_transformed["normal_vector"] = normal_wrf

                elif seg["type"] == "CIRCLE":
                    if "center" in seg:
                        # Create a temporary params dict to use the helper
                        seg_params = {"center": seg["center"]}
                        transform_center_trf_to_wrf(seg_params, tool_pose, seg_transformed)

                    if "plane" in seg:
                        normal_trf = PLANE_NORMALS_TRF[seg["plane"]]
                        normal_wrf = tool_pose.R @ normal_trf
                        seg_transformed["normal_vector"] = normal_wrf

                elif seg["type"] == "SPLINE":
                    if "waypoints" in seg:
                        transformed_wps = []
                        for wp in seg["waypoints"]:
                            transformed_wps.append(pose6_trf_to_wrf(wp, tool_pose))
                        seg_transformed["waypoints"] = transformed_wps

                transformed_segments.append(seg_transformed)
            transformed["segments"] = transformed_segments

    # Generic transformations for any command with these parameters
    if "start_pose" in params:
        transformed["start_pose"] = pose6_trf_to_wrf(params["start_pose"], tool_pose)

    return transformed


__all__ = [
    "PLANE_NORMALS_TRF",
    "point_trf_to_wrf_mm",
    "pose6_trf_to_wrf",
    "se3_to_pose6_mm_deg",
    "transform_center_trf_to_wrf",
    "transform_start_pose_if_needed",
    "transform_command_params_to_wrf",
]
