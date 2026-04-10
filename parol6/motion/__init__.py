"""
Motion pipeline for trajectory generation and execution.

This module provides a unified pipeline for all motion commands:
- Cartesian commands (MoveCart, MoveCartRelTrf)
- Joint commands (MovePose, MoveJoint)
- Smooth shapes (Circle, Arc, Helix, Spline)

All commands produce a JointPath that gets converted to a Trajectory
via time-optimal path parameterization (TOPP-RA).

Streaming executors provide real-time jerk-limited motion for jogging.

Geometry generators provide path geometry for visualization and preview.
"""

from parol6.motion.geometry import (
    CircularMotion,
    SplineMotion,
    blend_path_into,
    build_composite_cartesian_path,
    build_composite_joint_path,
    compute_circle_from_3_points,
    joint_path_to_tcp_poses,
)
from parol6.motion.streaming_executors import (
    CartesianStreamingExecutor,
    RuckigExecutorBase,
    StreamingExecutor,
)
from parol6.motion.trajectory import (
    JointPath,
    ProfileType,
    Trajectory,
    TrajectoryBuilder,
)

__all__ = [
    # Trajectory pipeline
    "JointPath",
    "Trajectory",
    "TrajectoryBuilder",
    "ProfileType",
    # Streaming executors
    "StreamingExecutor",
    "CartesianStreamingExecutor",
    "RuckigExecutorBase",
    # Geometry generators
    "CircularMotion",
    "SplineMotion",
    "joint_path_to_tcp_poses",
    # Blend infrastructure
    "blend_path_into",
    "build_composite_cartesian_path",
    "build_composite_joint_path",
    "compute_circle_from_3_points",
]
