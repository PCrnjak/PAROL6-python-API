"""
Commands Module for PAROL6 Robot Controller
Contains all command classes for robot control operations
"""

# Import helper functions and constants
from parol6.utils.ik import (
    CommandValue,
    normalize_angle,
    unwrap_angles,
    calculate_adaptive_tolerance,
    solve_ik_with_adaptive_tol_subdivision,
    quintic_scaling,
    AXIS_MAP
)

# Import basic commands
from .basic_commands import (
    HomeCommand,
    JogCommand,
    MultiJogCommand
)

# Import cartesian commands
from .cartesian_commands import (
    CartesianJogCommand,
    MovePoseCommand,
    MoveCartCommand
)

# Import joint commands
from .joint_commands import (
    MoveJointCommand
)

# Import gripper commands
from .gripper_commands import (
    GripperCommand
)

# Import utility commands
from .utility_commands import (
    DelayCommand
)

from .smooth_commands import (
    transform_command_params_to_wrf,
    BaseSmoothMotionCommand,
    SmoothTrajectoryCommand,
    SmoothCircleCommand,
    SmoothArcCenterCommand,
    SmoothArcParamCommand,
    SmoothHelixCommand,
    SmoothSplineCommand,
    SmoothBlendCommand,
    SmoothWaypointsCommand
)

# Export all command classes
__all__ = [
    # Helper functions
    'CommandValue',
    'normalize_angle',
    'unwrap_angles',
    'calculate_adaptive_tolerance',
    'solve_ik_with_adaptive_tol_subdivision',
    'quintic_scaling',
    'AXIS_MAP',

    # Basic commands
    'HomeCommand',
    'JogCommand',
    'MultiJogCommand',

    # Cartesian commands
    'CartesianJogCommand',
    'MovePoseCommand',
    'MoveCartCommand',

    # Joint commands
    'MoveJointCommand',

    # Gripper commands
    'GripperCommand',

    # Utility commands
    'DelayCommand',
    'transform_command_params_to_wrf',
    'BaseSmoothMotionCommand',
    'SmoothTrajectoryCommand',
    'SmoothCircleCommand',
    'SmoothArcCenterCommand',
    'SmoothArcParamCommand',
    'SmoothHelixCommand',
    'SmoothSplineCommand',
    'SmoothBlendCommand',
    'SmoothWaypointsCommand'
    ]
