"""
Tool Configuration Module

Defines available end-effector tools for the PAROL6 robot with their transforms and visualization data.
Tools are swappable at runtime and affect both kinematics and visualization.
"""

import numpy as np
from spatialmath import SE3
from typing import Dict, List, Any


TOOL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "NONE": {
        "name": "No Tool",
        "description": "Bare flange - no tool attached",
        "transform": np.eye(4),
        "stl_files": []
    },
    "PNEUMATIC": {
        "name": "Pneumatic Gripper",
        "description": "Pneumatic gripper assembly",
        "transform": (SE3(-0.04525, 0, 0) @ SE3.Rx(np.pi)).A,
        "stl_files": [
            {
                "file": "pneumatic_gripper_assembly.STL",
                "origin": [0, 0, 0],
                "rpy": [0, 0, 0]
            }
        ]
    }
}


def get_tool_transform(tool_name: str) -> np.ndarray:
    """
    Get the 4x4 transformation matrix for a tool.
    
    Parameters
    ----------
    tool_name : str
        Name of the tool (must be in TOOL_CONFIGS)
    
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix from flange to tool TCP
    
    Raises
    ------
    ValueError
        If tool_name is not recognized
    """
    if tool_name not in TOOL_CONFIGS:
        raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {list_tools()}")
    
    return TOOL_CONFIGS[tool_name]["transform"]


def list_tools() -> List[str]:
    """
    Get list of available tool names.
    
    Returns
    -------
    List[str]
        List of available tool configuration names
    """
    return list(TOOL_CONFIGS.keys())


def get_tool_info(tool_name: str) -> Dict[str, Any]:
    """
    Get complete configuration for a tool.
    
    Parameters
    ----------
    tool_name : str
        Name of the tool
    
    Returns
    -------
    Dict[str, Any]
        Complete tool configuration dictionary
    
    Raises
    ------
    ValueError
        If tool_name is not recognized
    """
    if tool_name not in TOOL_CONFIGS:
        raise ValueError(f"Unknown tool '{tool_name}'. Available tools: {list_tools()}")
    
    return TOOL_CONFIGS[tool_name]
