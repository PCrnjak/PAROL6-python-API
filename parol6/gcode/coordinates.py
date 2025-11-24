"""
Work Coordinate System Implementation for GCODE

Manages G54-G59 work coordinate systems and transformations between
work coordinates, machine coordinates, and robot coordinates.
"""

import json
import os


class WorkCoordinateSystem:
    """Manages work coordinate systems and transformations"""

    def __init__(self, config_file: str | None = None):
        """
        Initialize work coordinate system

        Args:
            config_file: Path to JSON file for persistent storage
        """
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), "work_coordinates.json"
        )

        # Initialize default work coordinate offsets (in mm)
        self.offsets = {
            "G54": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            "G55": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            "G56": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            "G57": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            "G58": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
            "G59": {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0, "B": 0.0, "C": 0.0},
        }

        # Tool offsets
        self.tool_offsets = {
            0: {"Z": 0.0},  # No tool
            1: {"Z": 0.0},  # Tool 1
            2: {"Z": 0.0},  # Tool 2
            # Add more tools as needed
        }

        # Current active coordinate system
        self.active_system = "G54"

        # Current tool number
        self.active_tool = 0

        # Load saved offsets if they exist
        self.load_offsets()

    def set_offset(self, coord_system: str, axis: str, value: float) -> bool:
        """
        Set work coordinate offset for a specific axis

        Args:
            coord_system: Work coordinate system (G54-G59)
            axis: Axis name (X, Y, Z, A, B, C)
            value: Offset value in mm

        Returns:
            True if successful, False otherwise
        """
        if coord_system not in self.offsets:
            return False

        if axis not in self.offsets[coord_system]:
            return False

        self.offsets[coord_system][axis] = value
        self.save_offsets()
        return True

    def get_offset(self, coord_system: str | None = None) -> dict[str, float]:
        """
        Get work coordinate offset

        Args:
            coord_system: Work coordinate system (G54-G59) or None for active

        Returns:
            Dictionary of axis offsets
        """
        system = coord_system or self.active_system
        return self.offsets.get(system, {}).copy()

    def set_active_system(self, coord_system: str) -> bool:
        """
        Set the active work coordinate system

        Args:
            coord_system: Work coordinate system (G54-G59)

        Returns:
            True if successful, False otherwise
        """
        if coord_system in self.offsets:
            self.active_system = coord_system
            return True
        return False

    def set_tool_offset(self, tool_number: int, z_offset: float) -> None:
        """
        Set tool length offset

        Args:
            tool_number: Tool number
            z_offset: Z-axis offset in mm
        """
        if tool_number not in self.tool_offsets:
            self.tool_offsets[tool_number] = {}
        self.tool_offsets[tool_number]["Z"] = z_offset
        self.save_offsets()

    def get_tool_offset(self, tool_number: int | None = None) -> float:
        """
        Get tool length offset

        Args:
            tool_number: Tool number or None for active tool

        Returns:
            Z-axis offset in mm
        """
        tool = tool_number if tool_number is not None else self.active_tool
        return self.tool_offsets.get(tool, {}).get("Z", 0.0)

    def work_to_machine(
        self,
        work_pos: dict[str, float],
        coord_system: str | None = None,
        apply_tool_offset: bool = True,
    ) -> dict[str, float]:
        """
        Convert work coordinates to machine coordinates

        Args:
            work_pos: Position in work coordinates
            coord_system: Work coordinate system or None for active
            apply_tool_offset: Whether to apply tool offset

        Returns:
            Position in machine coordinates
        """
        system = coord_system or self.active_system
        offset = self.get_offset(system)
        machine_pos = {}

        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            if axis in work_pos:
                machine_pos[axis] = work_pos[axis] + offset.get(axis, 0.0)

                # Apply tool offset to Z axis
                if axis == "Z" and apply_tool_offset:
                    machine_pos[axis] += self.get_tool_offset()

        return machine_pos

    def machine_to_work(
        self,
        machine_pos: dict[str, float],
        coord_system: str | None = None,
        apply_tool_offset: bool = True,
    ) -> dict[str, float]:
        """
        Convert machine coordinates to work coordinates

        Args:
            machine_pos: Position in machine coordinates
            coord_system: Work coordinate system or None for active
            apply_tool_offset: Whether to apply tool offset

        Returns:
            Position in work coordinates
        """
        system = coord_system or self.active_system
        offset = self.get_offset(system)
        work_pos = {}

        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            if axis in machine_pos:
                work_pos[axis] = machine_pos[axis] - offset.get(axis, 0.0)

                # Remove tool offset from Z axis
                if axis == "Z" and apply_tool_offset:
                    work_pos[axis] -= self.get_tool_offset()

        return work_pos

    def apply_incremental(
        self, current_pos: dict[str, float], incremental: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply incremental movement to current position

        Args:
            current_pos: Current position
            incremental: Incremental movement values

        Returns:
            New position after incremental movement
        """
        new_pos = current_pos.copy()

        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            if axis in incremental:
                new_pos[axis] = current_pos.get(axis, 0.0) + incremental[axis]

        return new_pos

    def robot_to_gcode_coords(self, robot_pos: list[float]) -> dict[str, float]:
        """
        Convert robot Cartesian position to GCODE coordinates

        Args:
            robot_pos: Robot position [X, Y, Z, RX, RY, RZ] in mm and degrees

        Returns:
            GCODE coordinates {X, Y, Z, A, B, C}
        """
        # For PAROL6, the mapping is straightforward
        # X, Y, Z are Cartesian positions
        # A, B, C are rotational axes (RX, RY, RZ)

        gcode_coords = {}

        if len(robot_pos) >= 3:
            gcode_coords["X"] = robot_pos[0]
            gcode_coords["Y"] = robot_pos[1]
            gcode_coords["Z"] = robot_pos[2]

        if len(robot_pos) >= 6:
            gcode_coords["A"] = robot_pos[3]  # RX
            gcode_coords["B"] = robot_pos[4]  # RY
            gcode_coords["C"] = robot_pos[5]  # RZ

        return gcode_coords

    def gcode_to_robot_coords(self, gcode_pos: dict[str, float]) -> list[float]:
        """
        Convert GCODE coordinates to robot Cartesian position

        Args:
            gcode_pos: GCODE coordinates {X, Y, Z, A, B, C}

        Returns:
            Robot position [X, Y, Z, RX, RY, RZ] in mm and degrees
        """
        robot_pos = [
            gcode_pos.get("X", 0.0),
            gcode_pos.get("Y", 0.0),
            gcode_pos.get("Z", 0.0),
            gcode_pos.get("A", 0.0),  # RX
            gcode_pos.get("B", 0.0),  # RY
            gcode_pos.get("C", 0.0),  # RZ
        ]

        return robot_pos

    def save_offsets(self) -> None:
        """Save offsets to JSON file"""
        data = {
            "work_offsets": self.offsets,
            "tool_offsets": self.tool_offsets,
            "active_system": self.active_system,
            "active_tool": self.active_tool,
        }

        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_offsets(self) -> None:
        """Load offsets from JSON file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file) as f:
                    data = json.load(f)

                self.offsets = data.get("work_offsets", self.offsets)
                self.tool_offsets = data.get("tool_offsets", self.tool_offsets)
                self.active_system = data.get("active_system", "G54")
                self.active_tool = data.get("active_tool", 0)
            except Exception as e:
                print(f"Error loading work coordinate offsets: {e}")

    def reset_offset(self, coord_system: str | None = None) -> None:
        """
        Reset work coordinate offset to zero

        Args:
            coord_system: System to reset, or None to reset all
        """
        if coord_system:
            if coord_system in self.offsets:
                self.offsets[coord_system] = {
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "A": 0.0,
                    "B": 0.0,
                    "C": 0.0,
                }
        else:
            # Reset all systems
            for system in self.offsets:
                self.offsets[system] = {
                    "X": 0.0,
                    "Y": 0.0,
                    "Z": 0.0,
                    "A": 0.0,
                    "B": 0.0,
                    "C": 0.0,
                }

        self.save_offsets()

    def set_current_as_zero(
        self, machine_pos: dict[str, float], coord_system: str | None = None
    ) -> None:
        """
        Set current machine position as zero in work coordinates

        Args:
            machine_pos: Current machine position
            coord_system: Work coordinate system or None for active
        """
        system = coord_system or self.active_system

        # Set offsets such that current machine position becomes 0,0,0
        for axis in ["X", "Y", "Z", "A", "B", "C"]:
            if axis in machine_pos:
                self.offsets[system][axis] = machine_pos[axis]

        self.save_offsets()
