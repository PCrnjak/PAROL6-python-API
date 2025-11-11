"""
GCODE Parser for PAROL6 Robot

Tokenizes and parses GCODE lines into structured data.
Supports standard GCODE syntax including G-codes, M-codes, and parameters.
"""

import re
from dataclasses import dataclass


@dataclass
class GcodeToken:
    """Represents a parsed GCODE token"""

    code_type: str  # 'G', 'M', 'T', 'N', etc.
    code_number: float  # The numeric value
    parameters: dict[str, float]  # Associated parameters (X, Y, Z, F, etc.)
    comment: str | None = None
    line_number: int | None = None
    raw_line: str = ""

    def __str__(self):
        result = f"{self.code_type}{self.code_number:.10g}".rstrip("0").rstrip(".")
        for key, val in self.parameters.items():
            result += f" {key}{val:.10g}".rstrip("0").rstrip(".")
        if self.comment:
            result += f" ; {self.comment}"
        return result


class GcodeParser:
    """GCODE parser that tokenizes and validates GCODE lines"""

    # Regex patterns for parsing
    COMMENT_PATTERN = re.compile(r"\((.*?)\)|;(.*)$")
    LINE_NUMBER_PATTERN = re.compile(r"^N(\d+)", re.IGNORECASE)
    CODE_PATTERN = re.compile(r"([GMT])(\d+(?:\.\d+)?)", re.IGNORECASE)
    PARAM_PATTERN = re.compile(r"([XYZABCIJKRFSPQD])([+-]?\d*\.?\d+)", re.IGNORECASE)

    # Valid GCODE commands we support
    SUPPORTED_G_CODES = {
        0: "Rapid positioning",
        1: "Linear interpolation",
        2: "Clockwise arc",
        3: "Counter-clockwise arc",
        4: "Dwell",
        17: "XY plane selection",
        18: "XZ plane selection",
        19: "YZ plane selection",
        20: "Inch units",
        21: "Millimeter units",
        28: "Return to home",
        54: "Work coordinate 1",
        55: "Work coordinate 2",
        56: "Work coordinate 3",
        57: "Work coordinate 4",
        58: "Work coordinate 5",
        59: "Work coordinate 6",
        90: "Absolute positioning",
        91: "Incremental positioning",
    }

    SUPPORTED_M_CODES = {
        0: "Program stop",
        1: "Optional stop",
        3: "Spindle/Gripper on CW",
        4: "Spindle/Gripper on CCW",
        5: "Spindle/Gripper off",
        30: "Program end",
    }

    def __init__(self):
        self.line_count = 0
        self.errors = []

    def parse_line(self, line: str) -> list[GcodeToken]:
        """
        Parse a single line of GCODE into tokens

        Args:
            line: Raw GCODE line

        Returns:
            List of GcodeToken objects parsed from the line
        """
        self.line_count += 1
        tokens: list[GcodeToken] = []

        # Store original line
        original_line = line.strip()
        if not original_line:
            return tokens

        # Extract and remove comments
        comment = None
        comment_match = self.COMMENT_PATTERN.search(line)
        if comment_match:
            comment = comment_match.group(1) or comment_match.group(2)
            line = self.COMMENT_PATTERN.sub("", line)

        # Extract line number if present
        line_number = None
        line_num_match = self.LINE_NUMBER_PATTERN.match(line)
        if line_num_match:
            line_number = int(line_num_match.group(1))
            line = line[line_num_match.end() :]

        # Convert to uppercase for parsing
        line = line.upper().strip()

        # Parse all parameters first
        parameters: dict[str, float] = {}
        for match in self.PARAM_PATTERN.finditer(line):
            param_letter = match.group(1)
            try:
                param_value = float(match.group(2))
                # Validate feed rate
                if param_letter == "F" and param_value <= 0:
                    self.errors.append(f"Line {self.line_count}: Invalid feed rate: {param_value}")
                    continue
                # Validate spindle speed
                if param_letter == "S" and param_value < 0:
                    self.errors.append(
                        f"Line {self.line_count}: Invalid spindle speed: {param_value}"
                    )
                    continue
                parameters[param_letter] = param_value
            except ValueError:
                self.errors.append(
                    f"Line {self.line_count}: Invalid numeric value for {param_letter}: {match.group(2)}"
                )

        # Parse G and M codes
        codes_found: list[tuple[str, float]] = []
        for match in self.CODE_PATTERN.finditer(line):
            code_type = match.group(1)
            code_number = float(match.group(2))
            codes_found.append((code_type, code_number))

        # Create tokens for each code found
        if codes_found:
            for code_type, code_number in codes_found:
                # For motion commands, include coordinate parameters
                if code_type == "G" and code_number in [0, 1, 2, 3]:
                    motion_params = {
                        k: v
                        for k, v in parameters.items()
                        if k in ["X", "Y", "Z", "A", "B", "C", "I", "J", "K", "R", "F"]
                    }
                    token = GcodeToken(
                        code_type=code_type,
                        code_number=code_number,
                        parameters=motion_params,
                        comment=comment,
                        line_number=line_number,
                        raw_line=original_line,
                    )
                else:
                    # Other codes get remaining parameters
                    token = GcodeToken(
                        code_type=code_type,
                        code_number=code_number,
                        parameters={
                            k: v
                            for k, v in parameters.items()
                            if k not in ["X", "Y", "Z", "A", "B", "C", "I", "J", "K", "R", "F"]
                        },
                        comment=comment,
                        line_number=line_number,
                        raw_line=original_line,
                    )
                tokens.append(token)

        # Handle standalone feed rate
        elif "F" in parameters and not codes_found:
            token = GcodeToken(
                code_type="F",
                code_number=parameters["F"],
                parameters={},
                comment=comment,
                line_number=line_number,
                raw_line=original_line,
            )
            tokens.append(token)

        # Handle comment-only lines
        elif comment:
            token = GcodeToken(
                code_type="COMMENT",
                code_number=0,
                parameters={},
                comment=comment,
                line_number=line_number,
                raw_line=original_line,
            )
            tokens.append(token)

        return tokens

    def validate_token(self, token: GcodeToken) -> tuple[bool, str | None]:
        """
        Validate a GCODE token

        Args:
            token: GcodeToken to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if token.code_type == "G":
            if token.code_number not in self.SUPPORTED_G_CODES:
                return False, f"Unsupported G-code: G{token.code_number}"

            # Validate required parameters for motion commands
            if token.code_number in [0, 1]:  # Linear motion
                if not any(k in token.parameters for k in ["X", "Y", "Z", "A", "B", "C"]):
                    return False, f"G{token.code_number} requires at least one coordinate"

            elif token.code_number in [2, 3]:  # Arc motion
                if not any(k in token.parameters for k in ["X", "Y", "Z"]):
                    return False, f"G{token.code_number} requires endpoint coordinates"
                if not (
                    ("I" in token.parameters or "J" in token.parameters) or "R" in token.parameters
                ):
                    return False, f"G{token.code_number} requires either IJK offsets or R radius"

            elif token.code_number == 4:  # Dwell
                if "P" not in token.parameters and "S" not in token.parameters:
                    return False, "G4 dwell requires P (milliseconds) or S (seconds) parameter"

        elif token.code_type == "M":
            if token.code_number not in self.SUPPORTED_M_CODES:
                return False, f"Unsupported M-code: M{token.code_number}"

        elif token.code_type in ["F", "T", "S", "COMMENT"]:
            # These are always valid if parsed
            pass

        else:
            return False, f"Unknown code type: {token.code_type}"

        return True, None

    def parse_program(self, program: str | list[str]) -> list[GcodeToken]:
        """
        Parse a complete GCODE program

        Args:
            program: Either a string with newlines or a list of lines

        Returns:
            List of all tokens in the program
        """
        if isinstance(program, str):
            lines = program.split("\n")
        else:
            lines = program

        all_tokens = []
        self.errors = []
        self.line_count = 0

        for line in lines:
            try:
                tokens = self.parse_line(line)
                for token in tokens:
                    is_valid, error_msg = self.validate_token(token)
                    if not is_valid:
                        self.errors.append(f"Line {self.line_count}: {error_msg}")
                    else:
                        all_tokens.append(token)
            except Exception as e:
                self.errors.append(f"Line {self.line_count}: Parse error - {e!s}")

        return all_tokens

    def get_errors(self) -> list[str]:
        """Get list of parsing errors"""
        return self.errors
