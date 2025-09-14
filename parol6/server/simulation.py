"""
Simulation module for FAKE_SERIAL mode.

This module provides simulation capabilities for testing without hardware,
simulating robot motion and state updates at 100Hz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import parol6.PAROL6_ROBOT as PAROL6_ROBOT


@dataclass
class SimulationState:
    """State for FAKE_SERIAL simulation mode."""
    enabled: bool = False
    position_in: List[int] = field(default_factory=lambda: [0] * 6)
    speed_in: List[int] = field(default_factory=lambda: [0] * 6)
    homed_in: List[int] = field(default_factory=lambda: [1] * 8)  # Simulate homed state
    temperature_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    position_error_in: List[int] = field(default_factory=lambda: [0] * 8)
    io_in: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 1, 0, 0, 0])  # E-stop released (bit 4)
    gripper_data_in: List[int] = field(default_factory=lambda: [0] * 6)
    update_rate: float = 0.01
    last_update: float = 0.0
    homing_countdown: int = 0
    
    def __post_init__(self):
        """Initialize with current time."""
        self.last_update = time.time()


def simulate_robot_step(state: SimulationState, 
                       command_code: int,
                       position_out: List[int],
                       speed_out: List[int],
                       dt: float) -> None:
    """
    Simulate one step of robot motion.
    
    Updates Position_in/Speed_in based on Command_out and Speed_out/Position_out.
    
    Args:
        state: Simulation state to update
        command_code: Current command code (123 for jog, 156 for position)
        position_out: Target positions
        speed_out: Target speeds
        dt: Time delta since last update
    """
    # Manage homing countdown: if active, count down and finalize homing when it reaches zero
    if state.homing_countdown > 0:
        state.homing_countdown -= 1
        if state.homing_countdown == 0:
            for i in range(6):
                state.homed_in[i] = 1

    # Ensure E-stop is not pressed (bit 4 = 1 means not pressed)
    if len(state.io_in) > 4:
        state.io_in[4] = 1
    
    # Simulate motion based on command type
    if command_code == 123:  # JOG command
        # Speed control: integrate Speed_out over dt
        for i in range(6):
            v = int(speed_out[i])
            # Clamp to max speed
            max_v = int(PAROL6_ROBOT.Joint_max_speed[i])
            if v > max_v:
                v = max_v
            elif v < -max_v:
                v = -max_v
            
            # Update position
            new_pos = int(state.position_in[i] + v * dt)
            
            # Clamp to joint limits
            jmin, jmax = PAROL6_ROBOT.Joint_limits_steps[i]
            if new_pos < jmin:
                new_pos = jmin
                v = 0
            elif new_pos > jmax:
                new_pos = jmax
                v = 0
            
            state.speed_in[i] = v
            state.position_in[i] = new_pos
            
    elif command_code == 156:  # MOVE command
        # Position control: move toward Position_out
        for i in range(6):
            target = position_out[i]
            current = state.position_in[i]
            err = int(target - current)
            
            if err == 0:
                state.speed_in[i] = 0
                continue
            
            # Calculate step size based on max speed
            max_step = int(PAROL6_ROBOT.Joint_max_speed[i] * dt)
            if max_step < 1:
                max_step = 1
            
            # Move toward target
            step = max(-max_step, min(max_step, err))
            new_pos = current + step
            
            # Clamp to joint limits
            jmin, jmax = PAROL6_ROBOT.Joint_limits_steps[i]
            if new_pos < jmin:
                new_pos = jmin
                step = 0
            elif new_pos > jmax:
                new_pos = jmax
                step = 0
            
            state.position_in[i] = int(new_pos)
            state.speed_in[i] = int(step / dt) if dt > 0 else 0
            
    elif command_code == 100:  # HOME command
        # Start homing: mark unhomed and set a short countdown to complete homing
        if state.homing_countdown == 0:
            for i in range(6):
                state.homed_in[i] = 0
            # complete homing after ~0.2s worth of simulation steps
            steps = int(0.2 / dt) if dt and dt > 0 else int(0.2 / state.update_rate) if state.update_rate > 0 else 20
            state.homing_countdown = max(1, steps)
        # Also ensure speeds go to zero while homing initiates
        for i in range(6):
            state.speed_in[i] = 0
        
    else:
        # Idle or other commands: hold position
        for i in range(6):
            state.speed_in[i] = 0


def simulate_homing(state: SimulationState) -> None:
    """
    Simulate homing sequence by scheduling a brief unhomed period
    followed by marking joints as homed.
    """
    # Bring joints to zero position and zero velocity
    for i in range(6):
        state.position_in[i] = 0
        state.speed_in[i] = 0
        state.homed_in[i] = 0
    # If not already counting down, schedule completion after ~0.2s
    if state.homing_countdown == 0:
        steps = int(0.2 / state.update_rate) if state.update_rate > 0 else 20
        state.homing_countdown = max(1, steps)


def simulate_motion(state: SimulationState,
                   command_code: int,
                   target_pos: List[int],
                   target_speed: List[int]) -> None:
    """
    High-level motion simulation.
    
    Updates simulation state based on command and targets.
    
    Args:
        state: Simulation state to update
        command_code: Command code to execute
        target_pos: Target positions for each joint
        target_speed: Target speeds for each joint
    """
    # Calculate time delta
    now = time.time()
    dt = now - state.last_update
    state.last_update = now
    
    # Simulate robot step
    simulate_robot_step(state, command_code, target_pos, target_speed, dt)


def create_simulation_state() -> SimulationState:
    """
    Create and initialize a simulation state.
    
    Returns:
        Initialized SimulationState instance
    """
    state = SimulationState(enabled=True)
    
    # Set initial positions to standby position (good for IK) instead of all zeros
    # Use PAROL6_ROBOT.Joints_standby_position_degree = [0,-90,180,0,0,180]
    standby_positions_steps = []
    for i in range(6):
        deg = PAROL6_ROBOT.Joints_standby_position_degree[i]
        steps = int(PAROL6_ROBOT.DEG2STEPS(deg, i))
        standby_positions_steps.append(steps)
        state.position_in[i] = steps
        state.homed_in[i] = 1
    
    # Ensure E-stop is not pressed
    state.io_in[4] = 1
    
    return state


def is_fake_serial_enabled() -> bool:
    """
    Check if FAKE_SERIAL mode is enabled via environment variable.
    
    Returns:
        True if FAKE_SERIAL is enabled, False otherwise
    """
    import os
    fake_serial = str(os.getenv("PAROL6_FAKE_SERIAL", "0")).lower()
    return fake_serial in ("1", "true", "yes", "on")
