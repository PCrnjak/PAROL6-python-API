# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PAROL6 Python API is a lightweight client and controller for PAROL6 6-DOF robot arms. It provides:
- **AsyncRobotClient**: Async UDP client for robot control
- **RobotClient**: Synchronous wrapper for imperative scripts
- **Controller**: Fixed-rate control loop with serial/simulator transport

## Architecture

```
┌─────────────────────────────────────────┐
│           Client Application            │
│  (AsyncRobotClient / RobotClient)       │
└──────────────────┬──────────────────────┘
         UDP commands (port 5001)
         Multicast status (239.255.0.101:50510)
┌──────────────────▼──────────────────────┐
│              Controller                 │
│  (parol6/server/controller.py)          │
│  - 100 Hz control loop (configurable)   │
│  - Command queue & execution            │
│  - Status multicast broadcasting        │
└──────────────────┬──────────────────────┘
         Serial (3 Mbaud) or MockSerial
┌──────────────────▼──────────────────────┐
│     Robot Hardware / Simulator          │
└─────────────────────────────────────────┘
```

## Build & Test Commands

```bash
# Development setup
pip install -e .[dev]
pre-commit install

# Linting & formatting
ruff check .
ruff format .
mypy parol6/

# Run all pre-commit hooks
pre-commit run -a

# Testing (simulator used by default via conftest.py — do NOT prefix with env vars)
pytest

# Run specific test file
pytest tests/unit/test_wire.py -v
```

**IMPORTANT: Do NOT prefix `pytest` commands with environment variables like `PAROL6_FAKE_SERIAL=1 pytest ...`. The conftest.py already configures `PAROL6_FAKE_SERIAL=1`. Just run `pytest` directly.**

## Controller CLI

```bash
# Start controller
parol6-server --log-level=INFO

# With explicit serial port
parol6-server --serial=/dev/ttyUSB0 --log-level=DEBUG

# Verbosity shortcuts: -v (INFO), -vv (DEBUG), -vvv (TRACE)
```

## Key Modules

- **`parol6/client/async_client.py`**: Primary API - async UDP client with motion commands, queries, and status streaming
- **`parol6/server/controller.py`**: Controller with fixed-rate loop and command execution
- **`parol6/commands/`**: Polymorphic command classes using `@register_command("NAME")` decorator
- **`parol6/protocol/wire.py`**: Binary frame packing/unpacking (START=0xFFFFFF, END=0x0102)
- **`parol6/PAROL6_ROBOT.py`**: Robot kinematics config, DH parameters, joint limits, tool transforms

## Adding a New Command

1. Create a class in `parol6/commands/` and decorate with `@register_command("NAME")`
2. Implement `match(parts)`, `setup(state)`, and `tick(state)` lifecycle methods
3. For motion commands, set `streamable=True` if supporting high-rate streaming
4. Use helpers from `parol6/commands/base.py` (parsers, motion profiles)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PAROL6_CONTROL_RATE_HZ` | 100 | Control loop frequency |
| `PAROL6_STATUS_RATE_HZ` | 50 | Status broadcast rate (tests use 20 Hz) |
| `PAROL6_FAKE_SERIAL` | 0 | Enable simulator (no hardware) |
| `PAROL6_COM_PORT` | auto | Serial port override |
| `PAROL_TRACE` | 0 | Enable TRACE logging |

## Simulator Mode

- Uses `MockSerialTransport` to emulate robot dynamics without hardware
- Toggle via client: `simulator_on()` / `simulator_off()`
- Controller updates `PAROL6_FAKE_SERIAL` and reinitializes transport seamlessly
- **Important**: Simulator cannot guarantee hardware success—motor/current limits may cause failures on real robot

## Kinematics Notes

- Uses numerical IK via pinokin (C++/Pinocchio with nanobind Python bindings)
- J4 is particularly sensitive—some cartesian targets may fail to solve
- Update `PAROL6_ROBOT.py` for modified hardware (gear ratios, limits, DH params)

## Test Markers

- `@pytest.mark.unit`: Isolated component tests
- `@pytest.mark.integration`: Component interaction tests (uses simulator)

## Performance Warnings

If you see `Control loop avg period degraded by +XX%`:
- Reduce `PAROL6_CONTROL_RATE_HZ`
- Disable TRACE logging (significant overhead)
- Avoid heavy background tasks during motion

## Hot Path Rules

`execute_step()` and `tick()` run at 100 Hz. **No heap allocations** in these methods.

For streamable commands (`streamable = True`), `do_setup()` also runs at high frequency (50 Hz from UI) via `assign_params()` + `do_setup()` fast-path. Treat it as a hot path too.

- No `list(...)`, `[x for x in ...]`, `dict(...)`, or other container construction
- No string formatting or f-strings (except in error/stop paths that run once)
- Pre-allocate all buffers in `__init__`
- `ndarray[:] = list` is fine — numpy writes into existing buffer in-place
- Use `dest[:] = src` for array copies. Only use `np.copyto(dest, src, casting=...)` when casting is needed — it's slower otherwise

## Code Style

- **Comments**: Describe the final code state, not what changed. Avoid "changed X to Y" or "added this because..." comments.
- **Git commits/PRs**: Keep messages concise and factual. No emoji, no "Generated by..." footers, no co-author boilerplate.
- **Type annotations**: Fix type errors properly instead of using `# type: ignore`. Prefer:
  - `@overload` decorators for functions with different return types based on input
  - `assert` statements to narrow types after None checks
  - `cast()` from typing when the type is known but mypy can't infer it
  - `np.atleast_1d()` or similar to guarantee array returns from numpy functions
  - Only use `# type: ignore` as a last resort for genuine mypy/library limitations (e.g., numpy's `ArrayLike` being too broad)

## Testing Guidelines

Prefer fewer, comprehensive integration tests that mimic manual testing over a large number of unit tests. We have no code coverage requirements—the goal is working features, not metrics.
- **NEVER** run long test suites and only capture a few lines of output (e.g. `| tail -5` or `| grep passed`). This wastes time when you have to re-run to see failures.
- Always capture enough output to see BOTH the summary line AND any failure tracebacks in a single run. Use `tail -40` or similar.
- For background test runs, just let the full output come through.
- **NEVER run parol6 and web commander test suites in parallel** — no proper isolation, they share resources and have timing issues when resource-constrained. Always run sequentially.
