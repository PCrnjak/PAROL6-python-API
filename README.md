# PAROL6 Python API

Python client and controller for PAROL6 robot arms, implementing the [waldoctl](https://github.com/Jepson2k/waldoctl) robot interface. Any application built against the waldoctl ABCs (e.g., [PAROL Web Commander](https://github.com/PCrnjak/PAROL-Web-Commander)) can use this package as a drop-in backend.

This package provides:
- **Robot** — unified entry point: lifecycle, kinematics (FK/IK), client factories (`waldoctl.Robot`)
- **AsyncRobotClient** — async UDP client for motion commands and status streaming (`waldoctl.RobotClient`)
- **RobotClient** — sync wrapper around the async client
- **DryRunRobotClient** — offline trajectory simulation
- **`parol6-server`** CLI for standalone controller operation

The controller speaks a msgpack-based UDP protocol and can run on the same machine or remotely.
Every command datagram carries a 4-byte request id ahead of the msgpack body, and the
OK / ERROR / RESPONSE reply echoes it, so a reply whose caller has already given up is
dropped instead of answering the next request. An id of 0 asks for no reply, which is
what streamed motion sends. Status broadcasts carry `PROTO_VERSION` in their second
slot: a client reading a status from another version raises `ProtocolVersionError`
naming both, rather than reporting the silence of a failed decode. Client and
controller are released together — there is no compatibility window between versions.

---

## Table of contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Architecture overview](#architecture-overview)
- [Control loop internals](#control-loop-internals)
- [Hot path rules](#hot-path-rules)
- [Motion profiles](#motion-profiles)
- [Command system](#command-system)
- [Kinematics and tools](#kinematics-and-tools)
- [Environment variables](#environment-variables)
- [Development setup](#development-setup)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Safety notes](#safety-notes)

---

## Installation

**Requirements:** Python >= 3.11 · Supported platforms: macOS (ARM64), Windows (AMD64), Linux (x86_64, aarch64)

```bash
pip install .
```

To launch the controller after installation:

```bash
parol6-server --log-level=INFO
```

## Quickstart

### Using the Robot class
```python
from parol6 import Robot, RobotClient

robot = Robot(host="127.0.0.1", port=5001)
robot.start()  # starts controller subprocess, blocks until ready
try:
    with RobotClient(host="127.0.0.1", port=5001) as client:
        print("ping:", client.ping())
        print("pose:", client.get_pose())
finally:
    robot.stop()
```

The `Robot` class can also be used as a context manager:
```python
with Robot() as robot:
    with RobotClient() as client:
        client.home(wait=True)
```

### Async client
```python
import asyncio
from parol6 import AsyncRobotClient

async def main():
    async with AsyncRobotClient(host="127.0.0.1", port=5001) as client:
        ready = await client.wait_ready(timeout=3)
        print("server ready:", ready)
        print("ping:", await client.ping())
        status = await client.get_status()
        print("status keys:", list(status.keys()) if status else None)

asyncio.run(main())
```

### Sync client (convenience wrapper)
```python
from parol6 import RobotClient

with RobotClient(host="127.0.0.1", port=5001) as client:
    print("ping:", client.ping())
    print("pose:", client.get_pose())
```

### Examples

See the [`examples/`](examples/) directory for runnable scripts:
- `sync_client_quickstart.py` -- basic sync client usage (ping, query)
- `async_client_quickstart.py` -- async client with status streaming and motion
- `manage_server_demo.py` -- starting/stopping the controller programmatically
- `pick_and_place.py` -- pick-and-place cycle with electric gripper
- `draw_circle.py` -- curved motion commands (moveC, moveS, moveP)
- `zigzag_scan.py` -- raster scan pattern with blend radius for smooth corners
- `speed_comparison.py` -- timing different speeds and motion profiles

## Architecture overview

```mermaid
flowchart TB
    subgraph Client["Client Application"]
        ROB["Robot<br/>(lifecycle, kinematics, factories)"]
        ARC["AsyncRobotClient / RobotClient"]
        DRC["DryRunRobotClient<br/>(offline simulation)"]
    end

    subgraph Controller["Controller Process"]
        direction TB

        subgraph UDP["UDP Layer"]
            UDP_RX["UDPTransport<br/>recv (port 5001)"]
            UDP_TX["ACK/Response"]
        end

        subgraph CmdProc["Command Processing"]
            REG["Command Registry<br/>(auto-discover)"]
            QUEUE["Command Queue<br/>(max 100)"]
        end

        subgraph Planner["MotionPlanner (subprocess)"]
            direction TB
            PLAN_IN["command_queue"]
            PLAN_WORK["TrajectoryPlanner<br/>(path gen → IK chain → TOPPRA)"]
            PLAN_OUT["segment_queue"]

            PLAN_IN --> PLAN_WORK --> PLAN_OUT
        end

        subgraph MainLoop["Main Control Loop (100 Hz)"]
            direction TB
            RX_SERIAL["1. Read Serial Frame"]
            POLL["2. Poll UDP Commands"]
            STATUS["3. Broadcast Status<br/>(change-detection cache)"]
            ESTOP["4. E-Stop Check"]
            EXEC["5. Execute<br/>(SegmentPlayer or StreamingExecutor)"]
            TX_SERIAL["6. Write Serial Frame"]
            TIMING["7. Deadline Wait"]

            RX_SERIAL --> POLL --> STATUS --> ESTOP --> EXEC --> TX_SERIAL --> TIMING
        end
    end

    subgraph Transports["Transport Layer"]
        FACTORY["TransportFactory"]
        SERIAL["SerialTransport<br/>(3 Mbaud)"]
        MOCK["MockSerialTransport<br/>(shared memory IPC)"]
    end

    subgraph HW["Hardware / Simulator"]
        BOARD["PAROL6 Board"]
        SIM["Simulated Dynamics<br/>(subprocess)"]
    end

    %% Client to Controller
    ROB -->|"start / stop"| Controller
    ARC -->|"UDP commands"| UDP_RX
    UDP_TX -->|"ACK/response"| ARC
    STATUS -->|"STATUS multicast<br/>239.255.0.101:50510"| ARC

    %% Command flow
    UDP_RX --> REG --> QUEUE

    %% Planned path: queue → planner subprocess → segment player
    QUEUE -->|"planned moves<br/>(MoveJ, MoveL, etc.)"| PLAN_IN
    PLAN_OUT -->|"TrajectorySegment"| EXEC

    %% Streaming path: queue → executor directly (bypasses planner)
    QUEUE -->|"streaming cmds<br/>(JogJ, ServoJ, etc.)"| EXEC

    %% Transport to hardware
    TX_SERIAL --> FACTORY
    FACTORY --> SERIAL --> BOARD
    FACTORY --> MOCK --> SIM
```

### Component summary

- **Robot** (`parol6.robot`): Unified entry point — server lifecycle, kinematics (FK/IK), client factories, configuration
- **Client** (`parol6.client`): `AsyncRobotClient` (async UDP with built-in multicast status listener), `RobotClient` (sync wrapper), `DryRunRobotClient` (offline simulation)
- **Controller** (`parol6.server.controller`): Main loop with phase-based execution at 100 Hz, UDP command server, status broadcasting
- **MotionPlanner** (`parol6.server.motion_planner`): Separate subprocess for trajectory computation (TOPPRA, IK chains) — keeps the 100 Hz loop free. Only planned moves (MoveJ, MoveL, MoveC, MoveS, MoveP) go through the planner; streaming commands (JogJ, ServoJ, etc.) execute directly in the main loop
- **SegmentPlayer** (`parol6.server.segment_player`): Consumes computed trajectory segments in the control loop — indexes one waypoint per tick with zero allocation
- **StreamingExecutor** (`parol6.motion.streaming_executors`): Joint-space and Cartesian Ruckig-based executors for real-time jog/servo commands
- **Motion pipeline** (`parol6.motion`): Offline trajectory generation (TOPPRA, Ruckig, Quintic, Trapezoid, Linear) and online streaming executors
- **Transports** (`parol6.server.transports`): `SerialTransport` (hardware, 3 Mbaud), `MockSerialTransport` (simulator via shared memory IPC)
- **StatusCache** (`parol6.server.status_cache`): Change-detection cache with async IK worker for cartesian/joint enablement computation

### Why multicast status?

The controller pushes status via UDP multicast to avoid client-side polling, reduce command-channel contention, and support multiple observers (GUI, logging). Falls back to unicast when multicast is unavailable (`PAROL6_STATUS_TRANSPORT=UNICAST`).

### Simulator mode

Uses `MockSerialTransport` with shared memory IPC for subprocess isolation. Toggle via `simulator_on()` / `simulator_off()`. The simulator syncs to controller state on enable for pose continuity. **Note**: Simulation cannot guarantee hardware success—motor/current limits may cause failures on the real robot.

---

## Control loop internals

The main loop (`controller.py`) runs a fixed sequence of phases every tick:

1. **Read serial frame** — poll transport for incoming telemetry (position, I/O, gripper)
2. **Poll UDP commands** — non-blocking receive up to 25 messages per tick
3. **Broadcast status** — multicast at `STATUS_RATE_HZ` (default 50 Hz), skipped if status cache is stale
4. **E-Stop check** — hardware pin polling; on activation: cancel all motion, clear queue, send DISABLE to firmware. Auto-recovers on release
5. **Execute** — run SegmentPlayer (planned moves) or StreamingExecutor (jog/servo)
6. **Write serial frame** — pack output into 58-byte frame and transmit
7. **Deadline wait** — hybrid sleep + busy-loop to hit exact tick boundary

### Timing: hybrid sleep + busy-loop

The loop timer (`loop_timer.py`) uses a two-phase strategy for precise tick timing:

```
deadline = now + interval (10ms at 100Hz)

if time_remaining > busy_threshold (1ms):
    time.sleep(time_remaining - busy_threshold)    # OS sleep for bulk of wait

while time.perf_counter() < deadline:
    pass                                           # Busy-loop for final 1ms
```

OS `time.sleep()` has ~1-4ms jitter depending on platform and load. The busy-loop absorbs this jitter to hit deadline with sub-millisecond precision. The `PAROL6_BUSY_THRESHOLD_MS` env var (default 1.0) controls the crossover point.

### Planned vs streaming command paths

**Planned moves** (MoveJ, MoveL, MoveC, MoveS, MoveP, Home):
1. Command arrives via UDP → decoded → queued
2. Submitted to MotionPlanner subprocess via `command_queue`
3. Planner runs TrajectoryBuilder (TOPPRA, IK) — can take 10-500ms
4. Result sent back as `TrajectorySegment` via `segment_queue`
5. SegmentPlayer indexes one waypoint per tick: `Position_out[:] = trajectory_steps[step]`

**Streaming commands** (JogJ, JogL, ServoJ, ServoL):
1. Command arrives via UDP → **stream fast-path** (no full decode if type matches active command)
2. `assign_params()` updates target on existing command instance
3. `do_setup()` re-runs (also a hot path at ~50Hz for UI-driven jog)
4. StreamingExecutor ticks Ruckig for smooth interpolation
5. Per-tick IK solve for Cartesian commands (JogL, ServoL)

The fast-path avoids command object creation entirely — it reuses the active command instance and re-assigns parameters. This is critical for 50Hz jog streams.

## Hot path rules

The control loop and streaming command paths are latency-critical. GC pauses are the primary cause of loop timing degradation. The codebase enforces strict allocation discipline:

### Zero-allocation zones

`execute_step()` and `tick()` run at 100Hz. `do_setup()` for streamable commands runs at ~50Hz (UI sends at status rate). These are **zero heap allocation** zones:

- **No container construction**: No `list(...)`, `[x for x in ...]`, `dict(...)`, `set(...)`, or comprehensions
- **No string formatting**: No f-strings or `%` formatting (except error paths that run once)
- **No object creation**: No `dataclass()`, `namedtuple()`, or class instantiation
- **Pre-allocate all buffers in `__init__`**: numpy arrays, lists, memoryviews
- **In-place array ops**: `dest[:] = src` (numpy writes into existing buffer)
- **`np.copyto(dest, src, casting=...)` only when casting is needed** — it's slower than `dest[:] = src` otherwise

Pre-allocate all buffers in `__init__` and reuse them every tick via `dest[:] = src`. StreamingExecutor's `tick()` returns reused `list[float]` — callers must copy if they need values across ticks.

Performance-critical functions (unit conversions, serial frame packing, IK checks, SE3 ops, statistics) are Numba JIT-compiled with `@numba.njit(cache=True)`. First run takes 3-10s for compilation; `warmup_jit()` pre-compiles at startup. Subsequent runs use the cache (~100ms).

---

## Motion profiles

Set the motion profile for all moves:

```python
client.set_profile("TOPPRA")  # Default: time-optimal path-following
```

**Available profiles** (`SETPROFILE`):

| Profile | Description |
|---------|-------------|
| **TOPPRA** | Time-optimal path-following (default) |
| **RUCKIG** | Jerk-limited point-to-point motion (joint moves only) |
| **QUINTIC** | C² smooth polynomial trajectories |
| **TRAPEZOID** | Linear segments with parabolic blends |
| **LINEAR** | Direct interpolation (no smoothing) |

Note: RUCKIG is point-to-point only and cannot follow Cartesian paths. When RUCKIG is set, Cartesian moves automatically use TOPPRA instead.

### Speed and acceleration

```python
client.moveJ(target, speed=0.5, accel=0.5)   # 50% of joint limits
client.moveL(target, speed=0.25, accel=1.0)   # 25% cart speed, full accel
client.moveL(target, duration=2.0)             # Fixed duration (uses TOPPRA)
```

Speed and accel are fractions of maximum (0.0–1.0), not percentages.

For Cartesian moves, joint limits stay at 100% as hard bounds—the speed fraction only affects the Cartesian velocity constraint.

### Queued execution speed and pause

`set_execution_speed(scale)` selects 10–100% of an already planned trajectory's
speed. The command's `speed`, `accel` and `duration` still define the original
plan. Jog and streamed servo commands retain their own timing.

Override transitions use a separate rate ramp and acceleration checks. The
nominal motion profile's jerk ceiling is not guaranteed during a transition.

Use `pause()` to retain the queue and decelerate queued motion to a hold, and
`resume()` to continue at the selected scale. Changing speed while paused keeps
the pause. The speed setter rejects zero. These controls return 1 when their
request is confirmed, or 0 when confirmation times out.

Fresh `execution_speed()` readback exposes `target_scale`, `applied_scale` and
`resume_scale`. Its `paused` property confirms the applied scale reached zero;
the pause request can be acknowledged while still decelerating. Queued delays
retain their remaining time while paused; positive speed changes do not retime
delays, tool actuators or homing routines already in progress.

Standalone `wait_command()` keeps its wall-clock timeout and returns false if
completion is unconfirmed. Blocking motion calls raise `TimeoutError` in that
case. A timed-out wait leaves the motion queued; `stop()` cancels it. Planning
preview retimes trajectories and reports paused queued operations as
`UnresolvedPreview` instead of claiming completion.

## Timed observations

`stream_status()` supplies `session_id`, `seq` and `mono_time_ns` for recording
observations. The session identifies the status publisher's lifetime and changes
on restart. Sequence gaps reveal missed publications; the monotonic timestamp
marks publication of the current controller snapshot, not simultaneous sensor
acquisition. Status without these fields reports zero metadata and cannot support
identified demonstration capture. The client advertises `observation.timed`.

Waldo Commander's `record_demonstration` stores this metadata and its host receipt
time with the observed joints and tool state. Its replay skill uses ordinary
native joint moves/delays, including native retiming, completion and collision
checks; no continuous recorded-trajectory command is added.

## Command system

Jog and servo commands (JogJ, JogL, ServoJ, ServoL) automatically use the streaming fast-path — the server de-duplicates stale inputs, reduces ACK chatter, and reuses the active command. Use jog/servo for UI-driven motion or teleoperation; use planned moves (MoveJ, MoveL, etc.) for discrete motions and queued programs.

### Command categories

| Category | Examples | Queue | ACK | Execution |
|----------|----------|-------|-----|-----------|
| **Query** | PING, GET_STATUS, GET_ANGLES | No | Request/response | Immediate |
| **System** | RESUME, HALT, SET_IO, SIMULATOR | No | Always | Immediate (even when disabled) |
| **Planned motion** | MOVEJ, MOVEL, MOVEC, MOVES, MOVEP, HOME | Yes | With command_index | MotionPlanner subprocess → SegmentPlayer |
| **Streaming motion** | JOGJ, JOGL, SERVOJ, SERVOL | Yes | Fire-and-forget | StreamingExecutor in main loop |
| **Utility** | DELAY, CHECKPOINT, SET_TOOL | Yes | With command_index | Inline via MotionPlanner (preserves ordering) |
| **Tool action** | TOOL_ACTION | Yes | With command_index | Inline via MotionPlanner |

### Command lifecycle

All commands implement the `CommandBase` protocol:
- `setup(state)` → calls `do_setup(state)`: one-time preparation (trajectory computation, target resolution)
- `tick(state)` → calls `execute_step(state)`: per-tick execution in control loop, returns `EXECUTING`, `COMPLETED`, or `FAILED`
- `assign_params(params)`: for streamable commands, updates target without recreating the command

### Adding a new command

1. Create a class under `parol6/commands/` and decorate with `@register_command(CmdType.YOUR_CMD)`
2. Define a `PARAMS_TYPE` msgspec Struct for wire validation
3. Implement `do_setup(state)` and `execute_step(state)` — obey hot path rules
4. Set `streamable = True` if the command supports high-rate streaming
5. Add client method to `async_client.py` and `sync_client.py`

## Kinematics and tools

Uses numerical IK via pinokin (C++/Pinocchio bindings). Some Cartesian targets may fail to solve — J4 is particularly sensitive. To adapt to modified hardware, update `parol6/PAROL6_ROBOT.py` (gear ratios, limits) and `parol6/tools.py` (tool transforms).

Currently supported tools (see `parol6/tools.py`):
- `NONE` (bare flange)
- `PNEUMATIC` (pneumatic gripper — vertical/horizontal variants)
- `SSG-48` (adaptive electric gripper — finger/pinch variants)
- `MSG` (compliant AI stepper gripper — 100mm/150mm/200mm rail variants)
- `VACUUM` (vacuum gripper)

Set tool at runtime from the client:
```python
from parol6 import RobotClient
with RobotClient() as c:
    c.set_tool("PNEUMATIC")                            # default variant
    c.set_tool("PNEUMATIC", variant_key="horizontal")  # horizontal pneumatic
    c.set_tool("SSG-48", variant_key="pinch")          # pinch grip
    c.set_tool("MSG", variant_key="150mm")             # 150mm rail
```

Add a new tool by creating a `ToolConfig` subclass (or using `ToolConfig` directly) and calling `register_tool("KEY", config)` in `parol6/tools.py`.


**Security note:** The controller has no authentication — it accepts any correctly parsed command on its UDP port. Multiple senders are supported by design (e.g., GUI + orchestrator), but deploy only on trusted networks.

## Environment variables
- `PAROL6_CONTROL_RATE_HZ` — control loop frequency in Hz (default 100)
- `PAROL6_STATUS_RATE_HZ` — STATUS broadcast rate in Hz (default 50; tests use 20 Hz to reduce CI load)
- `PAROL6_STATUS_STALE_S` — skip broadcast if cache is older than this (default 0.5)
- `PAROL6_BUSY_THRESHOLD_MS` — busy-loop threshold for loop timing in ms (default 1.0)
- `PAROL6_PATH_SAMPLES` — trajectory path sampling points (default 50)
- `PAROL6_MAX_BLEND_LOOKAHEAD` — command blending lookahead count (default 100)
- `PAROL6_MCAST_GROUP` — multicast group for status (default 239.255.0.101)
- `PAROL6_MCAST_PORT` — multicast port for status (default 50510)
- `PAROL6_MCAST_TTL` — multicast TTL (default 1)
- `PAROL6_MCAST_IF` — interface/IP for multicast (default 127.0.0.1)
- `PAROL6_STATUS_TRANSPORT` — MULTICAST (default) or UNICAST
- `PAROL6_STATUS_UNICAST_HOST` — unicast target host (default 127.0.0.1)
- `PAROL6_CONTROLLER_IP` / `PAROL6_CONTROLLER_PORT` — bind host/port for controller
- `PAROL6_FORCE_ACK` — force ACK for motion commands regardless of policy
- `PAROL6_FAKE_SERIAL` — enable simulator ("1"/"true"/"on"); used internally by simulator_on/off
- `PAROL6_COM_FILE` — path to persistent COM port file (default `~/.parol6/com_port.txt`)
- `PAROL6_COM_PORT` / `PAROL6_SERIAL` — explicit serial port override (e.g., `/dev/ttyUSB0` or `COM3`)
- `PAROL_TRACE` — `1` enables TRACE logging level unless overridden by CLI


## Development setup

For contributors working on this repository:

```bash
pip install -e .[dev]
pre-commit install
```

- Run all pre-commit hooks locally: `pre-commit run -a`
- Run tests with pytest:
  - `pytest`
  - Simulator is used by default (PAROL6_FAKE_SERIAL=1).

### Control rate and performance

The default control loop rate is **100 Hz** (`PAROL6_CONTROL_RATE_HZ=100`). Higher rates up to 250 Hz and even 500 Hz are achievable, but there are diminishing returns in motion smoothness as you go higher.

Even under complete IK failure (worst-case computation), the control loop typically completes in **under 2ms**. However, consistent high-rate performance requires consideration of OS scheduling—the operating system commonly interrupts user-space processes, which can cause jitter at higher rates.

**Note:** Rates above 250 Hz may require increasing IK solving tolerance, as the distance moved per tick becomes smaller and numerical precision becomes a factor.

### Tuning for higher rates

For consistent high-rate performance:
1. **Elevate process priority**: On Linux, use `nice -n -20` or `chrt -f 50` for real-time scheduling
2. **Disable logging**: TRACE and DEBUG logging add significant overhead
3. **Reduce background load**: Heavy background tasks compete for CPU time
4. **Consider CPU isolation**: Pin the controller to dedicated cores with `taskset`


## FAQ / Troubleshooting
- I see `Control loop avg period degraded by …` warnings
  - The loop is falling behind. Reduce `PAROL6_CONTROL_RATE_HZ` and ensure TRACE and DEBUG logging is disabled.
- Motion feels inconsistent or jittery on my machine
  - Lower the control rate; avoid heavy background tasks; disable TRACE and DEBUG logging.
- Some cartesian targets fail to solve (especially around J4)
  - Without null-space control in the backend, some poses are hard to reach. Re-plan the path, adjust the target, or change the starting posture. Future backend updates may add null-space manipulation.


## Safety notes
- Keep physical E‑Stop accessible at all times when connected to hardware
- The controller can halt motion via `halt()` and reacts to E‑Stop inputs when on real hardware
- Prefer `simulator_on()` for development without hardware and validate motions before switching to real serial

## TCP transforms

Use `set_tcp_transform(x, y, z, roll, pitch, yaw)` for a full user TCP correction,
in millimetres and intrinsic XYZ degrees (`Rx · Ry · Rz`) relative to the
registered tool. Async and sync clients return a queued command index; wait for
that index before treating the correction as applied or querying it.

```python
with RobotClient() as rbt:
    index = rbt.set_tcp_transform(0, 0, 25, 0, 90, 0)
    if not rbt.wait_command(index):
        raise RuntimeError("TCP application was not confirmed")
    applied = rbt.tcp_transform()
```

Live FK, Cartesian planning, TRF motion and dry-run preview use the same
transform. Pending blend paths are completed with their original TCP before a
configuration change. Cancelling a queued change preserves the applied value.
A different tool or variant clears the correction; reselecting the same tool
and variant preserves it. Physical collision meshes stay on their registered
links, independent of the user-defined tip and axes.

The existing `set_tcp_offset(x, y, z)` clears user rotation and now returns its
queued index for confirmation. `tcp_offset()` still reads three translations;
`tcp_transform()` reads all six values. Both raise `TimeoutError` when no valid
reply arrives instead of reporting a misleading zero correction.

Digital I/O reads and writes accept an optional per-call `timeout` in seconds:
`rbt.io(timeout=1.0)` returns `None` without a reply, while
`rbt.write_io(0, 1, timeout=1.0)` raises `TimeoutError` if acceptance remains
unconfirmed. The deadline includes transport setup and retries. Omitting it
retains the configured client timeout. The same options work on the sync client.
The client advertises `io.digital` for typed named-signal skills, which can be
imported from `waldo_commander.skills`; mappings are `waldoctl.signals.DigitalSignal`
values stored in a setup snapshot. Dry-run clients advertise `execution.preview`
so those skills require explicit observation fixtures during preview.
