# Architecture overview

Single source of truth for how the TWFarmBot codebase is wired after the
local-rewrite branch. The Pi now talks directly to the stock Farmduino
firmware over USB serial; there is no cloud, MQTT, or `farmbot-py` dependency.

## 1. Folder layout

```
TWFarmBot/
├── apps/                       # deployable entry points (orchestration only)
│   ├── api_server/             # FastAPI; POST /actions, GET /health, GET /position, ...
│   ├── ui/                     # Streamlit dashboard
│   └── worker/                 # scheduled / long-running jobs (skeleton)
│
├── core/                       # cross-cutting primitives
│   └── twfarmbot_core/
│       ├── domain/             # Action, Point3D, Rectangle, GardenEntity, GardenWorld
│       ├── config/             # YAML config + env-based settings
│       ├── logging/            # configure_logging, get_logger
│       └── events/             # internal event bus
│
├── services/                   # one concern per service
│   ├── safety_service/         # gates every action before execution
│   ├── watering_service/       # irrigation + hardware backend abstraction
│   ├── planning_service/       # LLM / VLM task planning
│   ├── spatial_service/        # garden coordinates + persistent world model
│
├── libs/                       # reusable, framework-agnostic utilities
│   ├── farmbot_serial/         # USB-serial G-code/F-code driver for Farmduino
│   └── ml_utils/               # ML helpers (segmentation, VLM, etc.)
│
├── configs/                    # YAML / JSON; loaded via core/config
├── docs/                       # architecture + ADRs (this file)
├── projects/                   # isolated student projects (must not modify shared code)
├── experiments/                # reproducible experiment runners
├── scripts/                    # operational helpers (systemd, smoke tests)
└── tests/                      # cross-cutting / integration tests
```

## 2. Hard rules

1. **Only `libs/farmbot_serial/` talks to the Farmduino.**
   - It is a plain USB-serial driver; no cloud protocol, no MQTT.
   - `services/watering_service/watering_service/backends/direct_serial.py` is the
     only backend that opens `/dev/ttyACM0`.
   - Nothing outside `watering_service` imports `farmbot_serial` directly.

2. **Every real-world action goes through `safety_service` before it hits hardware.**
   - `services/safety_service/safety_service/__init__.py` exposes `validate(action)`.
   - `ActionRegistry.dispatch` runs safety validation automatically.

3. **Apps orchestrate, services decide, libs compute.**
   - `apps/api_server` wires services together and handles I/O.
   - `services/<x>_service` owns the logic for one concern.
   - `libs/` contains pure utilities and the serial driver.

4. **`core/` defines the shared vocabulary.**
   - `Action`, `Point3D`, `Rectangle`, `GardenEntity`, `GardenWorld` live in `core/twfarmbot_core/domain/`.

5. **Configuration is data, not code.**
   - Robot serial port, movement limits, pin map, camera — all live in `configs/dev.yaml`.
   - Env vars are runtime overrides (see `safety_service.load_limits`).

6. **Projects stay isolated.**
   - `projects/<name>/` may import from `core/`, `libs/`, and call public service APIs.

7. **Tests live with the code.**
   - Unit tests next to the module; cross-service tests in `tests/` at the repo root.

## 3. The Action flow

```
Action (JSON)
    │
    ▼
apps/api_server        POST /actions
    │
    ▼
ActionRegistry.dispatch(action)        core/twfarmbot_core/actions.py
    │
    ├── safety_service.validate(action)     ALWAYS runs first
    │       ↳ raises UnsafeActionError → HTTP 400
    │
    ├── look up handler by action.kind
    │       ↳ UnknownActionError → HTTP 404
    │
    └── handler(action)                      apps/api_server/.../handlers/<name>.py
            │
            ▼
        watering_service.get_backend()
            │
            ▼
        DirectSerialBackend
            │
            ▼
        FarmduinoSerial over /dev/ttyACM0
            │
            ▼
        Farmduino firmware (G-code / F-code)
```

Key files:

- `core/twfarmbot_core/domain/action.py` — `Action(kind, params)`.
- `core/twfarmbot_core/actions.py` — `ActionRegistry` + `dispatch`.
- `apps/api_server/src/twfarmbot_api_server/handlers/<name>.py` — handlers.
- `services/safety_service/safety_service/__init__.py` — `validate(action)`.
- `services/watering_service/watering_service/__init__.py` — backend loader.
- `services/watering_service/watering_service/backends/base.py` — `RobotBackend` protocol.
- `services/watering_service/watering_service/backends/direct_serial.py` — local backend.
- `libs/farmbot_serial/farmbot_serial/client.py` — serial driver.

## 4. Hardware isolation layers

```
HTTP request (FastAPI handler)
    ↓
watering_service.get_backend()  →  RobotBackend protocol
    ↓
DirectSerialBackend
    ↓
FarmduinoSerial (/dev/ttyACM0)
    ↓
Farmduino firmware
    ↓
steppers / valves / sensors
```

Adding another backend (e.g. a simulator) means implementing the `RobotBackend`
protocol and changing `watering_service/__init__.py:_load_backend()` to load it.

## 5. Configuration

All hardware-specific values live in `configs/dev.yaml` under the `hardware:` block:

```yaml
hardware:
  version: genesis_v1.8
  board: farmduino_v32
  serial:
    port: /dev/ttyACM0
    baud: 115200
  movement:
    steps_per_mm: {x: 80, y: 80, z: 400}
    max_speed_mm_s: {x: 80, y: 80, z: 16}
  peripherals:
    water: {pin: 8, mode: digital}
  camera:
    index: 0
    save_dir: data/images
```

Runtime overrides via `.env` (see `.env.example`):

| Var | Purpose | Default |
|---|---|---|
| `FARMBOT_REQUIRED` | if `0`, api_server boots without a live bot | `1` |
| `WATERING_BACKEND` | backend module name | `direct_serial` |
| `FARMBOT_MAX_WATER_SECONDS` | safety cap | `300` |
| `FARMBOT_PUMP_PIN` | pump pin override | `8` |
| `FARMBOT_MAX_AXIS_{X,Y,Z}` | move action bounds in mm | from `configs/dev.yaml` |

## 6. Action kinds shipped today

| Kind | Params | Backend call | Safety rule |
|---|---|---|---|
| `water` | `seconds` | `backend.water(seconds)` | seconds ≤ `FARMBOT_MAX_WATER_SECONDS` |
| `move` | `x`, `y`, `z`, optional `speed` | `backend.move(x,y,z,speed)` | each axis within configured bounds |
| `move_path` | `waypoints`, optional `photo_at_waypoints` | `backend.move()` + `take_photo()` | each waypoint within bounds |
| `read_pin` | `pin`, optional `mode` | `backend.read_pin(pin, mode)` | — |
| `write_pin` | `pin`, `value`, optional `mode`/`seconds` | `backend.write_pin(...)` | — |
| `mount_tool` | `tool_name` | `backend.mount_tool(tool_name)` | — |
| `dismount_tool` | — | `backend.dismount_tool()` | — |
| `send_message` | `message`, optional `type`/`channels` | `backend.send_message(...)` | — |
| `e_stop` | — | `backend.e_stop()` | — |
| `find_home` | — | `backend.find_home()` | — |
| `take_photo` | — | `backend.take_photo()` | — |

### Read-only GET routes

Defined in `apps/api_server/.../read.py`, these skip `ActionRegistry` because
there is no `Action` envelope:

| Route | Returns |
|---|---|
| `GET /health` | status, action list, `farmbot` connection state |
| `GET /position` | `{xyz: {x, y, z}}` |
| `GET /status` | full backend status tree |
| `GET /pin/{pin}?mode=digital|analog` | `{pin, mode, value}` |
| `GET /messages` | `{last_messages: []}` (local stack has no MQTT queue) |
| `GET /images?limit=10` | `{images: [...]}` |
| `GET /garden` | bounds, camera pose, zones, entities, cached robot pose |

## 7. How to add …

### a new action kind

1. Create `apps/api_server/src/twfarmbot_api_server/handlers/<name>.py` with
   `handle_<name>(action: Action) -> Action`.
2. Register it in `apps/api_server/.../handlers/__init__.py`.
3. If it touches hardware, add the corresponding method to `RobotBackend` and
   implement it in `DirectSerialBackend`.
4. (Optional) add a safety rule in `services/safety_service/.../__init__.py:validate`.

### a new hardware backend

1. Implement `RobotBackend` in `services/watering_service/watering_service/backends/<name>.py`.
2. Expose a module-level `backend` instance.
3. Set `WATERING_BACKEND=<name>` in the environment.

### a new service

1. Create `services/<name>_service/<name>_service/__init__.py`.
2. Create `services/<name>_service/pyproject.toml` depending on `twfarmbot-core`.
3. Add the member to `[tool.uv.workspace]` and the package to root dependencies.
4. Wire its public API through one or more handlers.

### a new pump pin

Change `watering.pump_pin` in `configs/dev.yaml` or set `FARMBOT_PUMP_PIN` in `.env`.

## 8. UI

`apps/ui` is a single-page Streamlit app. It contains **zero business logic** —
every widget calls the API server:

- `GET /health` → connection status
- `GET /position` → gantry position
- `GET /images` → photo gallery
- `POST /actions` → all real-world commands

Run it while the API server is up:

```bash
uv run twfarmbot-api   # terminal 1
uv run twfarmbot-ui    # terminal 2 → http://localhost:8501
```

## 9. api_server boot contract

The API server eagerly connects to the Farmduino before binding the HTTP port:

1. `connect_to_farmduino()` opens `/dev/ttyACM0` via `get_backend().connect()`.
2. On success, `app.state.farmbot_status = "connected"`.
3. On failure with `FARMBOT_REQUIRED=1`, it raises `SystemExit`.
4. With `FARMBOT_REQUIRED=0`, failure is recorded but the server boots anyway.
5. `GET /health` returns the current status.

## 10. Testing

```bash
# Offline test suite
PYTHONPATH= uv run pytest tests/ -q

# Real hardware smoke test (read-only by default)
PYTHONPATH= uv run python scripts/test_farmduino_local.py

# Safe motion checks (only when the bed is clear)
PYTHONPATH= uv run python scripts/test_farmduino_local.py --home
PYTHONPATH= uv run python scripts/test_farmduino_local.py --move 0 0 10
```

## 11. Where things live — quick reference

| Want to … | Look in |
|---|---|
| Trigger watering manually | `apps/api_server` → `/actions` with `kind="water"` |
| Add a safety rule | `services/safety_service/safety_service/__init__.py:validate` |
| Talk to hardware directly | `libs/farmbot_serial/farmbot_serial/client.py` |
| Change serial port / pins | `configs/dev.yaml` |
| Add a backend | `services/watering_service/watering_service/backends/` |
| Add a CLI tool | `scripts/` |
| Add a shared type | `core/twfarmbot_core/domain/` |
| Add an experiment | `experiments/<name>/` |
| Build a student project | `projects/<name>/`, isolated from shared code |
