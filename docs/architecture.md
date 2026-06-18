# Architecture overview

Single source of truth for how the TWFarmBot codebase is wired. Read this
before adding code. Folder rules are duplicated from the top-level
`README.md` so an agent doesn't need to chase cross-references.

## 1. Folder layout

```
TWFarmBot/
├── apps/                       # deployable entry points (orchestration only)
│   ├── api_server/             # FastAPI; POST /actions, GET /health
│   ├── ui/                     # dashboard
│   └── worker/                 # scheduled / long-running jobs
│
├── core/                       # cross-cutting primitives
│   └── twfarmbot_core/
│       ├── domain/             # Action, Bed, Plant, SensorReading
│       ├── config/             # env-based settings (load_settings())
│       ├── logging/            # configure_logging, get_logger
│       └── events/             # internal event bus
│
├── services/                   # one concern per service
│   ├── farmbot_gateway/        # ONLY place that talks to FarmBot hardware
│   ├── safety_service/         # gates every action before execution
│   ├── watering_service/       # irrigation (FarmBotBackend only — see backends/farmbot.py)
│   ├── sensor_service/         # soil / temp / light
│   ├── vision_service/         # camera + models
│   ├── planning_service/       # LLM / VLM task planning
│   └── spatial_service/        # garden coordinates + persistent world model
│
├── libs/                       # pure, reusable utilities
│   └── farmbot_client/         # wraps farmbot-py; reusable client
│
├── configs/                    # YAML / JSON; loaded via core/config
├── docs/                       # architecture + ADRs (this file)
├── projects/                   # isolated student projects (must not modify shared code)
├── experiments/                # reproducible experiment runners
└── tests/                      # cross-cutting / integration tests
```

## 2. Hard rules

These are non-negotiable. They are the reason this repo is split the way it is.

1. **Only `services/farmbot_gateway/` talks to the FarmBot hardware.**
   - Wraps the `farmbot-py` library via `libs/farmbot_client`.
   - Other code calls `farmbot_gateway.get_farmbot()` or registers an action
     handler — never imports `farmbot` directly.
   - See `services/farmbot_gateway/farmbot_gateway/__init__.py`.

2. **Every real-world action goes through `safety_service` before it hits
   `farmbot_gateway`.**
   - `services/safety_service/safety_service/__init__.py` exposes
     `validate(action)` which raises `UnsafeActionError` on rejection.
   - The api_server's `ActionRegistry.dispatch` runs safety validation
     automatically; handlers do not re-validate.

3. **Apps orchestrate, services decide, libs compute.**
   - `apps/api_server` and `apps/worker` wire services together and handle
     I/O. They must not contain domain logic.
   - `services/<x>_service` owns the logic for one concern.
   - `libs/` contains pure utilities, no I/O, no global state.

4. **`core/` defines the shared vocabulary.**
   - `Action`, `Bed`, `Plant`, `SensorReading` live in `core/twfarmbot_core/domain/`.
   - Do not redefine equivalent types inside a service or project.

5. **Configuration is data, not code.**
   - Robot coordinates, sensor calibrations, watering limits — all live in
     `configs/*.yaml` and are loaded via `core/twfarmbot_core/config`.
   - Env vars are the runtime override mechanism (see `safety_service.load_limits`).

6. **Projects stay isolated.**
   - `projects/<name>/` may import from `core/`, `libs/`, and call the
     public APIs of `services/`. They must not modify shared code or talk
     to hardware directly.

7. **Tests live with the code.**
   - Unit tests next to the module; cross-service tests in `tests/` at the
     repo root.

## 3. The Action flow

Everything that affects the real world is an `Action`. There is exactly
one path an `Action` takes through the system:

```
Action (JSON)
    │
    ▼
apps/api_server        POST /actions
apps/worker            (scheduler triggers)    POST /actions over queue (future)
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
        service.water_bed / .move / .sense / ...
            │
            ▼
        farmbot_gateway.get_farmbot() via FarmBotBackend
            │
            ▼
        FarmBot over WiFi/MQTT (libs/farmbot_client)
```

Key files:

- `core/twfarmbot_core/domain/action.py` — `Action(kind, params)`.
- `core/twfarmbot_core/actions.py` — `ActionRegistry` + `dispatch`. Shared by api_server and worker.
- `apps/api_server/src/twfarmbot_api_server/handlers/<name>.py` — example handler.
- `services/safety_service/safety_service/__init__.py` — `validate(action)`.
- `apps/api_server/src/twfarmbot_api_server/handlers/watering.py` — example handler.
- `services/safety_service/safety_service/__init__.py` — `validate(action)`.
- `services/watering_service/watering_service/__init__.py` — example service.
- `services/watering_service/watering_service/backends/farmbot.py` — FarmBotBackend; the only backend.
- `services/farmbot_gateway/farmbot_gateway/__init__.py` — `get_farmbot()`.
- `libs/farmbot_client/farmbot_client/client.py` — wraps `farmbot-py`.

## 4. Core ↔ safety coupling

`core/twfarmbot_core/actions.py` imports `safety_service.validate` so
that *every* dispatch automatically goes through safety — no caller can
forget. This means `twfarmbot-core` declares a dependency on
`twfarmbot-safety-service`. That's intentional: safety is a precondition
of any real-world action, not an optional policy.

To avoid a circular import, `core/__init__.py` does **not** eagerly import
`core.actions`. Use the explicit form: `from twfarmbot_core.actions import ActionRegistry`.

## 5. Hardware isolation

There are three concentric layers between an Action and the FarmBot:

```
handlers (api_server)
    ↓
watering_service (decision: open valve X for Y seconds)
    ↓
watering_service.backends.farmbot   (the only backend; translates to farmbot-py)
    ↓
farmbot_gateway.get_farmbot()      (singleton, reconnecting link)
    ↓
farmbot-py                         (MQTT + REST via libs/farmbot_client)
    ↓
FarmBot over WiFi
```

`watering_service.backends.farmbot.FarmBotBackend` is the single place
that translates our vocabulary into `farmbot-py` calls. Adding a new
backend is not a current need — if it ever is, the import in
`watering_service/__init__.py:_load_backend()` is the only place that
needs to know.

## 5. Hardware isolation

## 6. WiFi connection

Set these env vars (or use `.env` with `uv run --env-file=.env …`):

| Var | Purpose | Default |
|---|---|---|
| `FARMBOT_EMAIL` | account email | required |
| `FARMBOT_PASSWORD` | account password | required |
| `FARMBOT_SERVER` | REST auth host | `https://my.farm.bot` |
| `FARMBOT_HOST` | MQTT broker | `farmbot.farm.bot` |
| `FARMBOT_TOKEN` | optional pre-fetched token JSON | — |
| `WATERING_BACKEND` | backend module name (default `farmbot`) | `farmbot` |
| `FARMBOT_MAX_WATER_SECONDS` | safety cap | `300` |
| `FARMBOT_ALLOWED_BEDS` | comma-separated bed allowlist | empty (allow all) |
| `FARMBOT_MAX_AXIS_{X,Y,Z}` | move action bounds in mm | `3000`/`1500`/`800` |
| `FARMBOT_PIN_<bed>` | valve pin override per bed (e.g. `FARMBOT_PIN_b1=13`) | from `configs/dev.yaml` |
| `FARMBOT_REQUIRED` | if `0`, api_server boots without a live bot (UI-only mode) | `1` |

Test the link: `uv run --env-file=.env python scripts/test_farmbot_connect.py`.

## 7. Action kinds shipped today

The api_server registers these action kinds via
`apps/api_server/.../handlers/__init__.py`:

| Kind | Params | Backend call | Safety rule |
|---|---|---|---|
| `water` | `bed_id`, `seconds` | `farmbot_backend.water()` (valve on/off) | seconds ≤ `FARMBOT_MAX_WATER_SECONDS`, bed in allowlist |
| `move` | `x`, `y`, `z`, optional `speed` | `farmbot_backend.move()` | each axis within `FARMBOT_MAX_AXIS_{X,Y,Z}` |
| `read_pin` | `pin`, optional `mode` | `farmbot_backend.read_pin()` | — |
| `write_pin` | `pin`, `value`, optional `mode` | `farmbot_backend.write_pin()` | — |
| `mount_tool` | `tool_name` | `farmbot_backend.mount_tool()` | — |
| `dismount_tool` | — | `farmbot_backend.dismount_tool()` | — |
| `send_message` | `message`, optional `type`/`channels` | `farmbot_backend.send_message()` | — |
| `e_stop` | — | `farmbot_backend.e_stop()` | — |
| `take_photo` | — | `farmbot_backend.take_photo()` | — |

`farmbot_backend` lives at
`services/watering_service/watering_service/backends/farmbot.py` and is
the **only** place that translates our vocabulary into `farmbot-py` calls.

### Read-only GET routes

The api_server also exposes GETs that read FarmBot state directly, used
by the UI for live status (defined in `apps/api_server/.../read.py`).
These skip `ActionRegistry` because there's no `Action` envelope and no
safety rule, but they still call into `FarmBotBackend` so the UI never
imports `farmbot-py` directly:

| Route | Returns |
|---|---|
| `GET /position` | `{xyz: {x, y, z}}` |
| `GET /status?path=<optional>` | `{path, state}` |
| `GET /pin/{pin}?mode=digital|analog` | `{pin, mode, value}` |
| `GET /messages` | `{last_messages: [...]}` |
| `GET /images?limit=10` | `{images: [{attachment_url, created_at, meta, ...}]}` |
| `GET /garden` | bed bounds, camera pose, entities, zones, and cached robot pose |

## 8. How to add …

### a new action kind

Two files plus zero plumbing:

1. Add a method to `services/watering_service/watering_service/backends/farmbot.py`
   if it touches hardware.
2. Create `apps/api_server/src/twfarmbot_api_server/handlers/<name>.py`
   with `handle_<name>(action: Action) -> Action` (one method call).
3. Register in `apps/api_server/.../handlers/__init__.py`:
   `registry.register("<name>", handle_<name>)`.
4. (Optional) add a safety rule in `services/safety_service/.../__init__.py:validate`.

The `/actions` route, pydantic payload validation, safety gate, and
registry dispatch already exist. Tests in `tests/test_farmbot_backend.py`.

### a new service (e.g. "weeding_service")

1. Create `services/<name>_service/<name>_service/__init__.py`.
2. Create `services/<name>_service/pyproject.toml` depending on `twfarmbot-core`.
3. Add `"services/<name>_service"` to `[tool.uv.workspace] members` in
   the root `pyproject.toml`, plus `"twfarmbot-<name>-service"` to root
   `dependencies` and `[tool.uv.sources]`.
4. Run `uv sync`.
5. Wire its public API through one or more handlers in
   `apps/api_server/src/twfarmbot_api_server/handlers/`.
6. Never import the FarmBot library directly — go through
   `farmbot_gateway.get_farmbot()` (or the `FarmBotBackend`).

### a new valve / new bed

Add an entry to `configs/dev.yaml` under `watering.pins`, or set
`FARMBOT_PIN_<bed_id>=<pin>` in `.env`. No code change.

### a new shared type (e.g. "FertilizerDose")

Add it to `core/twfarmbot_core/domain/<name>.py` and re-export from
`core/twfarmbot_core/domain/__init__.py`. Do not redefine it inside a service.

### a new config key

Add to `configs/dev.yaml` and read via `watering_service._load_yaml_config()`,
or env-only via `os.getenv(...)`.

## 9. UI

`apps/ui` is a single-page Streamlit app (`streamlit run apps/ui/src/twfarmbot_ui/app.py`).
It contains **zero business logic** — every widget is a thin HTTP proxy
to the api_server:

**Reads (buttons/forms in the UI):**
- "Check status" sidebar button → `GET /health`
- "Position" button → `GET /position`
- "Messages" button → `GET /messages`
- "Read pin" form → `GET /pin/{pin}`
- "Status" form → `GET /status?path=...`

**Writes:**
- "Water now" form → `POST /actions {"kind":"water", ...}`
- "Move" form → `POST /actions {"kind":"move", ...}`
- "Raw action" form → `POST /actions {"kind":..., "params":{...}}`

Run the UI in another terminal while the api_server is up:

```bash
uv run twfarmbot-api            # terminal 1 (auto-connects to FarmBot)
uv run twfarmbot-ui             # terminal 2 → http://localhost:8501
```

`TWFB_API_URL` overrides the default `http://127.0.0.1:8000`.

## 10. api_server boot contract

The api_server is the canonical "thing the user starts", so it eagerly
connects to the FarmBot before binding the HTTP port:

1. `apps/api_server/.../app.py:connect_to_farmbot()` is called from
   `main()` after settings are loaded and before uvicorn starts.
2. If the connect succeeds, `app.state.farmbot_status = "connected"`.
3. If it fails and `FARMBOT_REQUIRED=1` (default), `main()` raises
   `SystemExit` with a FATAL message — uvicorn never starts.
4. If `FARMBOT_REQUIRED=0`, the failure is recorded on `app.state` and
   the server boots anyway (useful for offline UI dev).
5. `GET /health` returns the current status, e.g.
   `{"status":"ok","actions":["water"],"farmbot":"connected"}`.

`get_farmbot()` itself remains lazy. Worker processes, tests, and scripts
that import `farmbot_gateway` still don't connect until first use.

## 11. Testing

```bash
uv run pytest tests/                    # all tests (offline)
uv run --env-file=.env pytest tests/    # also runs the live FarmBot test
```

The opt-in live test (`tests/test_farmbot_connection.py::test_farmbot_connection_live`)
only runs when `FARMBOT_LIVE_TEST=1` is set.

## 12. Where things live — quick reference

| Want to … | Look in |
|---|---|
| Trigger watering manually | `apps/api_server` → `/actions` with `kind="water"` |
| Add a safety rule | `services/safety_service/safety_service/__init__.py:validate` |
| Read FarmBot state | `farmbot_gateway.get_farmbot()` |
| Build a new sensor reading | `sensor_service` (skeleton) |
| Add a CLI tool | a `scripts/` entry, calling the public service API |
| Run scheduled jobs | `apps/worker` (skeleton) |
| Add a shared type | `core/twfarmbot_core/domain/` |
| Add an experiment | `experiments/<name>/` reading configs, calling services |
| Build a student project | `projects/<name>/`, isolated from shared code |
