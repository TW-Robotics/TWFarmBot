# TWFarmBot

Implementation of the FarmBot at UAS Technikum Wien.

This repository is a **monorepo** for the FarmBot system: the physical robot
integration, sensor pipelines, irrigation/vision/planning services, the API and
worker apps, and the student projects and experiments that build on top of
them. To keep all of these moving parts maintainable, every contribution is
expected to follow the **folder structure and separation of responsibilities**
described below.

---

## Folder structure

The repository is organised into the following top-level folders. Each one has
a clear purpose — please put new code in the folder that matches its role.

```
farmbot-research/
├── apps/                       # deployable applications (entry points)
│   ├── ui/                     # dashboard, sensor display, manual triggers
│   ├── api_server/             # exposes FarmBot + sensor + experiment API
│   └── worker/                 # background jobs, experiment execution
│
├── core/                       # shared building blocks used everywhere
│   ├── domain/                 # shared concepts: Plant, Bed, SensorReading, Action
│   ├── config/                 # typed settings, env loading
│   ├── logging/                # structured logging setup
│   └── events/                 # internal event bus / message contracts
│
├── services/                   # business-logic services, one concern each
│   ├── farmbot_gateway/        # ONLY place that talks to the FarmBot hardware
│   ├── sensor_service/         # reads soil, temperature, light, etc.
│   ├── watering_service/       # irrigation logic
│   ├── vision_service/         # camera + pure vision models
│   ├── planning_service/       # LLM/VLM task planning
│   └── safety_service/         # validates actions before execution
│
├── projects/                   # student / research projects living in the repo
│   ├── student_project_weed_detection/
│   ├── student_project_vlm_planner/
│   └── student_project_soil_mapping/
│
├── experiments/                # reproducible experiment runs & evaluations
│   ├── watering_strategy_eval/
│   ├── plant_state_estimation/
│   └── vlm_grounding/
│
├── libs/                       # reusable libraries (no business logic)
│   ├── farmbot_client/         # reusable client used by services/api
│   ├── vision_utils/
│   ├── ml_utils/
│   └── geometry/
│
├── docs/
├── tests/
└── configs/
```

### What goes where

| Folder | Purpose | Examples |
| --- | --- | --- |
| `apps/` | Runnable applications. Each subfolder has its own entry point. | `apps/api_server/main.py`, `apps/worker/main.py` |
| `core/` | Cross-cutting primitives shared by apps and services. | `Plant`, `Bed`, `SensorReading`, `Action` dataclasses; logging; config |
| `services/` | One service per concern. Services expose a clear API to `apps/` and to other services. | Reading sensors, watering a bed, running vision on a frame |
| `projects/` | Self-contained student/research projects. A project can wire services together but **must not** modify shared code in `core/`, `services/`, or `libs/`. | Weed detection prototype, VLM planner prototype |
| `experiments/` | Reproducible experiment definitions, harnesses, configs and result outputs. | `watering_strategy_eval/`, `vlm_grounding/` |
| `libs/` | Reusable, framework-agnostic utilities. No business logic, no FarmBot specifics beyond `farmbot_client`. | Geometry helpers, ML helpers |
| `docs/` | Design notes, ADRs, onboarding guides. See [`docs/architecture.md`](docs/architecture.md) for the source of truth on structure, the action flow, and how to add a new service / handler / action. | |
| `tests/` | Cross-cutting / integration tests. Unit tests live next to the code they test. | |
| `configs/` | YAML/JSON config files for environments, robots, sensors. | |

---

## Separation of responsibilities

These rules exist so that the system stays modular and so that hardware, AI
research and student projects don't step on each other.

1. **Only `services/farmbot_gateway/` talks to the FarmBot hardware.**
   Nothing else may import the FarmBot HTTP/MQTT/serial client directly. If
   you need to move the robot, call `farmbot_gateway`. The reusable low-level
   client lives in `libs/farmbot_client/`; the gateway is the *only* user of
   it inside `services/`.

2. **`apps/` orchestrates, `services/` decide, `libs/` compute.**
   - `apps/api_server` and `apps/worker` wire services together and handle
     I/O (HTTP, queues, schedulers). They should contain little or no
     domain logic.
   - `services/<x>_service` owns the logic for one concern (sensors,
     watering, vision, planning, safety).
   - `libs/` contains pure, reusable utilities. No I/O, no global state.

3. **`core/` defines the shared vocabulary.**
   `Plant`, `Bed`, `SensorReading`, `Action`, `Event`, … are defined here and
   imported by everything else. Do not redefine equivalent types inside a
   service or a project.

4. **`safety_service` gates every action that affects the real world.**
   Any code path that ultimately moves the FarmBot (watering, weeding,
   tool changes, …) must pass through `safety_service` before it reaches
   `farmbot_gateway`.

5. **Student projects live in `projects/` and must stay isolated.**
   A project may import from `core/`, `libs/` and call the public APIs of
   `services/`. A project **must not**:
   - modify code inside `core/`, `services/`, `libs/`, or `apps/`,
   - talk to the FarmBot hardware directly (go via `farmbot_gateway`),
   - mutate shared state owned by another project.
   If shared functionality is missing, propose it upstream in `libs/` or
   `services/` instead of forking it inside the project.

6. **Experiments are reproducible, not interactive.**
   Code under `experiments/` reads configs from `configs/`, runs through
   `services/`, writes results into its own subfolder, and does not modify
   production state. Use the worker (`apps/worker`) for long-running
   experiment jobs.

7. **Configuration is data, not code.**
   Robot coordinates, sensor calibrations, watering thresholds, experiment
   parameters — all live in `configs/` as YAML/JSON and are loaded via
   `core/config`. Do not hard-code these values inside services.

8. **Tests live with the code.**
   Unit tests sit next to the module they test (e.g.
   `services/watering_service/tests/`). Cross-service and integration tests
   live in `tests/` at the repo root.

9. **Documentation goes in `docs/`.**
   Architecture decisions, onboarding, service contracts and experiment
   write-ups belong in `docs/` and should be referenced from the README of
   the relevant subfolder.

---

## Adding a new component

- **New service?** Create `services/<name>_service/` with its own README,
  tests, and a public API consumed by `apps/`.
- **New project?** Create `projects/<descriptive_name>/` with a short README
  describing the goal, inputs (which services it uses) and outputs.
- **New experiment?** Create `experiments/<descriptive_name>/` with the
  config, runner and an `outputs/` folder for results.
- **New shared helper?** If it's pure and reusable, put it in `libs/`. If it
  depends on services or I/O, it belongs in a service instead.
- **New shared type?** Add it to `core/domain/` and re-export from there.

If you're unsure where something belongs, prefer the **most general** folder
that fits (`libs/` over a service, `core/` over a project) and ask in a PR.

---

## Development setup

The repository is a [uv](https://docs.astral.sh/uv/) workspace.

```bash
# install everything (creates .venv and editable-installs every subpackage)
uv sync

# run the apps
uv run twfarmbot-ui
uv run twfarmbot-api
uv run twfarmbot-worker

# run the test suite
uv run pytest tests/
```

### Subpackages

| Folder | Distribution name | Purpose |
| --- | --- | --- |
| `core/` | `twfarmbot-core` | Shared `twfarmbot_core.domain`, `.config`, `.logging`, `.events` |
| `apps/ui/` | `twfarmbot-ui` | Dashboard, sensor display, manual triggers |
| `apps/api_server/` | `twfarmbot-api-server` | HTTP API for FarmBot, sensors and experiments |
| `apps/worker/` | `twfarmbot-worker` | Background jobs and experiment execution |

Each subpackage has its own `pyproject.toml` and can be developed and
released independently.

> If your shell exports a `PYTHONPATH` that points at system site-packages
> (for example ROS), run uv commands with `PYTHONPATH= uv run …` so that
> the venv isn't poisoned by incompatible system packages.