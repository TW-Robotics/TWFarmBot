# TWFarmBot

FarmBot implementation at UAS Technikum Wien — an open research platform for autonomous, AI-assisted precision farming.

[![Pylint](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/pylint.yml/badge.svg)](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/pylint.yml)
[![Ruff](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/ruff.yml/badge.svg)](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/ruff.yml)
[![Mypy](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/mypy.yml/badge.svg)](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/mypy.yml)
[![Tests](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/tests.yml/badge.svg)](https://github.com/TW-Robotics/TWFarmBot/actions/workflows/tests.yml)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-blue?logo=githubactions&logoColor=white)](https://github.com/TW-Robotics/TWFarmBot/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Contributors](https://img.shields.io/github/contributors/TW-Robotics/TWFarmBot)](https://github.com/TW-Robotics/TWFarmBot/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/TW-Robotics/TWFarmBot)](https://github.com/TW-Robotics/TWFarmBot/commits/main)
[![Issues](https://img.shields.io/github/issues/TW-Robotics/TWFarmBot)](https://github.com/TW-Robotics/TWFarmBot/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/TW-Robotics/TWFarmBot)](https://github.com/TW-Robotics/TWFarmBot/pulls)

---

## What is TWFarmBot?

TWFarmBot is a modular, service-oriented control system for a [FarmBot](https://farm.bot/) research installation. The Pi talks directly to the stock Farmduino firmware over USB serial — no cloud account, MQTT broker, or WiFi credentials required. On top of that local hardware layer it adds computer vision, large-language-model planning, sensor fusion and safety validation, while keeping student projects and experiments isolated from core infrastructure.

The repository is a **monorepo** split into apps, services, shared core libraries, student projects and reproducible experiments.

## Quick start

This is a [uv](https://docs.astral.sh/uv/) workspace. After cloning:

```bash
# Install all packages and create the virtual environment
uv sync

# Start the API server (connects to /dev/ttyACM0 by default)
uv run twfarmbot-api

# In another terminal, start the dashboard
uv run twfarmbot-ui

# Or start the local ReSiReg-Mini vision server
uv run resireg-server
```

Open the UI at `http://localhost:8501`, the API docs at `http://localhost:8000/docs`, and the ReSiReg server at `http://localhost:8080`.

To run the API without a live bot (UI-only mode):

```bash
FARMBOT_REQUIRED=0 uv run twfarmbot-api
```

## Running as services (auto-start + auto-restart)

Install the systemd user services once:

```bash
./scripts/install_services.sh
# Then, so services start at boot before anyone logs in:
sudo loginctl enable-linger farmbot
```

Start, stop, or restart everything:

```bash
./scripts/start_all.sh
./scripts/stop_all.sh
./scripts/restart_all.sh
```

View live logs:

```bash
./scripts/logs.sh
# or for a single service:
journalctl --user -u twfarmbot-api -f
```

Each service restarts automatically on failure. To disable auto-start on boot:

```bash
systemctl --user disable twfarmbot-resireg twfarmbot-api twfarmbot-ui
```

## Hardware smoke test

With the Farmduino connected:

```bash
# Read-only checks (position, endstops, firmware params)
PYTHONPATH= uv run python scripts/test_farmduino_local.py

# Home all axes (only when the bed is clear)
PYTHONPATH= uv run python scripts/test_farmduino_local.py --home

# Move 10 mm up in Z
PYTHONPATH= uv run python scripts/test_farmduino_local.py --move 0 0 10
```

## Continuous Integration

Every push and pull request is checked by GitHub Actions:

| Workflow | File | Purpose |
| --- | --- | --- |
| Pylint | [`.github/workflows/pylint.yml`](.github/workflows/pylint.yml) | Static analysis with Pylint |
| Ruff | [`.github/workflows/ruff.yml`](.github/workflows/ruff.yml) | Formatting and linting with Ruff |
| Mypy | [`.github/workflows/mypy.yml`](.github/workflows/mypy.yml) | Static type checking |
| Tests | [`.github/workflows/tests.yml`](.github/workflows/tests.yml) | Test suite with pytest |

View all runs on the [Actions tab](https://github.com/TW-Robotics/TWFarmBot/actions).

## Local checks

All CI checks can be reproduced locally from the workspace root:

```bash
uv run ruff format --check apps/ services/ libs/ core/ tests/
uv run ruff check apps/ services/ libs/ core/ tests/
uv run mypy apps/ services/ libs/ core/ tests/
uv run pylint apps/ services/ libs/ core/ tests/
uv run pytest tests/ -q
```

## Repository structure

| Folder | Purpose |
| --- | --- |
| `apps/` | Runnable applications: `ui` (Streamlit), `api_server` (FastAPI), `worker` (background jobs) |
| `core/` | Shared primitives: `Action`, `Point3D`, `GardenWorld`, config, logging, events |
| `services/` | One service per concern: safety, watering, planning, spatial |
| `projects/` | Isolated student / research projects |
| `experiments/` | Reproducible evaluations with their own configs and outputs |
| `libs/` | Reusable utilities: `farmbot_serial` (USB-serial driver), `ml_utils` |
| `tests/` | Cross-cutting and integration tests (unit tests live next to the code) |
| `configs/` | YAML/JSON environment, robot and sensor configuration |
| `docs/` | Architecture, ADRs and onboarding guides |
| `scripts/` | Operational helpers: systemd install, smoke tests, start/stop/restart |

See [`docs/architecture.md`](docs/architecture.md) for the full system design, action flow and how to add a new service or handler.

## Design principles

1. **Only `libs/farmbot_serial/` talks to the Farmduino.**
   The `services/watering_service/backends/direct_serial.py` backend is the single point that opens `/dev/ttyACM0`.
2. **`apps/` orchestrates, `services/` decides, `libs/` computes.**
   Keep I/O in apps, domain logic in services, pure helpers and the serial driver in libs.
3. **`core/` defines the shared vocabulary.**
   `Action`, `Point3D`, `GardenEntity`, `GardenWorld`, `Event`, … live here.
4. **`safety_service` gates every real-world action.**
   Watering, moving, tooling — all validated before execution.
5. **Student projects stay isolated in `projects/`.**
   They import from `core/` and `libs/` and call public service APIs, but never modify shared code.
6. **Experiments are reproducible.**
   Config-driven runs under `experiments/`, results in their own `outputs/` folders.
7. **Configuration is data, not code.**
   Robot coordinates, thresholds, experiment params live in `configs/`.
8. **Tests live with the code.**
   Unit tests sit next to modules; cross-service tests live in `tests/`.

## Subpackages

| Folder | Distribution | Purpose |
| --- | --- | --- |
| `core/` | `twfarmbot-core` | Shared domain, config, logging, events |
| `apps/ui/` | `twfarmbot-ui` | Streamlit dashboard |
| `apps/api_server/` | `twfarmbot-api-server` | FastAPI HTTP API |
| `apps/worker/` | `twfarmbot-worker` | Background jobs / experiments (skeleton) |

Each subpackage has its own `pyproject.toml` and can be developed independently.

## License

This project is licensed under the **GNU General Public License v3.0 or later** — see [`LICENSE`](LICENSE) for the full text.

## Contributing

- Follow the folder structure above.
- Keep code formatted with Ruff and typed with Mypy.
- Add tests for new behaviour.
- Update [`docs/architecture.md`](docs/architecture.md) for structural changes.
- Open a pull request against `main`.

> If your shell exports a `PYTHONPATH` that points at system site-packages (for example ROS), run uv commands with `PYTHONPATH= uv run …` so the venv isn't poisoned by incompatible system packages.
