# Architecture overview

This document gives a high-level view of the TWFarmBot system. For folder
layout and responsibility rules see the top-level `README.md`.

## Apps

- `apps/ui` — dashboard, sensor display, manual triggers.
- `apps/api_server` — HTTP API for FarmBot, sensors and experiments.
- `apps/worker` — background jobs and long-running experiment execution.

Apps orchestrate. They wire services together and handle I/O.

## Core

- `core/domain` — shared types (`Plant`, `Bed`, `SensorReading`, `Action`).
- `core/config` — typed settings, environment loading.
- `core/logging` — structured logging setup.
- `core/events` — internal event bus.

## Services

One concern per service. Services decide.

- `farmbot_gateway` — only place that talks to the FarmBot hardware.
- `sensor_service` — soil, temperature, light, …
- `watering_service` — irrigation logic.
- `vision_service` — camera + pure vision models.
- `planning_service` — LLM / VLM task planning.
- `safety_service` — gates every real-world action.

## Projects

Self-contained student / research projects under `projects/`. They consume
`core/`, `libs/`, and the public APIs of `services/`, and must not modify
shared code or talk to hardware directly.

## Experiments

Reproducible experiment runners under `experiments/`. They read configs,
invoke services, and write results into their own subfolder.

## Libraries

Pure, reusable utilities under `libs/`. No business logic, no I/O, no global
state.
