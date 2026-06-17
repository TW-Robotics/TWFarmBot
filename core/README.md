# twfarmbot-core

Cross-cutting primitives shared by every app and service.

Subpackages:

- `domain/` — shared concepts: `Plant`, `Bed`, `SensorReading`, `Action`, …
- `config/` — typed settings and environment loading.
- `logging/` — structured logging setup.
- `events/` — internal event bus and message contracts.
