# twfarmbot-resireg-server

ReSiReg-Mini vision server integrated into the TWFarmBot project.

Runs an OpenAI-compatible HTTP server for dense similarity, zero-shot segmentation,
traversability, and embeddings using `SimonSchwaiger/resireg_mini`.

## Run

```bash
uv run resireg-server
```

Optional environment variables:
- `RESIREG_HOST` — bind host (default: `0.0.0.0`)
- `RESIREG_PORT` — bind port (default: `8080`)
