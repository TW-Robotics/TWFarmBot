# twfarmbot-api-server

HTTP API for FarmBot, sensors and experiments.

The API server is an orchestrator. It wires services together and exposes
them over HTTP. It must not contain domain logic of its own; that lives in
`services/`.
