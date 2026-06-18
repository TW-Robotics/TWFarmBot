# ml_utils

Reusable ML helpers (metrics, model loading, evaluation, hosted-model clients).
No FarmBot business logic or global state.

`HuggingFaceImageProcessor` is a thin adapter around the official
`gradio-client` package. It sends a public image URL and prompt to a named
Space endpoint and returns the downloaded result path. UI rendering remains
inside the UI app.
