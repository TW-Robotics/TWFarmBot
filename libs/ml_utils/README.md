# ml_utils

Reusable ML helpers (metrics, model loading, evaluation, hosted-model clients).
No FarmBot business logic or global state.

`VisionProcessor` is a thin client for the local OpenAI-compatible
ReSiReg-Mini vision server (`apps/resireg_server`). It sends an image URL
and prompt to `/v1/chat/completions` and returns the decoded result image
path(s). UI rendering remains inside the UI app.
