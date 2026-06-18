# twfarmbot-ui

Dashboard, sensor display, and manual triggers.

This app is a thin orchestration layer on top of `twfarmbot-core`. It should
not contain domain logic; it only renders state and forwards user actions
to the API server.

## Camera AI analysis

The Camera tab can send a selected FarmBot image and a research prompt to the
Eupe-Lang Hugging Face Space. The returned similarity image is displayed next
to its source image. The external model adapter lives in `libs/ml_utils` and
is not connected to watering, motion, sensors, or other robot actions.

The Space can be changed without code changes:

```bash
TWFB_AI_SPACE_ID=owner/space-name uv run twfarmbot-ui
```
