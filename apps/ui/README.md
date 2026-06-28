# twfarmbot-ui

Dashboard, sensor display, and manual triggers.

This app is a thin orchestration layer on top of `twfarmbot-core`. It should
not contain domain logic; it only renders state and forwards user actions
to the API server.

## Camera AI analysis

The Camera tab sends a selected FarmBot image to the local ReSiReg-Mini
vision server. It supports open-language similarity, zero-shot segmentation,
PCA feature visualization, and traversability estimation. The returned
analysis images are displayed next to the source image. The model adapter
lives in `libs/ml_utils` and is not connected to watering, motion, sensors,
or other robot actions.

The local server URL can be changed without code changes:

```bash
TWFB_RESIREG_URL=http://127.0.0.1:8080 uv run twfarmbot-ui
```
