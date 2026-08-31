# twfarmbot-ui

Native Astryx dashboard. The browser talks to `twfarmbot-api`; this package
serves the React app and a few host-local endpoints (garden YAML writes,
ReSiReg vision, session listing).

```bash
cd apps/ui && npm install && npm run build   # once, or after frontend changes
uv run twfarmbot-ui                          # http://localhost:8501
```

Frontend HMR during development (same port):

```bash
cd apps/ui && npm run dev   # http://localhost:8501
```

`TWFB_API_URL` defaults to `http://127.0.0.1:8000` in the browser settings page.
`TWFB_RESIREG_URL` is used by the Camera analysis proxy on the UI server.
