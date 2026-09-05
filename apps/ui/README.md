# twfarmbot-ui

Native Astryx dashboard. The browser talks to `twfarmbot-api`; this package
serves the React app and a few host-local endpoints (garden YAML writes,
ReSiReg vision, session listing).

```bash
cd apps/ui && npm install && npm run build   # once, or after frontend changes
uv run twfarmbot-ui                          # https://localhost:8501 (self-signed TLS)
```

Frontend HMR during development (same port):

```bash
cd apps/ui && npm run dev   # http://localhost:8501
```

The browser talks to the API through a same-origin `/api` proxy (avoids mixed
content on HTTPS). `TWFB_API_URL` is the UI server's upstream, default
`http://127.0.0.1:8000`. Set `TWFB_UI_TLS=0` to serve plain HTTP (remote
microphone will not work). `TWFB_RESIREG_URL` is used by the Camera analysis
proxy on the UI server.
