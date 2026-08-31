# twfarmbot-worker

Scheduled FarmBot jobs. The inspect loop POSTs `inspect_zone` for every
configured bed and writes a history card under `data/ui_sessions/`.

```
TWFB_API_URL=http://127.0.0.1:8000
TWFB_INSPECT_INTERVAL_S=21600   # 0 disables
TWFB_INSPECT_ON_START=0
```
