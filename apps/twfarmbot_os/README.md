# TWFarmbotOS

Runs on the Raspberry Pi. Owns the Farmduino serial port, waits for R-codes,
and exposes HTTP on `:3001` for `farmbot_client`.

```bash
# simulation (default)
uv run --env-file=.env twfarmbot-os

# real Farmduino — config approval (F22 P2 V1 Q0) runs automatically on boot
FARMBOT_TRANSPORT=serial FARMBOT_SERIAL_PATH=/dev/ttyACM0 uv run --env-file=.env twfarmbot-os
```

Or use `./scripts/start_all.sh`, which starts `twfarmbot-os` with `.env` already loaded.

Camera: `rpicam-still`, `libcamera-still`, or `fswebcam` if present; otherwise a stub JPEG.
Override with `FARMBOT_CAMERA_CMD` / `FARMBOT_PHOTO_DIR`.
