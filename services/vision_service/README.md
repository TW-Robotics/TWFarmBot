# vision_service

Payload USB camera capture. One primitive: `capture(band) → artifact_id`.

`band` is required and must be one of `rgb`, `nir`, `rededge`, `thermal`, `swir`.

- `rgb` / `nir` / `rededge` share one UVC backend. Open `/dev/camera-{band}` from config (udev `SYMLINK+=camera-{band}`). Missing node fails. Never `/dev/videoN`, never a fallback webcam.
- `thermal` / `swir` fail closed (bus not pinned). No USB assumption, no stub JPEG.
- Not a wrap of `take_photo` / `POST /photo`. Does not go through `farmbot_gateway`.

## udev

Example rules: [`configs/udev/99-twfarmbot-cameras.rules`](../../configs/udev/99-twfarmbot-cameras.rules).

Fill `ATTRS{serial}` from the physical camera (do not use that `/dev/videoN` node in software):

```bash
udevadm info --query=property --name=/dev/video0 | grep -E 'ID_SERIAL|ID_USB_SERIAL'
# or, walking the device tree:
udevadm info -a -n /dev/video0 | grep serial
```

Install on the Pi: copy the rules to `/etc/udev/rules.d/` and run `sudo udevadm control --reload-rules && sudo udevadm trigger`.
