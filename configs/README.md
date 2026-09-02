# configs

YAML / JSON configs for environments, robots, sensors, experiments.

Loaded at runtime via `twfarmbot_core.config`. Do not hard-code these values
inside services.

Udev example for payload USB cameras: `configs/udev/99-twfarmbot-cameras.rules`.
Fill `ATTRS{serial}` with `udevadm info --query=property --name=/dev/videoN`
(identify the camera first; software opens `/dev/camera-{band}`, never `/dev/videoN`).
