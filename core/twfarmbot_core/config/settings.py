from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    env: str = "dev"
    log_level: str = "INFO"


def load_settings() -> Settings:
    return Settings(
        env=os.getenv("FARMBOT_ENV", "dev"),
        log_level=os.getenv("FARMBOT_LOG_LEVEL", "INFO"),
    )


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dev.yaml"


def load_yaml_config(
    path: str | Path | None = None,
    env_var: str = "TWFB_CONFIG",
) -> dict[str, Any]:
    """Load the project YAML config (``configs/dev.yaml`` by default).

    Resolution order:
      1. ``path`` argument if provided.
      2. ``os.getenv(env_var)`` if set.
      3. ``configs/dev.yaml`` relative to the project root.

    Returns an empty dict if the file does not exist or is empty.
    """
    if path is None:
        path = Path(os.getenv(env_var, _DEFAULT_CONFIG_PATH))
    else:
        path = Path(path)
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}
