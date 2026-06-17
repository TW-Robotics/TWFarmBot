from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    env: str = "dev"
    log_level: str = "INFO"


def load_settings() -> Settings:
    return Settings(
        env=os.getenv("FARMBOT_ENV", "dev"),
        log_level=os.getenv("FARMBOT_LOG_LEVEL", "INFO"),
    )
