"""farmbot_gateway — the only place that talks to FarmBot hardware.

Wraps the TWFarmbotOS HTTP API via ``farmbot_client``. Other
services must import from here, never from ``farmbot_client`` directly.
"""

from __future__ import annotations

import logging

from farmbot_client import FarmBotConnectionError, get_client

from . import farmbot
from .farmbot import FarmBotBackend, backend

log = logging.getLogger(__name__)


class FarmBotLink:
    """Handle around the local HTTP client."""

    def cached_xyz(self) -> dict[str, float]:
        try:
            return get_client().get_xyz()
        except FarmBotConnectionError:
            return {}

    def cached_last_messages(self):
        return None

    def __getattr__(self, name: str):
        return getattr(get_client(), name)


_fb_singleton: FarmBotLink | None = None


def get_farmbot() -> FarmBotLink:
    """Singleton handle around the local HTTP client (connects on first request)."""
    global _fb_singleton
    if _fb_singleton is None:
        _fb_singleton = FarmBotLink()
        log.info("FarmBot local HTTP client ready (%s)", get_client().url)
    return _fb_singleton


__all__ = [
    "FarmBotBackend",
    "FarmBotConnectionError",
    "FarmBotLink",
    "backend",
    "farmbot",
    "get_client",
    "get_farmbot",
]
