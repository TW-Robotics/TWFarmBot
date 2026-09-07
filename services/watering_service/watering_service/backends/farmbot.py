"""Backward-compatibility shim. FarmBotBackend has moved to farmbot_gateway."""

from __future__ import annotations

from farmbot_gateway import farmbot
from farmbot_gateway.farmbot import FarmBotBackend, backend

__all__ = ["FarmBotBackend", "backend", "farmbot"]
