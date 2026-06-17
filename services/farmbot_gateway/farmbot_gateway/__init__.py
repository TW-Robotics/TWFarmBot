"""farmbot_gateway — the only place that talks to the FarmBot hardware.

Other services and apps must import from here, never from
``farmbot_client`` directly.
"""

from __future__ import annotations

import logging
import os

from farmbot import Farmbot

from farmbot_client import FarmBotConnectionError, FarmBotCredentials, connect

log = logging.getLogger(__name__)


def _creds_from_env() -> FarmBotCredentials:
    return FarmBotCredentials(
        email=os.environ["FARMBOT_EMAIL"],
        password=os.environ["FARMBOT_PASSWORD"],
        server=os.environ.get("FARMBOT_SERVER", "https://my.farm.bot"),
        host=os.environ.get("FARMBOT_HOST", "farmbot.farm.bot"),
    )


class FarmBotLink:
    """Reconnecting handle around a cached :class:`Farmbot` instance.

    Wraps every attribute access with a liveness check on the underlying
    paho-mqtt client; if the broker dropped, transparently rebuilds.
    """

    def __init__(self, fb: Farmbot) -> None:
        self._fb: Farmbot = fb

    def _is_alive(self) -> bool:
        try:
            client = self._fb.broker.client
        except AttributeError:
            return False
        is_connected = getattr(client, "is_connected", None)
        return bool(is_connected()) if callable(is_connected) else False

    def _ensure_alive(self) -> Farmbot:
        if not self._is_alive():
            log.warning("FarmBot MQTT link lost, reconnecting")
            self._fb = connect(_creds_from_env())
            log.info("FarmBot reconnected")
        return self._fb

    def __getattr__(self, name: str):
        # __getattr__ only fires when normal lookup fails, so `self._fb`
        # access above is safe (it lives on the instance, not via getattr).
        return getattr(self._ensure_alive(), name)


_fb_singleton: FarmBotLink | None = None


def get_farmbot() -> FarmBotLink:
    """Singleton, lazily-connected FarmBot handle (WiFi/MQTT)."""
    global _fb_singleton
    if _fb_singleton is None:
        log.info("connecting to FarmBot over WiFi")
        try:
            _fb_singleton = FarmBotLink(connect(_creds_from_env()))
        except FarmBotConnectionError:
            log.exception("initial FarmBot connection failed")
            raise
        log.info("FarmBot connected")
    return _fb_singleton
