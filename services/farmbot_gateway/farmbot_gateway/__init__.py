"""farmbot_gateway — the only place that talks to the FarmBot hardware.

Other services and apps must import from here, never from
``farmbot_client`` directly.
"""

from __future__ import annotations

import logging
import os

from farmbot import Farmbot
from farmbot.functions import broker as _broker

from farmbot_client import FarmBotConnectionError, FarmBotCredentials, connect

log = logging.getLogger(__name__)


def _creds_from_env() -> FarmBotCredentials:
    return FarmBotCredentials(
        email=os.environ["FARMBOT_EMAIL"],
        password=os.environ["FARMBOT_PASSWORD"],
        server=os.environ.get("FARMBOT_SERVER", "https://my.farm.bot"),
        host=os.environ.get("FARMBOT_HOST", "farmbot.farm.bot"),
    )


def _install_broker_loop_patches() -> None:
    """Make the paho-mqtt network loop persistent for the process lifetime.

    ``farmbot-py's`` ``BrokerConnect`` calls ``client.loop_start()`` at the
    start of every command and ``client.loop_stop()`` at the end. Paho-mqtt
    does not reliably recover from repeated ``loop_stop`` / ``loop_start``
    cycles — the network thread can be left stopped while the client still
    reports itself as connected, which silently breaks every subsequent
    command (RPCs time out, no keepalives are sent, and the broker drops the
    session without our code noticing).

    We patch two methods on ``BrokerConnect`` exactly once per process so
    that a single ``loop_start()`` runs for the entire ``Farmbot`` instance.
    """
    if getattr(_broker.BrokerConnect, "_loop_patched", False):
        return

    _original_start_listen = _broker.BrokerConnect.start_listen

    def start_listen(self, channel="#", message_options=None):
        _original_start_listen(self, channel, message_options)
        # The original method calls loop_start() unconditionally; we've
        # already started a persistent loop in connect(). If a loop is
        # already running, calling loop_start() again is a paho-mqtt
        # no-op (and prints a warning). Suppress the second call.
        client = self.client
        if client is None:
            return
        try:
            client.loop_start()
        except RuntimeError:
            # Paho raises "Loop already started" on some versions; safe to ignore.
            pass

    def stop_listen(self):
        # Intentionally a no-op: tearing down the paho network loop after
        # every command is what causes the intermittent disconnects. The
        # loop is shut down for good by BrokerConnect.disconnect().
        return

    _broker.BrokerConnect.start_listen = start_listen
    _broker.BrokerConnect.stop_listen = stop_listen
    _broker.BrokerConnect._loop_patched = True


_install_broker_loop_patches()


class FarmBotLink:
    """Reconnecting handle around a cached :class:`Farmbot` instance.

    Wraps every attribute access with a liveness check on the underlying
    paho-mqtt client; if the broker dropped, transparently rebuilds.
    """

    def __init__(self, fb: Farmbot) -> None:
        self._fb: Farmbot = fb
        self._install_disconnect_logger()

    def _install_disconnect_logger(self) -> None:
        """Log unexpected broker drops so they're visible in our logs."""
        try:
            client = self._fb.broker.client
        except AttributeError:
            return

        log_ref = log

        def on_disconnect(_client, _userdata, reason_code, _properties=None):
            log_ref.warning(
                "FarmBot MQTT disconnected (reason=%s); will reconnect on next use",
                reason_code,
            )

        try:
            client.on_disconnect = on_disconnect
        except Exception:  # noqa: BLE001 — never let logging break the link
            log.exception("could not register on_disconnect callback")

    def _is_alive(self) -> bool:
        try:
            client = self._fb.broker.client
        except AttributeError:
            return False
        if client is None:
            return False
        is_connected = getattr(client, "is_connected", None)
        return bool(is_connected()) if callable(is_connected) else False

    def _ensure_alive(self) -> Farmbot:
        if not self._is_alive():
            log.warning("FarmBot MQTT link lost, reconnecting")
            try:
                self._fb.broker.disconnect()
            except Exception:  # noqa: BLE001
                log.exception("error during stale broker disconnect")
            self._fb = connect(_creds_from_env())
            self._install_disconnect_logger()
            log.info("FarmBot reconnected")
        return self._fb

    def cached_xyz(self) -> dict[str, float]:
        """Return the last MQTT position without reconnecting or issuing a read."""
        try:
            statuses = self._fb.state.last_messages.get("status", [])
            position = statuses[-1]["content"]["location_data"]["position"]
        except (AttributeError, IndexError, KeyError, TypeError):
            return {}
        return {
            axis: float(position[axis])
            for axis in ("x", "y", "z")
            if position.get(axis) is not None
        }

    def cached_last_messages(self):
        """Return the last MQTT messages without triggering reconnection."""
        return getattr(self._fb.state, "last_messages", None)

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
