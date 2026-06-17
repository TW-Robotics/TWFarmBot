"""Thin wrapper around the official ``farmbot`` (farmbot-py) library.

Everything FarmBot-specific lives here. The rest of the system goes through
``services/farmbot_gateway`` and never touches this module directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep

from farmbot import Farmbot


class FarmBotConnectionError(RuntimeError):
    """Raised when a WiFi/MQTT connection to the FarmBot cannot be established."""


@dataclass(frozen=True)
class FarmBotCredentials:
    email: str
    password: str
    server: str = "https://my.farm.bot"   # REST API host for token issuance
    host: str = "farmbot.farm.bot"         # MQTT broker (cloud default)
    token: str | None = None               # optional pre-fetched token JSON


def connect(
    creds: FarmBotCredentials,
    *,
    debug: bool = False,
    timeout_s: float = 30.0,
    retries: int = 3,
    backoff_s: float = 1.0,
) -> Farmbot:
    """Connect to a FarmBot over WiFi via MQTT and return a ready client.

    Sequence (per farmbot-py):
      1. ``Farmbot()`` — instantiate the client.
      2. ``fb.api.get_token(email, password, server=...)`` — REST auth.
      3. ``fb.broker.connect()`` — open MQTT using the issued token.

    Raises :class:`FarmBotConnectionError` if no connection is established
    within ``timeout_s`` total, across up to ``retries`` attempts.
    """
    deadline = monotonic() + timeout_s
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        fb = Farmbot()
        if debug:
            fb.state.verbosity = "debug"
        try:
            if creds.token is None:
                result = fb.api.get_token(creds.email, creds.password, server=creds.server)
                if not isinstance(result, dict) or "token" not in result:
                    raise FarmBotConnectionError(
                        f"auth failed for {creds.email!r}: {result}"
                    )
            else:
                fb.state.token = creds.token
            fb.broker.connect()  # blocking until connected; raises on failure
            return fb
        except Exception as err:  # noqa: BLE001 — farmbot-py raises broad types
            last_err = err
            if monotonic() >= deadline or attempt == retries:
                break
            sleep(backoff_s * attempt)
    raise FarmBotConnectionError(
        f"could not connect to FarmBot {creds.email!r} on {creds.server!r} "
        f"after {retries} attempts"
    ) from last_err
