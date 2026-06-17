"""Smoke tests for the FarmBot WiFi client + gateway.

These do NOT talk to a real FarmBot. They cover:
- importability / public API shape,
- the bounded connect() path (bad host -> typed error, fast),
- credential construction and gateway cred reading.

Real end-to-end tests against a live FarmBot belong in ``tests/integration/``
and are gated behind env vars (see ``test_farmbot_connection_live``).
"""

from __future__ import annotations

import os

import pytest

from farmbot_client import (
    FarmBotConnectionError,
    FarmBotCredentials,
    connect,
)


def test_credentials_defaults() -> None:
    creds = FarmBotCredentials(email="a@b", password="pw")
    assert creds.server == "https://my.farm.bot"
    assert creds.host == "farmbot.farm.bot"
    assert creds.token is None


def test_connect_fails_fast_on_bad_server() -> None:
    creds = FarmBotCredentials(
        email="x@y", password="bad", server="http://127.0.0.1:1"
    )
    with pytest.raises(FarmBotConnectionError):
        connect(creds, timeout_s=2.0, retries=1, backoff_s=0.1)


def test_connect_fails_fast_on_bad_mqtt_host() -> None:
    creds = FarmBotCredentials(
        email="x@y", password="bad",
        server="https://my.farm.bot",
        host="127.0.0.1",
    )
    with pytest.raises(FarmBotConnectionError):
        connect(creds, timeout_s=2.0, retries=1, backoff_s=0.1)


def test_gateway_creds_require_email_and_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FARMBOT_EMAIL", raising=False)
    monkeypatch.delenv("FARMBOT_PASSWORD", raising=False)
    from farmbot_gateway import _creds_from_env
    with pytest.raises(KeyError):
        _creds_from_env()


def test_gateway_creds_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FARMBOT_EMAIL", "a@b")
    monkeypatch.setenv("FARMBOT_PASSWORD", "pw")
    monkeypatch.delenv("FARMBOT_SERVER", raising=False)
    monkeypatch.delenv("FARMBOT_HOST", raising=False)
    from farmbot_gateway import _creds_from_env
    creds = _creds_from_env()
    assert creds.email == "a@b"
    assert creds.password == "pw"
    assert creds.server == "https://my.farm.bot"
    assert creds.host == "farmbot.farm.bot"


@pytest.mark.skipif(
    not os.getenv("FARMBOT_LIVE_TEST"),
    reason="set FARMBOT_LIVE_TEST=1 to run against a real FarmBot",
)
def test_farmbot_connection_live() -> None:
    """Opt-in: actually talks to the FarmBot. Requires:

      FARMBOT_LIVE_TEST=1
      FARMBOT_EMAIL, FARMBOT_PASSWORD  (and optionally FARMBOT_SERVER/FARMBOT_HOST)
    """
    from farmbot_gateway import get_farmbot

    fb = get_farmbot()
    xyz = fb.get_xyz()
    assert xyz is not None
