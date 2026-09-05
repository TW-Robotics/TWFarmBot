"""Mint ephemeral OpenAI Realtime client secrets for the browser."""

from __future__ import annotations

from typing import Any

import requests

from .config import resolve_openai_api_key

REALTIME_MODEL = "gpt-realtime-2.1"
REALTIME_VOICE = "marin"
CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"

SESSION_CONFIG: dict[str, Any] = {
    "session": {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "instructions": (
            "You are a text-to-speech voice for a farm robot assistant. "
            "When asked to speak, read the provided text aloud verbatim. "
            "Do not add words, answer questions, or call tools."
        ),
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "transcription": {"model": "gpt-4o-mini-transcribe"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                    "create_response": False,
                    "interrupt_response": True,
                },
            },
            "output": {"voice": REALTIME_VOICE},
        },
    }
}


def create_client_secret() -> dict[str, Any]:
    """Return a short-lived Realtime token. Never includes the long-lived key."""
    key = resolve_openai_api_key()
    if not key:
        raise RuntimeError(
            "OpenAI API key required for voice. Save an OpenAI key in Settings."
        )
    try:
        response = requests.post(
            CLIENT_SECRETS_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=SESSION_CONFIG,
            timeout=20,
        )
    except requests.RequestException as err:
        raise RuntimeError(f"OpenAI Realtime request failed: {err}") from err
    if response.status_code >= 400:
        detail = (response.text or "")[:300]
        raise RuntimeError(f"OpenAI Realtime failed ({response.status_code}): {detail}")
    data = response.json() if response.content else {}
    if not isinstance(data, dict):
        raise RuntimeError("OpenAI Realtime returned an unexpected body")
    secret = data.get("value")
    if not secret:
        nested = data.get("client_secret")
        if isinstance(nested, dict):
            secret = nested.get("value")
    if not isinstance(secret, str) or not secret:
        raise RuntimeError("OpenAI Realtime response missing client secret")
    return {
        "value": secret,
        "expires_at": data.get("expires_at"),
        "model": REALTIME_MODEL,
        "voice": REALTIME_VOICE,
    }
