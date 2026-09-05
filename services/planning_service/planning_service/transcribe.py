"""Speech-to-text via OpenAI for the Assistant dictate button."""

from __future__ import annotations

from typing import Any

import requests

from .config import PlannerConfig, resolve_openai_api_key

TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"


def transcribe_audio(
    audio: bytes,
    *,
    fmt: str = "webm",
    config: PlannerConfig | None = None,
) -> str:
    if not audio:
        raise ValueError("audio is empty")
    fmt = (fmt or "webm").lower().lstrip(".")
    key = resolve_openai_api_key()
    if not key:
        raise RuntimeError("OpenAI API key required for transcription")
    timeout = max(90.0, config.timeout_s) if config else 90.0
    try:
        response = requests.post(
            TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (f"audio.{fmt}", audio, f"audio/{fmt}")},
            data={"model": TRANSCRIBE_MODEL},
            timeout=timeout,
        )
    except requests.RequestException as err:
        raise RuntimeError(f"transcription request failed: {err}") from err
    if response.status_code >= 400:
        detail = (response.text or "")[:300]
        raise RuntimeError(f"transcription failed ({response.status_code}): {detail}")
    data: Any = response.json() if response.content else {}
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str):
        raise RuntimeError("transcription response missing text")
    return text.strip()
