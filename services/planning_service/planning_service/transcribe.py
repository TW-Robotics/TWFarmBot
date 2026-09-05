"""Speech-to-text via OpenAI, with Muse Voice as fallback."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import PlannerConfig, resolve_openai_api_key, resolve_voice_api_key

TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"
TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"
_KEYWORDS = ["FarmBot", "NDRE", "NIR", "red edge", "gantry"]


def _api_key() -> str | None:
    return resolve_voice_api_key()


def _openai_transcribe(audio: bytes, fmt: str, timeout: float) -> str:
    key = resolve_openai_api_key()
    if not key:
        raise RuntimeError("no OpenAI API key configured for transcription")
    try:
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (f"audio.{fmt}", audio, f"audio/{fmt}")},
            data={"model": "gpt-4o-mini-transcribe"},
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


def transcribe_audio(
    audio: bytes,
    *,
    fmt: str = "wav",
    config: PlannerConfig | None = None,
) -> str:
    """Transcribe a clip via OpenAI, falling back to Muse Voice."""
    if not audio:
        raise ValueError("audio is empty")
    fmt = (fmt or "wav").lower().lstrip(".")
    timeout = max(90.0, config.timeout_s) if config else 90.0
    if resolve_openai_api_key():
        return _openai_transcribe(audio, fmt, timeout)
    if fmt != "wav":
        raise ValueError("Muse Voice Transcribe requires a 16 kHz or 24 kHz WAV clip")
    key = _api_key()
    if not key:
        raise RuntimeError("no API key configured for transcription")
    files = {
        "request": (
            None,
            json.dumps(
                {
                    "model": TRANSCRIBE_MODEL,
                    "audioEncoding": "WAV",
                    "mode": "PUSH_TO_TALK",
                    "keywords": _KEYWORDS,
                    "languageBias": ["English"],
                }
            ),
            "application/json",
        ),
        "audio": ("audio.wav", audio, "audio/wav"),
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    try:
        response = requests.post(
            TRANSCRIBE_URL, headers=headers, files=files, timeout=timeout
        )
        if response.status_code == 429:
            time.sleep(2)
            response = requests.post(
                TRANSCRIBE_URL, headers=headers, files=files, timeout=timeout
            )
    except requests.RequestException as err:
        raise RuntimeError(f"transcription request failed: {err}") from err
    if response.status_code >= 400:
        detail = (response.text or "")[:300]
        raise RuntimeError(f"transcription failed ({response.status_code}): {detail}")
    data: Any = response.json() if response.content else {}
    text = data.get("transcript") if isinstance(data, dict) else None
    if not isinstance(text, str):
        raise RuntimeError("transcription response missing transcript")
    return text.strip()
