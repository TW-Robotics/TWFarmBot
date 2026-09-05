"""Speech-to-text via OpenRouter (MAI-Transcribe 2)."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from .config import PlannerConfig, resolve_api_key_for

TRANSCRIBE_MODEL = "microsoft/mai-transcribe-2"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_FORMATS = frozenset({"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"})


def normalize_audio_format(value: str | None) -> str:
    raw = (value or "webm").strip().lower()
    if raw.startswith("audio/"):
        raw = raw.split("/", 1)[1]
    raw = raw.split(";")[0].strip()
    if raw in {"mpeg", "mpga"}:
        raw = "mp3"
    if raw in {"x-m4a", "mp4"}:
        raw = "m4a"
    if raw not in _FORMATS:
        raise ValueError(f"unsupported audio format {value!r}")
    return raw


def openrouter_stt_credentials(cfg: PlannerConfig) -> tuple[str, str]:
    """Base URL + API key for OpenRouter STT (chat may use another provider)."""
    key = resolve_api_key_for("openrouter")
    base = OPENROUTER_BASE
    if cfg.provider == "openrouter":
        base = (cfg.base_url or OPENROUTER_BASE).rstrip("/")
        key = cfg.api_key or key
    if not key:
        raise ValueError("OpenRouter API key required for transcription")
    return base, key


def transcribe_audio(
    audio_bytes: bytes,
    *,
    fmt: str,
    config: PlannerConfig,
    model: str = TRANSCRIBE_MODEL,
) -> str:
    """Return transcript text from MAI-Transcribe 2 (or another STT slug)."""
    fmt = normalize_audio_format(fmt)
    if not audio_bytes:
        raise ValueError("audio is empty")
    base, key = openrouter_stt_credentials(config)
    payload = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": fmt,
        },
    }
    response = httpx.post(
        f"{base}/audio/transcriptions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=max(30.0, float(config.timeout_s)),
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        detail = _error_detail(err.response)
        raise RuntimeError(detail) from err
    return _text_from_body(response.json())


def _text_from_body(body: Any) -> str:
    if isinstance(body, dict):
        text = body.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        nested = body.get("error")
        if isinstance(nested, dict) and nested.get("message"):
            raise RuntimeError(str(nested["message"]))
        if isinstance(nested, str) and nested:
            raise RuntimeError(nested)
    raise RuntimeError("transcription returned no text")


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return response.text or f"transcription failed ({response.status_code})"
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            message = str(err["message"])
            meta = err.get("metadata")
            raw = meta.get("raw") if isinstance(meta, dict) else None
            if raw:
                return f"{message}: {raw}"
            return message
        if isinstance(err, str):
            return err
        if body.get("detail"):
            return str(body["detail"])
    return f"transcription failed ({response.status_code})"
