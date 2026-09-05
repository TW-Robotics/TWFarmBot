"""Offline tests for OpenAI dictate transcription."""

from __future__ import annotations

from typing import Any

import pytest

from planning_service.config import PlannerConfig
from planning_service.transcribe import TRANSCRIBE_MODEL, TRANSCRIBE_URL, transcribe_audio


def _cfg() -> PlannerConfig:
    return PlannerConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="unused",
        api_key="llm-key",
        timeout_s=30.0,
        temperature=0.0,
    )


def test_transcribe_audio_posts_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    seen: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b'{"text":" water the tomatoes "}'
        text = '{"text":" water the tomatoes "}'

        def json(self) -> dict[str, str]:
            return {"text": " water the tomatoes "}

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        seen["data"] = kwargs.get("data")
        return _Resp()

    monkeypatch.setattr("planning_service.transcribe.requests.post", fake_post)
    text = transcribe_audio(b"RIFF", fmt="webm", config=_cfg())
    assert text == "water the tomatoes"
    assert seen["url"] == TRANSCRIBE_URL
    assert seen["headers"]["Authorization"] == "Bearer sk-openai"
    assert seen["data"]["model"] == TRANSCRIBE_MODEL


def test_transcribe_audio_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        transcribe_audio(b"", config=_cfg())


def test_transcribe_audio_requires_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        transcribe_audio(b"RIFF", fmt="webm", config=_cfg())
