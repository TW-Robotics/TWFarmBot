"""Offline tests for Muse Voice Transcribe."""

from __future__ import annotations

import json
from typing import Any

import pytest

from planning_service.config import PlannerConfig
from planning_service.transcribe import TRANSCRIBE_MODEL, TRANSCRIBE_URL, transcribe_audio


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PLANNING_LLM_API_KEY", raising=False)


def _cfg() -> PlannerConfig:
    return PlannerConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="unused",
        api_key="llm-key",
        timeout_s=30.0,
        temperature=0.0,
    )


def test_transcribe_audio_posts_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b'{"transcript":" water the tomatoes "}'
        text = '{"transcript":" water the tomatoes "}'

        def json(self) -> dict[str, str]:
            return {"transcript": " water the tomatoes "}

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        seen["files"] = kwargs.get("files")
        return _Resp()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MUSE_VOICE_API_KEY", "muse-key")
    monkeypatch.setattr("planning_service.transcribe.requests.post", fake_post)
    text = transcribe_audio(b"RIFF", fmt="wav", config=_cfg())
    assert text == "water the tomatoes"
    assert seen["url"] == TRANSCRIBE_URL
    assert seen["headers"]["Authorization"] == "Bearer muse-key"
    request = json.loads(seen["files"]["request"][1])
    assert request["model"] == TRANSCRIBE_MODEL
    assert request["audioEncoding"] == "WAV"
    assert request["mode"] == "PUSH_TO_TALK"
    assert seen["files"]["audio"][0] == "audio.wav"


def test_transcribe_audio_retries_once_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _Resp:
        def __init__(self, status: int, text: str = "") -> None:
            self.status_code = status
            self.text = text
            self.content = text.encode()

        def json(self) -> dict[str, str]:
            return {"transcript": "ok"}

    def fake_post(*_args: Any, **_kwargs: Any) -> _Resp:
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(429, "rate limited")
        return _Resp(200, '{"transcript":"ok"}')

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MUSE_VOICE_API_KEY", "muse-key")
    monkeypatch.setattr("planning_service.transcribe.requests.post", fake_post)
    monkeypatch.setattr("planning_service.transcribe.time.sleep", lambda _s: None)
    assert transcribe_audio(b"RIFF", fmt="wav", config=_cfg()) == "ok"
    assert calls["n"] == 2


def test_transcribe_audio_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        transcribe_audio(b"", config=_cfg())


def test_transcribe_audio_requires_wav() -> None:
    with pytest.raises(ValueError, match="WAV"):
        transcribe_audio(b"RIFF", fmt="webm", config=_cfg())


def test_transcribe_audio_requires_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_VOICE_API_KEY", raising=False)
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        transcribe_audio(b"RIFF", fmt="wav", config=_cfg())


def test_transcribe_uses_stored_voice_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from planning_service import config as planning_config

    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_VOICE_API_KEY", raising=False)
    monkeypatch.delenv("META_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    planning_config.write_stored_voice_key("muse-stored")

    class _Resp:
        status_code = 200
        content = b'{"transcript":"hello"}'
        text = '{"transcript":"hello"}'

        def json(self) -> dict[str, str]:
            return {"transcript": "hello"}

    seen: dict[str, Any] = {}

    def fake_post(_url: str, **kwargs: Any) -> _Resp:
        seen["headers"] = kwargs.get("headers")
        return _Resp()

    monkeypatch.setattr("planning_service.transcribe.requests.post", fake_post)
    assert transcribe_audio(b"RIFF", fmt="wav", config=_cfg()) == "hello"
    assert seen["headers"]["Authorization"] == "Bearer muse-stored"


def test_openai_transcribe_used_when_openai_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("TWFB_LLM_KEYS_FILE", str(tmp_path / "llm_keys.json"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    seen: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b'{"text":"hello openai"}'
        text = '{"text":"hello openai"}'

        def json(self) -> dict[str, str]:
            return {"text": "hello openai"}

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen["data"] = kwargs.get("data")
        return _Resp()

    monkeypatch.setattr("planning_service.transcribe.requests.post", fake_post)
    assert transcribe_audio(b"RIFF", fmt="wav", config=_cfg()) == "hello openai"
    assert seen["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert seen["data"]["model"] == "gpt-4o-mini-transcribe"
