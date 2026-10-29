"""Offline tests for dictate transcription: OpenAI, and the local LFM3 server."""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import replace
from fractions import Fraction
from typing import Any

import av
import numpy as np
import pytest

from planning_service import transcribe
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


# --- local LFM3 transcription -------------------------------------------------

def _encode(fmt: str, codec: str, rate: int = 48_000, seconds: float = 0.5) -> bytes:
    """A real recording in the given container, like MediaRecorder would produce."""
    t = np.arange(int(rate * seconds)) / rate
    samples = (np.sin(2 * np.pi * 440 * t) * 12_000).astype(np.int16)
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=fmt) as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = "mono"
        frame_size = 960
        for start in range(0, len(samples) - frame_size, frame_size):
            frame = av.AudioFrame.from_ndarray(
                samples[start : start + frame_size].reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = rate
            frame.time_base = Fraction(1, rate)
            frame.pts = start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return buffer.getvalue()


@pytest.mark.parametrize("fmt,codec", [("webm", "libopus"), ("mp4", "aac"), ("wav", "pcm_s16le")])
def test_to_wav_decodes_browser_formats(fmt: str, codec: str) -> None:
    wav = wave.open(io.BytesIO(transcribe.to_wav(_encode(fmt, codec))))
    assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16_000, 2)
    assert 0.3 < wav.getnframes() / 16_000 < 0.7  # about the half second we recorded


def test_to_wav_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        transcribe.to_wav(b"not audio at all")


def _config(provider: str) -> PlannerConfig:
    return PlannerConfig(
        provider=provider,
        base_url="http://127.0.0.1:8090/v1",
        model="lfm3",
        api_key=None,
        timeout_s=30,
        temperature=0.0,
    )


class _Reply:
    status_code = 200
    content = b"x"
    text = ""

    def __init__(self, body: dict) -> None:
        self._body = body

    def json(self) -> dict:
        return self._body


def test_local_provider_sends_wav_to_the_model_server(monkeypatch) -> None:
    sent: dict = {}

    def fake_post(url, **kwargs):
        sent.update(url=url, **kwargs)
        return _Reply({"choices": [{"message": {"content": " Front, center. "}}]})

    monkeypatch.setattr(transcribe.requests, "post", fake_post)
    text = transcribe.transcribe_audio(
        _encode("webm", "libopus"), fmt="webm", config=_config("local")
    )
    assert text == "Front, center."
    assert sent["url"] == "http://127.0.0.1:8090/v1/chat/completions"
    messages = sent["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "Perform ASR."}
    part = messages[1]["content"][0]["input_audio"]
    assert part["format"] == "wav"
    assert base64.b64decode(part["data"])[:4] == b"RIFF"


def test_other_providers_still_use_openai(monkeypatch) -> None:
    urls: list[str] = []

    def fake_post(url, **kwargs):
        urls.append(url)
        return _Reply({"text": "hello"})

    monkeypatch.setattr(transcribe.requests, "post", fake_post)
    monkeypatch.setattr(transcribe, "resolve_openai_api_key", lambda: "sk-test")
    cfg = replace(_config("openai"), base_url="https://api.openai.com/v1")
    assert transcribe.transcribe_audio(b"\x00\x01", fmt="webm", config=cfg) == "hello"
    assert urls == [transcribe.TRANSCRIBE_URL]
