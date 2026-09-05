from planning_service.config import PlannerConfig
from planning_service.transcribe import (
    TRANSCRIBE_MODEL,
    normalize_audio_format,
    transcribe_audio,
)


def test_normalize_audio_format() -> None:
    assert normalize_audio_format("audio/webm;codecs=opus") == "webm"
    assert normalize_audio_format("mp4") == "m4a"


def test_transcribe_audio_posts_mai_model(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"text": "move to y 100"}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["headers"] = kwargs["headers"]
        return FakeResponse()

    monkeypatch.setattr("planning_service.transcribe.httpx.post", fake_post)
    monkeypatch.setattr(
        "planning_service.transcribe.resolve_api_key_for",
        lambda _name: "sk-or-test",
    )
    cfg = PlannerConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="sk-openai",
        timeout_s=30,
        temperature=0,
    )
    text = transcribe_audio(b"RIFF....", fmt="wav", config=cfg)
    assert text == "move to y 100"
    assert captured["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert captured["json"]["model"] == TRANSCRIBE_MODEL
    assert captured["json"]["input_audio"]["format"] == "wav"
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
