"""Speech-to-text for the Assistant: the local LFM3 server, or OpenAI otherwise."""

from __future__ import annotations

import base64
import io
import wave
from typing import Any

import av
import requests

from .config import PlannerConfig, resolve_openai_api_key

TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"


SAMPLE_RATE = 16_000  # LFM3-Audio's encoder input rate


def to_wav(audio: bytes) -> bytes:
    """Decode any container the browser or a file picker produces (webm/opus, mp4/aac,
    ogg, wav, mp3...) into 16 kHz mono 16-bit WAV, the only thing llama.cpp reads
    besides mp3/flac."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm = bytearray()
    try:
        with av.open(io.BytesIO(audio)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("no audio track found")
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    pcm += out.to_ndarray().tobytes()
        for out in resampler.resample(None):
            pcm += out.to_ndarray().tobytes()
    except av.error.FFmpegError as err:
        raise ValueError(f"could not decode audio: {err}") from err
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(pcm))
    return buffer.getvalue()


def _transcribe_local(audio: bytes, config: PlannerConfig) -> str:
    """LFM3-Audio/Omni via llama-server's OpenAI-compatible chat endpoint."""
    wav_b64 = base64.b64encode(to_wav(audio)).decode()
    try:
        response = requests.post(
            f"{config.base_url.rstrip('/')}/chat/completions",
            json={
                "model": config.model,
                "temperature": 0,
                "max_tokens": 512,
                "messages": [
                    {"role": "system", "content": "Perform ASR."},  # language: inferred
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_b64, "format": "wav"},
                            }
                        ],
                    },
                ],
            },
            timeout=max(90.0, config.timeout_s),
        )
    except requests.RequestException as err:
        raise RuntimeError(f"local transcription request failed: {err}") from err
    if response.status_code >= 400:
        raise RuntimeError(
            f"local transcription failed ({response.status_code}): {(response.text or '')[:300]}"
        )
    try:
        text = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as err:
        raise RuntimeError("local transcription response missing text") from err
    return (text or "").strip()


def transcribe_audio(
    audio: bytes,
    *,
    fmt: str = "webm",
    config: PlannerConfig | None = None,
) -> str:
    if not audio:
        raise ValueError("audio is empty")
    if config is not None and config.provider == "local":
        return _transcribe_local(audio, config)
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
