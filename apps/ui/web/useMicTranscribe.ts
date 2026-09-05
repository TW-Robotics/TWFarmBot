import { useRef, useState } from "react";
import { api } from "./api";
import type { LlmOverrides } from "./settings";

function bytesToBase64(bytes: Uint8Array): string {
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function recorderMime(): string {
  if (typeof MediaRecorder === "undefined") return "audio/webm";
  for (const type of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "audio/webm";
}

function mixMono(buffer: AudioBuffer): Float32Array {
  const out = new Float32Array(buffer.length);
  const count = buffer.numberOfChannels;
  for (let channel = 0; channel < count; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < data.length; i += 1) {
      out[i] += data[i] / count;
    }
  }
  return out;
}

function writeWav(samples: Float32Array<ArrayBufferLike>, sampleRate: number): Blob {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
  return new Blob([bytes], { type: "audio/wav" });
}

async function blobToWav(blob: Blob): Promise<Blob> {
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();
  try {
    const decoded = await ctx.decodeAudioData((await blob.arrayBuffer()).slice(0));
    const rate = 16000;
    const frames = Math.max(1, Math.ceil((decoded.duration || 0) * rate));
    const offline = new OfflineAudioContext(1, frames, rate);
    const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    mono.copyToChannel(mixMono(decoded) as Float32Array<ArrayBuffer>, 0);
    const source = offline.createBufferSource();
    source.buffer = mono;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0);
    const pad = Math.round((rate * SHORT_PAD_MS) / 1000);
    if (samples.length < rate && pad > 0) {
      const padded = new Float32Array(samples.length + pad * 2);
      padded.set(samples, pad);
      return writeWav(padded, rate);
    }
    return writeWav(samples, rate);
  } finally {
    await ctx.close().catch(() => undefined);
  }
}

function formatFromBlob(blob: Blob, filename = ""): string {
  const mime = blob.type || "";
  if (mime.includes("mp4") || mime.includes("m4a")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp3") || mime.includes("mpeg")) return "mp3";
  if (mime.includes("wav")) return "wav";
  if (mime.includes("flac")) return "flac";
  if (mime.includes("aac")) return "aac";
  if (mime.includes("webm")) return "webm";
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  if (ext === "mp4") return "m4a";
  if (["wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"].includes(ext)) {
    return ext;
  }
  return "webm";
}

function canUseLiveMic(): boolean {
  return Boolean(
    typeof navigator !== "undefined" && navigator.mediaDevices?.getUserMedia,
  );
}

function pickAudioFromThisMachine(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*,.webm,.wav,.mp3,.m4a,.ogg,.aac,.flac";
    input.setAttribute("capture", "user");
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.click();
  });
}

const HINT_RMS = 0.028;
const SPEECH_RMS = 0.04;
const START_FRAMES = 3;
const END_MS = 850;
const FALSE_START_MS = 400;
const MIN_BLOB = 2_000;
const COOLDOWN_MS = 500;
const ARM_MS = 300;
const SHORT_PAD_MS = 120;

function timeDomainRms(analyser: AnalyserNode, buffer: Uint8Array<ArrayBuffer>): number {
  analyser.getByteTimeDomainData(buffer);
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    const sample = (buffer[i] - 128) / 128;
    sum += sample * sample;
  }
  return Math.sqrt(sum / buffer.length);
}

export function useMicTranscribe() {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const voiceRef = useRef<{
    listening: boolean;
    ctx: AudioContext | null;
    analyser: AnalyserNode | null;
    source: MediaStreamAudioSourceNode | null;
    raf: number;
    speaking: boolean;
    loudFrames: number;
    silentSince: number;
    startedAt: number;
    confirmed: boolean;
    readyAt: number;
    cooldownUntil: number;
    inflight: boolean;
    pending: Blob | null;
    paused: boolean;
    recorder: MediaRecorder | null;
    chunks: Blob[];
    llm: LlmOverrides | null;
    onText: (text: string) => void;
    onError: (message: string) => void;
  }>({
    listening: false,
    ctx: null,
    analyser: null,
    source: null,
    raf: 0,
    speaking: false,
    loudFrames: 0,
    silentSince: 0,
    startedAt: 0,
    confirmed: false,
    readyAt: 0,
    cooldownUntil: 0,
    inflight: false,
    pending: null,
    paused: false,
    recorder: null,
    chunks: [],
    llm: null,
    onText: () => undefined,
    onError: () => undefined,
  });
  const [recording, setRecording] = useState(false);
  const [listening, setListening] = useState(false);
  const [paused, setPaused] = useState(false);
  const [busy, setBusy] = useState(false);
  const [transcribing, setTranscribing] = useState(0);

  const stopTracks = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  const transcribeBlob = async (
    blob: Blob,
    llm: LlmOverrides | null,
    filename = "",
  ) => {
    let payload = blob;
    let format = formatFromBlob(blob, filename);
    try {
      payload = await blobToWav(blob);
      format = "wav";
    } catch {
      // Keep the original bytes if the browser cannot decode this container.
    }
    const bytes = new Uint8Array(await payload.arrayBuffer());
    const result = await api<{ text?: string }>("/chat/transcribe", {
      method: "POST",
      timeoutMs: 90_000,
      body: JSON.stringify({
        audio_base64: bytesToBase64(bytes),
        format,
        llm,
      }),
    });
    return (result.text || "").trim();
  };

  const transcribeError = (error: unknown) => {
    const message = error instanceof Error ? error.message : "Transcription failed";
    if (message.includes("429")) {
      return "Transcription is rate-limited. Pause a moment, then speak again.";
    }
    return message;
  };

  const sendBlob = (
    blob: Blob,
    llm: LlmOverrides | null,
    onText: (text: string) => void,
    onError: (message: string) => void,
    filename = "",
  ) => {
    if (!blob.size) return;
    const voice = voiceRef.current;
    if (voice.listening && blob.size < MIN_BLOB) return;
    if (voice.listening && voice.inflight) {
      voice.pending = blob;
      return;
    }
    voice.inflight = true;
    setBusy(true);
    setTranscribing((count) => count + 1);
    void transcribeBlob(blob, llm, filename)
      .then((text) => {
        if (text) onText(text);
      })
      .catch((error) => {
        onError(transcribeError(error));
      })
      .finally(() => {
        voice.inflight = false;
        voice.cooldownUntil = Date.now() + COOLDOWN_MS;
        setBusy(false);
        setTranscribing((count) => Math.max(0, count - 1));
        const queued = voice.pending;
        voice.pending = null;
        if (queued && voice.listening) {
          sendBlob(queued, voice.llm, voice.onText, voice.onError);
        }
      });
  };

  const finishUtterance = (flush: boolean) => {
    const voice = voiceRef.current;
    const recorder = voice.recorder;
    if (!recorder || recorder.state !== "recording") return;
    if (!flush && !voice.confirmed) {
      recorder.onstop = () => {
        voice.recorder = null;
        voice.chunks = [];
        voice.confirmed = false;
      };
      recorder.stop();
      return;
    }
    recorder.stop();
  };

  const beginUtterance = () => {
    const stream = streamRef.current;
    const voice = voiceRef.current;
    if (!stream || voice.recorder || voice.paused) return;
    if (Date.now() < voice.readyAt || Date.now() < voice.cooldownUntil) return;
    const mime = recorderMime();
    const recorder = new MediaRecorder(stream, { mimeType: mime });
    voice.chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) voice.chunks.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(voice.chunks, { type: recorder.mimeType || mime });
      voice.recorder = null;
      voice.chunks = [];
      voice.confirmed = false;
      if (!voice.listening) stopTracks();
      sendBlob(blob, voice.llm, voice.onText, voice.onError);
    };
    voice.recorder = recorder;
    voice.startedAt = Date.now();
    voice.confirmed = false;
    recorder.start(100);
  };

  const stopVoice = () => {
    const voice = voiceRef.current;
    voice.listening = false;
    if (voice.raf) cancelAnimationFrame(voice.raf);
    voice.raf = 0;
    const recorder = voice.recorder;
    finishUtterance(true);
    voice.source?.disconnect();
    voice.source = null;
    voice.analyser = null;
    void voice.ctx?.close().catch(() => undefined);
    voice.ctx = null;
    if (!recorder || recorder.state === "inactive") stopTracks();
    voice.paused = false;
    setPaused(false);
    setListening(false);
    setRecording(false);
  };

  const pollVoice = () => {
    const voice = voiceRef.current;
    if (!voice.listening || !voice.analyser) return;
    if (voice.paused) {
      voice.raf = requestAnimationFrame(pollVoice);
      return;
    }
    const buffer = new Uint8Array(voice.analyser.fftSize);
    const rms = timeDomainRms(voice.analyser, buffer);
    const now = Date.now();
    if (now < voice.readyAt) {
      voice.raf = requestAnimationFrame(pollVoice);
      return;
    }
    if (rms >= HINT_RMS) {
      beginUtterance();
    }
    if (rms >= SPEECH_RMS) {
      voice.loudFrames += 1;
      voice.silentSince = 0;
      if (voice.loudFrames >= START_FRAMES) {
        voice.speaking = true;
        voice.confirmed = true;
      }
    } else {
      voice.loudFrames = 0;
      if (voice.recorder && !voice.confirmed) {
        if (now - voice.startedAt >= FALSE_START_MS) {
          finishUtterance(false);
        }
      } else if (voice.speaking) {
        if (!voice.silentSince) voice.silentSince = now;
        if (now - voice.silentSince >= END_MS) {
          voice.speaking = false;
          voice.silentSince = 0;
          finishUtterance(true);
        }
      }
    }
    voice.raf = requestAnimationFrame(pollVoice);
  };

  const startVoice = async (
    llm: LlmOverrides | null,
    onText: (text: string) => void,
    onError: (message: string) => void,
  ) => {
    if (voiceRef.current.listening) return;
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    if (!canUseLiveMic()) {
      onError("Microphone unavailable");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      onError(error instanceof Error ? error.message : "Microphone unavailable");
      return;
    }
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.12;
    source.connect(analyser);
    streamRef.current = stream;
    const voice = voiceRef.current;
    voice.listening = true;
    voice.ctx = ctx;
    voice.source = source;
    voice.analyser = analyser;
    voice.speaking = false;
    voice.loudFrames = 0;
    voice.silentSince = 0;
    voice.confirmed = false;
    voice.readyAt = Date.now() + ARM_MS;
    voice.cooldownUntil = 0;
    voice.inflight = false;
    voice.pending = null;
    voice.paused = false;
    voice.llm = llm;
    voice.onText = onText;
    voice.onError = onError;
    setPaused(false);
    setListening(true);
    voice.raf = requestAnimationFrame(pollVoice);
  };

  const pauseVoice = () => {
    const voice = voiceRef.current;
    if (!voice.listening || voice.paused) return;
    voice.paused = true;
    voice.speaking = false;
    finishUtterance(false);
    setPaused(true);
  };

  const resumeVoice = () => {
    const voice = voiceRef.current;
    if (!voice.listening || !voice.paused) return;
    voice.paused = false;
    voice.readyAt = Date.now() + ARM_MS;
    setPaused(false);
  };

  const toggle = async (
    llm: LlmOverrides | null,
    onText: (text: string) => void,
    onError: (message: string) => void,
  ) => {
    if (busy || voiceRef.current.listening) return;
    const active = recorderRef.current;
    if (active && active.state === "recording") {
      active.stop();
      return;
    }

    if (!canUseLiveMic()) {
      try {
        const file = await pickAudioFromThisMachine();
        if (!file) return;
        sendBlob(file, llm, onText, onError, file.name);
      } catch (error) {
        onError(error instanceof Error ? error.message : "Could not read audio from this machine");
      }
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      onError(error instanceof Error ? error.message : "Microphone unavailable");
      return;
    }
    streamRef.current = stream;
    const mime = recorderMime();
    const recorder = new MediaRecorder(stream, { mimeType: mime });
    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || mime });
      recorderRef.current = null;
      stopTracks();
      setRecording(false);
      sendBlob(blob, llm, onText, onError);
    };
    recorderRef.current = recorder;
    recorder.start(250);
    setRecording(true);
  };

  return {
    recording,
    busy,
    listening,
    paused,
    transcribing: transcribing > 0,
    toggle,
    startVoice,
    stopVoice,
    pauseVoice,
    resumeVoice,
    liveMic: canUseLiveMic(),
  };
}
