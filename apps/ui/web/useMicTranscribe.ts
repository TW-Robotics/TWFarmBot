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

function writeWav(samples: Float32Array, sampleRate: number): Blob {
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
    mono.copyToChannel(mixMono(decoded), 0);
    const source = offline.createBufferSource();
    source.buffer = mono;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return writeWav(rendered.getChannelData(0), rate);
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

export function useMicTranscribe() {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);

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

  const toggle = async (
    llm: LlmOverrides | null,
    onText: (text: string) => void,
    onError: (message: string) => void,
  ) => {
    if (busy) return;
    const active = recorderRef.current;
    if (active && active.state === "recording") {
      active.stop();
      return;
    }

    const sendBlob = (blob: Blob, filename = "") => {
      if (!blob.size) return;
      setBusy(true);
      void transcribeBlob(blob, llm, filename)
        .then((text) => {
          if (text) onText(text);
        })
        .catch((error) => {
          onError(error instanceof Error ? error.message : "Transcription failed");
        })
        .finally(() => setBusy(false));
    };

    if (!canUseLiveMic()) {
      try {
        const file = await pickAudioFromThisMachine();
        if (!file) return;
        sendBlob(file, file.name);
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
      sendBlob(blob);
    };
    recorderRef.current = recorder;
    recorder.start(250);
    setRecording(true);
  };

  return { recording, busy, toggle, liveMic: canUseLiveMic() };
}
