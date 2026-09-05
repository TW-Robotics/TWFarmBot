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
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "audio/webm";
}

function formatFromBlob(blob: Blob, filename = ""): string {
  const mime = blob.type || "";
  if (mime.includes("mp4") || mime.includes("m4a")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp3") || mime.includes("mpeg")) return "mp3";
  if (mime.includes("wav")) return "wav";
  if (mime.includes("webm")) return "webm";
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  if (ext === "mp4") return "m4a";
  if (["wav", "mp3", "m4a", "ogg", "webm"].includes(ext)) return ext;
  return "webm";
}

function canUseLiveMic(): boolean {
  return Boolean(typeof navigator !== "undefined" && navigator.mediaDevices?.getUserMedia);
}

function pickAudioFromThisMachine(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "audio/*,.webm,.wav,.mp3,.m4a,.ogg";
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

  const sendBlob = (
    blob: Blob,
    llm: LlmOverrides | null,
    onText: (text: string) => void,
    onError: (message: string) => void,
    filename = "",
  ) => {
    if (!blob.size) return;
    setBusy(true);
    void (async () => {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const result = await api<{ text?: string }>("/chat/transcribe", {
        method: "POST",
        timeoutMs: 90_000,
        body: JSON.stringify({
          audio_base64: bytesToBase64(bytes),
          format: formatFromBlob(blob, filename),
          llm,
        }),
      });
      const text = (result.text || "").trim();
      if (text) onText(text);
    })()
      .catch((error) => {
        onError(error instanceof Error ? error.message : "Transcription failed");
      })
      .finally(() => setBusy(false));
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

  return { recording, busy, toggle, liveMic: canUseLiveMic() };
}
