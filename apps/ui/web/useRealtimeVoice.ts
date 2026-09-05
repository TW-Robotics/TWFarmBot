import { useRef, useState } from "react";
import { api } from "./api";

function speakable(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[#*_>~]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isMetricSpeak(text: string): boolean {
  return /vegetation[_\s-]?fraction|stress[_\s-]?fraction|\bndre mean\b|mean\s*[:=]\s*-?\d|\(\s*nir\s*[−-]\s*red/i.test(
    text,
  );
}

function takeSentences(buffer: string): { ready: string[]; rest: string } {
  const ready: string[] = [];
  let rest = buffer;
  const match = /[.!?…](\s+|$)/;
  while (true) {
    const found = match.exec(rest);
    if (!found) break;
    const end = found.index + found[0].length;
    const piece = rest.slice(0, end).trim();
    if (piece) ready.push(piece);
    rest = rest.slice(end);
  }
  return { ready, rest };
}

export function useRealtimeVoice() {
  const sessionRef = useRef<{
    pc: RTCPeerConnection;
    dc: RTCDataChannel;
    stream: MediaStream;
    audio: HTMLAudioElement;
    buffer: string;
    activeResponse: boolean;
    spokenCount: number;
  } | null>(null);
  const handlersRef = useRef<{
    onText: (text: string) => void;
    onError: (message: string) => void;
  }>({ onText: () => undefined, onError: () => undefined });
  const [listening, setListening] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const sendEvent = (event: Record<string, unknown>) => {
    const dc = sessionRef.current?.dc;
    if (!dc || dc.readyState !== "open") return;
    dc.send(JSON.stringify(event));
  };

  const cancelSpeak = () => {
    const session = sessionRef.current;
    if (session) session.buffer = "";
    if (session) session.spokenCount = 0;
    if (session?.activeResponse) {
      sendEvent({ type: "response.cancel" });
      session.activeResponse = false;
    }
    setSpeaking(false);
  };

  const speak = (text: string) => {
    const session = sessionRef.current;
    const clean = speakable(text);
    if (!clean || !session || isMetricSpeak(clean)) return;
    if (session.spokenCount >= 2) return;
    session.spokenCount += 1;
    if (session.activeResponse) {
      sendEvent({ type: "response.cancel" });
    }
    session.activeResponse = true;
    setSpeaking(true);
    sendEvent({
      type: "response.create",
      response: {
        conversation: "none",
        output_modalities: ["audio"],
        instructions: `Speak this text verbatim, with no extra words:\n\n${clean}`,
      },
    });
  };

  const speakDelta = (text: string) => {
    const session = sessionRef.current;
    if (!session || !text) return;
    session.buffer += text;
    const { ready, rest } = takeSentences(session.buffer);
    session.buffer = rest;
    for (const sentence of ready) speak(sentence);
  };

  const flushSpeak = () => {
    const session = sessionRef.current;
    if (!session) return;
    const leftover = session.buffer;
    session.buffer = "";
    if (leftover.trim()) speak(leftover);
  };

  const setMicEnabled = (enabled: boolean) => {
    sessionRef.current?.stream.getAudioTracks().forEach((track) => {
      track.enabled = enabled;
    });
  };

  const stopVoice = () => {
    const session = sessionRef.current;
    sessionRef.current = null;
    session?.stream.getTracks().forEach((track) => track.stop());
    session?.pc.close();
    session?.audio.pause();
    session && (session.audio.srcObject = null);
    setListening(false);
    setPaused(false);
    setSpeaking(false);
  };

  const handleEvent = (raw: string) => {
    let event: { type?: string; transcript?: string; delta?: string };
    try {
      event = JSON.parse(raw);
    } catch {
      return;
    }
    if (event.type === "input_audio_buffer.speech_started") {
      cancelSpeak();
      return;
    }
    if (
      event.type === "conversation.item.input_audio_transcription.completed" ||
      event.type === "input_audio_transcription.completed"
    ) {
      const text = (event.transcript || "").trim();
      if (text) handlersRef.current.onText(text);
      return;
    }
    if (event.type === "response.created") {
      if (sessionRef.current) sessionRef.current.activeResponse = true;
      return;
    }
    if (event.type === "response.done" || event.type === "response.cancelled") {
      if (sessionRef.current) sessionRef.current.activeResponse = false;
      setSpeaking(false);
      return;
    }
    if (event.type === "error") {
      const message =
        typeof (event as { error?: { message?: string } }).error?.message === "string"
          ? (event as { error: { message: string } }).error.message
          : "Voice session error";
      if (/no active response/i.test(message) || /cancellation failed/i.test(message)) {
        if (sessionRef.current) sessionRef.current.activeResponse = false;
        setSpeaking(false);
        return;
      }
      handlersRef.current.onError(message);
    }
  };

  const startVoice = async (
    onText: (text: string) => void,
    onError: (message: string) => void,
  ) => {
    if (sessionRef.current) return;
    handlersRef.current = { onText, onError };
    try {
      const token = await api<{ value?: string; client_secret?: { value?: string } }>(
        "/voice/session",
        { method: "POST" },
      );
      const key = token.value || token.client_secret?.value;
      if (!key) throw new Error("Voice session did not return a token");

      const pc = new RTCPeerConnection();
      const audio = new Audio();
      audio.autoplay = true;
      pc.ontrack = (event) => {
        audio.srcObject = event.streams[0] || null;
      };
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => pc.addTrack(track, stream));
      const dc = pc.createDataChannel("oai-events");
      dc.addEventListener("message", (event) => handleEvent(String(event.data)));
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const sdpResponse = await fetch("https://api.openai.com/v1/realtime/calls", {
        method: "POST",
        body: offer.sdp,
        headers: {
          Authorization: `Bearer ${key}`,
          "Content-Type": "application/sdp",
        },
      });
      if (!sdpResponse.ok) {
        throw new Error((await sdpResponse.text()) || `Realtime connect failed (${sdpResponse.status})`);
      }
      await pc.setRemoteDescription({ type: "answer", sdp: await sdpResponse.text() });
      sessionRef.current = { pc, dc, stream, audio, buffer: "", activeResponse: false, spokenCount: 0 };
      setPaused(false);
      setListening(true);
    } catch (error) {
      stopVoice();
      onError(error instanceof Error ? error.message : "Could not start OpenAI voice");
    }
  };

  const pauseVoice = () => {
    if (!sessionRef.current) return;
    cancelSpeak();
    setMicEnabled(false);
    setPaused(true);
  };

  const resumeVoice = () => {
    if (!sessionRef.current) return;
    setMicEnabled(true);
    setPaused(false);
  };

  return {
    listening,
    paused,
    speaking,
    startVoice,
    stopVoice,
    pauseVoice,
    resumeVoice,
    speakDelta,
    flushSpeak,
    cancelSpeak,
  };
}
