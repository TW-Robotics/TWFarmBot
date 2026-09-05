import type { ReactNode } from "react";
import { Markdown } from "@astryxdesign/core/Markdown";
import { PauseIcon, PlayIcon, XMarkIcon } from "@heroicons/react/24/solid";
import { resolveChatImages } from "./chat";

type VoiceOverlayProps = {
  paused: boolean;
  status: string;
  you?: string;
  reply?: string;
  activity?: string;
  onHold: () => void;
  onEnd: () => void;
};

export function VoiceOverlay({
  paused,
  status,
  you,
  reply,
  activity,
  onHold,
  onEnd,
}: VoiceOverlayProps) {
  return (
    <div
      className="twfb-voice"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 80,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "28px 24px 48px",
        color: "#161616",
        background: "linear-gradient(180deg, #ffffff 0%, #e4e4e4 42%, #9c9c9c 100%)",
      }}
    >
      <style>
        {`
          @keyframes twfb-voice-glow {
            0%, 100% { opacity: 0.55; transform: translateY(6px) scale(1); }
            50% { opacity: 0.95; transform: translateY(0) scale(1.06); }
          }
          .twfb-voice-reply, .twfb-voice-reply * {
            color: #161616 !important;
          }
        `}
      </style>
      <div style={{ fontSize: 17, fontWeight: 500, letterSpacing: 0.2, color: "#2a2a2a" }}>
        Voice
      </div>
      <div
        style={{
          width: 180,
          height: 180,
          margin: "20px 0 12px",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, #ffffff 0%, #d5d5d5 42%, rgba(160, 160, 160, 0.15) 70%, transparent 74%)",
          animation: paused ? undefined : "twfb-voice-glow 2.8s ease-in-out infinite",
          flexShrink: 0,
        }}
      />
      <div style={{ fontSize: 20, fontWeight: 500, marginBottom: 18, color: "#1a1a1a" }}>
        {status}
      </div>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          width: "min(640px, 100%)",
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {you ? (
          <div style={{ color: "#5a5a5a", fontSize: 15, lineHeight: 1.45 }}>{you}</div>
        ) : null}
        {activity ? (
          <div style={{ color: "#6a6a6a", fontSize: 14 }}>{activity}</div>
        ) : null}
        {reply ? (
          <div className="twfb-voice-reply" style={{ fontSize: 16, lineHeight: 1.55, color: "#161616" }}>
            <Markdown density="compact" isStreaming={status === "Working…"}>
              {resolveChatImages(reply)}
            </Markdown>
          </div>
        ) : null}
      </div>
      <div style={{ display: "flex", gap: 48, marginTop: 28 }}>
        <VoiceButton
          label={paused ? "Resume" : "Hold"}
          tone="hold"
          onClick={onHold}
          icon={
            paused ? (
              <PlayIcon style={{ width: 26, height: 26 }} />
            ) : (
              <PauseIcon style={{ width: 26, height: 26 }} />
            )
          }
        />
        <VoiceButton
          label="End"
          tone="end"
          onClick={onEnd}
          icon={<XMarkIcon style={{ width: 26, height: 26 }} />}
        />
      </div>
    </div>
  );
}

function VoiceButton({
  label,
  tone,
  icon,
  onClick,
}: {
  label: string;
  tone: "hold" | "end";
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 10,
        background: "none",
        border: 0,
        color: "#2a2a2a",
        cursor: "pointer",
      }}
    >
      <span
        style={{
          width: 72,
          height: 72,
          borderRadius: "50%",
          display: "grid",
          placeItems: "center",
          color: "#fff",
          background: tone === "end" ? "#e53935" : "#4a4a4a",
        }}
      >
        {icon}
      </span>
      <span style={{ fontSize: 13, color: "#3a3a3a" }}>{label}</span>
    </button>
  );
}
