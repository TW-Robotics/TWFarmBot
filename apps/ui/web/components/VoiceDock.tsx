import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { TextArea } from "@astryxdesign/core/TextArea";
import { MicrophoneIcon, StopIcon } from "@heroicons/react/24/outline";

const ICON = { width: 22, height: 22 };

/**
 * Touch-first replacement for the text composer: speak one request, check the
 * "You said:" bubble, then Send or Discard. Nothing is sent automatically.
 */
export function VoiceDock({
  recording,
  transcribing,
  busy,
  pendingText,
  onSpeak,
  onSendPending,
  onDiscardPending,
  onEditPending,
  onStop,
}: {
  recording: boolean;
  transcribing: boolean;
  busy: boolean;
  pendingText: string | null;
  onSpeak: () => void;
  onSendPending: () => void;
  onDiscardPending: () => void;
  onEditPending: (text: string) => void;
  onStop: () => void;
}) {
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (pendingText === null) setEditing(false);
  }, [pendingText]);
  const status = pendingText
    ? "Check what you said, then tap Send"
    : recording
      ? "Recording — tap Done when you're finished"
      : transcribing
        ? "Writing that down…"
        : busy
          ? "Working…"
          : "Talk to the assistant";
  return (
    <VStack gap={3} style={{ width: "100%", alignItems: "center", padding: "8px 0 4px" }}>
      {pendingText ? (
        <div
          className="twfb-goal-bubble"
          style={{ width: "100%", alignItems: "center", textAlign: "center" }}
        >
          <Text type="supporting" color="secondary">
            {"You said:"}
          </Text>
          {editing ? (
            <TextArea
              label="Your message"
              isLabelHidden
              value={pendingText}
              onChange={onEditPending}
            />
          ) : (
            <Text>{pendingText}</Text>
          )}
          <HStack gap={2} wrap="wrap" style={{ justifyContent: "center" }}>
            <Button label="Send" variant="primary" size="lg" onClick={onSendPending} />
            <Button
              label={editing ? "Done" : "Edit"}
              size="lg"
              onClick={() => setEditing((value) => !value)}
            />
            <Button label="Discard" size="lg" onClick={onDiscardPending} />
          </HStack>
        </div>
      ) : null}
      <Text type="supporting" color="secondary">
        {status}
      </Text>
      <HStack gap={3} wrap="wrap" style={{ justifyContent: "center" }}>
        <Button
          label={recording ? "Done" : "Speak"}
          icon={recording ? <StopIcon style={ICON} /> : <MicrophoneIcon style={ICON} />}
          size="lg"
          variant={recording ? "primary" : "secondary"}
          isDisabled={transcribing || pendingText !== null}
          onClick={onSpeak}
        />
        {busy ? <Button label="Stop" size="lg" onClick={onStop} /> : null}
      </HStack>
    </VStack>
  );
}
