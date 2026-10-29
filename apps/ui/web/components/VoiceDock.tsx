import { Button } from "@astryxdesign/core/Button";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { MicrophoneIcon, SpeakerWaveIcon, StopIcon } from "@heroicons/react/24/outline";

const ICON = { width: 22, height: 22 };

/** Touch-first replacement for the text composer: live voice or record-and-send. */
export function VoiceDock({
  liveOn,
  recording,
  transcribing,
  busy,
  onLive,
  onTranscribe,
  onStop,
}: {
  liveOn: boolean;
  recording: boolean;
  transcribing: boolean;
  busy: boolean;
  onLive: () => void;
  onTranscribe: () => void;
  onStop: () => void;
}) {
  const status = liveOn
    ? "Live voice is on"
    : recording
      ? "Recording — tap again to send"
      : transcribing
        ? "Transcribing…"
        : busy
          ? "Working…"
          : "Talk to the assistant";
  return (
    <VStack gap={3} style={{ width: "100%", alignItems: "center", padding: "8px 0 4px" }}>
      <Text type="supporting" color="secondary">
        {status}
      </Text>
      <HStack gap={3} wrap="wrap" style={{ justifyContent: "center" }}>
        <Button
          label={liveOn ? "End live voice" : "Live voice"}
          icon={<SpeakerWaveIcon style={ICON} />}
          size="lg"
          variant={liveOn ? "primary" : "secondary"}
          isDisabled={recording || transcribing}
          onClick={onLive}
        />
        <Button
          label={recording ? "Send" : "Transcribe"}
          icon={recording ? <StopIcon style={ICON} /> : <MicrophoneIcon style={ICON} />}
          size="lg"
          variant={recording ? "primary" : "secondary"}
          isDisabled={liveOn || transcribing}
          onClick={onTranscribe}
        />
        {busy ? <Button label="Stop" size="lg" onClick={onStop} /> : null}
      </HStack>
    </VStack>
  );
}
