import { useEffect, useState } from "react";
import { Card } from "@astryxdesign/core/Card";
import { VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { localApi } from "../api";
import { PageHeader } from "../components/PageHeader";

export function HistoryPage() {
  const [sessions, setSessions] = useState<any[]>([]);

  useEffect(() => {
    void localApi<{ sessions?: any[] }>("/local/sessions")
      .then((r) => setSessions(r.sessions || []))
      .catch(() => setSessions([]));
  }, []);

  return (
    <VStack gap={4}>
      <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="History" />
      {sessions.length === 0 ? (
        <Text color="secondary">No saved sessions yet. Chat is stored in this browser.</Text>
      ) : (
        sessions.map((sess) => (
          <Card key={sess.session_id} padding={3}>
            <VStack gap={1}>
              <Text weight="semibold">
                {sess.kind === "inspect" ? "Inspect · " : ""}
                {sess.label || sess.session_id}
              </Text>
              <Text type="supporting">
                {sess.updated_at} {sess.preview ? `· ${sess.preview}` : ""}
              </Text>
            </VStack>
          </Card>
        ))
      )}
    </VStack>
  );
}
