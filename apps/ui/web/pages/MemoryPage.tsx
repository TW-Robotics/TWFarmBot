import { useCallback, useEffect, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";
import { Card } from "@astryxdesign/core/Card";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { HStack, VStack } from "@astryxdesign/core/Stack";
import { Text } from "@astryxdesign/core/Text";
import { useToast } from "@astryxdesign/core/Toast";
import { api } from "../api";
import { PageHeader } from "../components/PageHeader";

export type MemoryItem = {
  id: number;
  created_at?: string;
  content: string;
  category?: string;
  x?: number | null;
  y?: number | null;
  z?: number | null;
  importance?: number;
  tags?: string[];
};

function formatCoords(item: MemoryItem): string | null {
  const parts: string[] = [];
  if (typeof item.x === "number") parts.push(`x ${Math.round(item.x)}`);
  if (typeof item.y === "number") parts.push(`y ${Math.round(item.y)}`);
  if (typeof item.z === "number") parts.push(`z ${Math.round(item.z)}`);
  return parts.length ? `${parts.join(" · ")} mm` : null;
}

function formatWhen(iso: string | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export function MemoryPage() {
  const toast = useToast();
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const body = await api<{ memories?: MemoryItem[] }>("/memory?limit=500");
      setItems(body.memories || []);
    } catch (error) {
      setItems([]);
      toast({
        body: error instanceof Error ? error.message : "Could not load memories",
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <VStack gap={4}>
      <HStack gap={3} vAlign="end" wrap="wrap">
        <div style={{ flex: 1, minWidth: 200 }}>
          <PageHeader kicker="TWFarmBot · UAS Technikum Wien" title="Memory" />
        </div>
        <Button label="Refresh" variant="secondary" onClick={() => void load()} />
      </HStack>
      {loading && items.length === 0 ? (
        <Text color="secondary">Loading memories…</Text>
      ) : items.length === 0 ? (
        <EmptyState
          title="No memories yet"
          description="The assistant stores noteworthy observations, treatments, and rules here."
        />
      ) : (
        items.map((item) => {
          const where = formatCoords(item);
          const when = formatWhen(item.created_at);
          return (
            <Card key={item.id} padding={3}>
              <VStack gap={2}>
                <HStack gap={2} wrap="wrap" vAlign="center">
                  {item.category ? (
                    <Badge variant="neutral" label={item.category} />
                  ) : null}
                  {(item.tags || []).map((tag) => (
                    <Badge key={tag} variant="neutral" label={tag} />
                  ))}
                </HStack>
                <Text>{item.content}</Text>
                <Text type="supporting" color="secondary">
                  {[when, where].filter(Boolean).join(" · ")}
                </Text>
              </VStack>
            </Card>
          );
        })
      )}
    </VStack>
  );
}
