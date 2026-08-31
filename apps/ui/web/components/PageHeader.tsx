import { Heading, Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/Stack";
import { Card } from "@astryxdesign/core/Card";

export function PageHeader({ kicker, title }: { kicker: string; title: string }) {
  return (
    <VStack gap={1}>
      <Text type="supporting" color="secondary">
        {kicker}
      </Text>
      <Heading level={1}>{title}</Heading>
    </VStack>
  );
}

export function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card padding={3}>
      <VStack gap={1}>
        <Text type="supporting" color="secondary">
          {label}
        </Text>
        <Heading level={3}>{value}</Heading>
      </VStack>
    </Card>
  );
}
