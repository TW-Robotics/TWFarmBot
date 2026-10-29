import { Heading, Text } from "@astryxdesign/core/Text";
import { VStack } from "@astryxdesign/core/Stack";
import { Card } from "@astryxdesign/core/Card";
import { ClickableCard } from "@astryxdesign/core/ClickableCard";

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

export function Metric({
  label,
  value,
  onClick,
}: {
  label: string;
  value: string;
  onClick?: () => void;
}) {
  const body = (
    <VStack gap={1}>
      <Text type="supporting" color="secondary">
        {label}
      </Text>
      <Heading level={3}>{value}</Heading>
    </VStack>
  );
  return onClick ? (
    <ClickableCard label={`${label} history`} padding={3} onClick={onClick}>
      {body}
    </ClickableCard>
  ) : (
    <Card padding={3}>{body}</Card>
  );
}
