import { DropdownMenu } from "@astryxdesign/core/DropdownMenu";
import { Selector } from "@astryxdesign/core/Selector";
import {
  effortFromBudget,
  reasoningEffortsFor,
} from "../reasoning";

function effortIndex(model: string, budget: number): number {
  const efforts = reasoningEffortsFor(model);
  const exact = efforts.findIndex((item) => item.id === effortFromBudget(budget));
  if (exact >= 0) return exact;
  return Math.max(0, efforts.length - 1);
}

export function ReasoningEffortMenu({
  model,
  budget,
  onChange,
}: {
  model: string;
  budget: number;
  onChange: (budget: number) => void;
}) {
  const efforts = reasoningEffortsFor(model);
  if (!efforts.length) return null;
  const current = efforts[effortIndex(model, budget)];
  return (
    <DropdownMenu
      hasChevron
      placement="above"
      menuWidth={180}
      button={{
        label: current?.label ?? "Reasoning",
        variant: "ghost",
        size: "sm",
      }}
      items={efforts.map((item) => ({
        id: item.id,
        label: item.label,
        onClick: () => onChange(item.budget),
      }))}
    />
  );
}

export function ReasoningEffortSelector({
  model,
  budget,
  onChange,
  isDisabled,
}: {
  model: string;
  budget: number;
  onChange: (budget: number) => void;
  isDisabled?: boolean;
}) {
  const efforts = reasoningEffortsFor(model);
  if (!efforts.length) return null;
  const current = efforts[effortIndex(model, budget)];
  return (
    <Selector
      label="Reasoning"
      options={efforts.map((item) => item.label)}
      value={current?.label ?? "Medium"}
      isDisabled={isDisabled}
      onChange={(label) => {
        const next = efforts.find((item) => item.label === label);
        if (next) onChange(next.budget);
      }}
    />
  );
}
