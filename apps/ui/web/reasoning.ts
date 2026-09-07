export type ReasoningEffort = "none" | "low" | "medium" | "high" | "xhigh";

export const REASONING_EFFORTS: { id: ReasoningEffort; label: string; budget: number }[] = [
  { id: "none", label: "None", budget: 0 },
  { id: "low", label: "Low", budget: 2048 },
  { id: "medium", label: "Medium", budget: 8192 },
  { id: "high", label: "High", budget: 16384 },
  { id: "xhigh", label: "Extra high", budget: 32768 },
];

export function effortFromBudget(budget: number): ReasoningEffort {
  if (budget <= 0) return "none";
  if (budget <= 2048) return "low";
  if (budget <= 8192) return "medium";
  if (budget <= 16384) return "high";
  return "xhigh";
}

export function budgetFromEffort(effort: ReasoningEffort): number {
  return REASONING_EFFORTS.find((item) => item.id === effort)?.budget ?? 8192;
}

export function modelUsesReasoning(model: string): boolean {
  const name = model.toLowerCase();
  if (!name) return false;
  if (name.includes("gpt-4") && !name.includes("gpt-5")) return false;
  if (name.includes("gpt-3")) return false;
  if (name.includes("gpt-5") && !name.includes("chat")) return true;
  if (/(^|\/)o[1-4]/.test(name)) return true;
  if (name.includes("gemini-2.5") || name.includes("gemini-3")) return true;
  if (
    name.includes("claude") &&
    (name.includes("thinking") ||
      name.includes("4.") ||
      name.includes("sonnet-4") ||
      name.includes("opus-4") ||
      name.includes("haiku-4"))
  ) {
    return true;
  }
  if (name.includes("deepseek") && (name.includes("r1") || name.includes("v4") || name.includes("reason"))) {
    return true;
  }
  if (name.includes("glm-5") || name.includes("glm-4.5") || name.includes("glm-4.6")) return true;
  if (name.includes("grok")) return true;
  if (name.includes("thinking") || name.includes("reasoner")) return true;
  return false;
}

export function reasoningEffortsFor(model: string): typeof REASONING_EFFORTS {
  if (!modelUsesReasoning(model)) return [];
  const name = model.toLowerCase();
  const gpt5Classic =
    name.includes("gpt-5") &&
    !name.includes("gpt-5.6") &&
    !name.includes("gpt-5.1") &&
    !name.includes("gpt-5.2") &&
    !name.includes("gpt-5-pro");
  return gpt5Classic ? REASONING_EFFORTS.filter((item) => item.id !== "xhigh") : REASONING_EFFORTS;
}
