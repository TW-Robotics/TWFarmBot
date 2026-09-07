"""Map a shared reasoning effort onto each provider's API.

The UI exposes none / low / medium / high / extra-high. Providers disagree
on the wire format, so we keep one integer ``reasoning_budget`` in config
and translate at the client.
"""

from __future__ import annotations

EFFORTS = ("none", "low", "medium", "high", "xhigh")
_BUDGET = {
    "none": 0,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}


def effort_from_budget(budget: int) -> str:
    if budget <= 0:
        return "none"
    if budget <= 2048:
        return "low"
    if budget <= 8192:
        return "medium"
    if budget <= 16384:
        return "high"
    return "xhigh"


def budget_from_effort(effort: str) -> int:
    return _BUDGET.get(effort, _BUDGET["medium"])


def openai_effort(model: str, effort: str) -> str:
    """GPT-5.0-class models reject ``xhigh``; keep extra-high for 5.1+ / 5.6."""
    if effort != "xhigh":
        return effort
    name = model.lower()
    if "gpt-5.6" in name or "gpt-5.1" in name or "gpt-5.2" in name or "gpt-5-pro" in name:
        return "xhigh"
    if name.startswith("gpt-5") or "/gpt-5" in name:
        return "high"
    return "xhigh"


def model_uses_reasoning(model: str) -> bool:
    """Whether this model id accepts a reasoning/thinking control."""
    name = model.lower()
    if not name:
        return False
    if "gpt-4" in name and "gpt-5" not in name:
        return False
    if "gpt-3" in name:
        return False
    if "gpt-5" in name and "chat" not in name:
        return True
    if name.startswith(("o1", "o3", "o4")) or "/o1" in name or "/o3" in name or "/o4" in name:
        return True
    if "gemini-2.5" in name or "gemini-3" in name:
        return True
    if "claude" in name and (
        "thinking" in name or "4." in name or "sonnet-4" in name or "opus-4" in name or "haiku-4" in name
    ):
        return True
    if "deepseek" in name and ("r1" in name or "v4" in name or "reason" in name):
        return True
    if "glm-5" in name or "glm-4.5" in name or "glm-4.6" in name:
        return True
    if "grok" in name:
        return True
    if "thinking" in name or "reasoner" in name:
        return True
    return False
