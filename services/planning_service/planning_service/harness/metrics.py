"""Lightweight performance metrics collected during a planning turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Metrics:
    """Performance metrics for a single assistant turn."""

    total_latency_s: float = 0.0
    llm_latency_s: float = 0.0
    ttft_s: float = 0.0
    tokens_per_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    resireg_latency_s: float = 0.0
    turn_count: int = 0

    def merge(self, other: "Metrics") -> None:
        """Add another turn's metrics into this one."""
        self.total_latency_s += other.total_latency_s
        self.llm_latency_s += other.llm_latency_s
        self.ttft_s = max(self.ttft_s, other.ttft_s)
        self.resireg_latency_s += other.resireg_latency_s
        self.turn_count += other.turn_count
        # Token counts are cumulative across turns; keep the largest cumulative value.
        if other.prompt_tokens is not None:
            self.prompt_tokens = max(self.prompt_tokens or 0, other.prompt_tokens)
        if other.completion_tokens is not None:
            self.completion_tokens = max(
                self.completion_tokens or 0, other.completion_tokens
            )
        if other.total_tokens is not None:
            self.total_tokens = max(self.total_tokens or 0, other.total_tokens)
        self._recalc_tokens_per_s()

    def _recalc_tokens_per_s(self) -> None:
        if self.llm_latency_s and self.completion_tokens:
            self.tokens_per_s = round(self.completion_tokens / self.llm_latency_s, 1)
        elif self.llm_latency_s and self.total_tokens:
            self.tokens_per_s = round(self.total_tokens / self.llm_latency_s, 1)

    def add_llm_usage(
        self, response: Any, latency_s: float, ttft_s: float = 0.0
    ) -> None:
        """Extract token usage from a LangChain response and update stats."""
        self.llm_latency_s += latency_s
        # For non-streaming calls, TTFT is the same as total LLM latency.
        self.ttft_s = max(self.ttft_s, ttft_s or latency_s)
        self.turn_count += 1

        token_usage = self._extract_usage(response)
        if token_usage:
            self.prompt_tokens = max(
                self.prompt_tokens or 0, token_usage.get("prompt_tokens", 0)
            )
            self.completion_tokens = max(
                self.completion_tokens or 0, token_usage.get("completion_tokens", 0)
            )
            self.total_tokens = max(
                self.total_tokens or 0, token_usage.get("total_tokens", 0)
            )
        self._recalc_tokens_per_s()

    def add_resireg_latency(self, latency_s: float) -> None:
        self.resireg_latency_s += latency_s

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int] | None:
        if response is None:
            return None
        metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(token_usage, dict):
            out: dict[str, int] = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = token_usage.get(key)
                if isinstance(value, (int, float)):
                    out[key] = int(value)
            return out or None
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            out = {}
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage_metadata.get(key)
                if isinstance(value, (int, float)):
                    out[key] = int(value)
            # Normalize names so the rest of the code is consistent.
            if out:
                return {
                    "prompt_tokens": out.get("input_tokens", 0),
                    "completion_tokens": out.get("output_tokens", 0),
                    "total_tokens": out.get("total_tokens", 0),
                }
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_latency_s": round(self.total_latency_s, 2),
            "llm_latency_s": round(self.llm_latency_s, 2),
            "ttft_s": round(self.ttft_s, 2),
            "tokens_per_s": self.tokens_per_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "resireg_latency_s": round(self.resireg_latency_s, 2),
            "turn_count": self.turn_count,
        }
