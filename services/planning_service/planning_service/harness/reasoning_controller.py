"""Extract and surface model reasoning / thinking.

Provider-specific reasoning fields (DeepSeek ``reasoning_content``,
OpenRouter ``reasoning``, Claude ``thinking``) and explicit ``<think>``
tags are normalized into plain ``thinking`` events.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class ReasoningController:
    """Extract reasoning from LangChain messages and streaming chunks."""

    @staticmethod
    def extract(message: Any) -> str | None:
        """Extract thinking from a complete LangChain message."""
        content = str(getattr(message, "content", "") or "")
        match = THINK_TAG_RE.search(content)
        if match:
            thinking = match.group(1).strip()
            return thinking if thinking else None

        for key in ("reasoning_content", "thinking", "reasoning"):
            value = getattr(message, "response_metadata", {}).get(key) or getattr(
                message, "additional_kwargs", {}
            ).get(key)
            if value:
                return str(value).strip() or None
        return None

    @staticmethod
    def strip_from_text(text: str) -> str:
        """Remove ``<think>...</think>`` blocks from visible text."""
        return THINK_TAG_RE.sub("", text).strip()

    @classmethod
    def stream_chunks(
        cls,
        chunk: Any,
        *,
        accumulated_reasoning: list[str],
        emitted: bool,
    ) -> Iterator[dict[str, Any]]:
        """Yield any ``thinking`` event that should be emitted for this chunk.

        The reasoning is yielded once, right before the first content chunk
        after reasoning begins. This keeps reasoning visible above the final
        answer while still allowing streaming text.
        """
        reasoning = getattr(chunk, "additional_kwargs", {}).get("reasoning")
        if reasoning:
            accumulated_reasoning.append(str(reasoning))
        content = getattr(chunk, "content", None)
        if content and accumulated_reasoning and not emitted:
            yield {"type": "thinking", "content": "".join(accumulated_reasoning)}

    @classmethod
    def split_stream(cls, text: str) -> tuple[list[dict[str, Any]], str]:
        """Split text into events, holding back an unclosed ``<think>`` block.

        Returns ``(events, remainder)`` where ``remainder`` is either empty
        or starts at an unclosed ``<think>`` tag. Feed ``remainder`` plus
        the next chunk back into this method; complete pairs always flush.
        """
        events: list[dict[str, Any]] = []
        buffer = text
        while True:
            start = buffer.find("<think>")
            end = buffer.find("</think>")
            if start != -1 and end != -1 and end > start:
                prefix = buffer[:start]
                think = buffer[start + 7 : end]
                suffix = buffer[end + 8 :]
                if prefix:
                    events.append({"type": "delta", "content": prefix})
                if think:
                    events.append({"type": "thinking", "content": think})
                buffer = suffix
                continue
            break
        tag = buffer.find("<think>")
        if tag == -1:
            if buffer:
                events.append({"type": "delta", "content": buffer})
            return events, ""
        if tag > 0:
            events.append({"type": "delta", "content": buffer[:tag]})
        return events, buffer[tag:]

    @classmethod
    def split_text(cls, text: str) -> Iterator[dict[str, Any]]:
        """Split text into alternating delta and thinking events.

        Yields ``{"type": "delta", ...}`` and ``{"type": "thinking", ...}``
        pieces so that ``<think>`` blocks appear as reasoning rather than
        visible output.
        """
        events, rest = cls.split_stream(text)
        yield from events
        if rest:
            yield {"type": "delta", "content": rest}
