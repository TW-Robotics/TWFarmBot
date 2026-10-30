"""Scripted Pydantic AI model for harness, planner and autonomy tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models.function import DeltaToolCall, FunctionModel

_Call = tuple[str, dict[str, Any], str]


@dataclass
class Turn:
    """One scripted model reply: optional text plus optional tool calls."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def AIMessage(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> Turn:  # noqa: N802
    return Turn(content, list(tool_calls or []))


def tool_call(name: str, args: dict[str, Any] | None = None, call_id: str = "call_1") -> Turn:
    return Turn(tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


class ScriptModel(FunctionModel):
    """Plays ``turns`` in order (the last one repeats).

    A turn is a string, a ``Turn``, or a list of ``(name, args, id)`` calls.
    With ``answer_think`` the model answers a forced ``think`` step itself;
    otherwise it ignores ``tool_choice`` and just follows the script.
    """

    def __init__(
        self,
        turns: list[Any] | None = None,
        *,
        responses: list[Any] | None = None,
        answer_think: bool = False,
    ) -> None:
        self._script: list[Any] = list(turns if turns is not None else responses or [""])
        self._index = 0
        self.seen: list[Any] = []
        self.choices: list[Any] = []
        self.on_call: Any = None  # called with the step index as the model is asked

        async def stream(messages: Any, info: Any):
            self.seen.append(messages)
            self.choices.append(((info.model_settings or {}).get('tool_choice'), sorted(t.name for t in info.function_tools)))
            if self.on_call:
                self.on_call(self._index)
            tools = {t.name for t in info.function_tools}
            if answer_think and (info.model_settings or {}).get("tool_choice") == "required" and "think" in tools:
                yield {0: DeltaToolCall(name="think", json_args='{"thought": "ok"}', tool_call_id="think")}
                return
            item = self._script[min(self._index, len(self._script) - 1)]
            self._index += 1
            if isinstance(item, list):
                item = Turn(tool_calls=[{"name": n, "args": a, "id": i} for n, a, i in item])
            elif isinstance(item, str):
                item = Turn(item)
            if item.content or not item.tool_calls:
                yield item.content
            if item.tool_calls:
                yield {
                    i: DeltaToolCall(
                        name=c["name"], json_args=json.dumps(c.get("args") or {}), tool_call_id=str(c.get("id") or f"call_{i}")
                    )
                    for i, c in enumerate(item.tool_calls)
                }

        super().__init__(stream_function=stream)

    def set_responses(self, responses: list[Any]) -> None:
        self._script, self._index = list(responses), 0
