"""Conversational chat interface for the FarmBot.

The heavy lifting is delegated to the harness ``AgentLoop``; this module
just wires it to the public ``chat()`` / ``stream_chat()`` signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from twfarmbot_core.actions import ActionRegistry

from .agent import build_base_model
from .config import PlannerConfig
from .harness import (
    AgentLoop,
    ApprovalGate,
    ContextBuilder,
    ReasoningController,
    ToolRegistry,
)
from .introspection import SystemStateProvider


@dataclass(frozen=True)
class ChatResult:
    """Result of a single chat turn."""

    response: str
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    thinking: str | None = None



def _make_loop(
    messages: list[dict[str, Any]],
    *,
    registry: ActionRegistry,
    world: Any = None,
    system_state: SystemStateProvider | None = None,
    model: BaseChatModel | None = None,
    config: PlannerConfig | None = None,
    allow_actions: bool = True,
    propose_only: bool = False,
    model_name: str | None = None,
) -> AgentLoop:
    cfg, base_model = build_base_model(model=model, config=config, model_name=model_name)
    tool_registry = ToolRegistry(registry, system_state)
    approval_gate = ApprovalGate(registry)
    context_builder = ContextBuilder(
        tool_registry, world=world, propose_only=propose_only
    )
    chat_model = base_model.bind_tools(tool_registry.langchain_tools())
    selected_model = model_name or cfg.model
    return AgentLoop(
        model=chat_model,
        tool_registry=tool_registry,
        approval_gate=approval_gate,
        context_builder=context_builder,
        reasoning=ReasoningController(),
        model_name=selected_model,
        propose_only=propose_only,
        allow_actions=allow_actions,
        include_reasoning=_include_reasoning_for(selected_model),
    )


def _include_reasoning_for(model_name: str) -> bool:
    return "deepseek" in model_name.lower() and "v4" in model_name.lower()


def chat(
    messages: list[dict[str, Any]],
    *,
    registry: ActionRegistry,
    world: Any = None,
    system_state: SystemStateProvider | None = None,
    model: BaseChatModel | None = None,
    config: PlannerConfig | None = None,
    allow_actions: bool = True,
    propose_only: bool = False,
    model_name: str | None = None,
) -> ChatResult:
    """Run one conversational turn with tool use.

    ``messages`` should contain the conversation so far (user + assistant
    turns, no system message). The function prepends a system prompt,
    runs the model, executes any tool calls, and returns the final
    assistant text plus a log of tool calls made.
    """
    loop = _make_loop(
        messages,
        registry=registry,
        world=world,
        system_state=system_state,
        model=model,
        config=config,
        allow_actions=allow_actions,
        propose_only=propose_only,
        model_name=model_name,
    )
    result = loop.run(messages)
    out_messages = list(messages)
    out_messages.append({"role": "assistant", "content": result.response})
    return ChatResult(
        response=result.response,
        proposed_actions=result.proposed_actions,
        tool_calls=result.tool_calls,
        messages=out_messages,
        thinking=result.thinking,
    )


def stream_chat(
    messages: list[dict[str, Any]],
    *,
    registry: ActionRegistry,
    world: Any = None,
    system_state: SystemStateProvider | None = None,
    model: BaseChatModel | None = None,
    config: PlannerConfig | None = None,
    allow_actions: bool = True,
    propose_only: bool = False,
    model_name: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Streaming conversational assistant.

    Yields events:
      - ``{"type": "tool_call", ...}`` after a tool is executed.
      - ``{"type": "meta", "tool_calls": [...], "proposed_actions": [...]}``.
      - ``{"type": "thinking", "content": "..."}`` for reasoning traces.
      - ``{"type": "delta", "content": "..."}`` for the final answer text.
    """
    loop = _make_loop(
        messages,
        registry=registry,
        world=world,
        system_state=system_state,
        model=model,
        config=config,
        allow_actions=allow_actions,
        propose_only=propose_only,
        model_name=model_name,
    )
    yield from loop.stream(messages)
