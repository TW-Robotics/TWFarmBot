"""planning_service — turn natural-language requests into Action[].

Public surface:
    plan(request, *, world=None, registry=None, model=None) -> PlanResult
    chat(...), stream_chat(...)

The planner and chat interfaces are now driven by the same harness:
``ToolRegistry`` + ``ApprovalGate`` + ``AgentLoop``. The harness keeps
safety/approval policy in one place and lets the model reason across
multiple tool-call turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from .agent import build_base_model
from .chat import ChatResult, chat, stream_chat
from .config import PlannerConfig, load_config
from .harness import (
    AgentLoop,
    ApprovalGate,
    ContextBuilder,
    ReasoningController,
    ToolRegistry,
)
from .introspection import (
    InMemorySystemStateProvider,
    SystemStateProvider,
    build_introspection_tools,
)
from .parser import PlanError, _extract_json, parse_plan
from .prompt import PlannerResponse
from .tools import build_tools, extract_tool_calls, tool_calls_to_actions

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanResult:
    """The output of a single planning call."""

    request: str
    actions: list[Action] = field(default_factory=list)
    rationale: str = ""
    raw_text: str = ""


def get_default_registry() -> ActionRegistry:
    """Build an ActionRegistry with the same handlers the API uses."""
    from twfarmbot_api_server.handlers import register_default_handlers

    registry = ActionRegistry()
    register_default_handlers(registry)
    return registry


def plan(
    request: str,
    *,
    world: Any = None,
    registry: ActionRegistry | None = None,
    model: BaseChatModel | None = None,
    config: PlannerConfig | None = None,
    system_state: SystemStateProvider | None = None,
    model_name: str | None = None,
) -> PlanResult:
    """Translate a natural-language ``request`` into a validated PlanResult.

    The harness lets the model call introspection tools and collect action
    proposals across multiple turns. Physical actions are never executed
    inside ``plan()``; they are validated and returned for the caller to
    execute or preview.
    """
    cfg, base_model = build_base_model(model=model, config=config, model_name=model_name)
    registry = registry or get_default_registry()

    tool_registry = ToolRegistry(registry, system_state)
    approval_gate = ApprovalGate(registry, planning_mode=True)
    context_builder = ContextBuilder(tool_registry, world=world)
    planner_model = base_model.bind_tools(tool_registry.langchain_tools())
    selected_model = model_name or cfg.model

    loop = AgentLoop(
        model=planner_model,
        tool_registry=tool_registry,
        approval_gate=approval_gate,
        context_builder=context_builder,
        reasoning=ReasoningController(),
        model_name=selected_model,
        propose_only=False,
        allow_actions=False,
        include_reasoning=False,
    )

    log.info("planning request via %s/%s", cfg.base_url, selected_model)
    result = loop.plan_request(request)

    actions = _extract_actions(result.tool_calls, registry, approval_gate)
    raw_text = result.response
    rationale = result.response.strip()

    if not actions and raw_text:
        # Fallback: non-tool-calling models may return free-form JSON.
        try:
            actions, rationale = _parse_with_rationale(raw_text, registry)
        except PlanError as err:
            log.warning("JSON fallback failed (%s); returning empty plan", err)
            actions = []
            rationale = raw_text.strip() or "planner could not produce a plan"

    if not rationale:
        rationale = "Plan built from tool calls."

    return PlanResult(
        request=request,
        actions=actions,
        rationale=rationale,
        raw_text=raw_text,
    )


def _extract_actions(
    tool_calls: list[dict[str, Any]],
    registry: ActionRegistry,
    approval_gate: ApprovalGate,
) -> list[Action]:
    """Turn action tool calls from the planning loop into validated Actions."""
    known = set(registry.kinds())
    actions: list[Action] = []
    for call in tool_calls:
        name = call.get("name")
        if name not in known:
            continue
        result = call.get("result", {})
        params = (
            result.get("params", call.get("args", {}))
            if isinstance(result, dict)
            else call.get("args", {})
        )
        approval_gate.check_safety(name, params)
        actions.append(Action(kind=name, params=dict(params)))
    return actions


def _parse_with_rationale(
    text: str, registry: ActionRegistry
) -> tuple[list[Action], str]:
    """Parse the LLM output, returning (actions, rationale)."""
    actions = parse_plan(text, registry)
    rationale = ""
    try:
        raw = _extract_json(text)
        response = PlannerResponse.model_validate(raw)
        rationale = response.rationale
    except Exception:  # noqa: BLE001
        pass
    return actions, rationale


__all__ = [
    "Action",
    "ChatResult",
    "InMemorySystemStateProvider",
    "PlanError",
    "PlanResult",
    "PlannerConfig",
    "SystemStateProvider",
    "build_base_model",
    "build_introspection_tools",
    "build_tools",
    "chat",
    "get_default_registry",
    "load_config",
    "parse_plan",
    "plan",
    "stream_chat",
]
