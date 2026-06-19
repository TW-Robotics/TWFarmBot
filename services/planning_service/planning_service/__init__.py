"""planning_service — turn natural-language requests into Action[].

Public surface:
    plan(request, *, world=None, registry=None, model=None) -> PlanResult

The default ``model`` is built from :func:`config.load_config` and uses
LangChain's ``ChatOpenAI`` against any OpenAI-compatible endpoint
(OpenRouter, llama.cpp, vLLM, Ollama, etc.).

The model is bound to LangChain tools for every action kind (see
:mod:`tools`). Models that support tool-calling (most modern ones) will
produce structured ``tool_calls``; models that don't fall back to the
JSON path. Every plan still runs through ``safety_service.validate``
before being returned.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from spatial_service import format_world_context

from .agent import build_base_model, build_tool_set
from .chat import ChatResult, chat, stream_chat
from .config import PlannerConfig, load_config
from .introspection import (
    InMemorySystemStateProvider,
    SystemStateProvider,
    build_introspection_tools,
)
from .parser import PlanError, parse_plan
from .prompt import build_chat_system_prompt, build_system_prompt, build_user_prompt
from .tools import build_tools, extract_tool_calls, tool_calls_to_actions

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanResult:
    """The output of a single planning call.

    ``actions`` is empty when the model returned an empty plan (ambiguous
    or refused request) — in that case ``rationale`` always explains
    why. ``raw_text`` holds the model's exact output for debugging.
    """

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
) -> PlanResult:
    """Translate a natural-language ``request`` into a validated PlanResult.

    Args:
        request: Free-form task description, e.g. "water the tomato bed
            for 90 seconds, then go home".
        world: Optional world model (a ``GardenWorld`` or any object
            with ``to_dict()``). When supplied, a compact summary is
            included in the prompt so the model can ground names like
            "tomato bed" to actual entities.
        registry: Action vocabulary. Defaults to the api_server's
            default registry.
        model: Pre-built LangChain chat model. Defaults to one built
            from ``config.load_config()``.
        config: Optional planner config (overrides env-derived config).
        system_state: Optional provider for live system state. When
            supplied, the planner binds introspection tools
            (``get_position``, ``list_zones``, …) so
            the model can query the system during a planning call.
            Pass ``InMemorySystemStateProvider(...)`` in tests.
    """
    cfg, base_model = build_base_model(model=model, config=config)
    registry = registry or get_default_registry()

    world_context = format_world_context(world) if world is not None else None
    user_msg = build_user_prompt(
        request,
        world_context=world_context,
    )

    # Bind all tools: action tools + (optional) introspection tools.
    all_tools = build_tool_set(
        registry, system_state, for_chat=False, propose_only=False, allow_actions=True
    )
    chat_model = base_model.bind_tools(all_tools) if all_tools else base_model

    # Preferred path: structured output via Pydantic. Forces the model
    # to fill the PlannerResponse fields directly — eliminates the
    # "I'll just chat instead" failure mode.
    structured = _try_structured_output(base_model, all_tools)
    if structured is not None:
        messages = [
            {"role": "system", "content": build_system_prompt(registry.kinds())},
            {"role": "user", "content": user_msg},
        ]
        log.info(
            "planning via with_structured_output (%s/%s, tools=%d)",
            cfg.base_url, cfg.model, len(all_tools),
        )
        try:
            parsed = structured.invoke(messages)
        except Exception as err:  # noqa: BLE001 — fall through to legacy path
            log.warning("structured-output path failed (%s); falling back", err)
            parsed = None
        if parsed is not None:
            actions = _actions_from_structured(parsed, registry)
            rationale = getattr(parsed, "rationale", "") or "Plan built from structured output."
            raw_text = json.dumps(parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.__dict__)
            return PlanResult(
                request=request,
                actions=actions,
                rationale=rationale,
                raw_text=raw_text,
            )

    messages: list[tuple[str, str]] = [
        ("system", build_system_prompt(registry.kinds())),
        ("user", user_msg),
    ]

    log.info(
        "planning request via %s/%s (tools=%d)", cfg.base_url, cfg.model, len(all_tools),
    )
    response = chat_model.invoke(messages)
    text = _text_from_response(response)

    # Preferred path: tool calls from the bound model.
    tool_calls = extract_tool_calls(response)
    if tool_calls:
        try:
            actions, introspect_results = _actions_from_tool_calls_with_introspection(
                tool_calls, registry, system_state,
            )
        except (PlanError, ValueError) as err:
            log.warning("tool-calling path failed (%s); falling back to JSON", err)
            actions = []
            introspect_results = []
        if actions:
            rationale = _extract_rationale_from_text(text)
            log.info("planner produced %d action(s) via tool calls", len(actions))
            return PlanResult(
                request=request,
                actions=actions,
                rationale=rationale or "Plan built from tool calls.",
                raw_text=text,
            )

    # Fallback path: free-form JSON in the model's text.
    try:
        actions, rationale = _parse_with_rationale(text, registry)
    except PlanError as err:
        log.warning("JSON fallback failed (%s); returning empty plan", err)
        actions = []
        rationale = text.strip() or "planner could not produce a JSON plan"
    log.info("planner produced %d action(s) via JSON fallback", len(actions))
    return PlanResult(
        request=request,
        actions=actions,
        rationale=rationale,
        raw_text=text,
    )


def _try_structured_output(
    base_model: BaseChatModel, tools: list[BaseTool]
) -> Any | None:
    """Wrap the model with ``with_structured_output(PlannerResponse)``.

    Returns ``None`` if the model doesn't support it, so the caller can
    fall back to the tool-calling / JSON paths.
    """
    try:
        return base_model.with_structured_output(_PlannerResponseModel)
    except (NotImplementedError, AttributeError, TypeError, ValueError):
        return None


class _PlannerActionModel(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class _PlannerResponseModel(BaseModel):
    actions: list[_PlannerActionModel] = Field(default_factory=list)
    rationale: str = ""


def _actions_from_structured(
    parsed: Any, registry: ActionRegistry
) -> list[Action]:
    from safety_service import validate as safety_validate

    known = set(registry.kinds())
    actions: list[Action] = []
    raw_actions = getattr(parsed, "actions", []) or []
    for item in raw_actions:
        kind = getattr(item, "kind", None) or (item.get("kind") if isinstance(item, dict) else None)
        params = getattr(item, "params", None) or (item.get("params", {}) if isinstance(item, dict) else {})
        if kind not in known:
            raise PlanError(f"structured output has unknown kind {kind!r}")
        action = Action(kind=kind, params=dict(params or {}))
        safety_validate(action)
        actions.append(action)
    return actions


def _actions_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    registry: ActionRegistry,
) -> list[Action]:
    """Convert LangChain tool_calls into safety-validated ``Action``s.

    Only action tools (move, water, …) are returned. Introspection tool
    calls are dropped here — they're handled by
    :func:`_actions_from_tool_calls_with_introspection` when a system
    state provider is supplied.
    """
    from safety_service import validate as safety_validate

    known = set(registry.kinds())
    pairs = tool_calls_to_actions(tool_calls)
    actions: list[Action] = []
    for kind, params in pairs:
        if kind not in known:
            raise PlanError(f"tool produced unknown action kind {kind!r}")
        action = Action(kind=kind, params=params)
        safety_validate(action)
        actions.append(action)
    return actions


def _actions_from_tool_calls_with_introspection(
    tool_calls: list[dict[str, Any]],
    registry: ActionRegistry,
    system_state: SystemStateProvider | None,
) -> tuple[list[Action], list[dict[str, Any]]]:
    """Resolve mixed tool calls: run introspection tools, keep action tools.

    Returns ``(actions, introspection_results)`` so the caller can log
    what the model looked at.
    """
    action_pairs: list[tuple[str, dict[str, Any]]] = []
    introspect_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        name = call["name"]
        if name in {
            "move", "water", "find_home", "read_pin", "write_pin",
            "take_photo", "send_message", "mount_tool",
            "dismount_tool", "e_stop",
        }:
            action_pairs.append((name, call.get("args", {})))
        else:
            introspect_calls.append(call)

    introspect_results: list[dict[str, Any]] = []
    if system_state is not None and introspect_calls:
        tools = {t.name: t for t in build_introspection_tools(system_state)}
        for call in introspect_calls:
            tool = tools.get(call["name"])
            if tool is None:
                introspect_results.append(
                    {"tool": call["name"], "error": "unknown introspection tool"}
                )
                continue
            try:
                result = tool.invoke(call.get("args", {}))
            except Exception as err:  # noqa: BLE001
                result = {"error": f"{type(err).__name__}: {err}"}
            introspect_results.append({"tool": call["name"], "result": result})

    actions = _actions_from_tool_calls(
        [{"name": k, "args": p} for k, p in action_pairs], registry
    )
    return actions, introspect_results


def _parse_with_rationale(text: str, registry: ActionRegistry) -> tuple[list[Action], str]:
    """Parse the LLM output, returning (actions, rationale)."""
    rationale = _extract_rationale_from_text(text)
    actions = parse_plan(text, registry)
    return actions, rationale


def _extract_rationale_from_text(text: str) -> str:
    for match in _JSON_FENCE_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("rationale"), str):
            return obj["rationale"]
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return ""
                if isinstance(obj, dict) and isinstance(obj.get("rationale"), str):
                    return obj["rationale"]
                return ""
    return ""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)



_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _world_context(world: Any) -> str:
    """Render a compact, model-friendly summary of the world model.

    Mirrors the YAML: name, id, kind, bounds, entity positions. Nothing
    derived, nothing invented — the model does the arithmetic if it needs
    a center.
    """
    snapshot = world.to_dict() if hasattr(world, "to_dict") else dict(world)
    lines: list[str] = []
    for zone in snapshot.get("zones", []):
        bounds = zone.get("bounds", {})
        x = bounds.get("x", 0)
        y = bounds.get("y", 0)
        w = bounds.get("width", 0)
        h = bounds.get("height", 0)
        name = zone.get("name", zone.get("id"))
        lines.append(
            f"- zone {name!r} "
            f"(kind={zone.get('kind')}, id={zone.get('id')}, "
            f"x={x}, y={y}, width={w}, height={h})"
        )
    for entity in snapshot.get("entities", []):
        pos = entity.get("position", {})
        lines.append(
            f"- entity {entity.get('name', entity.get('id'))!r} "
            f"(kind={entity.get('kind')}, id={entity.get('id')}, "
            f"x={pos.get('x')}, y={pos.get('y')}, z={pos.get('z')})"
        )
    return "\n".join(lines) if lines else "(no zones or entities configured)"


def _text_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


__all__ = [
    "Action",
    "ChatResult",
    "InMemorySystemStateProvider",
    "PlanError",
    "PlanResult",
    "PlannerConfig",
    "SystemStateProvider",
    "build_chat_model",
    "build_chat_system_prompt",
    "build_introspection_tools",
    "build_tools",
    "chat",
    "get_default_registry",
    "load_config",
    "parse_plan",
    "plan",
    "stream_chat",
]
