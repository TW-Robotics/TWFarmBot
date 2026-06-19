"""Conversational chat interface for the FarmBot.

The model is bound to both introspection (read-only) and execution tools so it
can answer "what is the status?" and also water, take photos, move, etc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from twfarmbot_core.actions import ActionRegistry

from .client import build_chat_model
from .config import PlannerConfig, load_config
from .execution_tools import build_execution_tools
from .introspection import SystemStateProvider, build_introspection_tools
from .prompt import build_chat_system_prompt

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatResult:
    """Result of a single chat turn."""

    response: str
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)


def _to_langchain_message(message: dict[str, Any]) -> SystemMessage | HumanMessage | AIMessage:
    role = message.get("role", "")
    content = message.get("content", "")
    if role == "user":
        return HumanMessage(content=str(content))
    if role == "assistant":
        return AIMessage(content=str(content))
    # Unknown roles are treated as system prompts.
    return SystemMessage(content=str(content))


def chat(
    messages: list[dict[str, Any]],
    *,
    registry: ActionRegistry,
    world: Any = None,
    system_state: SystemStateProvider | None = None,
    model: Any = None,
    config: PlannerConfig | None = None,
    allow_actions: bool = True,
    propose_only: bool = False,
    max_iterations: int = 5,
) -> ChatResult:
    """Run one conversational turn with tool use.

    ``messages`` should contain the conversation so far (user + assistant
    turns, no system message). The function prepends a system prompt,
    runs the model, executes any tool calls, and returns the final
    assistant text plus a log of tool calls made.

    When ``propose_only=True`` action tools do not mutate the robot. They
    only record proposed actions so the UI can ask the user for approval.
    """
    cfg = config or load_config()
    base_model = model or build_chat_model(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        temperature=cfg.temperature,
    )

    introspection_tools = (
        build_introspection_tools(system_state) if system_state is not None else []
    )
    execution_tools = (
        build_execution_tools(registry, propose_only=propose_only)
        if allow_actions else []
    )

    # Avoid duplicate read_pin tools: the execution read_pin does the same job
    # as the introspection one and actually returns the value.
    introspection_names = {t.name for t in introspection_tools}
    execution_names = {t.name for t in execution_tools}
    all_tools: list[BaseTool] = list(execution_tools)
    for tool in introspection_tools:
        if tool.name not in execution_names:
            all_tools.append(tool)

    chat_model = base_model.bind_tools(all_tools) if all_tools else base_model
    tool_map = {t.name: t for t in all_tools}

    system_prompt = build_chat_system_prompt(registry.kinds(), propose_only=propose_only)
    if world is not None:
        world_context = _world_context(world)
        if world_context:
            system_prompt += "\n\nCurrent world model:\n" + world_context

    langchain_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        langchain_messages.append(_to_langchain_message(msg))

    tool_log: list[dict[str, Any]] = []
    proposed_actions: list[dict[str, Any]] = []
    final_response = ""

    for _ in range(max_iterations):
        response = chat_model.invoke(langchain_messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            final_response = str(response.content or "")
            break

        langchain_messages.append(response)
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            tool_call_id = call.get("id", "")
            tool = tool_map.get(name)
            if tool is None:
                result = {"error": f"unknown tool {name!r}"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as err:  # noqa: BLE001
                    result = {"error": f"{type(err).__name__}: {err}"}
            tool_log.append({"name": name, "args": args, "result": result})
            if isinstance(result, dict) and result.get("status") == "proposed":
                proposed_actions.append({
                    "kind": result.get("kind", name),
                    "params": result.get("params", args),
                })
            langchain_messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call_id,
                    name=name,
                )
            )
    else:
        # Hit the iteration limit; return the last model text if any.
        final_response = str(getattr(response, "content", "") or "")
        if not final_response:
            final_response = "I ran too many tool calls without finishing. Please try again."

    out_messages = list(messages)
    out_messages.append({"role": "assistant", "content": final_response})
    return ChatResult(
        response=final_response,
        proposed_actions=proposed_actions,
        tool_calls=tool_log,
        messages=out_messages,
    )


def stream_chat(
    messages: list[dict[str, Any]],
    *,
    registry: ActionRegistry,
    world: Any = None,
    system_state: SystemStateProvider | None = None,
    model: Any = None,
    config: PlannerConfig | None = None,
    allow_actions: bool = True,
    propose_only: bool = False,
    max_iterations: int = 5,
):
    """Streaming version of :func:`chat`.

    Yields events:
      - ``{"type": "delta", "content": "..."}`` for each piece of the
        final assistant text.
      - ``{"type": "tool_call", "name": ..., "args": ..., "result": ...}``
        after a tool is executed.
      - ``{"type": "meta", "tool_calls": [...], "proposed_actions": [...]}``
        at the very end.

    Tool calls are resolved server-side; the text stream only starts after
    the model has finished using read-only tools and decided on a final
    answer.
    """
    cfg = config or load_config()
    base_model = model or build_chat_model(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        temperature=cfg.temperature,
    )

    introspection_tools = (
        build_introspection_tools(system_state) if system_state is not None else []
    )
    execution_tools = (
        build_execution_tools(registry, propose_only=propose_only)
        if allow_actions else []
    )

    introspection_names = {t.name for t in introspection_tools}
    execution_names = {t.name for t in execution_tools}
    all_tools: list[BaseTool] = list(execution_tools)
    for tool in introspection_tools:
        if tool.name not in execution_names:
            all_tools.append(tool)

    chat_model = base_model.bind_tools(all_tools) if all_tools else base_model
    tool_map = {t.name: t for t in all_tools}

    system_prompt = build_chat_system_prompt(registry.kinds(), propose_only=propose_only)
    if world is not None:
        world_context = _world_context(world)
        if world_context:
            system_prompt += "\n\nCurrent world model:\n" + world_context

    langchain_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        langchain_messages.append(_to_langchain_message(msg))

    tool_log: list[dict[str, Any]] = []
    proposed_actions: list[dict[str, Any]] = []
    action_tool_names = {t.name for t in execution_tools}
    last_response = None

    for _ in range(max_iterations):
        response = chat_model.invoke(langchain_messages)
        last_response = response
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        langchain_messages.append(response)
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            tool_call_id = call.get("id", "")
            tool = tool_map.get(name)
            if tool is None:
                result = {"error": f"unknown tool {name!r}"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as err:  # noqa: BLE001
                    result = {"error": f"{type(err).__name__}: {err}"}
            tool_log.append({"name": name, "args": args, "result": result})
            if isinstance(result, dict) and result.get("status") == "proposed":
                proposed_actions.append({
                    "kind": result.get("kind", name),
                    "params": result.get("params", args),
                })
            yield {"type": "tool_call", "name": name, "args": args, "result": result}
            langchain_messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call_id,
                    name=name,
                )
            )

    # Fallback: some models describe the proposed action in text without ever
    # calling the action tool. If no action tool was invoked but the answer
    # looks like a proposal, ask the planner for a concrete action list so the
    # UI can render Approve/Reject buttons.
    if (
        not any(tc["name"] in action_tool_names for tc in tool_log)
        and last_response is not None
        and _response_describes_action(str(last_response.content or ""), messages)
    ):
        from planning_service import plan as planner_plan

        last_user = next(
            (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        try:
            plan_result = planner_plan(
                last_user,
                registry=registry,
                world=world,
                system_state=system_state,
                model=base_model,
                config=cfg,
            )
        except Exception:  # noqa: BLE001
            plan_result = None
        if plan_result and plan_result.actions:
            proposed_actions = [
                {"kind": a.kind, "params": a.params} for a in plan_result.actions
            ]
            tool_log.append({
                "name": "planner_fallback",
                "args": {"request": last_user},
                "result": {
                    "status": "proposed",
                    "actions": proposed_actions,
                    "rationale": plan_result.rationale,
                },
            })

    yield {"type": "meta", "tool_calls": tool_log, "proposed_actions": proposed_actions}

    for chunk in chat_model.stream(langchain_messages):
        content = getattr(chunk, "content", None)
        if content:
            yield {"type": "delta", "content": str(content)}


def _world_context(world: Any) -> str:
    """Render a compact summary of the configured world model."""
    snapshot = world.to_dict() if hasattr(world, "to_dict") else dict(world)
    lines: list[str] = []
    for zone in snapshot.get("zones", []):
        bounds = zone.get("bounds", {})
        lines.append(
            f"- zone {zone.get('name', zone.get('id'))!r} "
            f"(x={bounds.get('x', 0)}, y={bounds.get('y', 0)}, "
            f"width={bounds.get('width', 0)}, height={bounds.get('height', 0)})"
        )
    for entity in snapshot.get("entities", []):
        pos = entity.get("position", {})
        lines.append(
            f"- entity {entity.get('name', entity.get('id'))!r} "
            f"(x={pos.get('x')}, y={pos.get('y')}, z={pos.get('z')})"
        )
    return "\n".join(lines)


def _response_describes_action(response_text: str, messages: list[dict[str, Any]]) -> bool:
    """Detect when the model described an action but skipped the tool call."""
    text = response_text.lower()
    last_user = next(
        (str(m.get("content", "")).lower() for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    action_words = {"move", "water", "irrigate", "find_home", "home", "e_stop", "mount_tool", "dismount_tool"}
    if not any(word in last_user or word in text for word in action_words):
        return False
    # Concrete proposal indicators
    if re.search(r"→|->", text):
        return True
    if re.search(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)", text):
        return True
    if ("bed" in text or "seconds" in text) and ("water" in text or "irrigate" in text):
        return True
    return False
