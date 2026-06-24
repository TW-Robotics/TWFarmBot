"""Shared setup helpers for chat and planner agents."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from twfarmbot_core.actions import ActionRegistry

from .client import build_chat_model
from .config import PlannerConfig, load_config
from .execution_tools import build_execution_tools
from .introspection import SystemStateProvider, build_introspection_tools
from .tools import build_tools


def build_base_model(
    model: Any | None = None,
    config: PlannerConfig | None = None,
) -> tuple[PlannerConfig, BaseChatModel]:
    """Resolve config and build the chat model."""
    cfg = config or load_config()
    base_model = model or build_chat_model(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        temperature=cfg.temperature,
        extra_body=cfg.extra_body,
    )
    return cfg, base_model


def build_tool_set(
    registry: ActionRegistry,
    system_state: SystemStateProvider | None = None,
    *,
    for_chat: bool = True,
    propose_only: bool = False,
    allow_actions: bool = True,
) -> list[BaseTool]:
    """Build the combined tool list for a chat/planner run.

    Execution tools win name collisions against introspection tools (e.g.
    ``read_pin``) so the robot actually changes state.
    """
    if for_chat:
        execution_tools = (
            build_execution_tools(registry, propose_only=propose_only)
            if allow_actions
            else []
        )
    else:
        execution_tools = build_tools(registry)

    introspection_tools = (
        build_introspection_tools(system_state) if system_state is not None else []
    )
    execution_names = {t.name for t in execution_tools}
    return list(execution_tools) + [
        t for t in introspection_tools if t.name not in execution_names
    ]
