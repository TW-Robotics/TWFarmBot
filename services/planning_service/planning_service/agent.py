"""Shared setup helpers for chat and planner agents."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from twfarmbot_core.actions import ActionRegistry

from .client import build_chat_model
from .config import PlannerConfig, load_config
from .harness import ToolRegistry
from .introspection import SystemStateProvider


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

    The harness owns approval/execution semantics; this helper now just
    returns schema-complete LangChain tools generated from the unified
    ``ToolRegistry``.

    The ``for_chat``, ``propose_only``, and ``allow_actions`` parameters
    are kept for backward compatibility but no longer change the returned
    tool schemas — policy is applied at invocation time by ``AgentLoop``.
    """
    del for_chat, propose_only, allow_actions
    tool_registry = ToolRegistry(registry, system_state)
    return tool_registry.langchain_tools()
