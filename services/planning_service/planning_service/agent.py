"""Shared setup helpers for chat and planner agents."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from twfarmbot_core.actions import ActionRegistry

from .config import PlannerConfig, load_config
from .harness import ToolRegistry
from .harness.tracing import init_weave
from .introspection import SystemStateProvider
from .providers import get_provider


def build_base_model(
    model: Any | None = None,
    config: PlannerConfig | None = None,
    model_name: str | None = None,
) -> tuple[PlannerConfig, BaseChatModel]:
    """Resolve config and build the chat model.

    ``model_name`` overrides the configured model name for this call only.
    """
    cfg = config or load_config()
    init_weave(cfg.weave_project)
    provider = get_provider(cfg.provider)
    selected_model = model_name or cfg.model
    base_model = model or provider.build_chat_model(selected_model, cfg)
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
