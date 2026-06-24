"""Tool policy primitives.

Every tool exposed to the LLM carries a ``ToolPolicy`` that declares:
- what category it belongs to (read, act, analyze),
- whether it requires user approval,
- which safety validators apply,
- a model-facing description.

This is the single source of truth for "can this tool mutate the robot?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


class ToolCategory(str, Enum):
    READ = "read"
    ACT = "act"
    ANALYZE = "analyze"


@dataclass(frozen=True)
class ToolPolicy:
    """Behavioral policy for one tool."""

    category: ToolCategory
    requires_approval: bool = False
    allow_without_user: bool = True  # if False, even allow_actions=True cannot auto-run
    safety_rules: tuple[str, ...] = ()  # names of safety_service validators
    description: str = ""


@dataclass(frozen=True)
class ToolDescriptor:
    """Complete description of one LLM-facing tool."""

    name: str
    args_schema: type[BaseModel]
    policy: ToolPolicy
    execute: Callable[..., dict[str, Any]] | None = None
    is_introspection: bool = False

    @property
    def is_read_only(self) -> bool:
        return self.policy.category in (ToolCategory.READ, ToolCategory.ANALYZE)
