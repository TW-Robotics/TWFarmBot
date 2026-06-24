"""FarmBot LLM harness primitives.

A small, reusable layer on top of LangChain that owns the agent loop,
tool policy, approval gate, reasoning extraction, and prompt context.
"""

from __future__ import annotations

from .agent_loop import AgentLoop, AgentTurnResult
from .approval_gate import ApprovalGate, ProposedResult, ToolResult
from .context_builder import ContextBuilder
from .reasoning_controller import ReasoningController
from .tool_policy import ToolCategory, ToolDescriptor, ToolPolicy
from .tool_registry import ToolRegistry

__all__ = [
    "AgentLoop",
    "AgentTurnResult",
    "ApprovalGate",
    "ContextBuilder",
    "Event",
    "ProposedResult",
    "ReasoningController",
    "ToolCategory",
    "ToolDescriptor",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
]
