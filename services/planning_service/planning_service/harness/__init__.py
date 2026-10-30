"""FarmBot LLM harness primitives.

A small, reusable layer on top of LangChain and LangGraph that owns the agent
loop, tool policy, approval gate, reasoning extraction, and prompt context.
"""

from __future__ import annotations

from .loop import STOP_DONE, STOP_MAX_ERRORS, STOP_MAX_TURNS, AgentLoop, AgentTurnResult, RunState
from .approval_gate import ApprovalGate, ProposedResult, ToolResult
from .context_builder import ContextBuilder
from .farm_script import FarmScriptError, FarmScriptRuntime, extract_farm_scripts
from .tool_policy import ToolCategory, ToolDescriptor, ToolPolicy
from .tool_registry import ToolRegistry

__all__ = [
    "AgentLoop",
    "AgentTurnResult",
    "ApprovalGate",
    "ContextBuilder",
    "FarmScriptError",
    "FarmScriptRuntime",
    "extract_farm_scripts",
    "ProposedResult",
    "RunState",
    "STOP_DONE",
    "STOP_MAX_ERRORS",
    "STOP_MAX_TURNS",
    "ToolCategory",
    "ToolDescriptor",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
]
