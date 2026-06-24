"""LangChain tool definitions for the planner.

Each action kind in the :class:`ActionRegistry` becomes a ``@tool`` with
a Pydantic-typed schema. The model is bound to these tools via
``model.bind_tools(tools)`` and produces a structured tool call instead
of free-form JSON. The parser still accepts the JSON path for models
that don't support tool calling.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from twfarmbot_core.actions import ActionRegistry

# ── Tool argument schemas ────────────────────────────────────────────────


class MoveArgs(BaseModel):
    x: float = Field(..., description="X coordinate in millimetres.")
    y: float = Field(..., description="Y coordinate in millimetres.")
    z: float = Field(..., description="Z coordinate in millimetres.")


class WaterArgs(BaseModel):
    seconds: float = Field(..., description="How long to keep the pump on (1..300).")


class FindHomeArgs(BaseModel):
    axis: str = Field(
        default="all",
        description="Which axis to home: 'x', 'y', 'z', or 'all'.",
    )
    speed: int = Field(default=100, description="Homing speed (1..100).")


class ReadPinArgs(BaseModel):
    pin: int = Field(..., description="GPIO pin number to read.")
    mode: str = Field(default="digital", description="'digital' or 'analog'.")


class WritePinArgs(BaseModel):
    pin: int = Field(..., description="GPIO pin number to write.")
    value: int = Field(..., description="0 or 1.")
    mode: str = Field(default="digital", description="'digital' or 'analog'.")


class SendMessageArgs(BaseModel):
    message: str = Field(..., description="Message text to show.")
    message_type: str = Field(
        default="info",
        description="'info', 'success', 'warn', or 'error'.",
    )


class MountToolArgs(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to mount.")


class Waypoint(BaseModel):
    x: float = Field(..., description="X coordinate in millimetres.")
    y: float = Field(..., description="Y coordinate in millimetres.")
    z: float = Field(..., description="Z coordinate in millimetres.")


class MovePathArgs(BaseModel):
    waypoints: list[Waypoint] = Field(
        ...,
        description="Sequence of waypoints to visit.",
    )
    speed: float | None = Field(
        default=None,
        description="Optional movement speed override.",
    )
    photo_at_waypoints: bool = Field(
        default=False,
        description="If true, take a photo at every waypoint.",
    )


# ── Tool builder ────────────────────────────────────────────────────────


def build_tools(registry: ActionRegistry) -> list[BaseTool]:
    """Build LangChain tool objects for every registered action kind.

    This is now a thin compatibility wrapper around the harness
    ``ToolRegistry``. The returned tools carry the correct schemas and
    descriptions; execution semantics are applied later by ``AgentLoop``.
    """
    from .harness import ToolRegistry

    return ToolRegistry(registry, system_state=None).langchain_tools()


def extract_tool_calls(response: Any) -> list[dict[str, Any]] | None:
    """Pull ``tool_calls`` out of a LangChain AIMessage.

    Returns ``None`` when the model did not call any tools (e.g. it
    returned plain text instead). The caller falls back to JSON parsing
    in that case.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return None
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        name = (
            call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        )
        args = (
            call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        )
        if name is None:
            continue
        out.append({"name": name, "args": dict(args or {})})
    return out or None


def tool_calls_to_actions(
    tool_calls: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Map tool-call names back to (kind, params) for the registry.

    The tool body returns the same shape so this is a simple pass-through.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {})
        if name in {
            "move",
            "move_path",
            "water",
            "find_home",
            "read_pin",
            "write_pin",
            "take_photo",
            "send_message",
            "mount_tool",
            "dismount_tool",
            "e_stop",
        }:
            out.append((name, args))
    return out
