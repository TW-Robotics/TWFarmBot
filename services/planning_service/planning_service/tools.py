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


# ── Tool builder ────────────────────────────────────────────────────────


def build_tools(registry: ActionRegistry) -> list[BaseTool]:
    """Build LangChain tool objects for every registered action kind.

    Each tool's body is a no-op: invoking it is not how the planner
    executes — it just gives the model a structured schema to fill in.
    The actual execution happens via :class:`ActionRegistry.dispatch`
    in the API layer.
    """
    kinds = set(registry.kinds())
    tools: list[BaseTool] = []

    if "move" in kinds:
        @tool(args_schema=MoveArgs)
        def move(x: float, y: float, z: float) -> dict[str, Any]:
            """Move the gantry to absolute coordinates (mm)."""
            return {"kind": "move", "params": {"x": x, "y": y, "z": z}}
        tools.append(move)

    if "water" in kinds:
        @tool(args_schema=WaterArgs)
        def water(seconds: float) -> dict[str, Any]:
            """Turn the pump on for the given seconds."""
            return {"kind": "water", "params": {"seconds": seconds}}
        tools.append(water)

    if "find_home" in kinds:
        @tool(args_schema=FindHomeArgs)
        def find_home(axis: str = "all", speed: int = 100) -> dict[str, Any]:
            """Run the end-stop homing sequence to calibrate axes."""
            return {"kind": "find_home", "params": {"axis": axis, "speed": speed}}
        tools.append(find_home)

    if "read_pin" in kinds:
        @tool(args_schema=ReadPinArgs)
        def read_pin(pin: int, mode: str = "digital") -> dict[str, Any]:
            """Read a GPIO pin value."""
            return {"kind": "read_pin", "params": {"pin": pin, "mode": mode}}
        tools.append(read_pin)

    if "write_pin" in kinds:
        @tool(args_schema=WritePinArgs)
        def write_pin(pin: int, value: int, mode: str = "digital") -> dict[str, Any]:
            """Write a GPIO pin to the given value (0 or 1)."""
            return {"kind": "write_pin", "params": {"pin": pin, "value": value, "mode": mode}}
        tools.append(write_pin)

    if "take_photo" in kinds:
        @tool
        def take_photo() -> dict[str, Any]:
            """Trigger the camera to take a photo."""
            return {"kind": "take_photo", "params": {}}
        tools.append(take_photo)

    if "send_message" in kinds:
        @tool(args_schema=SendMessageArgs)
        def send_message(message: str, message_type: str = "info") -> dict[str, Any]:
            """Show a message to the user."""
            return {
                "kind": "send_message",
                "params": {"message": message, "message_type": message_type},
            }
        tools.append(send_message)

    if "mount_tool" in kinds:
        @tool(args_schema=MountToolArgs)
        def mount_tool(tool_name: str) -> dict[str, Any]:
            """Mount a named tool on the gantry."""
            return {"kind": "mount_tool", "params": {"tool_name": tool_name}}
        tools.append(mount_tool)

    if "dismount_tool" in kinds:
        @tool
        def dismount_tool() -> dict[str, Any]:
            """Dismount whatever tool is currently mounted."""
            return {"kind": "dismount_tool", "params": {}}
        tools.append(dismount_tool)

    if "e_stop" in kinds:
        @tool
        def e_stop() -> dict[str, Any]:
            """Emergency stop — halt the robot immediately."""
            return {"kind": "e_stop", "params": {}}
        tools.append(e_stop)

    return tools


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
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
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
        if name in {"move", "water", "find_home", "read_pin", "write_pin",
                    "take_photo", "send_message", "mount_tool",
                    "dismount_tool", "e_stop"}:
            out.append((name, args))
    return out
