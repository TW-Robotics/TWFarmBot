"""Real execution tools for the chatbot.

Unlike the schema-only action tools in :mod:`tools`, these actually call
``ActionRegistry.dispatch`` so the LLM can water, move, take photos, etc.,
during a conversation. Every dispatch runs through the safety validator.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from twfarmbot_core.actions import ActionRegistry
from twfarmbot_core.domain import Action

from .tools import (
    FindHomeArgs,
    MountToolArgs,
    MoveArgs,
    ReadPinArgs,
    SendMessageArgs,
    WaterArgs,
    WritePinArgs,
)


def build_execution_tools(
    registry: ActionRegistry, *, propose_only: bool = False
) -> list[BaseTool]:
    """Build LangChain tools that either execute or propose actions.

    When ``propose_only=True`` the tools do **not** mutate the robot. They
    return a proposed-action marker so the caller can ask the user for
    approval before executing.
    """

    def _dispatch_or_propose(kind: str, params: dict[str, Any]) -> dict[str, Any]:
        if propose_only:
            return {
                "status": "proposed",
                "kind": kind,
                "params": params,
                "note": "This action is proposed and requires user approval.",
            }
        try:
            result = registry.dispatch(Action(kind=kind, params=params))
            return {"status": "ok", "kind": kind, "params": result.params}
        except Exception as err:  # noqa: BLE001
            return {"status": "error", "kind": kind, "error": f"{type(err).__name__}: {err}"}

    tools: list[BaseTool] = []

    if "move" in registry.kinds():
        @tool(args_schema=MoveArgs)
        def move(x: float, y: float, z: float) -> dict[str, Any]:
            """Move the gantry to absolute coordinates (mm)."""
            return _dispatch_or_propose("move", {"x": x, "y": y, "z": z})
        tools.append(move)

    if "water" in registry.kinds():
        @tool(args_schema=WaterArgs)
        def water(bed_id: str, seconds: float) -> dict[str, Any]:
            """Open a bed's irrigation valve for the given seconds."""
            return _dispatch_or_propose("water", {"bed_id": bed_id, "seconds": seconds})
        tools.append(water)

    if "find_home" in registry.kinds():
        @tool(args_schema=FindHomeArgs)
        def find_home(axis: str = "all", speed: int = 100) -> dict[str, Any]:
            """Run the end-stop homing sequence to calibrate axes."""
            return _dispatch_or_propose("find_home", {"axis": axis, "speed": speed})
        tools.append(find_home)

    if "read_pin" in registry.kinds():
        @tool(args_schema=ReadPinArgs)
        def read_pin(pin: int, mode: str = "digital") -> dict[str, Any]:
            """Read a GPIO pin value."""
            return _dispatch_or_propose("read_pin", {"pin": pin, "mode": mode})
        tools.append(read_pin)

    if "write_pin" in registry.kinds():
        @tool(args_schema=WritePinArgs)
        def write_pin(pin: int, value: int, mode: str = "digital") -> dict[str, Any]:
            """Write a GPIO pin to the given value (0 or 1)."""
            return _dispatch_or_propose("write_pin", {"pin": pin, "value": value, "mode": mode})
        tools.append(write_pin)

    if "take_photo" in registry.kinds():
        @tool
        def take_photo() -> dict[str, Any]:
            """Trigger the camera to take a photo."""
            return _dispatch_or_propose("take_photo", {})
        tools.append(take_photo)

    if "send_message" in registry.kinds():
        @tool(args_schema=SendMessageArgs)
        def send_message(message: str, message_type: str = "info") -> dict[str, Any]:
            """Show a message to the user."""
            return _dispatch_or_propose(
                "send_message", {"message": message, "message_type": message_type}
            )
        tools.append(send_message)

    if "mount_tool" in registry.kinds():
        @tool(args_schema=MountToolArgs)
        def mount_tool(tool_name: str) -> dict[str, Any]:
            """Mount a named tool on the gantry."""
            return _dispatch_or_propose("mount_tool", {"tool_name": tool_name})
        tools.append(mount_tool)

    if "dismount_tool" in registry.kinds():
        @tool
        def dismount_tool() -> dict[str, Any]:
            """Dismount whatever tool is currently mounted."""
            return _dispatch_or_propose("dismount_tool", {})
        tools.append(dismount_tool)

    if "e_stop" in registry.kinds():
        @tool
        def e_stop() -> dict[str, Any]:
            """Emergency stop — halt the robot immediately."""
            return _dispatch_or_propose("e_stop", {})
        tools.append(e_stop)

    return tools
