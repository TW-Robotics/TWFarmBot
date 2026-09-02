"""Argument schemas for FarmBot tools.

Each action kind in the :class:`ActionRegistry` has a Pydantic schema.
The agent loop exposes these as JSON function/tool calls. A free-form
JSON action plan remains a fallback for ``plan()``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from twfarmbot_core.actions import ActionRegistry

# ── Tool argument schemas ────────────────────────────────────────────────


class MoveArgs(BaseModel):
    x: float = Field(..., description="X coordinate in millimetres.")
    y: float = Field(..., description="Y coordinate in millimetres.")
    z: float = Field(..., description="Z coordinate in millimetres.")
    speed: float | None = Field(
        default=None,
        description="Optional speed as 1..100 percent of axis max (or steps/s if >100).",
    )


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
    mode: str = Field(
        default="digital",
        description=(
            "'digital' or 'analog'. If omitted, default to the mode configured "
            "for this pin in the system config."
        ),
    )


class WritePinArgs(BaseModel):
    pin: int = Field(..., description="GPIO pin number to write.")
    value: int = Field(
        ...,
        description=(
            "Value to write. For digital pins use 0 or 1. For analog pins use "
            "0..255 (PWM)."
        ),
    )
    mode: str = Field(
        default="digital",
        description=(
            "'digital' or 'analog'. If omitted, default to the mode configured "
            "for this pin in the system config."
        ),
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
        description="Optional speed as 1..100 percent of axis max (or steps/s if >100).",
    )
    photo_at_waypoints: bool = Field(
        default=False,
        description="If true, take a photo at every waypoint.",
    )
    water_pin: int | None = Field(
        default=None,
        description="If set, keep this GPIO pin HIGH while moving and turn it OFF once the final position is reached.",
    )


class CaptureArgs(BaseModel):
    band: str = Field(
        ...,
        description=(
            "Payload USB camera band. Required: 'rgb', 'nir', or 'rededge'. "
            "'thermal' and 'swir' are not available."
        ),
    )


class CaptureNdreArgs(BaseModel):
    return_to_start: bool = Field(
        default=True,
        description=(
            "After capturing red-edge at the calibrated offset, move back to the "
            "starting gantry pose (default true)."
        ),
    )


class InspectZoneArgs(BaseModel):
    zone_id: str = Field(..., description="Zone id or loose name, e.g. 'tomato'.")
    step_mm: float = Field(default=200.0, description="Raster spacing in millimetres.")
    z: float = Field(default=0.0, description="Gantry Z for the sweep, in millimetres.")
    classes: str = Field(
        default="plant, weed, soil",
        description="Comma-separated segmentation classes for the scorecard.",
    )


class WaterZoneArgs(BaseModel):
    zone_id: str = Field(..., description="Zone id or loose name to water.")
    seconds: float = Field(..., description="How long to keep the pump on (1..300).")
    z: float = Field(
        default=0.0, description="Gantry Z at the zone centre, in millimetres."
    )


class GotoNamedArgs(BaseModel):
    name: str = Field(
        ..., description="Zone, plant, or preset name, e.g. 'tomatoes' or 'Home'."
    )
    z: float | None = Field(
        default=None,
        description="Optional Z override in millimetres. Uses the resolved target Z when omitted.",
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
            "capture",
            "capture_ndre",
            "mount_tool",
            "dismount_tool",
            "e_stop",
            "unlock",
            "inspect_zone",
            "water_zone",
            "goto_named",
        }:
            out.append((name, args))
    return out
