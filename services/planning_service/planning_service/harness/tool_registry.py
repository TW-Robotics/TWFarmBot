"""Build a unified tool registry from the ActionRegistry and introspection tools."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from twfarmbot_core.actions import ActionRegistry

from .. import introspection
from ..tools import (
    CaptureArgs,
    CaptureNdreArgs,
    ScanNdreArgs,
    FindHomeArgs,
    MountToolArgs,
    MoveArgs,
    MovePathArgs,
    ReadPinArgs,
    WaterArgs,
    WritePinArgs,
)
from .tool_policy import ToolCategory, ToolDescriptor, ToolPolicy

# Single source of truth for action-tool policies.
_ACTION_POLICIES: dict[str, ToolPolicy] = {
    "move": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=False,
        safety_rules=("move",),
        description="Move the gantry to absolute X/Y/Z mm. Jogs under 50 mm run immediately; larger moves pause for approval in chat.",
    ),
    "water": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=True,
        safety_rules=("water",),
        description="Turn the pump on for N seconds. Pauses for operator approval in chat.",
    ),
    "find_home": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=True,
        description="Run the end-stop homing sequence. Pauses for operator approval in chat.",
    ),
    "write_pin": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=False,
        description="Write a value (0/1) to a GPIO pin.",
    ),
    "mount_tool": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=False,
        description="Mount a named tool on the gantry.",
    ),
    "dismount_tool": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=False,
        description="Dismount the currently mounted tool.",
    ),
    "read_pin": ToolPolicy(
        ToolCategory.READ,
        requires_approval=False,
        description="Read a GPIO pin value.",
    ),
    "take_photo": ToolPolicy(
        ToolCategory.READ,
        requires_approval=False,
        description=(
            "Trigger the FarmBot onboard camera. Prefer capture(band=rgb) when "
            "you need a still you can inspect yourself."
        ),
    ),
    "capture": ToolPolicy(
        ToolCategory.READ,
        requires_approval=False,
        safety_rules=("capture",),
        description=(
            "Grab one still from a payload USB camera. band is required "
            "(rgb, nir, or rededge). Does not move the gantry. The still is "
            "attached for you to inspect — look at it yourself; do not call "
            "a separate image-analysis service."
        ),
    ),
    "capture_ndre": ToolPolicy(
        ToolCategory.ANALYZE,
        requires_approval=False,
        safety_rules=("capture_ndre",),
        description=(
            "Capture NIR at the current pose, move by the calibrated band "
            "offset, capture red-edge, apply saved image_align, and return "
            "NDRE metrics plus an interpretation "
            "(label, action_hint, advice). "
            "You MUST use interpretation.action_hint and advice to decide "
            "what to do next (ok / monitor / consider_water / recheck / "
            "reposition) — do not only list the numbers. "
            "Move to the target pose first if needed. The NDRE map is attached "
            "as an image you can see; also use ndre.* and interpretation.*"
        ),
    ),
    "scan_ndre": ToolPolicy(
        ToolCategory.ANALYZE,
        requires_approval=True,
        safety_rules=("scan_ndre",),
        description=(
            "Sweep X or Y in millimetre steps and run capture_ndre at every "
            "stop. Use this for bed transects (e.g. y=0..300 step 100). "
            "Do not loop move+capture_ndre yourself. Pauses for approval in chat."
        ),
    ),
    "move_path": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=True,
        safety_rules=("move_path",),
        description="Move the gantry through a sequence of waypoints, optionally taking photos. Pauses for operator approval in chat.",
    ),
    "e_stop": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=False,
        allow_without_user=True,
        description="Emergency stop — halt the robot immediately.",
    ),
    "unlock": ToolPolicy(
        ToolCategory.ACT,
        requires_approval=False,
        description="Clear emergency stop so the gantry can move again.",
    ),
}

# Zone/garden map tooling is parked until beds are reconfigured.
# Resi heatmaps/PCA stay on the Camera page; the chat model looks at
# attached RGB stills itself. segment_image remains for pixel masks.
_HIDDEN_AGENT_TOOLS = frozenset(
    {
        "inspect_zone",
        "water_zone",
        "goto_named",
        "list_zones",
        "get_garden",
        "scan_zone",
        "analyze_image",
        "visualize_image_features",
        "estimate_traversability",
    }
)

_ACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "move": MoveArgs,
    "move_path": MovePathArgs,
    "water": WaterArgs,
    "find_home": FindHomeArgs,
    "read_pin": ReadPinArgs,
    "write_pin": WritePinArgs,
    "mount_tool": MountToolArgs,
    "capture": CaptureArgs,
    "capture_ndre": CaptureNdreArgs,
    "scan_ndre": ScanNdreArgs,
}

_INTROSPECTION_CATEGORIES: dict[str, ToolCategory] = {
    "list_endpoints": ToolCategory.READ,
    "get_health": ToolCategory.READ,
    "get_position": ToolCategory.READ,
    "get_status": ToolCategory.READ,
    "get_messages": ToolCategory.READ,
    "get_garden": ToolCategory.READ,
    "list_zones": ToolCategory.READ,
    "list_endpoints_action": ToolCategory.READ,
    "get_pins": ToolCategory.READ,
    "get_positions": ToolCategory.READ,
    "get_images": ToolCategory.READ,
    "plan_path": ToolCategory.ANALYZE,
    "scan_zone": ToolCategory.ANALYZE,
    "analyze_image": ToolCategory.ANALYZE,
    "segment_image": ToolCategory.ANALYZE,
    "visualize_image_features": ToolCategory.ANALYZE,
    "estimate_traversability": ToolCategory.ANALYZE,
}


class ToolRegistry:
    """Unified registry of LLM-facing tools.

    Combines the physical action vocabulary from ``ActionRegistry`` with
    the read-only introspection tools. Each tool is represented by a
    ``ToolDescriptor`` that carries its policy, schema, and execution logic.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        system_state: introspection.SystemStateProvider | None = None,
    ) -> None:
        self._registry = registry
        self._system_state = system_state
        self._descriptors: list[ToolDescriptor] | None = None
        self._by_name: dict[str, ToolDescriptor] | None = None

    def descriptors(self) -> list[ToolDescriptor]:
        if self._descriptors is None:
            # Action and introspection vocabularies overlap (e.g. read_pin).
            # Providers like Gemini reject duplicate function names, so merge
            # by name — later (introspection) wins, matching by_name().
            merged: dict[str, ToolDescriptor] = {}
            for descriptor in list(self._build_action_descriptors()) + list(
                self._build_introspection_descriptors()
            ):
                merged[descriptor.name] = descriptor
            self._descriptors = list(merged.values())
        return self._descriptors

    def by_name(self) -> dict[str, ToolDescriptor]:
        if self._by_name is None:
            self._by_name = {d.name: d for d in self.descriptors()}
        return self._by_name

    def langchain_tools(
        self,
        resolve: (
            Callable[[ToolDescriptor, dict[str, Any]], dict[str, Any]] | None
        ) = None,
    ) -> list[BaseTool]:
        """Return LangChain tools for all registered descriptors.

        ``resolve`` is called for every invocation; it receives the
        descriptor and parsed arguments and returns the tool result. When
        omitted, introspection tools execute normally and action tools
        return a placeholder marker.
        """
        out: list[BaseTool] = []
        for descriptor in self.descriptors():
            out.append(_descriptor_to_langchain(descriptor, resolve))
        return out

    def _build_action_descriptors(self) -> list[ToolDescriptor]:
        out: list[ToolDescriptor] = []
        for kind in self._registry.kinds():
            if kind in _HIDDEN_AGENT_TOOLS:
                continue
            policy = _ACTION_POLICIES.get(kind)
            if policy is None:
                # Unknown action kind: still expose it read-only with a safe default.
                policy = ToolPolicy(
                    ToolCategory.READ,
                    requires_approval=False,
                    description=f"Action kind '{kind}'.",
                )
            out.append(
                ToolDescriptor(
                    name=kind,
                    args_schema=_ACTION_SCHEMAS.get(kind, _NoArgs),
                    policy=policy,
                    execute=None,
                    is_introspection=False,
                )
            )
        return out

    def _build_introspection_descriptors(self) -> list[ToolDescriptor]:
        out: list[ToolDescriptor] = []
        if self._system_state is None:
            return out
        for lc_tool in introspection.build_introspection_tools(self._system_state):
            name = lc_tool.name
            if name in _HIDDEN_AGENT_TOOLS:
                continue
            schema = _schema_from_tool(lc_tool)
            category = _INTROSPECTION_CATEGORIES.get(name, ToolCategory.READ)
            policy = ToolPolicy(
                category,
                requires_approval=False,
                description=lc_tool.description or "",
            )
            out.append(
                ToolDescriptor(
                    name=name,
                    args_schema=schema,
                    policy=policy,
                    execute=lambda args, t=lc_tool: t.invoke(args),
                    is_introspection=True,
                )
            )
        return out


class _NoArgs(BaseModel):
    """Fallback schema for tools that take no arguments."""

    pass


def _schema_from_tool(lc_tool: BaseTool) -> type[BaseModel]:
    """Best-effort extraction of the args schema from a LangChain tool."""
    if isinstance(lc_tool.args_schema, type) and issubclass(
        lc_tool.args_schema, BaseModel
    ):
        return lc_tool.args_schema
    # Build a permissive schema from the tool's JSON schema if needed.
    return _NoArgs


def _descriptor_to_langchain(
    descriptor: ToolDescriptor,
    resolve: Callable[[ToolDescriptor, dict[str, Any]], dict[str, Any]] | None = None,
) -> BaseTool:
    """Wrap a descriptor as an invocable LangChain tool."""

    def _run(**kwargs: Any) -> dict[str, Any]:
        if descriptor.is_introspection and descriptor.execute is not None:
            return descriptor.execute(kwargs)
        if resolve is not None:
            return resolve(descriptor, kwargs)
        # Schema-only action tool: should never be invoked directly.
        return {"status": "noop", "kind": descriptor.name, "params": kwargs}

    # Preserve the descriptor description unless it is empty.
    description = descriptor.policy.description or descriptor.name

    return StructuredTool.from_function(
        func=_run,
        name=descriptor.name,
        description=description,
        args_schema=descriptor.args_schema,
        return_direct=False,
    )
