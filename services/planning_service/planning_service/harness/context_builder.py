"""Build system prompts and conversation context for the agent loop.

Tool lists and approval notes are generated from ``ToolRegistry`` so the
prompts stay in sync with the code.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from spatial_service import format_world_context
from twfarmbot_core.config import load_yaml_config

from .farm_script import format_tool_catalog
from .tool_registry import ToolRegistry

_CHAT_HEADER = """You are TWFarmBot Assistant, a helpful, concise farm-robot operator.

You can chat naturally with the user, answer questions about the robot and
garden, and use the available tools.

Use Programmatic Tool Calling for bounded read-only workflows
(list_zones, get_position, get_garden, get_images, get_health, get_pins,
segment_image, analyze_image, plan_path, scan_zone). Run independent reads
concurrently, reduce the results to a small JSON object, then answer. Do not
expose generated programs or internal reasoning to the user.

For physical work, call one direct action so the safety/approval gate can
register it: inspect_zone, water_zone, goto_named, move, water, move_path.
Never merely describe an action that the user asked you to perform. Always
respond in the same language the user writes in.

"""

_CHAT_FOOTER = """
Guidelines:
- To inspect a named bed, call `inspect_zone`. To water it, call `water_zone`.
  To go to a named bed, plant, or preset, call `goto_named`.
- Before a custom move to a named zone, call `list_zones` to get its centre.
- Keep answers short and actionable. Confirm what you did and any relevant
  sensor/position readings.
- If a request is unsafe or impossible, refuse and explain why.
- When asked about a specific zone or bed (e.g. "radischen", "tomatoes"),
  ALWAYS move the camera to that zone first, then call `take_photo`, and
  only then run an analysis tool like `segment_image` or `analyze_image`.
  Do not analyze the most recent image if it was taken somewhere else.
- When you call analysis tools (`analyze_image`, `segment_image`,
  `visualize_image_features`, `estimate_traversability`), you cannot see the
  returned images yourself. Use the numeric metrics and class lists the tools
  provide, then state what analysis was run and that the images are shown to
  the user.
- Some actions require user approval (see function list). When a call returns
  a proposed-action marker, the interface shows Approve/Reject buttons. For
  multi-step tasks, keep the complete proposal together so the full plan is
  shown at once.
- After every physical action, call the relevant live read tool (especially
  `get_position` after movement) and compare the observed result with the
  requested target before answering. If it is not satisfied, continue the
  bounded tool loop with a corrective action when safe; never claim success
  from an action acknowledgement alone.
- When a question depends on the live garden state, gather and cross-check
  evidence in one script. For example:
  - If an image is dark or segmentation shows nothing, call `take_photo` for
    a fresh frame and/or `get_position` to see where the camera is.
  - Combine `get_position`, `list_zones`, and `get_garden` to know which zone
    the camera is pointing at and whether the view matches expectations.
  - Use `segment_image` when you need numeric presence/absence of classes.
  - If evidence is still unclear after a few calls, say so and propose a
    concrete next step (e.g. move to a zone with better lighting).
- Use the reasoning/thinking space to plan the script before giving the
  final answer; the user will see the reasoning as a collapsible pill.
"""

_PROPOSE_ONLY_APPENDIX = """
IMPORTANT: You are in proposal mode. When the user asks you to perform one
or more actions (move, water, take_photo, etc.), you MUST call the
  corresponding action tools to register the proposal(s). If the request
  involves multiple steps, register ALL of them in the correct order.
For example, “move it to the middle” requires an actual `move` function call
with coordinates, followed by a final explanation. Your next model operation
must be the function call; a text-only plan is invalid.
Each call will return a proposed-action marker; collect them, briefly state
the full plan, note that it requires approval, and stop.
Do NOT describe the action in text without calling the function first. Do NOT
ask the user a yes/no approval question and do NOT say the action is done —
the interface shows Approve/Reject buttons for the whole plan.
"""

_PLANNER_HEADER = """You are a task planner for an autonomous farm robot.

Translate natural-language requests into an ordered plan using the available
tools. Use Programmatic Tool Calling for bounded read-only data gathering and
reduction. Physical actions must remain direct, safety-validated proposals.
Do not emit a JSON action plan or generated program as the user-facing answer.

"""

_PLANNER_FOOTER = """
Grounding names to coordinates:
- Match names LOOSELY: "the tomatoes", "tomato", "tomato zone", and
  "Tomato Zone" all refer to the same entry. Match by stem
  (tomato/herbs/camera) not by exact string.
- To "move to a named zone", use its `center` from `list_zones`.
- To "move to a named entity", use its `(x, y, z)` from the world model.
- DEFAULT to producing a plan. Only return `actions: []` when the request is
  genuinely impossible.
- If the request is ambiguous, pick the most specific match and explain in
  `rationale`.
- If the request is unsafe or impossible, return `actions: []` and explain
  in `rationale`.
"""


class ContextBuilder:
    """Build prompts and LangChain message lists."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        world: Any = None,
        propose_only: bool = False,
    ) -> None:
        self._registry = tool_registry
        self._world = world
        self._propose_only = propose_only

    def chat_system_prompt(self) -> str:
        parts = [_CHAT_HEADER]
        parts.append(self._render_tool_section())
        parts.append(_format_pin_context())
        parts.append(_CHAT_FOOTER)
        if self._propose_only:
            parts.append(_PROPOSE_ONLY_APPENDIX)
        parts.append(
            "\nRegistered action kinds you can use: "
            + ", ".join(sorted(self._registry.by_name()))
            + "."
        )
        return "\n".join(parts)

    def planner_system_prompt(self) -> str:
        parts = [_PLANNER_HEADER]
        parts.append(self._render_tool_section(for_planner=True))
        parts.append(_format_pin_context())
        parts.append(_PLANNER_FOOTER)
        return "\n".join(parts)

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        include_reasoning: bool = False,
    ) -> list[SystemMessage | HumanMessage | AIMessage | ToolMessage]:
        system = self.chat_system_prompt()
        world_context = (
            format_world_context(self._world) if self._world is not None else None
        )
        if world_context:
            system += "\n\nCurrent world model:\n" + world_context
        out: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [
            SystemMessage(content=system)
        ]
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "") or "")
            if role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant":
                kwargs: dict[str, Any] = {}
                if include_reasoning:
                    thinking = msg.get("thinking")
                    if thinking:
                        kwargs["additional_kwargs"] = {
                            "reasoning_content": str(thinking)
                        }
                out.append(AIMessage(content=content, **kwargs))
            elif role == "tool":
                # Preserve tool results across multi-turn conversation.
                out.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=str(msg.get("tool_call_id", "")),
                        name=str(msg.get("name", "")),
                    )
                )
        return out

    def planner_messages(
        self, request: str
    ) -> list[SystemMessage | HumanMessage | ToolMessage]:
        system = self.planner_system_prompt()
        world_context = (
            format_world_context(self._world) if self._world is not None else None
        )
        if world_context:
            system += "\n\nCurrent world model:\n" + world_context
        return [
            SystemMessage(content=system),
            HumanMessage(content=request),
        ]

    def _render_tool_section(self, for_planner: bool = False) -> str:
        del for_planner
        return format_tool_catalog(self._registry.descriptors())


def _format_pin_context() -> str:
    """Load named pins from config and format them for the system prompt."""
    try:
        pins = load_yaml_config().get("pins", []) or []
    except Exception:  # noqa: BLE001
        pins = []
    if not pins:
        return ""
    lines = ["\nConfigured GPIO pins (single source of truth):"]
    for p in pins:
        label = p.get("label", "unknown")
        pin = p.get("pin", "?")
        mode = p.get("mode", "digital")
        kind = p.get("kind", "io")
        group = p.get("group", "")
        group_text = f" · {group}" if group else ""
        presets = p.get("presets") or {}
        preset_text = ""
        if presets:
            preset_items = ", ".join(
                f"{v}={lbl}"
                for v, lbl in sorted(presets.items(), key=lambda x: int(x[0]))
            )
            preset_text = f" · presets: {preset_items}"
        lines.append(
            f"- pin {pin} · {label} · mode={mode} · kind={kind}{group_text}{preset_text}"
        )
    lines.append(
        "When calling read_pin or write_pin, use the configured mode for the pin "
        "unless the user explicitly asks for a different mode. For analog pins, "
        "use the named preset values when the user refers to them."
    )
    return "\n".join(lines)
