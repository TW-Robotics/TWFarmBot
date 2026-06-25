"""Build system prompts and conversation context for the agent loop.

Tool lists and approval notes are generated from ``ToolRegistry`` so the
prompts stay in sync with the code.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from spatial_service import format_world_context
from twfarmbot_core.config import load_yaml_config

from .tool_policy import ToolCategory
from .tool_registry import ToolRegistry

_CHAT_HEADER = """You are TWFarmBot Assistant, a helpful, concise farm-robot operator.

You can chat naturally with the user, answer questions about the robot and
garden, and perform actions by calling tools. Always respond in the same
language the user writes in.

"""

_CHAT_FOOTER = """
Guidelines:
- Before moving to a named zone, call `list_zones` to get its centre.
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
- Some actions require user approval (see tool list). When a tool returns a
  proposed-action marker, the interface shows Approve/Reject buttons. For
  multi-step tasks, call ALL required action tools in a single turn so the
  full plan is shown at once; do not propose one step at a time.
- When a question depends on the live garden state, do not rely on a single
  tool result. Gather and cross-check evidence across multiple tools and
  reason about the combined picture. For example:
  - If an image is dark or segmentation shows nothing, call `take_photo` for
    a fresh frame and/or `get_position` to see where the camera is.
  - Combine `get_position`, `list_zones`, and `get_garden` to know which zone
    the camera is pointing at and whether the view matches expectations.
  - Use `segment_image` when you need numeric presence/absence of classes.
  - If evidence is still unclear after a few tool calls, say so and propose a
    concrete next step (e.g. move to a zone with better lighting).
- Use the reasoning/thinking space to plan your tool calls before giving the
  final answer; the user will see the reasoning as a collapsible pill.
"""

_PROPOSE_ONLY_APPENDIX = """
IMPORTANT: You are in proposal mode. When the user asks you to perform one
or more actions (move, water, take_photo, etc.), you MUST call the
corresponding action tool(s) to register the proposal(s). If the request
involves multiple steps, call ALL required action tools in the correct order
within the same turn. Each tool will return a proposed-action marker; collect
them, briefly state the full plan, note that it requires approval, and stop.
Do NOT describe the action in text without calling the tool first. Do NOT ask
the user a yes/no approval question and do NOT say the action is done — the
interface shows Approve/Reject buttons for the whole plan.
"""

_PLANNER_HEADER = """You are a task planner for an autonomous farm robot.

You translate natural-language requests into a strict, ordered list of
machine actions. The robot has a fixed action vocabulary; you MUST only emit
kinds that appear in the vocabulary below. Do not invent kinds.

Output format (REQUIRED):
- Return a single JSON object with two keys: `actions` and `rationale`.
- `actions` is a JSON array, in execution order.
- Each action is {"kind": <string>, "params": <object>}.
- Keep `rationale` to one short sentence.
- Do not wrap the JSON in markdown. Do not add commentary outside the JSON.

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
        out: list[SystemMessage | HumanMessage | AIMessage] = [
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

    def planner_messages(self, request: str) -> list[SystemMessage | HumanMessage]:
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
        lines: list[str] = []
        descriptors = self._registry.descriptors()

        read_tools = [d for d in descriptors if d.policy.category == ToolCategory.READ]
        analyze_tools = [
            d for d in descriptors if d.policy.category == ToolCategory.ANALYZE
        ]
        act_tools = [d for d in descriptors if d.policy.category == ToolCategory.ACT]

        if for_planner:
            if act_tools:
                lines.append("Available action kinds:")
                for d in act_tools:
                    lines.append(f"- `{d.name}` — {d.policy.description}")
            if read_tools:
                lines.append("\nRead-only introspection tools:")
                for d in read_tools:
                    lines.append(f"- `{d.name}` — {d.policy.description}")
            return "\n".join(lines)

        if read_tools:
            lines.append("Read-only tools (use these to answer questions):")
            for d in read_tools:
                approval = (
                    " **Requires user approval.**" if d.policy.requires_approval else ""
                )
                lines.append(f"- `{d.name}` — {d.policy.description}{approval}")

        if analyze_tools:
            lines.append("\nAnalysis tools:")
            for d in analyze_tools:
                lines.append(f"- `{d.name}` — {d.policy.description}")

        if act_tools:
            lines.append("\nExecution tools (use these to change the robot state):")
            for d in act_tools:
                approval = (
                    " **Requires user approval.**"
                    if d.policy.requires_approval
                    else " Executes immediately."
                )
                lines.append(f"- `{d.name}` — {d.policy.description}{approval}")

        return "\n".join(lines)


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
                f"{v}={lbl}" for v, lbl in sorted(presets.items(), key=lambda x: int(x[0]))
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
