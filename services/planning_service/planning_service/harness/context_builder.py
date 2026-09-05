"""Build system prompts and conversation context for the agent loop.

Tool lists and approval notes are generated from ``ToolRegistry`` so the
prompts stay in sync with the code.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from spatial_service import format_world_context
from twfarmbot_core.config import load_yaml_config

from ..tool_results import compact_input_text
from .farm_script import format_tool_catalog
from .tool_registry import ToolRegistry

_CHAT_HEADER = """You are TWFarmBot Assistant, a helpful, concise farm-robot operator.

You can chat naturally with the user, answer questions about the robot, and
use the available tools.

Call tools through JSON function/tool calling. Read tools
(get_position, get_images, get_health, get_pins, segment_image, analyze_image,
plan_path) are ordinary JSON tools. Call independent reads together when
useful, then answer from the results. Do not write Python scripts or
farm_script fences.

For physical work, call the action tools directly (move, water, find_home,
move_path, take_photo, capture, capture_ndre, etc.). Prefer absolute
`move(x,y,z)` — there are no named garden zones right now.

Never merely describe an action that the user asked you to perform. Always
respond in the same language the user writes in.

"""

_CHAT_FOOTER = """
Guidelines:
- Keep answers short and actionable. Confirm what you did and any relevant
  sensor/position readings.
- Do NOT refuse moves because of workspace/garden bounds. The operator is
  present; only refuse clearly impossible requests (e.g. missing coordinates
  when asked to move somewhere specific) or hard tool errors.
- When the user names a plant/area without coordinates, ask for mm targets
  or use the current pose — do not invent zone centres or cite garden bounds.
- When you call `capture`, `take_photo`, or `capture_ndre`, the latest still is
  attached as an image you can see. Use it. Analysis overlays
  (`analyze_image`, `segment_image`, …) stay user-facing unless a still is
  attached the same way.
- For canopy vigor / water-stress questions: move to the requested pose
  (or stay if already there), then call `capture_ndre`.
- After `capture_ndre`, do NOT just restate the numbers. You must:
  1. Read `interpretation.label`, `action_hint`, and `advice`.
  2. Check `get_position` so you know where the reading was taken.
  3. Decide a next step: if action_hint is `consider_water`, offer or run
     `water` when the user wants action; if `reposition`/`recheck`, move and
     capture again; if `ok`/`monitor`, say the canopy looks fine and stop.
  4. Answer in 2–4 short sentences: verdict + why (1–2 metrics) + what you
     will do / recommend. The NDRE map is shown to the user automatically.
- Do not use `segment_image` on NIR/red-edge grayscale.
- Water duration is still safety-limited. If a tool returns an error, explain
  it briefly — do not tell the user to "fix garden bounds".
- After every physical action, call `get_position` and compare with the
  requested target before the final answer.
- Do not stop after `get_position` or other reads. If the user asked to move,
  photograph, or measure NDRE, keep calling those tools until the sequence is
  done. Never reply with a plan of remaining steps instead of calling tools.
- A Y/X bed transect with photos or NDRE: call `scan_ndre` once
  (`axis`, `end_mm`, `step_mm`). Example: y to 300 mm every 100 mm →
  `scan_ndre(axis="y", end_mm=300, step_mm=100, start_mm=0)`. Do not
  substitute a single `capture_ndre` at the current pose.
- Use the reasoning/thinking space to plan tool calls before giving the
  final answer; the user will see the reasoning as a collapsible pill.
"""

_PLANNER_HEADER = """You are a task planner for an autonomous farm robot.

Translate natural-language requests into an ordered plan using JSON
function/tool calling. Call read tools to gather live state. Physical
actions must remain safety-validated proposals. Do not write Python
scripts, and do not emit a JSON action plan as the user-facing answer
when you can call tools instead.

"""

_PLANNER_FOOTER = """
Grounding coordinates:
- There are no named zones. Use absolute millimetre coordinates from the
  user, or `get_position` for the current pose.
- DEFAULT to producing a plan. Only return `actions: []` when the request is
  genuinely impossible (e.g. missing required numbers).
- Do not refuse because of workspace/garden bounds.
- If the request is ambiguous, ask for coordinates in `rationale` rather
  than inventing a zone centre.
"""


class ContextBuilder:
    """Build prompts and LangChain message lists."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        world: Any = None,
    ) -> None:
        self._registry = tool_registry
        self._world = world

    def chat_system_prompt(self) -> str:
        parts = [_CHAT_HEADER]
        parts.append(self._render_tool_section())
        parts.append(_format_pin_context())
        parts.append(_CHAT_FOOTER)
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
                out.append(HumanMessage(content=compact_input_text(content)))
            elif role == "assistant":
                kwargs: dict[str, Any] = {}
                if include_reasoning:
                    thinking = msg.get("thinking")
                    if thinking:
                        kwargs["additional_kwargs"] = {
                            "reasoning_content": str(thinking)
                        }
                out.append(AIMessage(content=compact_input_text(content), **kwargs))
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
