"""Closed-loop autonomous crop care engine.

Implements the core **See → Reason → Act → Reason (Verify)** cycle:
1. **See**: Observe the target area using perception tools (e.g. ``capture_ndre``,
   ``capture``, or sensor ``read_pin``).
2. **Reason**: Evaluate agronomic data (NDRE vigor, stress fraction, advice).
3. **Act**: Execute physical interventions (e.g. ``water``) safely if needed.
4. **Reason (Verify)**: Evaluate the outcome of actions and synthesize findings.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.models import Model
from twfarmbot_core.actions import ActionRegistry

from .chat import ChatResult, chat
from .config import PlannerConfig
from .introspection import SystemStateProvider

log = logging.getLogger(__name__)

PERCEPTION_TOOLS = frozenset(
    {
        "capture",
        "capture_ndre",
        "scan_ndre",
        "take_photo",
        "read_pin",
        "get_position",
        "get_status",
        "get_images",
        "get_messages",
    }
)

ACTION_TOOLS = frozenset(
    {
        "water",
        "set_light",
        "move",
        "move_axis",
        "move_path",
        "write_pin",
        "e_stop",
        "unlock",
    }
)

MEMORY_TOOLS = frozenset(
    {
        "remember",
        "recall_memories",
        "forget_memory",
    }
)

DEFAULT_AUTONOMOUS_PROMPT = (
    "Execute an autonomous closed-loop crop care cycle:\n\n"
    "1. [RECALL]: Check past memories for this location using `recall_memories(near_x=..., near_y=...)` to ground your assessment in prior treatments or observations.\n"
    "2. [SEE]: Observe the crop at the current location using perception tools:\n"
    "   - Call `capture_ndre` to assess canopy vigor and water stress.\n"
    "   - Call `read_pin` (e.g. pin 14 for soil moisture) if relevant.\n"
    "   - Call `get_meteorology` if you need current sun altitude/weather context.\n"
    "3. [REASON]: Analyze perception results in context of past memories:\n"
    "   - Check NDRE metrics (mean, vegetation_fraction, stress_fraction).\n"
    "   - Follow `interpretation.action_hint` and evaluate whether canopy has improved, maintained, or degraded since past visits.\n"
    "4. [ACT]: If and only if the data warrants intervention:\n"
    "   - If stressed (e.g. action_hint is 'consider_water'), call `water` with an appropriate duration.\n"
    "   - If healthy (action_hint is 'ok' or 'monitor') or environmental conditions advise delay (midday heat/glare), DO NOT water unnecessarily.\n"
    "5. [REASON / REMEMBER]: Conclude with a clear synthesis:\n"
    "   - State what you observed (NDRE, vigor, stress level).\n"
    "   - State your reasoning and what action was taken (or why no action was needed).\n"
    "   - ACTIVELY REASON whether this outcome is noteworthy enough to store in long-term memory. If significant (new stress detected, water delivered, crop recovered), call `remember(...)` to save the takeaway for future cycles."
)


@dataclass(frozen=True)
class AutonomousCycleResult:
    """Outcome of a single See → Reason → Act → Reason autonomous cycle."""

    goal: str
    status: str  # "ok" | "error"
    perceptions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    memories: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    session_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "perceptions": self.perceptions,
            "actions": self.actions,
            "observations": self.observations,
            "memories": self.memories,
            "summary": self.summary,
            "thinking": self.thinking,
            "tool_calls": self.tool_calls,
            "metrics": self.metrics,
            "session_id": self.session_id,
            "session_path": self.session_path,
        }


def _session_data_dir() -> Path:
    path = Path(os.getenv("TWFB_UI_DATA_DIR", Path.cwd() / "data" / "ui_sessions"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_autonomous_session(
    goal: str,
    chat_result: ChatResult,
    perceptions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    *,
    memories: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    suffix = secrets.token_hex(3)
    session_id = f"{stamp}-autonomous-{suffix}"
    now_iso = datetime.now(timezone.utc).isoformat()

    summary = error or chat_result.response or "Autonomous cycle completed"

    snapshot = {
        "session_id": session_id,
        "label": f"Autonomy · {stamp}",
        "kind": "autonomous",
        "created_at": now_iso,
        "updated_at": now_iso,
        "goal": goal,
        "assistant_messages": [
            {"role": "user", "content": goal},
            {"role": "assistant", "content": summary},
        ],
        "autonomous_cycle": {
            "status": "error" if error else "ok",
            "perceptions": perceptions,
            "actions": actions,
            "memories": memories or [],
            "summary": summary,
            "thinking": chat_result.thinking,
            "tool_calls": chat_result.tool_calls,
            "metrics": chat_result.metrics,
            "error": error,
        },
    }
    path = _session_data_dir() / f"{session_id}.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return session_id, path


def run_autonomous_cycle(
    *,
    registry: ActionRegistry,
    goal: str | None = None,
    world: Any = None,
    system_state: SystemStateProvider | None = None,
    model: Model | None = None,
    config: PlannerConfig | None = None,
    model_name: str | None = None,
    save_session: bool = True,
    ensure_homed: bool = True,
) -> AutonomousCycleResult:
    """Execute a See → Reason → Act → Reason autonomous crop care cycle."""
    target_goal = goal or DEFAULT_AUTONOMOUS_PROMPT
    log.info("Starting autonomous cycle: %s", target_goal[:80])

    if ensure_homed:
        try:
            from farmbot_gateway import farmbot
            from twfarmbot_core.domain import Action

            if not farmbot.backend.is_homed():
                log.warning(
                    "Homing Guard: Gantry is unhomed after restart. "
                    "Calibrating to physical endstops (0,0,0) before autonomous cycle..."
                )
                if "find_home" in registry.kinds():
                    registry.dispatch(Action(kind="find_home", params={"axis": "all"}))
                else:
                    farmbot.backend.ensure_homed()
        except Exception as err:  # noqa: BLE001
            log.warning("Homing Guard check/execution note: %s", err)

    try:
        chat_result = chat(
            [{"role": "user", "content": target_goal}],
            registry=registry,
            world=world,
            system_state=system_state,
            model=model,
            config=config,
            allow_actions=True,
            skip_approval=True,
            model_name=model_name,
            core_tools=frozenset({"capture_ndre", "find_home"}),  # the cycle's fixed workflow
        )
    except Exception as err:
        log.exception("Autonomous cycle failed")
        session_id, path = (None, None)
        if save_session:
            session_id, path = _save_autonomous_session(
                target_goal,
                ChatResult(response="", tool_calls=[]),
                [],
                [],
                error=f"{type(err).__name__}: {err}",
            )
        return AutonomousCycleResult(
            goal=target_goal,
            status="error",
            summary=f"Failed: {err}",
            session_id=session_id,
            session_path=str(path) if path else None,
        )

    perceptions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []

    action_seen = False
    for tc in chat_result.tool_calls:
        name = str(tc.get("name"))
        if name in MEMORY_TOOLS:
            memories.append(tc)
        elif name in ACTION_TOOLS:
            actions.append(tc)
            action_seen = True
        elif name in PERCEPTION_TOOLS:
            if not action_seen:
                perceptions.append(tc)
            else:
                observations.append(tc)
        else:
            (actions if action_seen else perceptions).append(tc)

    session_id, path = (None, None)
    if save_session:
        session_id, path = _save_autonomous_session(
            target_goal,
            chat_result,
            perceptions,
            actions,
            memories=memories,
        )

    return AutonomousCycleResult(
        goal=target_goal,
        status="ok",
        perceptions=perceptions,
        actions=actions,
        observations=observations,
        memories=memories,
        summary=chat_result.response,
        thinking=chat_result.thinking,
        tool_calls=chat_result.tool_calls,
        metrics=chat_result.metrics,
        session_id=session_id,
        session_path=str(path) if path else None,
    )
