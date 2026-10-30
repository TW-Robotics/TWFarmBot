"""Run-scoped goal checklist, re-read by the model every turn."""

from __future__ import annotations

from typing import Any

GOAL_MARKER = "[goal]"


def apply_goal_update(current: dict[str, Any] | None, args: dict[str, Any]) -> dict[str, Any]:
    goal = {
        "objective": str((current or {}).get("objective") or ""),
        "remaining": [str(s) for s in ((current or {}).get("remaining") or [])],
        "done": [str(s) for s in ((current or {}).get("done") or [])],
    }
    objective = args.get("objective")
    if objective:
        goal["objective"] = str(objective).strip()
    if args.get("remaining") is not None:
        goal["remaining"] = [
            str(step).strip() for step in args["remaining"] if str(step).strip()
        ]
    completed = args.get("completed")
    if completed:
        step = str(completed).strip()
        goal["remaining"] = [item for item in goal["remaining"] if item != step]
        if step and step not in goal["done"]:
            goal["done"].append(step)
    return goal


def format_goal(goal: dict[str, Any] | None) -> str:
    if not goal or not (goal.get("objective") or goal.get("remaining") or goal.get("done")):
        return ""
    lines = [GOAL_MARKER, "Current goal (re-read every turn; update with set_goal):"]
    if goal.get("objective"):
        lines.append(f"- Objective: {goal['objective']}")
    remaining = goal.get("remaining") or []
    done = goal.get("done") or []
    if remaining:
        lines.append("- Remaining: " + "; ".join(remaining))
    if done:
        lines.append("- Done: " + "; ".join(done))
    return "\n".join(lines)
