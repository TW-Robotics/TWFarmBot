"""Handler for Action(kind='end_effector'): one generic call for any configured tool head."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service import end_effectors


def handle_end_effector(action: Action) -> Action:
    result = end_effectors.run(str(action.params.get("name", "")), str(action.params.get("command", "")))
    return Action(kind="end_effector", params={**action.params, **result})
