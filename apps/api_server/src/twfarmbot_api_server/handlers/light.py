"""Handler for Action(kind='set_light')."""

from __future__ import annotations

from twfarmbot_core.domain import Action

from watering_service import lighting


def handle_set_light(action: Action) -> Action:
    result = lighting.set_light(bool(action.params.get("on")))
    return Action(kind="set_light", params={**action.params, **result})
