"""Camera action handlers."""

from __future__ import annotations

from twfarmbot_core.domain import Action
from vision_service import capture
from watering_service.backends import farmbot


def handle_take_photo(action: Action) -> Action:
    farmbot.backend.take_photo()
    wait_for_new_photo = getattr(farmbot.backend, "wait_for_new_photo", None)
    if callable(wait_for_new_photo):
        wait_for_new_photo()
    return action


def handle_capture(action: Action) -> Action:
    artifact_id = capture(str(action.params["band"]))
    return Action(
        kind=action.kind,
        params={**action.params, "artifact_id": artifact_id},
    )
