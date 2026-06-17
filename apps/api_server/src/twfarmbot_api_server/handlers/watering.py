"""Handler for Action(kind='water', params={'bed_id', 'seconds'})."""

from __future__ import annotations

from twfarmbot_core.domain import Action

import watering_service


def handle_water(action: Action) -> Action:
    bed_id = str(action.params["bed_id"])
    seconds = float(action.params["seconds"])
    watering_service.water_bed(bed_id, seconds)
    return action
