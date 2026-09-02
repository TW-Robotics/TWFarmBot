"""JSON output schemas for programmatic (read-only) tools.

These tell the hosted PTC runtime which fields generated JavaScript can use
after a tool returns. Keep them small and stable; extra keys in a live
payload are fine.
"""

from __future__ import annotations

from typing import Any

_STR = {"type": ["string", "null"]}
_NUM = {"type": ["number", "null"]}
_INT = {"type": ["integer", "null"]}
_ERR = {"error": _STR}


def _obj(properties: dict[str, Any], extra: Any = False) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": extra,
    }


_POINT = _obj({"x": _NUM, "y": _NUM, "z": _NUM})
_CENTER = {"type": "array", "items": {"type": "number"}}
_ZONE = _obj(
    {
        "name": {"type": "string"},
        "id": _STR,
        "kind": _STR,
        "x": _NUM,
        "y": _NUM,
        "width": _NUM,
        "height": _NUM,
        "center": _CENTER,
    }
)
_WAYPOINT = _obj(
    {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}
)

TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_zones": _obj(
        {"zones": {"type": "array", "items": _ZONE}, "count": _INT, **_ERR}
    ),
    "get_position": _obj({"x": _NUM, "y": _NUM, "z": _NUM, "xyz": _POINT, **_ERR}),
    "get_health": _obj(
        {
            "status": _STR,
            "farmbot": _STR,
            "actions": {"type": ["array", "null"], "items": {"type": "string"}},
            **_ERR,
        }
    ),
    "get_images": _obj(
        {
            "images": {
                "type": "array",
                "items": _obj(
                    {
                        "id": {"type": ["string", "number", "null"]},
                        "attachment_url": _STR,
                        "created_at": _STR,
                    }
                ),
            },
            **_ERR,
        }
    ),
    "plan_path": _obj(
        {"waypoints": {"type": "array", "items": _WAYPOINT}, "count": _INT, **_ERR}
    ),
    "scan_zone": _obj(
        {
            "waypoints": {"type": "array", "items": _WAYPOINT},
            "count": _INT,
            "zone_id": _STR,
            **_ERR,
        }
    ),
    "get_pins": _obj(
        {
            "pins": {
                "type": "array",
                "items": _obj(
                    {
                        "label": _STR,
                        "pin": _INT,
                        "mode": _STR,
                        "kind": _STR,
                        "group": _STR,
                    }
                ),
            },
            **_ERR,
        }
    ),
    "get_positions": _obj(
        {
            "positions": {
                "type": "array",
                "items": _obj({"label": _STR, "x": _NUM, "y": _NUM, "z": _NUM}),
            },
            **_ERR,
        }
    ),
    "list_endpoints": _obj(
        {
            "endpoints": {
                "type": "array",
                "items": _obj({"method": _STR, "path": _STR, "summary": _STR}),
            },
            "count": _INT,
            **_ERR,
        }
    ),
    "segment_image": _obj(
        {
            "class_scores": _obj({}, extra={"type": "number"}),
            "dominant_class": _STR,
            "detected_classes": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "source_url": _STR,
            **_ERR,
        }
    ),
    "analyze_image": _obj({"prompt": _STR, "source_url": _STR, **_ERR}),
    "estimate_traversability": _obj({"prompt": _STR, "source_url": _STR, **_ERR}),
}
