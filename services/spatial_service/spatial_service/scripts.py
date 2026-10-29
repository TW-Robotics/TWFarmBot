"""Named waypoint scripts (tool pickup, etc.) stored in YAML."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from twfarmbot_core.config import load_yaml_config, yaml_config_path

_AXES = ("x", "y", "z")


def _yaml_num(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _waypoint(raw: Mapping[str, Any]) -> dict[str, float]:
    missing = [axis for axis in _AXES if raw.get(axis) is None]
    if missing:
        raise ValueError(f"waypoint needs {', '.join(missing)}")
    return {axis: float(raw[axis]) for axis in _AXES}


def normalize_script(item: Mapping[str, Any]) -> dict[str, Any]:
    label = str(item.get("label") or "").strip()
    if not label:
        raise ValueError("script needs a label")
    waypoints = [_waypoint(row) for row in item.get("waypoints") or [] if isinstance(row, Mapping)]
    return {"label": label, "waypoints": waypoints}


def normalize_scripts(raw: Sequence[Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw or []:
        if not isinstance(row, Mapping):
            raise ValueError("each script must be a mapping")
        item = normalize_script(row)
        key = item["label"].casefold()
        if key in seen:
            raise ValueError(f"duplicate script label {item['label']!r}")
        seen.add(key)
        items.append(item)
    return items


def list_scripts(path: str | Path | None = None) -> list[dict[str, Any]]:
    return normalize_scripts(load_yaml_config(path).get("scripts"))


def resolve_script(name: str, path: str | Path | None = None) -> dict[str, Any]:
    query = (name or "").strip().casefold()
    if not query:
        raise ValueError("name is empty")
    items = list_scripts(path)
    for item in items:
        if item["label"].casefold() == query:
            return item
    partial = [
        item
        for item in items
        if query in item["label"].casefold() or item["label"].casefold() in query
    ]
    if len(partial) == 1:
        return partial[0]
    if not items:
        raise ValueError("no scripts configured — add one on the Scripts page")
    labels = ", ".join(item["label"] for item in items)
    raise ValueError(f"no script named {name!r}. Try: {labels}")


def save_scripts(
    items: Sequence[Mapping[str, Any]],
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_scripts(items)
    config_path = yaml_config_path(path)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    if config_path.exists():
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.load(fh)
    else:
        data = None
    if data is None:
        data = CommentedMap()
    dumped: list[CommentedMap] = []
    for item in normalized:
        row = CommentedMap()
        row["label"] = DoubleQuotedScalarString(item["label"])
        wps: list[CommentedMap] = []
        for wp in item["waypoints"]:
            point = CommentedMap()
            for axis in _AXES:
                point[axis] = _yaml_num(wp[axis])
            point.fa.set_flow_style()
            wps.append(point)
        row["waypoints"] = wps
        dumped.append(row)
    data["scripts"] = dumped
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return normalized
