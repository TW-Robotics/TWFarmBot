"""Named gantry presets shared by the Motion UI and goto_named."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from twfarmbot_core.config import load_yaml_config, yaml_config_path

_AXES = ("x", "y", "z")


def _num(value: Any) -> float:
    return float(value)


def _yaml_num(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def position_is_set(item: Mapping[str, Any]) -> bool:
    return all(item.get(axis) is not None for axis in _AXES)


def normalize_position(item: Mapping[str, Any]) -> dict[str, Any]:
    label = str(item.get("label") or "").strip()
    if not label:
        raise ValueError("position needs a label")
    group = str(item.get("group") or "Travel").strip() or "Travel"
    out: dict[str, Any] = {"label": label, "group": group, "set": False}
    if position_is_set(item):
        out["x"] = _num(item["x"])
        out["y"] = _num(item["y"])
        out["z"] = _num(item["z"])
        out["set"] = True
    return out


def normalize_positions(raw: Sequence[Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in raw or []:
        if not isinstance(row, Mapping):
            raise ValueError("each position must be a mapping")
        item = normalize_position(row)
        key = item["label"].casefold()
        if key in seen:
            raise ValueError(f"duplicate position label {item['label']!r}")
        seen.add(key)
        items.append(item)
    return items


def list_positions(path: str | Path | None = None) -> list[dict[str, Any]]:
    return normalize_positions(load_yaml_config(path).get("positions"))


def save_positions(
    items: Sequence[Mapping[str, Any]],
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_positions(items)
    config_path = yaml_config_path(path)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    data: Any
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
        row["group"] = DoubleQuotedScalarString(item["group"])
        if item["set"]:
            row["x"] = _yaml_num(item["x"])
            row["y"] = _yaml_num(item["y"])
            row["z"] = _yaml_num(item["z"])
        row.fa.set_flow_style()
        dumped.append(row)
    data["positions"] = dumped
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return normalized
