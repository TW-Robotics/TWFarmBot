"""Local UI helpers that stay on the dashboard host (YAML writes, vision)."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

_CONFIG_PATH = Path(os.getenv("TWFB_CONFIG", "configs/dev.yaml"))
_NUMBER_RE = re.compile(r"[^a-z0-9_]+")


def _entity_id(name: str) -> str:
    base = _NUMBER_RE.sub("_", name.lower()).strip("_")
    return base or "entity"


def add_garden_entity(x: float, y: float, kind: str, name: str) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    path = Path(_CONFIG_PATH)
    with path.open(encoding="utf-8") as fh:
        data = yaml.load(fh)
    spatial = data.setdefault("spatial", {})
    entities = spatial.setdefault("entities", [])
    entities.append(
        {
            "id": _entity_id(name),
            "kind": kind,
            "name": name,
            "x": float(x),
            "y": float(y),
            "z": 0.0,
            "radius_mm": 50,
            "metadata": {},
        }
    )
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh)


def path_to_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def run_vision(payload: dict[str, Any]) -> dict[str, Any]:
    from twfarmbot_ml_utils import VisionProcessor, parse_segmentation_labels

    resireg = os.getenv("TWFB_RESIREG_URL", "http://127.0.0.1:8080")
    processor = VisionProcessor(resireg)
    mode = payload.get("mode") or ""
    image_url = payload["image_url"]
    images: list[dict[str, str]] = []
    extra: dict[str, Any] = {}
    if mode == "Open Language Similarity":
        result = processor.process(image_url, str(payload.get("prompt") or "").strip())
        images.append({"url": path_to_data_url(result), "caption": "Similarity map"})
    elif mode == "Zero-Shot Segmentation":
        raw = processor.predict(
            image_url,
            api_name="/run_seg",
            classes=str(payload.get("classes") or "").strip(),
            negative=str(payload.get("negative") or "").strip(),
        )
        labels = [str(raw[2]), str(raw[3])]
        images.extend(
            [
                {"url": path_to_data_url(raw[0]), "caption": "Segmentation overlay"},
                {"url": path_to_data_url(raw[1]), "caption": "Segmentation map"},
            ]
        )
        extra["labels"] = labels
        extra["class_scores"] = parse_segmentation_labels(labels)
    elif mode == "PCA Feature Visualization":
        raw = processor.predict(
            image_url,
            api_name="/run_pca",
            n_clusters=int(payload.get("n_clusters") or 6),
        )
        for idx, item in enumerate(raw[:3]):
            images.append(
                {
                    "url": path_to_data_url(item),
                    "caption": f"PCA visualization {idx + 1}",
                }
            )
    elif mode == "Traversability Estimation":
        result = processor.predict(
            image_url,
            api_name="/run_trav",
            prompt=str(payload.get("prompt") or "").strip(),
            negatives=str(payload.get("negatives") or "").strip(),
        )
        images.append(
            {"url": path_to_data_url(result), "caption": "Traversability map"}
        )
    else:
        raise ValueError(f"Unknown vision mode: {mode}")
    return {"images": images, **extra}
