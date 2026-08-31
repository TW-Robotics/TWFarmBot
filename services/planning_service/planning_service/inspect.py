"""Compact vision scorecards for zone inspection. No hardware I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from twfarmbot_ml_utils import parse_segmentation_labels

from .introspection import _get_image_processor, _parse_class_list

log = logging.getLogger(__name__)


def score_image_urls(
    image_urls: Sequence[str],
    classes: str = "plant, weed, soil",
) -> list[dict[str, Any]]:
    """Segment each image and keep numeric class scores, not overlay pixels."""
    processor = _get_image_processor()
    frames: list[dict[str, Any]] = []
    for image_url in image_urls:
        if not image_url:
            continue
        try:
            result = processor.predict(
                image_url, api_name="/run_seg", classes=classes, negative=""
            )
            if not isinstance(result, tuple) or len(result) < 4:
                frames.append(
                    {"source_url": image_url, "error": "unexpected segmentation result"}
                )
                continue
            labels = [str(result[2]), str(result[3])]
            class_scores = parse_segmentation_labels(labels)
            dominant = (
                max(class_scores.items(), key=lambda item: item[1])[0]
                if class_scores
                else None
            )
            frames.append(
                {
                    "source_url": image_url,
                    "class_scores": class_scores,
                    "dominant_class": dominant,
                    "detected_classes": _parse_class_list(str(result[2])),
                }
            )
            for path in result[:2]:
                try:
                    Path(path).unlink(missing_ok=True)
                except (TypeError, ValueError, OSError):
                    pass
        except Exception as err:  # noqa: BLE001
            log.warning("inspect score failed for %s: %s", image_url, err)
            frames.append({"source_url": image_url, "error": f"{type(err).__name__}: {err}"})
    return frames


def summarize_frames(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average class scores across frames into one inspect summary."""
    buckets: dict[str, list[float]] = {}
    for frame in frames:
        scores = frame.get("class_scores") or {}
        if not isinstance(scores, dict):
            continue
        for name, value in scores.items():
            try:
                buckets.setdefault(str(name), []).append(float(value))
            except (TypeError, ValueError):
                continue
    means = {
        name: round(sum(values) / len(values), 3)
        for name, values in buckets.items()
        if values
    }
    dominant = max(means, key=means.get) if means else None
    return {
        "mean_scores": means,
        "dominant_class": dominant,
        "frame_count": len(frames),
        "error_count": sum(1 for frame in frames if frame.get("error")),
    }


def summary_text(zone_name: str, summary: dict[str, Any]) -> str:
    """One-line inspect result for history / chat."""
    dominant = summary.get("dominant_class") or "unknown"
    means = summary.get("mean_scores") or {}
    pct = means.get(dominant)
    share = f" ({round(float(pct) * 100)}%)" if isinstance(pct, (int, float)) else ""
    return (
        f"{zone_name}: {dominant}{share} over "
        f"{summary.get('frame_count', 0)} frames"
    )
