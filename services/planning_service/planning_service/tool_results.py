"""Helpers for tool-call payloads sent back to the model or the user."""

from __future__ import annotations

import re
from typing import Any

_IMAGE_DATA_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")


def compact_tool_result(value: Any) -> Any:
    """Avoid sending binary/base64 vision output back into the next LLM turn."""
    if isinstance(value, list):
        return [compact_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[image available to the user]"
                if key in {"image_url", "image_urls"} and isinstance(item, (str, list))
                else compact_tool_result(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image available to the user]"
    return value


def compact_input_text(text: str) -> str:
    """Strip persisted base64 images from old browser conversation history."""
    return _IMAGE_DATA_RE.sub("[previous image available to the user]", text)


def append_result_images(text: str, tool_log: list[dict[str, Any]]) -> str:
    """Expose final vision artifacts without dumping intermediate camera history."""
    artifacts: list[tuple[str, str]] = []
    labels = {
        "analyze_image": ("Similarity map",),
        "segment_image": ("Segmentation overlay", "Segmentation mask"),
        "visualize_image_features": (
            "PCA feature map",
            "PCA component map",
            "PCA cluster map",
        ),
        "estimate_traversability": ("Traversability map",),
    }
    for call in tool_log:
        name = str(call.get("name", ""))
        if name not in labels:
            continue
        result = call.get("result", {})
        if not isinstance(result, dict):
            continue
        images: list[str] = []
        value = result.get("image_url")
        if isinstance(value, str):
            images.append(value)
        values = result.get("image_urls")
        if isinstance(values, list):
            images.extend(value for value in values if isinstance(value, str))
        for index, url in enumerate(images):
            label_set = labels[name]
            label = label_set[index] if index < len(label_set) else "Analysis result"
            artifacts.append((label, url))

    # A plain photo request is normally verified with get_images. Show only the
    # newest frame from the final lookup, not every historical frame fetched by
    # intermediate agent turns.
    if not artifacts:
        for call in reversed(tool_log):
            if call.get("name") != "get_images":
                continue
            result = call.get("result", {})
            images = result.get("images", []) if isinstance(result, dict) else []
            if images and isinstance(images[0], dict):
                attachment = images[0].get("attachment_url")
                if isinstance(attachment, str):
                    artifacts.append(("FarmBot photo", attachment))
            break

    unique = dict.fromkeys(artifacts)
    additions = [f"![{label}]({url})" for label, url in unique if url not in text]
    return text + ("\n\n" + "\n\n".join(additions) if additions else "")
