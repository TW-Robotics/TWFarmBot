"""Helpers for tool-call payloads sent back to the model or the user."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

_IMAGE_DATA_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")
_MAX_PROVIDER_IMAGE_BYTES = 1_200_000

_CAPTURE_BAND_LABELS = {
    "rgb": "RGB capture",
    "nir": "NIR capture",
    "rededge": "Red-edge capture",
}


def _capture_attachment_url(artifact_id: str, band: str) -> str:
    return f"/captures/{artifact_id}/{band}"


def compact_tool_result(value: Any) -> Any:
    """Avoid sending binary/base64 vision output back into the next LLM turn."""
    if isinstance(value, list):
        return [compact_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[image available to the user]"
                if key
                in {
                    "image_url",
                    "image_urls",
                    "ndre_preview",
                    "align_preview",
                }
                and isinstance(item, (str, list))
                else compact_tool_result(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image available to the user]"
    return value


def provider_tool_content(name: str, result: Any) -> Any:
    """JSON for the tool message, plus one still when the provider can see it."""
    text = json.dumps(compact_tool_result(result), default=str)
    image = _provider_image_url(name, result)
    if not image:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image}},
    ]


def _provider_image_url(name: str, result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    params = result.get("params")
    params = params if isinstance(params, dict) else {}
    if name == "capture_ndre":
        preview = params.get("ndre_preview") or result.get("ndre_preview")
        loaded = _load_image_ref(preview)
        if loaded:
            return loaded
    artifact_id = params.get("artifact_id")
    band = params.get("band")
    if artifact_id and band:
        loaded = _capture_file_uri(str(artifact_id), str(band))
        if loaded:
            return loaded
    for key in ("image_url", "ndre_preview"):
        loaded = _load_image_ref(result.get(key))
        if loaded:
            return loaded
    images = result.get("images")
    if isinstance(images, list) and images and isinstance(images[0], dict):
        return _load_image_ref(images[0].get("attachment_url"))
    if name == "take_photo":
        return _latest_capture_uri()
    return None


def _capture_file_uri(artifact_id: str, band: str) -> str | None:
    try:
        from vision_service import capture_path
    except Exception:  # noqa: BLE001
        return None
    path = capture_path(artifact_id, band)
    return _path_to_data_uri(path) if path is not None else None


def _latest_capture_uri() -> str | None:
    try:
        from vision_service import list_captures
    except Exception:  # noqa: BLE001
        return None
    rows = list_captures()
    if not rows:
        return None
    raw = rows[0].get("path")
    return _path_to_data_uri(Path(raw)) if isinstance(raw, str) else None


def _load_image_ref(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:image/"):
        return value if len(value) <= _MAX_PROVIDER_IMAGE_BYTES * 2 else None
    match = re.fullmatch(r"/captures/([^/]+)/([^/]+)", value)
    if match:
        return _capture_file_uri(match.group(1), match.group(2))
    if value.startswith("/") and Path(value).is_file():
        return _path_to_data_uri(Path(value))
    return None


def _path_to_data_uri(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or len(data) > _MAX_PROVIDER_IMAGE_BYTES:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


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

    # Prefer the NDRE map alone when this turn ran capture_ndre — skip raw
    # NIR / red-edge stills even if the agent also called capture.
    has_ndre = any(call.get("name") == "capture_ndre" for call in tool_log)

    for call in tool_log:
        if call.get("name") != "capture":
            continue
        result = call.get("result", {})
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        params = result.get("params", {})
        if not isinstance(params, dict):
            continue
        artifact_id = params.get("artifact_id")
        band = params.get("band")
        if not artifact_id or not band:
            continue
        band_key = str(band).lower()
        if has_ndre and band_key in {"nir", "rededge"}:
            continue
        label = _CAPTURE_BAND_LABELS.get(band_key, f"{band} capture")
        artifacts.append((label, _capture_attachment_url(str(artifact_id), band_key)))

    for call in tool_log:
        if call.get("name") != "capture_ndre":
            continue
        result = call.get("result", {})
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        params = result.get("params", {})
        if not isinstance(params, dict):
            continue
        preview = params.get("ndre_preview")
        if isinstance(preview, str) and preview:
            artifacts.append(("NDRE map", preview))

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
