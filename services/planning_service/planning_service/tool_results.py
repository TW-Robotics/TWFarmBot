"""Helpers for tool-call payloads sent back to the model or the user."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

_IMAGE_DATA_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")
_MAX_PROVIDER_IMAGE_BYTES = 800_000
_MAX_PROVIDER_IMAGE_SIDE = 1280

_CAPTURE_BAND_LABELS = {
    "rgb": "RGB capture",
    "nir": "NIR capture",
    "rededge": "Red-edge capture",
}


def _capture_attachment_url(artifact_id: str, band: str) -> str:
    return f"/captures/{artifact_id}/{band}"


def _ndre_preview_from_sample(sample: dict[str, Any]) -> str | None:
    """Resolve a user-loadable NDRE preview URL for one scan sample."""
    preview = sample.get("ndre_preview")
    if isinstance(preview, str) and preview.startswith("/captures/"):
        return preview
    nir = sample.get("nir")
    if isinstance(nir, dict) and nir.get("artifact_id"):
        return _capture_attachment_url(str(nir["artifact_id"]), "ndre")
    return None


def _sse_images(name: str, result: Any) -> list[dict[str, str]]:
    """Compact ``{label, url}`` list for chat SSE events."""
    return [
        {"label": label, "url": url}
        for label, url in vision_artifacts([{"name": name, "result": result}])
        if isinstance(url, str)
        and (url.startswith("/") or url.startswith("data:image/"))
    ]


def compact_tool_result(value: Any) -> Any:
    """Drop inline base64; keep /captures/ paths the UI can load."""
    if isinstance(value, list):
        return [compact_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_tool_result(item) for key, item in value.items()}
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image available to the user]"
    return value


def provider_tool_content(name: str, result: Any) -> str:
    """JSON for the ``role: tool`` message. Stills go in ``provider_vision_message``."""
    del name
    return json.dumps(compact_tool_result(result), default=str)


def provider_vision_message(name: str, result: Any) -> HumanMessage | None:
    """User-turn still so Chat Completions actually sees the capture.

    OpenAI ignores ``image_url`` on ``role: tool``. Put the JPEG on a
    following user message after all tool results for this batch.
    """
    image = _provider_image_url(name, result)
    if not image:
        return None
    return HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"{name} still — inspect this image yourself and answer "
                    "from what you see. Do not tell the user to look at it."
                ),
            },
            {"type": "image_url", "image_url": {"url": image}},
        ]
    )


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
        if len(value) <= _MAX_PROVIDER_IMAGE_BYTES * 2:
            return value
        try:
            raw = base64.b64decode(value.split(",", 1)[1])
        except (IndexError, ValueError):
            return None
        encoded = _encode_still(raw)
        if not encoded:
            return None
        return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"
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
    encoded = _encode_still(data)
    if not encoded:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}"


def _encode_still(data: bytes) -> bytes | None:
    """JPEG small enough for the chat model; downscale instead of dropping."""
    if not data:
        return None
    if data[:2] == b"\xff\xd8" and len(data) <= _MAX_PROVIDER_IMAGE_BYTES:
        return data
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    height, width = img.shape[:2]
    scale = min(1.0, _MAX_PROVIDER_IMAGE_SIDE / max(height, width))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    for quality in (80, 55):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        out = buf.tobytes()
        if len(out) <= _MAX_PROVIDER_IMAGE_BYTES:
            return out
    return None


def compact_input_text(text: str) -> str:
    """Strip persisted base64 images from old browser conversation history."""
    return _IMAGE_DATA_RE.sub("[previous image available to the user]", text)


def vision_artifacts(tool_log: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """User-facing stills from a tool log (paths, not base64)."""
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

    # Prefer NDRE maps when this turn ran capture_ndre / scan_ndre — skip raw
    # NIR / red-edge stills even if the agent also called capture.
    has_ndre = any(call.get("name") in {"capture_ndre", "scan_ndre"} for call in tool_log)

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

    for call in tool_log:
        if call.get("name") != "scan_ndre":
            continue
        result = call.get("result", {})
        if not isinstance(result, dict) or result.get("status") == "error":
            continue
        params = result.get("params", {})
        samples = params.get("samples") if isinstance(params, dict) else None
        if not isinstance(samples, list):
            continue
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            preview = _ndre_preview_from_sample(sample)
            if not preview:
                continue
            axis_pos = sample.get("y", sample.get("x"))
            label = (
                f"NDRE {index + 1} ({axis_pos:g} mm)"
                if isinstance(axis_pos, (int, float))
                else f"NDRE {index + 1}"
            )
            artifacts.append((label, preview))

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
    return list(dict.fromkeys(artifacts))


def append_result_images(text: str, tool_log: list[dict[str, Any]]) -> str:
    """Expose final vision artifacts without dumping intermediate camera history."""
    unique = vision_artifacts(tool_log)
    additions = [f"![{label}]({url})" for label, url in unique if url not in text]
    return text + ("\n\n" + "\n\n".join(additions) if additions else "")
