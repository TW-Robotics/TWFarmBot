"""vision_service — payload USB stills.

Public primitive: ``capture(band) -> artifact_id``. One function, one still.
Does not talk to FarmBot serial or ``farmbot_gateway``.
"""

from .capture import (
    CAPTURE_BANDS,
    UNPINNED_CAPTURE_BANDS,
    USB_CAPTURE_BANDS,
    CaptureError,
    capture,
    capture_path,
    list_captures,
)

__all__ = [
    "CAPTURE_BANDS",
    "USB_CAPTURE_BANDS",
    "UNPINNED_CAPTURE_BANDS",
    "CaptureError",
    "capture",
    "capture_path",
    "list_captures",
]
