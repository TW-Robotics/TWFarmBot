"""Capture calibrated NIR + red-edge pair and return NDRE metrics."""

from __future__ import annotations

from twfarmbot_core.domain import Action


def handle_capture_ndre(action: Action) -> Action:
    from twfarmbot_api_server.spectral import capture_spectral_pair

    return_to_start = bool(action.params.get("return_to_start", True))
    result = capture_spectral_pair(return_to_start=return_to_start)
    metrics = result.get("metrics") or {}

    # Keep band artifact ids for debugging, but omit attachment URLs so the
    # agent reasons from interpretation + metrics (NDRE map is user-only).
    def _band_ref(band: dict | None) -> dict | None:
        if not isinstance(band, dict):
            return None
        out = {k: band[k] for k in ("artifact_id", "band") if k in band}
        return out or None

    return Action(
        kind="capture_ndre",
        params={
            **action.params,
            "summary": metrics.get("summary"),
            "ndre": metrics.get("ndre"),
            "interpretation": metrics.get("interpretation"),
            "nir": _band_ref(result.get("nir")),
            "rededge": _band_ref(result.get("rededge")),
            "gantry_mm": result.get("gantry_mm"),
            "image_align": metrics.get("image_align"),
            "ndre_preview": metrics.get("ndre_preview"),
            "calibration": result.get("calibration"),
        },
    )
