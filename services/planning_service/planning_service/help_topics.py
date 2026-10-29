"""Reference notes the assistant loads on demand via the ``get_help`` tool.

Kept out of the system prompt so every turn stays small (pi-style: docs are read
when the task needs them).
"""

from __future__ import annotations

HELP: dict[str, str] = {
    "spectral": """Spectral cameras (this robot — do not confuse with RGB or NDVI):
- NIR (near-infrared, ~800–900 nm): a dedicated grayscale USB camera (`capture` band=nir).
  Healthy leaves reflect strongly in NIR because of internal leaf structure. NIR is
  not thermal, not night-vision, and not an RGB channel.
- Red-edge (~700–750 nm): a second grayscale USB camera (`capture` band=rededge).
  It sits on the chlorophyll absorption → NIR reflection transition, so it is
  sensitive to chlorophyll / early stress. It is not the RGB red channel and
  not “just red”.
- The two cameras are not co-located. `capture_ndre` shoots NIR, moves the
  calibrated gantry offset, shoots red-edge, then warps red-edge onto NIR.
- NDRE = (NIR − red-edge) / (NIR + red-edge), per pixel, typically −1…1.
  This farm uses NDRE, not NDVI. NDVI uses red; NDRE uses red-edge and holds
  up better on dense canopy. Never invent NDVI or treat a single NIR/red-edge
  still as a colour photo.
- How to read this farm’s NDRE metrics (trust `interpretation.*` over vibes):
  mean ≥ 0.25 and vegetation_fraction ≥ 0.4 → healthy, do not water from NDRE
  alone; mean ≥ 0.15 → moderate, water only with other evidence; high
  stress_fraction with mean < 0.15 → likely stressed; mean < 0.05 and little
  vegetation → soil/empty/misaimed, do not water.
- After `capture_ndre` or `scan_ndre`, the user-facing reply is ONE short
  sentence paraphrasing `interpretation.advice` (e.g. "Canopy looks healthy —
  I wouldn't water from this."). Do not list mean, fractions, or the formula.
  The NDRE map is already on screen. For a scan, summarise the whole bed in
  that one sentence, not each stop.
- Do not run `segment_image` on NIR or red-edge grayscale. For plant ID use RGB.""",
}


def get_help(topic: str | None = None) -> str:
    """Return one topic, or the topic list when ``topic`` is empty/unknown."""
    key = (topic or "").strip().lower()
    if key in HELP:
        return HELP[key].strip()
    return "Topics: " + ", ".join(sorted(HELP))
