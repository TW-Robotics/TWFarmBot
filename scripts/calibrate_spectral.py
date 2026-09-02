#!/usr/bin/env python3
"""Fast NIR / red-edge calibration using one ArUco marker on the bed.

Fastest workflow (~20 minutes, RGB ignored):

1. Print an ArUco marker (DICT_4X4_50, id=0, 100 mm square).
   Generate: https://chev.me/arucogen/  (Dictionary 4x4, marker 0, 100 mm)

2. Tape it on the bed.

3. Jog until the marker is centered in NIR, then:
     uv run python scripts/calibrate_spectral.py --band nir --gantry X,Y,Z --capture

4. Jog along the payload axis until the same marker is centered in red-edge, then:
     uv run python scripts/calibrate_spectral.py --band rededge --gantry X,Y,Z --capture

5. Write configs/spectral_calibration.yaml:
     uv run python scripts/calibrate_spectral.py --finalize

Use gantry coordinates from GET /position or the Move page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision_service.capture import capture  # noqa: E402
from vision_service.spectral_calib import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_calibration,
    observe_band,
    save_calibration,
)

SESSION_PATH = ROOT / "data" / "spectral_calib_session.json"


def _parse_gantry(raw: str) -> dict[str, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {2, 3}:
        raise SystemExit("gantry must be X,Y or X,Y,Z in millimetres")
    try:
        x, y = float(parts[0]), float(parts[1])
        z = float(parts[2]) if len(parts) == 3 else 0.0
    except ValueError as err:
        raise SystemExit(f"invalid gantry numbers: {raw}") from err
    return {"x": x, "y": y, "z": z}


def _load_session() -> dict:
    if not SESSION_PATH.is_file():
        return {"observations": []}
    data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"observations": []}
    data.setdefault("observations", [])
    return data


def _save_session(data: dict) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", choices=["nir", "rededge"])
    parser.add_argument("--gantry", help="gantry X,Y[,Z] in mm")
    parser.add_argument("--image", help="existing capture JPEG")
    parser.add_argument("--capture", action="store_true", help="grab a fresh still first")
    parser.add_argument("--finalize", action="store_true", help="write calibration yaml")
    parser.add_argument("--status", action="store_true", help="show session progress")
    parser.add_argument("--output", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--reference-band", default="nir")
    parser.add_argument("--marker-size-mm", type=float, default=100.0)
    parser.add_argument("--marker-id", type=int, default=0)
    args = parser.parse_args()

    if args.status:
        session = _load_session()
        for item in session.get("observations") or []:
            print(
                f"{item['band']}: gantry={item['gantry_mm']} "
                f"gsd={item['gsd_mm_per_px']} image={item['image_path']}"
            )
        return

    if args.finalize:
        session = _load_session()
        doc = build_calibration(
            session.get("observations") or [],
            reference_band=args.reference_band,
            marker_size_mm=args.marker_size_mm,
        )
        out = save_calibration(doc, args.output)
        sep = doc["computed"]["band_separation_mm"]
        print(f"wrote {out}")
        print(
            "band separation mm (rededge vs nir): "
            f"x={sep['x']} y={sep['y']} z={sep.get('z', 0)}"
        )
        print(f"shared gsd_mm_per_px: {doc['shared']['gsd_mm_per_px']}")
        return

    if not args.band or not args.gantry:
        parser.error("provide --band and --gantry, or use --finalize / --status")

    gantry = _parse_gantry(args.gantry)
    image_path = Path(args.image) if args.image else None
    if args.capture:
        artifact_id = capture(args.band)
        image_path = ROOT / "data" / "captures" / f"{artifact_id}-{args.band}.jpg"
        print(f"captured {args.band} artifact_id={artifact_id}")
    if image_path is None:
        parser.error("provide --image or --capture")

    observation = observe_band(
        args.band,
        image_path,
        gantry,
        marker_size_mm=args.marker_size_mm,
        marker_id=args.marker_id,
    )
    session = _load_session()
    session["observations"] = [
        item for item in session["observations"] if item.get("band") != args.band
    ]
    session["observations"].append(observation)
    _save_session(session)

    print(f"stored {args.band}")
    print(f"  gsd_mm_per_px: {observation['gsd_mm_per_px']}")
    bands = {item["band"] for item in session["observations"]}
    if bands >= {"nir", "rededge"}:
        print("run: uv run python scripts/calibrate_spectral.py --finalize")
    else:
        missing = sorted({"nir", "rededge"} - bands)
        print(f"still need: {', '.join(missing)}")


if __name__ == "__main__":
    main()
