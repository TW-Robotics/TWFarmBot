"""Pose-stamped observation field for the garden bed.

Raw samples are the map. Latest cells are derived by binning millimetre
coordinates (default 250 mm). No occupancy-grid library — sqlite3 + rounding.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

DEFAULT_CELL_MM = 250.0
PREDICATES = frozenset({"plant", "weed", "soil", "healthy", "stressed"})
_RIPE = frozenset({"ripe", "ripeness", "ripe_plants"})
_DEFAULT_DB_REL = Path("data") / "garden.db"
_STRESSED_HINTS = frozenset({"consider_water", "recheck"})

_CLASS_ALIASES = {
    "plant": ("plant", "plants"),
    "weed": ("weed", "weeds"),
    "soil": ("soil",),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_db_path() -> Path:
    env_path = os.getenv("FARMBOT_GARDEN_DB")
    if env_path:
        return Path(env_path)
    try:
        from twfarmbot_core.config import load_yaml_config

        cfg_path = load_yaml_config().get("spatial", {}).get("db_path")
        if cfg_path:
            return Path(str(cfg_path))
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_DB_REL


def xyz_from_pose(data: Any) -> tuple[float, float, float] | None:
    """Extract gantry millimetres from a pose dict.

    Accepts ``{x,y,z}``, ``{xyz: {x,y,z}}``, and the paired spectral shape
    ``{nir: {x,y,z}, rededge: {x,y,z}}`` (stamp the NIR / start pose).
    """
    if not isinstance(data, Mapping):
        return None
    x = _float(data.get("x"))
    y = _float(data.get("y"))
    if x is not None and y is not None:
        z = _float(data.get("z"))
        return x, y, 0.0 if z is None else z
    for key in ("nir", "xyz", "rededge"):
        nested = data.get(key)
        if isinstance(nested, Mapping):
            parsed = xyz_from_pose(nested)
            if parsed is not None:
                return parsed
    return None


def class_score(scores: Mapping[str, Any] | None, kind: str) -> float | None:
    """Read plant/weed/soil from Resi labels (singular or plural)."""
    if not isinstance(scores, Mapping):
        return None
    lowered = {str(key).strip().casefold(): value for key, value in scores.items()}
    for alias in _CLASS_ALIASES.get(kind, (kind,)):
        if alias in lowered:
            return _float(lowered[alias])
    return None


class ObservationStore:
    """SQLite-backed pose-stamped observations."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            self.db_path: str | Path = _default_db_path()
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=15.0
            )
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    source TEXT NOT NULL,
                    ndre_mean REAL,
                    stress_fraction REAL,
                    veg_fraction REAL,
                    plant REAL,
                    weed REAL,
                    soil REAL,
                    dominant TEXT,
                    action_hint TEXT,
                    summary TEXT
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_obs_pos_time "
                "ON observations (x, y, timestamp);"
            )

    def record(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        source: str = "capture_ndre",
        ndre_mean: float | None = None,
        stress_fraction: float | None = None,
        veg_fraction: float | None = None,
        plant: float | None = None,
        weed: float | None = None,
        soil: float | None = None,
        dominant: str | None = None,
        action_hint: str | None = None,
        summary: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        when = timestamp or _utc_now_iso()
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO observations (
                    timestamp, x, y, z, source,
                    ndre_mean, stress_fraction, veg_fraction,
                    plant, weed, soil, dominant, action_hint, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    when,
                    float(x),
                    float(y),
                    float(z),
                    str(source),
                    ndre_mean,
                    stress_fraction,
                    veg_fraction,
                    plant,
                    weed,
                    soil,
                    dominant,
                    action_hint,
                    summary,
                ),
            )
        return {"id": cur.lastrowid, "timestamp": when, "x": x, "y": y, "z": z}

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()
        return int(row["n"] if row else 0)

    def latest_cells(self, cell_mm: float = DEFAULT_CELL_MM) -> list[dict[str, Any]]:
        """Merge samples into one dict per cell; later non-null fields win."""
        step = max(1.0, float(cell_mm))
        rows = self._conn.execute(
            "SELECT * FROM observations ORDER BY timestamp ASC, id ASC"
        ).fetchall()
        cells: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            key = (
                int(round(float(row["x"]) / step)),
                int(round(float(row["y"]) / step)),
            )
            cell = cells.setdefault(key, {})
            cell["x"] = float(row["x"])
            cell["y"] = float(row["y"])
            cell["z"] = float(row["z"] or 0)
            cell["observed_at"] = row["timestamp"]
            cell["source"] = row["source"]
            for field in (
                "ndre_mean",
                "stress_fraction",
                "veg_fraction",
                "plant",
                "weed",
                "soil",
            ):
                value = row[field]
                if value is not None:
                    cell[field] = round(float(value), 4)
            if row["dominant"]:
                cell["dominant"] = row["dominant"]
            if row["action_hint"]:
                cell["action_hint"] = row["action_hint"]
            if row["summary"]:
                cell["summary"] = row["summary"]
        return list(cells.values())

    def find_locations(
        self, predicate: str, cell_mm: float = DEFAULT_CELL_MM
    ) -> dict[str, Any]:
        query = (predicate or "").strip().casefold().replace(" ", "_")
        if query in _RIPE:
            return {
                "predicate": query,
                "cell_mm": float(cell_mm),
                "locations": [],
                "count": 0,
                "error": "ripeness is not a map layer yet",
                "try": ["plant", "healthy", "stressed"],
            }
        if query not in PREDICATES:
            raise ValueError(
                f"unknown predicate {predicate!r}; "
                "use plant, weed, soil, healthy, or stressed"
            )
        ranked: list[dict[str, Any]] = []
        for cell in self.latest_cells(cell_mm=cell_mm):
            score = _predicate_score(cell, query)
            if score is None:
                continue
            ranked.append(
                {
                    "x": cell["x"],
                    "y": cell["y"],
                    "z": cell.get("z", 0),
                    "score": round(score, 4),
                    "observed_at": cell.get("observed_at"),
                    "action_hint": cell.get("action_hint"),
                    "dominant": cell.get("dominant"),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["x"], item["y"]))
        return {
            "predicate": query,
            "cell_mm": float(cell_mm),
            "locations": ranked[:20],
            "count": min(len(ranked), 20),
        }

    def close(self) -> None:
        self._conn.close()


def _predicate_score(cell: Mapping[str, Any], predicate: str) -> float | None:
    if predicate in {"plant", "weed", "soil"}:
        return _float(cell.get(predicate))
    hint = str(cell.get("action_hint") or "").strip().casefold()
    ndre = _float(cell.get("ndre_mean"))
    stress = _float(cell.get("stress_fraction"))
    if predicate == "healthy":
        if ndre is not None:
            score = ndre
        elif stress is not None:
            score = 1.0 - stress
        else:
            return None
        if hint == "ok":
            score += 0.05
        elif hint in _STRESSED_HINTS:
            score -= 0.05
        return score
    if ndre is None and stress is None:
        return None
    score = stress if stress is not None else -ndre  # type: ignore[operator]
    if hint in _STRESSED_HINTS:
        score += 0.1
    return score


_GLOBAL_STORE: ObservationStore | None = None


def get_observation_store(db_path: str | Path | None = None) -> ObservationStore:
    """Return the process-wide store, or a one-off store when ``db_path`` is set."""
    global _GLOBAL_STORE
    if db_path is not None:
        return ObservationStore(db_path)
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = ObservationStore()
    return _GLOBAL_STORE


def record_observation(**kwargs: Any) -> dict[str, Any] | None:
    """Insert one sample. Returns None if coordinates are missing."""
    x = _float(kwargs.get("x"))
    y = _float(kwargs.get("y"))
    if x is None or y is None:
        return None
    z = _float(kwargs.get("z")) or 0.0
    return get_observation_store().record(
        x=x,
        y=y,
        z=z,
        source=str(kwargs.get("source") or "capture_ndre"),
        ndre_mean=_float(kwargs.get("ndre_mean")),
        stress_fraction=_float(kwargs.get("stress_fraction")),
        veg_fraction=_float(kwargs.get("veg_fraction")),
        plant=_float(kwargs.get("plant")),
        weed=_float(kwargs.get("weed")),
        soil=_float(kwargs.get("soil")),
        dominant=kwargs.get("dominant"),
        action_hint=kwargs.get("action_hint"),
        summary=kwargs.get("summary"),
        timestamp=kwargs.get("timestamp"),
    )


def latest_cells(cell_mm: float = DEFAULT_CELL_MM) -> list[dict[str, Any]]:
    return get_observation_store().latest_cells(cell_mm=cell_mm)


def find_locations(
    predicate: str, cell_mm: float = DEFAULT_CELL_MM
) -> dict[str, Any]:
    return get_observation_store().find_locations(predicate, cell_mm=cell_mm)


def try_record_ndre(params: Mapping[str, Any]) -> None:
    """Stamp NDRE metrics at ``gantry_mm``. Never raises."""
    try:
        pose = xyz_from_pose(params.get("gantry_mm")) or xyz_from_pose(params)
        if pose is None:
            log.warning("ndre observation skipped: gantry_mm has no x/y")
            return
        ndre = params.get("ndre") if isinstance(params.get("ndre"), Mapping) else {}
        interp = (
            params.get("interpretation")
            if isinstance(params.get("interpretation"), Mapping)
            else {}
        )
        record_observation(
            x=pose[0],
            y=pose[1],
            z=pose[2],
            source="capture_ndre",
            ndre_mean=ndre.get("mean"),
            stress_fraction=ndre.get("stress_fraction") or interp.get("stress_fraction"),
            veg_fraction=ndre.get("vegetation_fraction")
            or interp.get("vegetation_fraction"),
            action_hint=interp.get("action_hint"),
            summary=params.get("summary") or interp.get("advice"),
        )
    except Exception:  # noqa: BLE001
        log.warning("ndre observation stamp failed", exc_info=True)


def try_record_segment(
    pose: Any, class_scores: Mapping[str, Any] | None, dominant: str | None = None
) -> None:
    """Stamp plant/weed/soil coverage at the current gantry pose. Never raises."""
    try:
        xyz = xyz_from_pose(pose)
        if xyz is None:
            return
        record_observation(
            x=xyz[0],
            y=xyz[1],
            z=xyz[2],
            source="segment_image",
            plant=class_score(class_scores, "plant"),
            weed=class_score(class_scores, "weed"),
            soil=class_score(class_scores, "soil"),
            dominant=dominant,
        )
    except Exception:  # noqa: BLE001
        log.warning("segment observation stamp failed", exc_info=True)


def format_cells(cells: list[Mapping[str, Any]] | None) -> str:
    """Compact bed-map text for the planner prompt."""
    lines = [
        "Bed map (latest 250 mm cells). Query with find_locations(predicate) "
        "then move to the top (x,y). Predicates: plant, weed, soil, healthy, "
        "stressed. Ripeness is not a map layer yet — if the user says ripe, "
        "say so and offer plant / healthy / stressed."
    ]
    if not cells:
        lines.append(
            "(map empty — run scan_ndre or segment_image at known poses to fill it; "
            "do not invent coordinates)"
        )
        return "\n".join(lines)
    lines.append("Latest cells:")
    for cell in cells:
        bits = [f"({round(cell['x'])}, {round(cell['y'])})"]
        if cell.get("ndre_mean") is not None:
            bits.append(f"ndre={cell['ndre_mean']}")
        if cell.get("stress_fraction") is not None:
            bits.append(f"stress={cell['stress_fraction']}")
        if cell.get("action_hint"):
            bits.append(f"hint={cell['action_hint']}")
        for kind in ("plant", "weed", "soil"):
            if cell.get(kind) is not None:
                bits.append(f"{kind}={cell[kind]}")
        if cell.get("dominant"):
            bits.append(f"dominant={cell['dominant']}")
        lines.append("- " + " ".join(bits))
    return "\n".join(lines)
