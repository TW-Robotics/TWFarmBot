"""Long-term agentic memory store for TWFarmBot.

Persists discrete agronomic observations, treatments, and operational rules
in a lightweight SQLite database. Allows the LLM to actively decide what
is noteworthy enough to store, recall memories spatially (by coordinates)
and semantically (by text/category), and delete obsolete entries.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_DB_REL = Path("data") / "memory.db"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """SQLite-backed long-term memory for agentic FarmBot."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            env_path = os.getenv("FARMBOT_MEMORY_DB")
            if env_path:
                self.db_path = Path(env_path)
            else:
                try:
                    from twfarmbot_core.config import load_yaml_config

                    cfg_path = load_yaml_config().get("memory", {}).get("db_path")
                    self.db_path = Path(cfg_path) if cfg_path else _DEFAULT_DB_REL
                except Exception:
                    self.db_path = _DEFAULT_DB_REL
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
            # WAL mode is persistent and improves concurrent read/write performance
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'agronomy',
                    x REAL,
                    y REAL,
                    z REAL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    tags TEXT NOT NULL DEFAULT '[]',
                    active INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_coords ON memories(x, y);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);"
            )

    def remember(
        self,
        content: str,
        category: str = "agronomy",
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> dict[str, Any]:
        """Commit a noteworthy insight, observation, treatment, or rule to memory."""
        text = content.strip()
        if not text:
            raise ValueError("Memory content cannot be empty.")

        importance_clamped = max(0.0, min(1.0, float(importance)))
        tags_list = [t.strip().lower() for t in (tags or []) if t and t.strip()]
        tags_json = json.dumps(tags_list)
        now = _utc_now_iso()

        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO memories (
                    created_at, updated_at, content, category, x, y, z, importance, tags, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1);
                """,
                (
                    now,
                    now,
                    text,
                    category.strip().lower(),
                    float(x) if x is not None else None,
                    float(y) if y is not None else None,
                    float(z) if z is not None else None,
                    importance_clamped,
                    tags_json,
                ),
            )
            row_id = cur.lastrowid

        return {
            "id": row_id,
            "created_at": now,
            "content": text,
            "category": category.strip().lower(),
            "x": float(x) if x is not None else None,
            "y": float(y) if y is not None else None,
            "z": float(z) if z is not None else None,
            "importance": importance_clamped,
            "tags": tags_list,
            "status": "stored",
        }

    def recall(
        self,
        query: str | None = None,
        near_x: float | None = None,
        near_y: float | None = None,
        radius_mm: float = 300.0,
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve memories matching spatial coordinates, query text, or category."""
        sql = [
            "SELECT id, created_at, updated_at, content, category, x, y, z, importance, tags",
            "FROM memories WHERE active = 1",
        ]
        params: list[Any] = []

        if category:
            sql.append("AND category = ?")
            params.append(category.strip().lower())

        is_spatial = near_x is not None and near_y is not None
        if is_spatial:
            rad = max(1.0, float(radius_mm))
            sql.append(
                "AND x IS NOT NULL AND y IS NOT NULL "
                "AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?"
            )
            params.extend([near_x - rad, near_x + rad, near_y - rad, near_y + rad])

        if query and query.strip():
            terms = query.strip().lower().split()
            for term in terms:
                term_pattern = f"%{term}%"
                sql.append(
                    "AND (LOWER(content) LIKE ? OR LOWER(tags) LIKE ? OR LOWER(category) LIKE ?)"
                )
                params.extend([term_pattern, term_pattern, term_pattern])

        cur = self._conn.execute(" ".join(sql), params)
        rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            try:
                parsed_tags = json.loads(r["tags"])
            except Exception:
                parsed_tags = []

            item: dict[str, Any] = {
                "id": r["id"],
                "created_at": r["created_at"],
                "content": r["content"],
                "category": r["category"],
                "x": r["x"],
                "y": r["y"],
                "z": r["z"],
                "importance": r["importance"],
                "tags": parsed_tags,
            }

            if is_spatial and r["x"] is not None and r["y"] is not None:
                dist = math.hypot(r["x"] - near_x, r["y"] - near_y)
                if dist > radius_mm:
                    continue
                item["distance_mm"] = round(dist, 1)

            results.append(item)

        if is_spatial:
            # Sort by distance first, then higher importance, then newest
            results.sort(
                key=lambda m: (
                    m.get("distance_mm", 999999.0),
                    -m.get("importance", 0.5),
                    m.get("created_at", ""),
                )
            )
        else:
            # Sort by highest importance, then newest
            results.sort(
                key=lambda m: (
                    -m.get("importance", 0.5),
                    m.get("created_at", ""),
                ),
                reverse=False,
            )

        return results[: max(1, limit)]

    def list_recent(
        self, limit: int = 10, category: str | None = None
    ) -> list[dict[str, Any]]:
        """List recent active memories, newest first."""
        sql = [
            "SELECT id, created_at, updated_at, content, category, x, y, z, importance, tags",
            "FROM memories WHERE active = 1",
        ]
        params: list[Any] = []
        if category:
            sql.append("AND category = ?")
            params.append(category.strip().lower())
        sql.append("ORDER BY created_at DESC LIMIT ?")
        params.append(max(1, limit))

        cur = self._conn.execute(" ".join(sql), params)
        out: list[dict[str, Any]] = []
        for r in cur.fetchall():
            try:
                tags = json.loads(r["tags"])
            except Exception:
                tags = []
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "content": r["content"],
                    "category": r["category"],
                    "x": r["x"],
                    "y": r["y"],
                    "z": r["z"],
                    "importance": r["importance"],
                    "tags": tags,
                }
            )
        return out

    def get(self, memory_id: int) -> dict[str, Any] | None:
        """Fetch a single active memory by ID."""
        cur = self._conn.execute(
            """
            SELECT id, created_at, updated_at, content, category, x, y, z, importance, tags
            FROM memories WHERE id = ? AND active = 1;
            """,
            (int(memory_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            tags = json.loads(row["tags"])
        except Exception:
            tags = []
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "content": row["content"],
            "category": row["category"],
            "x": row["x"],
            "y": row["y"],
            "z": row["z"],
            "importance": row["importance"],
            "tags": tags,
        }

    def forget(self, memory_id: int) -> bool:
        """Mark a memory inactive (soft-delete)."""
        now = _utc_now_iso()
        with self._conn:
            cur = self._conn.execute(
                "UPDATE memories SET active = 0, updated_at = ? WHERE id = ? AND active = 1;",
                (now, int(memory_id)),
            )
            return cur.rowcount > 0

    def clear(self) -> None:
        """Delete all records from the database (testing utility)."""
        with self._conn:
            self._conn.execute("DELETE FROM memories;")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


_GLOBAL_STORE: MemoryStore | None = None


def get_memory_store(db_path: str | Path | None = None) -> MemoryStore:
    """Return the global MemoryStore singleton, or a custom one if db_path is given."""
    global _GLOBAL_STORE
    if db_path is not None:
        return MemoryStore(db_path)
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = MemoryStore()
    return _GLOBAL_STORE
