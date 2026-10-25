"""A small sqlite cache of raw judge responses, so re-running a report is free.

Keyed on everything that changes the answer: judge identity, model, the exact prompt (or the
case for callables), and the run index. Pass --fresh to bypass it and make new calls.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".judgeflake")


def key_for(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x1f")
    return h.hexdigest()


class Cache:
    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.path = path or (DEFAULT_DIR / "cache.sqlite")
        self._conn: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL)"
            )
        return self._conn

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        row = self._db().execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        db = self._db()
        db.execute(
            "INSERT OR REPLACE INTO responses (key, value, created) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), time.time()),
        )
        db.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
