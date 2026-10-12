"""SQLite response cache, so a re-run with the same inputs costs zero SerpApi credits.

Keys are the SHA-256 of the request params sorted by key, with ``api_key`` removed.
The full JSON response is stored with a timestamp; each caller decides how old an
entry may be (``max_age_seconds=None`` means it never expires).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""


def make_key(params: dict[str, Any]) -> str:
    """Stable cache key: SHA-256 of the params sorted by key, ignoring api_key."""
    clean = {k: v for k, v in params.items() if k != "api_key"}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    payload: Any
    created_at: float  # unix timestamp

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class ResponseCache:
    """A tiny thread-safe key/value store on SQLite (one short-lived connection per call)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str, max_age_seconds: float | None = None, *, now: float | None = None) -> CacheEntry | None:
        """Return the entry, or None if missing or older than ``max_age_seconds``."""
        with self._connect() as conn:
            row = conn.execute("SELECT payload, created_at FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        payload, created_at = row
        current = time.time() if now is None else now
        if max_age_seconds is not None and current - created_at > max_age_seconds:
            return None
        return CacheEntry(payload=json.loads(payload), created_at=created_at)

    def set(self, key: str, kind: str, payload: Any, *, now: float | None = None) -> None:
        created_at = time.time() if now is None else now
        blob = json.dumps(payload, ensure_ascii=False, default=str)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                (key, kind, blob, created_at),
            )

    def count(self, kind: str | None = None) -> int:
        with self._connect() as conn:
            if kind is None:
                return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM cache WHERE kind = ?", (kind,)).fetchone()[0]

    def clear(self, kind: str | None = None) -> None:
        with self._connect() as conn:
            if kind is None:
                conn.execute("DELETE FROM cache")
            else:
                conn.execute("DELETE FROM cache WHERE kind = ?", (kind,))
