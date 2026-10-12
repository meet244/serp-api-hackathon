"""Shared test helpers. Tests never call SerpApi or a real LLM."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from cfp_compass.cache import ResponseCache  # noqa: E402
from cfp_compass.context import RunContext  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 10, 12)


def load_fixture(name: str) -> dict:
    """Load tests/fixtures/<name>.json (SerpApi response examples)."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache.sqlite3")


@pytest.fixture
def ctx(cache: ResponseCache) -> RunContext:
    """A run context with an empty temp cache, a fixed 'today' and no keys (offline)."""
    return RunContext(cache=cache, today=TODAY)
