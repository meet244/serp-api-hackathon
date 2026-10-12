"""Shared state for one pipeline run: the cache, keys, counters and flags.

Worker threads update counters through ``incr``/``add_error``, which take a lock.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .cache import ResponseCache

log = logging.getLogger("cfp_compass")

_SECRET_ENV = (
    "SERPAPI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY",
)
_KEY_PARAM = re.compile(r"((?:api_?key|key|token)=)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str, extra: Iterable[str | None] = ()) -> str:
    """Remove API keys (known values and key=... URL parameters) from text shown or logged."""
    cleaned = _KEY_PARAM.sub(r"\1***", text)
    for value in [*(os.environ.get(name) for name in _SECRET_ENV), *extra]:
        if value and len(value) >= 8:
            cleaned = cleaned.replace(value, "***")
    return cleaned


@dataclass
class RunContext:
    cache: ResponseCache
    serpapi_api_key: str | None = field(default=None, repr=False)
    llm_model: Any = None  # a Pydantic AI model string, or a Model instance in tests
    cached_only: bool = False
    today: date = field(default_factory=date.today)
    max_workers: int = 4
    serp_client_factory: Callable[[], Any] | None = field(default=None, repr=False)  # tests inject a fake
    api_disabled_reason: str | None = None
    llm_disabled_reason: str | None = None
    counters: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _inflight: dict[str, threading.Lock] = field(default_factory=dict, repr=False)

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + amount

    def count(self, name: str) -> int:
        with self._lock:
            return self.counters.get(name, 0)

    def add_error(self, message: str) -> None:
        message = redact(message, [self.serpapi_api_key])
        with self._lock:
            self.errors.append(message)
        log.warning(message)

    def disable_api(self, reason: str) -> None:
        """Stop calling SerpApi for the rest of the run (e.g. out of credits): cached data only."""
        with self._lock:
            if self.api_disabled_reason is None:
                self.api_disabled_reason = reason
        log.warning("Switching to cached results only: %s", reason)

    def disable_llm(self, reason: str) -> None:
        """Stop calling the LLM for the rest of the run (quota gone); cached LLM answers still work."""
        with self._lock:
            if self.llm_disabled_reason is None:
                self.llm_disabled_reason = reason
        log.warning("LLM disabled for this run: %s", reason)

    def key_lock(self, key: str) -> threading.Lock:
        """One lock per cache key, so parallel identical searches make a single paid call."""
        with self._lock:
            return self._inflight.setdefault(key, threading.Lock())

    @property
    def serp_offline(self) -> bool:
        return self.cached_only or self.api_disabled_reason is not None or not (
            self.serpapi_api_key or self.serp_client_factory
        )
