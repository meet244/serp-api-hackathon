"""The only module that talks to SerpApi.

Every call goes through :func:`search`. It checks the cache, calls the official client
on a miss, stores the response and counts real API calls, so the UI can show the credits
used. One failed lookup never crashes a run: problems come back as ``SerpResponse.error``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import serpapi

from .cache import make_key
from .context import RunContext, log

HOUR = 60 * 60
TTL_DISCOVERY = 24 * HOUR
TTL_TRAVEL_POLICY = 24 * HOUR
TTL_FLIGHTS = 6 * HOUR
TTL_HOTELS = 6 * HOUR
TTL_AIRPORTS: float | None = None  # a city's airports do not change: never expires

SEARCH_TIMEOUT = 90  # seconds; Google searches through SerpApi can take 30 s or more
SEARCH_ATTEMPTS = 2  # a timed-out search usually finished at SerpApi: the retry is served from its free 1 h cache
_RETRYABLE_STATUS = {-1, 500, 502, 503, 504}

_NO_RESULTS = re.compile(r"(hasn't|has not) returned any results|no results", re.IGNORECASE)
_KEY_IN_URL = re.compile(r"(api_key=)[^&\s'\"]+")


@dataclass(frozen=True)
class SerpResponse:
    data: dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False
    fetched_at: datetime | None = None
    error: str | None = None  # set when no usable data could be obtained
    no_results: bool = False  # SerpApi answered, but the search found nothing

    @property
    def ok(self) -> bool:
        return self.error is None


def scrub(text: str, api_key: str | None) -> str:
    """Remove the API key from any text before it is shown or logged."""
    cleaned = _KEY_IN_URL.sub(r"\1***", text)
    if api_key:
        cleaned = cleaned.replace(api_key, "***")
    return cleaned


def _make_client(ctx: RunContext) -> Any:
    if ctx.serp_client_factory is not None:
        return ctx.serp_client_factory()
    # One client per call: each client owns a requests.Session, which is not thread-safe.
    return serpapi.Client(api_key=ctx.serpapi_api_key, timeout=SEARCH_TIMEOUT)


def _describe(params: dict[str, Any]) -> str:
    engine = params.get("engine", "google")
    detail = params.get("q") or f"{params.get('departure_id', '')}->{params.get('arrival_id', '')}"
    return f"{engine} [{detail}]"


def search(params: dict[str, Any], ctx: RunContext, *, ttl_seconds: float | None) -> SerpResponse:
    """Run one SerpApi search through the cache. ``ttl_seconds=None`` means never expire."""
    params = {k: v for k, v in params.items() if k != "api_key"}
    key = make_key(params)
    engine = str(params.get("engine", "google"))

    # Parallel identical searches (e.g. two conferences in one city) wait for the first one
    # and are then served from the cache, so they cost one credit, not several.
    with ctx.key_lock(key):
        return _search(params, key, engine, ctx, ttl_seconds)


def _search(params: dict[str, Any], key: str, engine: str, ctx: RunContext, ttl_seconds: float | None) -> SerpResponse:
    # Offline (cached-only toggle, credits ran out): a stale answer beats no answer.
    offline = ctx.serp_offline
    hit = ctx.cache.get(key, None if offline else ttl_seconds)
    if hit is not None:
        ctx.incr("serpapi_cache_hits")
        data = hit.payload if isinstance(hit.payload, dict) else {}
        return SerpResponse(
            data=data,
            from_cache=True,
            fetched_at=datetime.fromtimestamp(hit.created_at),
            no_results=bool(_NO_RESULTS.search(str(data.get("error", "")))),
        )

    if offline:
        reason = ctx.api_disabled_reason or "cached results only"
        return SerpResponse(error=f"unavailable: not in cache ({reason})")

    data: dict[str, Any] = {}
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        last_try = attempt == SEARCH_ATTEMPTS
        try:
            result = _make_client(ctx).search(dict(params))  # a copy: the client writes api_key into it
            data = result.as_dict() if hasattr(result, "as_dict") else dict(result)
            break
        except serpapi.HTTPError as exc:
            status = getattr(exc, "status_code", None)
            if (status is None or status in _RETRYABLE_STATUS) and not last_try:
                log.info("Retrying SerpApi %s after HTTP %s", _describe(params), status)
                continue
            fallback = "connection error" if status is None or status < 0 else f"HTTP {status}"
            message = scrub(str(getattr(exc, "error", None) or fallback), ctx.serpapi_api_key)
            if status in (401, 403, 429) or "run out of searches" in message.lower():
                ctx.disable_api(f"SerpApi said: {message}")
            ctx.add_error(f"SerpApi {_describe(params)} failed: {message}")
            return SerpResponse(error=message)
        except Exception as exc:  # timeouts, connection errors, bad JSON: degrade, never crash
            if not last_try:
                log.info("Retrying SerpApi %s after %s", _describe(params), type(exc).__name__)
                continue
            message = scrub(f"{type(exc).__name__}: {exc}", ctx.serpapi_api_key)[:300]
            ctx.add_error(f"SerpApi {_describe(params)} failed: {message}")
            return SerpResponse(error=message)

    ctx.incr("serpapi_searches")
    log.info("SerpApi search: %s", _describe(params))
    error = data.get("error")
    if error and not _NO_RESULTS.search(str(error)):
        message = scrub(str(error), ctx.serpapi_api_key)
        if "run out of searches" in message.lower() or "invalid api key" in message.lower():
            ctx.disable_api(f"SerpApi said: {message}")
        ctx.add_error(f"SerpApi {_describe(params)} returned an error: {message}")
        return SerpResponse(error=message)

    ctx.cache.set(key, f"serp:{engine}", data)
    return SerpResponse(data=data, from_cache=False, fetched_at=datetime.now(), no_results=bool(error))
