"""serp.search: caching, credit counting, error handling and key scrubbing. Uses a fake client only."""

from __future__ import annotations

import copy
import json
import logging
import time

import pytest
import requests
import serpapi

from cfp_compass.cache import make_key
from cfp_compass.serp import TTL_FLIGHTS, scrub, search

SECRET = "SECRET123"
NO_RESULTS = "Google hasn't returned any results for this query."


class FakeSerp:
    """Stands in for serpapi.Client. Like the real client, it writes api_key into the dict it gets."""

    def __init__(self, response: dict | None = None, raises: Exception | None = None):
        self.response = response if response is not None else {"organic_results": [{"title": "t", "link": "https://a.example"}]}
        self.raises = raises
        self.calls: list[dict] = []

    def factory(self) -> FakeSerp:
        return self

    def search(self, params: dict) -> dict:
        self.calls.append(dict(params))
        params["api_key"] = SECRET  # the real client mutates the caller's dict
        if self.raises is not None:
            raise self.raises
        return copy.deepcopy(self.response)


def http_error(status: int, body: dict | None = None) -> serpapi.HTTPError:
    """Build a serpapi.HTTPError the way the client does: wrapping a requests HTTPError."""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode() if body is not None else b""
    response.url = f"https://serpapi.com/search?engine=google&q=x&api_key={SECRET}"
    original = requests.exceptions.HTTPError(f"{status} Client Error: for url: {response.url}", response=response)
    return serpapi.HTTPError(original)


def params(q: str = "pycon cfp") -> dict:
    return {"engine": "google", "q": q, "hl": "en", "gl": "us"}


@pytest.fixture
def online(ctx):
    fake = FakeSerp()
    ctx.serp_client_factory = fake.factory
    ctx.serpapi_api_key = SECRET
    return ctx, fake


def test_miss_calls_client_once_counts_and_stores(online):
    ctx, fake = online
    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.ok and not response.from_cache and not response.no_results
    assert response.data == fake.response
    assert response.fetched_at is not None
    assert len(fake.calls) == 1
    assert ctx.count("serpapi_searches") == 1
    assert ctx.count("serpapi_cache_hits") == 0
    assert ctx.cache.count() == 1
    assert ctx.cache.get(make_key(params())).payload == fake.response


def test_same_params_again_is_a_cache_hit(online):
    ctx, fake = online
    search(params(), ctx, ttl_seconds=TTL_FLIGHTS)
    again = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert again.ok and again.from_cache
    assert again.data == fake.response
    assert again.fetched_at is not None
    assert len(fake.calls) == 1
    assert ctx.count("serpapi_searches") == 1
    assert ctx.count("serpapi_cache_hits") == 1


def test_param_order_does_not_matter(online):
    ctx, fake = online
    search({"engine": "google", "q": "x", "hl": "en"}, ctx, ttl_seconds=None)
    hit = search({"hl": "en", "q": "x", "engine": "google"}, ctx, ttl_seconds=None)
    assert hit.from_cache
    assert len(fake.calls) == 1


def test_callers_params_are_not_mutated(online):
    ctx, fake = online
    mine = params()
    snapshot = dict(mine)
    search(mine, ctx, ttl_seconds=TTL_FLIGHTS)
    assert mine == snapshot
    assert "api_key" not in mine


def test_cache_key_ignores_api_key(online):
    ctx, fake = online
    first = search({**params(), "api_key": "key-one"}, ctx, ttl_seconds=TTL_FLIGHTS)
    second = search({**params(), "api_key": "key-two"}, ctx, ttl_seconds=TTL_FLIGHTS)
    third = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert first.ok and not first.from_cache
    assert second.from_cache and third.from_cache
    assert len(fake.calls) == 1
    assert make_key({**params(), "api_key": "key-one"}) == make_key(params())
    assert ctx.cache.get(make_key(params())) is not None


def test_ttl_expiry_respected_online(online):
    ctx, fake = online
    old = time.time() - TTL_FLIGHTS - 60
    ctx.cache.set(make_key(params()), "serp:google", {"stale": True}, now=old)

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.ok and not response.from_cache
    assert response.data == fake.response
    assert len(fake.calls) == 1
    assert ctx.count("serpapi_searches") == 1


def test_fresh_entry_within_ttl_is_used(online):
    ctx, fake = online
    recent = time.time() - TTL_FLIGHTS + 600
    ctx.cache.set(make_key(params()), "serp:google", {"fresh": True}, now=recent)

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.from_cache and response.data == {"fresh": True}
    assert fake.calls == []


def test_cached_only_returns_stale_entries_regardless_of_ttl(online):
    ctx, fake = online
    ctx.cached_only = True
    very_old = time.time() - 30 * 24 * 60 * 60
    ctx.cache.set(make_key(params()), "serp:google", {"stale": True}, now=very_old)

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.ok and response.from_cache
    assert response.data == {"stale": True}
    assert fake.calls == []
    assert ctx.count("serpapi_cache_hits") == 1


def test_cached_only_miss_is_unavailable_without_calling_client(online):
    ctx, fake = online
    ctx.cached_only = True

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert not response.ok
    assert response.error.startswith("unavailable")
    assert fake.calls == []
    assert ctx.count("serpapi_searches") == 0


def test_no_key_and_no_factory_is_offline(ctx):
    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)
    assert response.error.startswith("unavailable")


def test_google_no_results_answer_is_cached_and_ok(online):
    ctx, fake = online
    fake.response = {"search_information": {"organic_results_state": "Fully empty"}, "error": NO_RESULTS}

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.ok is True
    assert response.no_results is True
    assert response.error is None
    assert ctx.cache.count() == 1
    assert ctx.errors == []

    again = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)
    assert again.from_cache and again.ok and again.no_results
    assert len(fake.calls) == 1


def test_other_error_value_is_not_cached(online):
    ctx, fake = online
    fake.response = {"error": "Unsupported `foo` search parameter."}

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.ok is False
    assert "Unsupported" in response.error
    assert response.no_results is False
    assert ctx.cache.count() == 0
    assert len(ctx.errors) == 1
    assert ctx.api_disabled_reason is None

    search(params(), ctx, ttl_seconds=TTL_FLIGHTS)
    assert len(fake.calls) == 2  # not cached, so asked again


def test_http_200_out_of_searches_disables_api(online):
    ctx, fake = online
    fake.response = {"error": "Your account has run out of searches."}

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)
    assert not response.ok
    assert ctx.api_disabled_reason is not None

    later = search(params("other"), ctx, ttl_seconds=TTL_FLIGHTS)
    assert later.error.startswith("unavailable")
    assert len(fake.calls) == 1


@pytest.mark.parametrize("status", [429, 401, 403])
def test_rate_limit_and_auth_errors_disable_api(online, status):
    ctx, fake = online
    fake.raises = http_error(status, {"error": "Your account has run out of searches." if status == 429 else "Invalid API key."})

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert not response.ok
    assert ctx.api_disabled_reason is not None
    assert len(fake.calls) == 1
    assert ctx.count("serpapi_searches") == 0
    assert ctx.cache.count() == 0

    later = search(params("another query"), ctx, ttl_seconds=TTL_FLIGHTS)
    assert later.error.startswith("unavailable")
    assert len(fake.calls) == 1  # the client is never reached again


def test_http_error_without_json_body_uses_status(online):
    ctx, fake = online
    fake.raises = http_error(401)  # empty body: .error is None

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.error == "HTTP 401"
    assert ctx.api_disabled_reason is not None
    assert all(SECRET not in e for e in ctx.errors)


def test_server_error_does_not_disable_api(online):
    ctx, fake = online
    fake.raises = http_error(500, {"error": "Internal error"})

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert not response.ok
    assert ctx.api_disabled_reason is None
    fake.raises = None
    assert search(params(), ctx, ttl_seconds=TTL_FLIGHTS).ok
    assert len(fake.calls) == 3  # the 500 was retried once, then the next search succeeded


def test_connection_error_status_minus_one(online):
    ctx, fake = online
    original = requests.exceptions.ConnectionError(
        f"Max retries exceeded with url: /search?engine=google&q=x&api_key={SECRET}"
    )
    fake.raises = serpapi.HTTPConnectionError(original)
    assert fake.raises.status_code == -1

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert response.error == "connection error"
    assert ctx.api_disabled_reason is None
    assert len(ctx.errors) == 1
    assert SECRET not in ctx.errors[0]


def test_generic_exception_never_leaks_the_key(online, caplog):
    ctx, fake = online
    fake.raises = RuntimeError(
        f"boom while fetching https://serpapi.com/search?q=x&api_key={SECRET}&engine=google (key {SECRET})"
    )

    with caplog.at_level(logging.DEBUG):
        response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert not response.ok
    assert response.error
    assert SECRET not in response.error
    assert ctx.errors, "the failure must be recorded"
    assert all(SECRET not in e for e in ctx.errors)
    assert SECRET not in caplog.text
    assert ctx.cache.count() == 0


def test_http_error_body_with_key_is_scrubbed(online):
    ctx, fake = online
    fake.raises = http_error(400, {"error": f"Bad request for api_key={SECRET}"})

    response = search(params(), ctx, ttl_seconds=TTL_FLIGHTS)

    assert SECRET not in response.error
    assert all(SECRET not in e for e in ctx.errors)


def test_scrub_removes_key_in_url_and_raw():
    text = f"GET /search?api_key={SECRET}&q=1 failed; key={SECRET}"
    cleaned = scrub(text, SECRET)
    assert SECRET not in cleaned
    assert "api_key=***" in cleaned
    assert scrub("api_key=abc&q=1", None) == "api_key=***&q=1"
