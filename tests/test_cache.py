"""SQLite response cache: keys, round trips, expiry and thread safety."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cfp_compass.cache import ResponseCache, make_key


# --- make_key ---------------------------------------------------------------------------

def test_make_key_ignores_api_key():
    params = {"engine": "google", "q": "pycon cfp", "hl": "en", "gl": "us"}
    assert make_key(params) == make_key({**params, "api_key": "secret-1"})
    assert make_key({**params, "api_key": "secret-1"}) == make_key({**params, "api_key": "secret-2"})


def test_make_key_ignores_parameter_order():
    a = {"engine": "google_flights", "departure_id": "COK", "arrival_id": "LAX", "currency": "INR"}
    b = {"currency": "INR", "arrival_id": "LAX", "engine": "google_flights", "departure_id": "COK"}
    assert make_key(a) == make_key(b)


def test_make_key_differs_for_different_values():
    base = {"engine": "google", "q": "pycon cfp"}
    assert make_key(base) != make_key({"engine": "google", "q": "djangocon cfp"})
    assert make_key(base) != make_key({**base, "gl": "us"})
    assert make_key({"engine": "google"}) != make_key({"engine": "google_hotels"})


def test_make_key_is_sha256_hex_and_does_not_leak_the_key():
    key = make_key({"engine": "google", "api_key": "super-secret"})
    assert len(key) == 64
    int(key, 16)
    assert "super-secret" not in key


def test_make_key_does_not_mutate_params():
    params = {"engine": "google", "api_key": "x"}
    make_key(params)
    assert params == {"engine": "google", "api_key": "x"}


# --- ResponseCache --------------------------------------------------------------------------

def test_round_trip(cache: ResponseCache):
    payload = {"organic_results": [{"title": "PyCon ₹", "link": "https://x.example"}], "n": 3, "ok": True}
    cache.set("k1", "serp:google", payload)
    entry = cache.get("k1")
    assert entry is not None
    assert entry.payload == payload
    assert abs(entry.created_at - time.time()) < 60
    assert entry.age_seconds >= 0


def test_missing_key_returns_none(cache: ResponseCache):
    assert cache.get("nope") is None
    assert cache.get("nope", 10) is None


def test_ttl_expiry_with_now(cache: ResponseCache):
    cache.set("k", "serp:google", {"a": 1}, now=1_000.0)
    assert cache.get("k", 60, now=1_030.0).payload == {"a": 1}
    assert cache.get("k", 60, now=1_060.0) is not None  # exactly max_age old: still fresh
    assert cache.get("k", 60, now=1_060.5) is None
    assert cache.get("k", 60, now=5_000.0) is None
    # Expiry does not delete: a longer max_age (or none) still sees it.
    assert cache.get("k", 10_000, now=5_000.0) is not None
    assert cache.get("k").created_at == pytest.approx(1_000.0)


def test_max_age_none_never_expires(cache: ResponseCache):
    cache.set("airports", "serp:google_flights_autocomplete", {"airports": ["COK"]}, now=0.0)
    entry = cache.get("airports", None, now=10**12)
    assert entry is not None
    assert entry.payload == {"airports": ["COK"]}


def test_overwrite_replaces(cache: ResponseCache):
    cache.set("k", "serp:google", {"v": 1}, now=100.0)
    cache.set("k", "serp:google_hotels", {"v": 2}, now=200.0)
    entry = cache.get("k")
    assert entry.payload == {"v": 2}
    assert entry.created_at == pytest.approx(200.0)
    assert cache.count() == 1
    assert cache.count("serp:google") == 0
    assert cache.count("serp:google_hotels") == 1


def test_count_and_clear_by_kind(cache: ResponseCache):
    for i in range(3):
        cache.set(f"g{i}", "serp:google", {"i": i})
    for i in range(2):
        cache.set(f"h{i}", "serp:google_hotels", {"i": i})
    cache.set("x", "llm:extract", {"name": "PyCon"})
    assert cache.count() == 6
    assert cache.count("serp:google") == 3
    assert cache.count("serp:google_hotels") == 2
    assert cache.count("missing-kind") == 0

    cache.clear("serp:google")
    assert cache.count("serp:google") == 0
    assert cache.count() == 3
    assert cache.get("g0") is None
    assert cache.get("h0") is not None

    cache.clear()
    assert cache.count() == 0


def test_non_dict_payloads(cache: ResponseCache):
    cache.set("list", "x", [1, 2, 3])
    cache.set("str", "x", "hello")
    cache.set("none", "x", None)
    assert cache.get("list").payload == [1, 2, 3]
    assert cache.get("str").payload == "hello"
    entry = cache.get("none")
    assert entry is not None and entry.payload is None


def test_persists_across_instances(tmp_path):
    path = tmp_path / "nested" / "dir" / "cache.sqlite3"
    first = ResponseCache(path)
    first.set("k", "serp:google", {"kept": True})
    second = ResponseCache(path)
    assert second.get("k").payload == {"kept": True}


def test_concurrent_set_and_get_from_8_threads(cache: ResponseCache):
    per_thread = 25
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def work(t: int) -> int:
        try:
            barrier.wait()
            found = 0
            for i in range(per_thread):
                key = make_key({"engine": "google", "thread": t, "i": i})
                cache.set(key, f"kind{t % 2}", {"thread": t, "i": i})
                entry = cache.get(key)
                assert entry is not None and entry.payload == {"thread": t, "i": i}
                found += 1
            return found
        except BaseException as exc:  # surface failures from worker threads
            errors.append(exc)
            raise

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(8)))

    assert not errors
    assert results == [per_thread] * 8
    assert cache.count() == 8 * per_thread
    assert cache.count("kind0") == 4 * per_thread
    assert cache.count("kind1") == 4 * per_thread
