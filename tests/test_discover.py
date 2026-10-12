"""Discovery queries, organic result parsing, candidate merging and the keyword pre-filter."""

from __future__ import annotations

from datetime import date

import pytest

from cfp_compass.discover import (
    MAX_CANDIDATES,
    build_queries,
    discover,
    discovery_params,
    merge_candidates,
    normalize_url,
    parse_organic,
    prefilter,
)
from cfp_compass.models import Candidate, SpeakerProfile
from conftest import TODAY, load_fixture


# --- queries ---------------------------------------------------------------------------------

def test_four_queries_per_topic():
    queries = build_queries(["Python concurrency"], TODAY)
    assert queries == [
        '"call for proposals" Python concurrency conference 2027',
        '"call for papers" Python concurrency conference 2027',
        "site:sessionize.com Python concurrency",
        "site:pretalx.com Python concurrency cfp",
    ]


def test_at_most_two_topics():
    queries = build_queries(["Python concurrency", "AI agents", "Rust"], TODAY)
    assert len(queries) == 8
    assert len(set(queries)) == 8
    assert not any("Rust" in q for q in queries)
    assert sum("AI agents" in q for q in queries) == 4
    assert sum(q.startswith("site:sessionize.com ") for q in queries) == 2
    assert sum(q.startswith("site:pretalx.com ") for q in queries) == 2


def test_years_follow_today():
    queries = build_queries(["Python"], TODAY)
    # October: this year's CFPs have mostly closed, so search next year's events only.
    assert sum(q.endswith("conference 2027") for q in queries) == 2
    # January to June: this year and next.
    assert all("2027 OR 2028" in q for q in build_queries(["Python"], date(2027, 1, 3))[:2])
    assert all("2027 OR 2028" in q for q in build_queries(["Python"], date(2027, 6, 30))[:2])
    assert all(q.endswith("conference 2028") for q in build_queries(["Python"], date(2027, 7, 1))[:2])


def test_no_topics_no_queries():
    assert build_queries([], TODAY) == []


def test_discovery_params():
    params = discovery_params("site:pretalx.com Python cfp")
    assert params == {"engine": "google", "q": "site:pretalx.com Python cfp", "hl": "en", "gl": "us"}
    assert "num" not in params
    assert "api_key" not in params
    assert "no_cache" not in params


# --- parse_organic -----------------------------------------------------------------------------

def test_parse_organic_fixture():
    data = load_fixture("google_search")
    candidates = parse_organic(data, "coffee")
    expected = [r for r in data["organic_results"] if str(r.get("link", "")).startswith("http")]
    assert len(candidates) == len(expected) > 0
    first = candidates[0]
    assert first.url == expected[0]["link"]
    assert first.title == expected[0]["title"]
    assert first.snippet == expected[0].get("snippet", "")
    assert first.source_query == "coffee"
    assert first.position == expected[0]["position"]
    dated = [c for c in candidates if c.date]
    assert [c.date for c in dated] == [r["date"] for r in expected if r.get("date")]
    assert all(c.url.startswith(("http://", "https://")) for c in candidates)


def test_parse_organic_skips_malformed_items():
    data = {
        "organic_results": [
            "not a dict",
            None,
            42,
            {"title": "No link here", "snippet": "call for proposals"},
            {"title": "Empty link", "link": ""},
            {"title": "FTP", "link": "ftp://files.example/cfp"},
            {"title": "JS", "link": "javascript:alert(1)"},
            {"title": "Relative", "link": "/cfp"},
            {"link": "https://ok.example/cfp", "position": "3", "date": ""},
            {"title": None, "link": "http://plain.example/talks", "snippet": None, "position": 7, "date": "Sep 1, 2026"},
        ]
    }
    candidates = parse_organic(data, "q")
    assert [c.url for c in candidates] == ["https://ok.example/cfp", "http://plain.example/talks"]
    ok, plain = candidates
    assert ok.title == "" and ok.snippet == "" and ok.position is None and ok.date is None
    assert plain.title == "" and plain.snippet == "" and plain.position == 7 and plain.date == "Sep 1, 2026"


@pytest.mark.parametrize("data", [{}, {"organic_results": None}, {"organic_results": []}, {"error": "x"}])
def test_parse_organic_empty(data):
    assert parse_organic(data, "q") == []


# --- merge_candidates -------------------------------------------------------------------------

def _c(url: str, query: str = "q") -> Candidate:
    return Candidate(title=url, url=url, snippet="call for proposals", source_query=query)


def test_merge_is_round_robin():
    a = [_c("https://a.example/1", "A"), _c("https://a.example/2", "A"), _c("https://a.example/3", "A")]
    b = [_c("https://b.example/1", "B"), _c("https://b.example/2", "B")]
    c: list[Candidate] = []
    merged = merge_candidates([a, b, c])
    assert [m.url for m in merged] == [
        "https://a.example/1",
        "https://b.example/1",
        "https://a.example/2",
        "https://b.example/2",
        "https://a.example/3",
    ]


def test_merge_dedupes_www_and_trailing_slash():
    a = [_c("https://www.PyCon.example/cfp/"), _c("https://unique.example/x")]
    b = [_c("https://pycon.example/cfp"), _c("http://www.unique.example/x/")]
    merged = merge_candidates([a, b])
    assert [m.url for m in merged] == ["https://www.PyCon.example/cfp/", "https://unique.example/x"]


def test_merge_keeps_query_strings_distinct():
    merged = merge_candidates([[_c("https://x.example/e?id=1"), _c("https://x.example/e?id=2")]])
    assert len(merged) == 2


def test_merge_caps_at_40_and_keeps_every_query():
    lists = [[_c(f"https://site{q}.example/{i}", f"Q{q}") for i in range(10)] for q in range(8)]
    merged = merge_candidates(lists)
    assert len(merged) == MAX_CANDIDATES == 40
    assert {m.source_query for m in merged[:8]} == {f"Q{q}" for q in range(8)}
    assert {m.source_query for m in merged} == {f"Q{q}" for q in range(8)}
    assert len({normalize_url(m.url) for m in merged}) == 40


def test_merge_custom_limit_and_empty():
    lists = [[_c(f"https://a.example/{i}") for i in range(5)]]
    assert len(merge_candidates(lists, limit=3)) == 3
    assert merge_candidates([]) == []
    assert merge_candidates([[], []]) == []


def test_normalize_url():
    assert normalize_url("https://www.Example.com/CFP/") == "example.com/CFP"
    assert normalize_url("http://example.com") == "example.com/"
    assert normalize_url("https://example.com/?a=1") == "example.com/?a=1"


# --- prefilter ------------------------------------------------------------------------------------

def _cand(url: str, title: str = "", snippet: str = "") -> Candidate:
    return Candidate(title=title, url=url, snippet=snippet)


def test_prefilter_keeps_cfp_text():
    keep = [
        _cand("https://us.pycon.org/2027/speaking/", "PyCon US 2027: Call for Proposals"),
        _cand("https://sessionize.com/pydata-x", "PyData X", "Submit a session by Nov 30"),
        _cand("https://pretalx.com/djangocon/cfp", "DjangoCon CFP"),
        _cand("https://conf.example/", "Speakers wanted", ""),
        _cand("https://conf.example/talks", "Conf", "Give a TALK at our event"),
        _cand("https://conf.example/papers", "Conf", "Papers due soon"),
        _cand("https://linkx.com/cfp", "Call for speakers"),  # not x.com
        _cand("https://conf.example/pdf-guide", "Proposal guide"),  # not a .pdf
    ]
    assert prefilter(keep) == keep


def test_prefilter_drops_pdfs_and_login_walls():
    drop = [
        _cand("https://conf.example/cfp.pdf", "Call for proposals"),
        _cand("https://conf.example/CFP.PDF?download=1", "Call for papers"),
        _cand("https://www.linkedin.com/posts/pycon-cfp", "PyCon CFP is open"),
        _cand("https://x.com/pycon/status/1", "Our CFP is open"),
        _cand("https://twitter.com/pycon/status/1", "Submit your talk"),
        _cand("https://www.youtube.com/watch?v=abc", "How to write a talk proposal"),
        _cand("https://m.youtube.com/watch?v=abc", "Call for speakers"),
        _cand("https://www.facebook.com/events/1", "Call for speakers"),
    ]
    assert prefilter(drop) == []


def test_prefilter_drops_texts_without_keywords():
    drop = [
        _cand("https://en.wikipedia.org/wiki/Coffee", "Coffee - Wikipedia", "Coffee is a brewed drink"),
        _cand("https://conf.example/", "", ""),
    ]
    assert prefilter(drop) == []


def test_prefilter_fixture_has_no_cfps():
    assert prefilter(parse_organic(load_fixture("google_search"), "coffee")) == []


# --- discover() through the serp wrapper with a fake client (no network) ---------------------------

class _FakeClient:
    calls: list[dict] = []

    def search(self, params: dict) -> dict:
        _FakeClient.calls.append(dict(params))
        q = params["q"]
        slug = abs(hash(q)) % 10_000
        return {
            "organic_results": [
                {"position": 1, "title": "Shared CFP", "link": "https://www.shared.example/cfp/", "snippet": "cfp"},
                {"position": 2, "title": f"Hit {q}", "link": f"https://hits.example/{slug}", "snippet": "talk"},
            ]
        }


def test_discover_with_fake_client(ctx):
    _FakeClient.calls = []
    ctx.serp_client_factory = _FakeClient
    profile = SpeakerProfile(
        topics="Python concurrency, AI agents",
        home_airport="COK",
        budget_inr=150_000,
        window_start=date(2026, 11, 12),
        window_end=date(2027, 10, 12),
    )
    candidates, n_queries = discover(profile, ctx)
    assert n_queries == 8
    assert len(_FakeClient.calls) == 8
    assert all(call["engine"] == "google" and call["hl"] == "en" and call["gl"] == "us" for call in _FakeClient.calls)
    assert all("api_key" not in call for call in _FakeClient.calls)
    urls = [c.url for c in candidates]
    assert urls.count("https://www.shared.example/cfp/") == 1
    assert len(candidates) == 1 + 8
    assert ctx.count("serpapi_searches") == 8

    # Same inputs again: served from the cache, no new calls.
    _FakeClient.calls = []
    again, _ = discover(profile, ctx)
    assert _FakeClient.calls == []
    assert [c.url for c in again] == urls
    assert ctx.count("serpapi_cache_hits") == 8
