"""Offline end-to-end tests for the ten-step pipeline and the CLI.

SerpApi is replaced by a fake client (``RunContext.serp_client_factory``) and the LLM by a
Pydantic AI ``FunctionModel`` (``RunContext.llm_model``). Nothing here touches the network.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests
import serpapi
from pydantic_ai import models as pai_models
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import cfp_compass.config as config
import cfp_compass.extract as extract
import cfp_compass.pipeline as pipeline
from cfp_compass.cache import ResponseCache
from cfp_compass.context import RunContext
from cfp_compass.models import CFP, SpeakerProfile
from cfp_compass.pipeline import Progress, main, run_pipeline

LLM_ENV_VARS = (
    "SERPAPI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "CFP_COMPASS_MODEL",
)


# --- Fake web pages: what Google "finds" and what the "LLM" extracts from each hit ------------

PYCONF = "https://sessionize.com/pyconf-hyd-2027/"
AICONF = "https://pretalx.com/aiconf-2027/cfp"
ONLINE = "https://asyncpy.example.org/cfp"
CLOSED = "https://concurrencydays.example.com/call-for-papers"
PYCONF_DUP = "https://pyconf-hyd.example.in/call-for-proposals"

PAGES: dict[str, dict[str, Any]] = {
    PYCONF: {
        "title": "PyConf Hyderabad 2027: Call for Speakers",
        "snippet": "Submit your talk on Python concurrency. CFP closes 30 Nov 2026. Conference 15-16 Feb 2027.",
        "record": {
            "conference_name": "PyConf Hyderabad 2027", "cfp_url": PYCONF, "cfp_deadline": "2026-11-30",
            "event_start": "2027-02-15", "event_end": "2027-02-16", "city": "Hyderabad", "country": "India",
            "topics": ["Python", "concurrency"], "confidence": 0.9, "evidence": "CFP closes 30 Nov 2026",
        },
    },
    AICONF: {
        "title": "AI Agents Summit 2027 call for papers",
        "snippet": "Talks on AI agents and LLM tooling. Singapore.",  # thin snippet: the page is fetched
        "record": {
            "conference_name": "AI Agents Summit 2027", "cfp_url": AICONF, "cfp_deadline": "2026-12-15",
            "event_start": "2027-03-10", "event_end": "2027-03-12", "city": "Singapore", "country": "Singapore",
            "topics": ["AI agents", "LLMs"], "confidence": 0.85, "evidence": "Submissions close 15 December 2026",
        },
    },
    ONLINE: {
        "title": "Async Python Online 2026 - CfP open",
        "snippet": "A free, online-only conference about asyncio. Submit a proposal by October 30.",
        "record": {
            "conference_name": "Async Python Online 2026", "cfp_url": ONLINE, "cfp_deadline": "2026-10-30",
            "event_start": "2026-12-05", "is_online": True, "topics": ["asyncio", "Python"],
            "confidence": 0.8, "evidence": "CfP deadline: October 30, 2026",
        },
    },
    CLOSED: {
        "title": "Concurrency Days 2027 call for papers",
        "snippet": "The call for papers is now closed. Thank you for your submissions.",
        "record": {
            "conference_name": "Concurrency Days 2027", "cfp_url": CLOSED, "cfp_deadline": "2026-11-20",
            "event_start": "2027-04-01", "city": "Berlin", "country": "Germany", "is_cfp_open": False,
            "topics": ["concurrency"], "confidence": 0.9, "evidence": "The call for papers is now closed",
        },
    },
    PYCONF_DUP: {
        "title": "PyConf Hyderabad call for proposals",
        "snippet": "Proposals due 30 November 2026. PyConf Hyderabad, 15 Feb 2027.",
        "record": {
            "conference_name": "PyConf Hyderabad", "cfp_url": PYCONF_DUP, "cfp_deadline": "2026-11-30",
            "event_start": "2027-02-15", "event_end": "2027-02-16", "city": "Hyderabad", "country": "India",
            "topics": ["Python", "AsyncIO"], "confidence": 0.7, "evidence": "Proposals due 30 November 2026",
        },
    },
}
# Hits the cheap pre-filter must drop before any LLM call.
JUNK_HITS = [
    {"title": "Python threads tutorial", "link": "https://pytutorials.example.com/threads",
     "snippet": "Learn threading and asyncio basics."},
    {"title": "Call for papers (PDF)", "link": "https://example.org/files/cfp-2027.pdf",
     "snippet": "Call for papers, Python conference 2027."},
]

AIRPORTS = {"Hyderabad": ("/m/09f07", "HYD"), "Singapore": ("/m/06t2t", "SIN")}
FLIGHT_PRICES = {"HYD": 9_000, "SIN": 32_000}
HOTEL_RATES = {"Hyderabad": [3_000, 5_000, None, 4_000], "Singapore": [9_000, 11_000, 10_000]}
FIT = {"PyConf": 0.9, "AI Agents": 0.8, "Async Python": 0.7}
TRAVEL_STATUS = {"PyConf": "grant_available", "AI Agents": "covered"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def serp_answer(params: dict[str, Any]) -> dict[str, Any]:
    """What SerpApi would answer, by engine."""
    engine = params.get("engine")
    q = str(params.get("q", ""))
    if engine == "google":
        if "travel grant" in q:
            name = q.split('"')[1]
            return {"organic_results": [
                {"position": 1, "title": f"{name}: speaker travel and financial aid",
                 "link": f"https://{_slug(name)}.example.org/travel-policy",
                 "snippet": f"{name} helps speakers with travel."},
                {"position": 2, "title": "Unrelated travel blog", "link": "https://blog.example.com/travel",
                 "snippet": "Ten tips for cheap flights."},
            ]}
        if "call for" in q or "site:" in q:
            hits = [{"position": i, "title": p["title"], "link": url, "snippet": p["snippet"]}
                    for i, (url, p) in enumerate(PAGES.items(), start=1)]
            return {"organic_results": hits + JUNK_HITS}
        raise AssertionError(f"unexpected google query: {q}")
    if engine == "google_flights_autocomplete":
        city = q.split(",")[0].strip()
        kgmid, code = AIRPORTS[city]
        return {"suggestions": [
            {"position": 1, "name": city, "type": "city", "description": f"City in {city}", "id": kgmid,
             "airports": [{"name": f"{city} International Airport", "id": code, "city": city, "city_id": kgmid}]},
        ]}
    if engine == "google_flights":
        assert params["departure_id"] == "COK"
        return {"price_insights": {"lowest_price": FLIGHT_PRICES[params["arrival_id"]]},
                "best_flights": [{"price": FLIGHT_PRICES[params["arrival_id"]] + 500}]}
    if engine == "google_hotels":
        city = next(c for c in HOTEL_RATES if c in q)
        return {"properties": [
            {"name": f"Hotel {i}", **({"rate_per_night": {"extracted_lowest": rate}} if rate else {})}
            for i, rate in enumerate(HOTEL_RATES[city])
        ]}
    raise AssertionError(f"unexpected engine: {engine}")


def http_error(status: int, message: str) -> serpapi.HTTPError:
    """A serpapi.HTTPError as the real client raises it for a 4xx/5xx answer."""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"error": message}).encode()
    response.headers["Content-Type"] = "application/json"
    response.url = "https://serpapi.com/search?engine=google&api_key=SECRET_TEST_KEY"
    original = requests.exceptions.HTTPError(f"{status} Client Error for url: {response.url}", response=response)
    return serpapi.HTTPError(original)


class FakeSerp:
    """Shared call log for every fake client the factory hands out (one client per search)."""

    def __init__(self, fail_after: int | None = None):
        self.calls: list[dict[str, Any]] = []
        self.fail_after = fail_after
        self._lock = threading.Lock()

    def factory(self) -> FakeSerpClient:
        return FakeSerpClient(self)

    @property
    def count(self) -> int:
        return len(self.calls)

    def engines(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for call in self.calls:
            out[call["engine"]] = out.get(call["engine"], 0) + 1
        return out


class FakeSerpClient:
    def __init__(self, owner: FakeSerp):
        self.owner = owner

    def search(self, params: dict[str, Any]) -> dict[str, Any]:
        assert "api_key" not in params or params["api_key"] != "SECRET_TEST_KEY"
        with self.owner._lock:
            self.owner.calls.append(dict(params))
            n = len(self.owner.calls)
        if self.owner.fail_after is not None and n > self.owner.fail_after:
            raise http_error(429, "Too many requests: hourly searches limit reached.")
        return serp_answer(params)


# --- Fake LLM: one FunctionModel that dispatches on the output schema ----------------------

def last_user_prompt(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    raise AssertionError("no user prompt")


def _lookup(table: dict[str, Any], name: str, default: Any) -> Any:
    return next((value for key, value in table.items() if key in name), default)


class FakeLLM:
    def __init__(self, pages: dict[str, dict[str, Any]] | None = None):
        self.pages = pages or PAGES
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def model(self) -> FunctionModel:
        return FunctionModel(self)

    def __call__(self, messages: list[Any], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        properties = tool.parameters_json_schema.get("properties", {})
        prompt = last_user_prompt(messages)
        if "cfps" in properties:
            kind = "extract"
            url = re.search(r"^URL: (\S+)$", prompt, re.MULTILINE).group(1)
            args = {"cfps": [self.pages[url]["record"]]}
        elif "scores" in properties:
            kind = "fit"
            items = re.findall(r"^(\d+)\. (.+?) \| topics:", prompt, re.MULTILINE)
            assert items, prompt
            args = {"scores": [
                {"id": int(i), "fit": _lookup(FIT, name, 0.3), "reason": f"{name} matches your topics"}
                for i, name in items
            ]}
        elif "status" in properties:
            kind = "travel"
            name = re.search(r"^Conference: (.+)$", prompt, re.MULTILINE).group(1)
            link = re.search(r"^Link: (\S+)$", prompt, re.MULTILINE).group(1)
            status = _lookup(TRAVEL_STATUS, name, "unknown")
            args = {"status": status, "evidence_url": link if status != "unknown" else None,
                    "note": f"{name}: {status.replace('_', ' ')}."}
        elif "terms" in properties:
            kind = "plan"  # search planner: keep each topic as its own search term
            args = {"terms": re.findall(r"^- (.+)$", prompt, re.MULTILINE)}
        else:
            raise AssertionError(f"unexpected output schema: {sorted(properties)}")
        with self._lock:
            self.calls.append(kind)
        return ModelResponse(parts=[ToolCallPart(tool.name, json.dumps(args))])


# --- Fixtures ----------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """No network, no real LLM: page fetches return None, real HTTP clients refuse to send."""
    fetched: list[str] = []

    def fake_fetch(url: str, **_kwargs: Any) -> None:
        fetched.append(url)
        return None

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(extract, "fetch_page_text", fake_fetch)
    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
    monkeypatch.setattr(requests.Session, "send", refuse)
    monkeypatch.setattr(pai_models, "ALLOW_MODEL_REQUESTS", False)
    return fetched


def make_profile() -> SpeakerProfile:
    return SpeakerProfile(
        topics="Python concurrency, AI agents", home_airport="COK", budget_inr=150_000,
        window_start=date(2026, 11, 12), window_end=date(2027, 10, 12),
    )


class ProgressLog:
    def __init__(self) -> None:
        self.events: list[tuple[Progress, int]] = []

    def __call__(self, p: Progress) -> None:
        self.events.append((p, threading.get_ident()))


def names(result: Any) -> list[str]:
    return [row.cfp.conference_name for row in result.ranked]


# --- The end-to-end run --------------------------------------------------------------------------

def test_full_run_then_rerun_costs_nothing(ctx: RunContext, cache: ResponseCache, offline: list[str]) -> None:
    serp, llm, progress = FakeSerp(), FakeLLM(), ProgressLog()
    ctx.serp_client_factory = serp.factory
    ctx.llm_model = llm.model()

    result = run_pipeline(make_profile(), ctx, progress=progress)
    stats = result.stats

    assert stats.errors == []
    # Credits: 8 discovery + 3 travel policy + 2 airports + 2 flights + 2 hotels (the online event is free).
    assert stats.serpapi_searches == serp.count == 17
    assert serp.engines() == {"google": 11, "google_flights_autocomplete": 2, "google_flights": 2, "google_hotels": 2}
    assert stats.serpapi_cache_hits == 0
    # LLM: 1 search plan, 5 extractions (junk + PDF pre-filtered), 1 fit call, 3 travel classifications.
    assert sorted(llm.calls) == ["extract"] * 5 + ["fit"] + ["plan"] + ["travel"] * 3
    assert stats.llm_calls == 10
    assert stats.queries == 8 and stats.candidates == 7 and stats.candidates_after_prefilter == 5
    assert stats.extracted == 5 and stats.valid == 3 and stats.shortlisted == 3 and stats.priced == 3
    assert AICONF in offline and PYCONF not in offline  # page fetched only for thin snippets

    # Ranked rows: sorted by final score, numbered 1..n, 2-4 reasons each.
    rows = result.ranked
    assert [r.rank for r in rows] == list(range(1, len(rows) + 1))
    scores = [r.final_score for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert all(2 <= len(r.reasons) <= 4 for r in rows)
    by_name = {r.cfp.conference_name: r for r in rows}

    # Closed CFP dropped; the PyConf duplicate merged into the higher-confidence record.
    assert set(by_name) == {"PyConf Hyderabad 2027", "AI Agents Summit 2027", "Async Python Online 2026"}
    assert not result.unpriced
    pyconf = by_name["PyConf Hyderabad 2027"]
    assert pyconf.cfp.source_url == PYCONF
    assert "AsyncIO" in pyconf.cfp.topics  # filled in from the duplicate

    # Priced trips: flight + hotel (median nightly x nights) + total, in INR.
    assert pyconf.trip.status == "ok" and pyconf.trip.airport_codes == ["HYD"]
    assert (pyconf.trip.flight_inr, pyconf.trip.nights, pyconf.trip.hotel_nightly_inr) == (9_000, 3, 4_000)
    assert (pyconf.trip.hotel_inr, pyconf.trip.total_inr) == (12_000, 21_000)
    assert pyconf.travel.status == "grant_available" and pyconf.net_cost_inr == 21_000
    ai = by_name["AI Agents Summit 2027"]
    assert (ai.trip.flight_inr, ai.trip.hotel_inr, ai.trip.total_inr) == (32_000, 40_000, 72_000)
    assert ai.travel.status == "covered" and ai.net_cost_inr == 0
    assert ai.travel.evidence_url == "https://ai-agents-summit.example.org/travel-policy"
    for row in rows:
        if row.trip.status == "ok":
            assert None not in (row.trip.flight_inr, row.trip.hotel_inr, row.trip.total_inr)
    online = by_name["Async Python Online 2026"]
    assert online.cfp.is_online and online.net_cost_inr == 0 and online.trip.total_inr == 0
    assert not any(r.over_budget for r in rows)
    assert result.prices_fetched_at is not None

    # Progress: every step 1..10 reported done, all callbacks on the calling thread.
    main_thread = threading.get_ident()
    done_steps = {p.step for p, _ in progress.events if p.done}
    assert done_steps == set(range(1, 11))
    assert all(p.total == 10 for p, _ in progress.events)
    assert {ident for _, ident in progress.events} == {main_thread}

    # Re-run with a new context on the same cache: zero SerpApi credits, zero LLM calls, same ranking.
    serp2, llm2 = FakeSerp(), FakeLLM()
    ctx2 = RunContext(cache=cache, today=ctx.today, serp_client_factory=serp2.factory, llm_model=llm2.model())
    again = run_pipeline(make_profile(), ctx2)
    assert serp2.count == 0 and again.stats.serpapi_searches == 0
    assert again.stats.serpapi_cache_hits == 17
    assert llm2.calls == [] and again.stats.llm_calls == 0 and again.stats.llm_cache_hits == 10
    assert names(again) == names(result)
    assert [r.final_score for r in again.ranked] == scores
    assert [r.net_cost_inr for r in again.ranked] == [r.net_cost_inr for r in rows]
    assert again.stats.errors == []


def test_rerun_is_free_whatever_order_extractions_finish(
    ctx: RunContext, cache: ResponseCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extractions finish in thread-completion order. With two equally confident duplicates, the
    merged record (name, venue) must not depend on that order, or the fit/hotel cache keys change
    and a re-run spends credits and LLM calls."""
    pages = {url: {**page, "record": dict(page["record"])} for url, page in PAGES.items()}
    pages[PYCONF_DUP]["record"].update(confidence=0.9, venue="HICC")  # a tie with PYCONF
    monkeypatch.setitem(PAGES, PYCONF_DUP, pages[PYCONF_DUP])

    real_extract_all = pipeline.extract_all
    order = {"reverse": False}

    def extract_in_order(*args: Any, **kwargs: Any) -> list[CFP]:
        cfps = sorted(real_extract_all(*args, **kwargs), key=lambda c: c.source_url, reverse=order["reverse"])
        return cfps

    monkeypatch.setattr(pipeline, "extract_all", extract_in_order)

    serp = FakeSerp()
    ctx.serp_client_factory, ctx.llm_model = serp.factory, FakeLLM(pages).model()
    first = run_pipeline(make_profile(), ctx)
    assert first.stats.errors == [] and first.stats.valid == 3

    order["reverse"] = True
    serp2, llm2 = FakeSerp(), FakeLLM(pages)
    ctx2 = RunContext(cache=cache, today=ctx.today, serp_client_factory=serp2.factory, llm_model=llm2.model())
    second = run_pipeline(make_profile(), ctx2)
    assert serp2.calls == [] and second.stats.serpapi_searches == 0
    assert llm2.calls == [] and second.stats.llm_calls == 0
    assert names(second) == names(first)


def test_cached_only_on_empty_cache_spends_nothing(ctx: RunContext) -> None:
    serp, llm = FakeSerp(), FakeLLM()
    ctx.serp_client_factory, ctx.llm_model, ctx.cached_only = serp.factory, llm.model(), True
    progress = ProgressLog()

    result = run_pipeline(make_profile(), ctx, progress=progress)

    assert serp.count == 0 and llm.calls == []
    assert result.ranked == [] and result.stats.serpapi_searches == 0 and result.stats.cached_only
    assert {p.step for p, _ in progress.events if p.done} == set(range(1, 11))


def test_rate_limit_mid_run_switches_to_cached_only(ctx: RunContext) -> None:
    serp, llm = FakeSerp(fail_after=3), FakeLLM()
    ctx.serp_client_factory, ctx.llm_model = serp.factory, llm.model()

    result = run_pipeline(make_profile(), ctx)  # must not raise

    stats = result.stats
    assert stats.api_disabled_reason and "Too many requests" in stats.api_disabled_reason
    assert stats.serpapi_searches == 3
    assert 4 <= serp.count <= 8  # only discovery queries reached the client; later lookups stayed offline
    assert all(call["engine"] == "google" for call in serp.calls)
    assert stats.errors and not any("SECRET_TEST_KEY" in e for e in stats.errors)
    for row in result.ranked:  # nothing could be priced, but the rows are still ranked
        assert row.cfp.is_online or row.trip.status == "unavailable"
        assert 2 <= len(row.reasons) <= 4


# --- CLI -----------------------------------------------------------------------------------------

@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    cache_path = tmp_path / "cli-cache.sqlite3"
    monkeypatch.setenv("CFP_COMPASS_CACHE", str(cache_path))
    return cache_path


def test_cli_run_cached_only(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--topics", "Python", "--cached-only"]) == 0
    out = capsys.readouterr()
    assert "SerpApi searches used this run: 0" in out.out
    assert "[10/10] Rank" in out.err
    assert cli_env.exists()


@pytest.mark.parametrize(
    ("extra", "expected"),
    [(["--topics", ""], "talk topic"), (["--topics", "Python", "--home", "12"], "3-letter IATA")],
)
def test_cli_run_rejects_bad_input(
    cli_env: Path, capsys: pytest.CaptureFixture[str], extra: list[str], expected: str
) -> None:
    assert main(["run", *extra, "--cached-only"]) == 2
    err = capsys.readouterr().err
    assert "Invalid input" in err and expected in err
    assert "Traceback" not in err


def test_cli_search_cached_only_miss(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "python cfp", "--cached-only"]) in (0, 1)
    out = capsys.readouterr()
    assert "SerpApi searches used: 0" in out.out
