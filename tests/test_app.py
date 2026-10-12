"""Streamlit smoke tests with streamlit.testing.v1.AppTest (no server, no browser, no network).

The pipeline is replaced by a fake that reports progress for all ten steps and returns a canned
result, so these tests check the UI wiring only.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import models as pai_models
from streamlit.testing.v1 import AppTest

import cfp_compass.config as config
import cfp_compass.pipeline as pipeline
from cfp_compass.context import RunContext
from cfp_compass.models import CFP, PipelineResult, RunStats, ScoreWeights, SpeakerProfile, TravelSupport, TripCost
from cfp_compass.pipeline import Progress
from cfp_compass.scoring import score_cfp, sort_and_number

APP = Path(__file__).resolve().parent.parent / "app.py"
TIMEOUT = 60
LLM_ENV_VARS = (
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
    "MISTRAL_API_KEY", "CFP_COMPASS_MODEL",
)
COLUMNS = [
    "Rank", "Conference", "Deadline", "Event", "Fit", "Travel support", "Flight (₹)", "Hotel (₹)",
    "Net cost (₹)", "Score", "Why",
]


@pytest.fixture(autouse=True)
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Dummy keys, no .env file, a temp cache, and no real model requests."""
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "dummy-serpapi-key")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    cache_path = tmp_path / "app-cache.sqlite3"
    monkeypatch.setenv("CFP_COMPASS_CACHE", str(cache_path))
    monkeypatch.setattr(pai_models, "ALLOW_MODEL_REQUESTS", False)
    return cache_path


def canned_result(profile: SpeakerProfile, today: date) -> PipelineResult:
    """Three rows: one over budget, one with speaker travel covered, one online."""

    def cfp(name: str, slug: str, deadline_days: int, start_days: int, **extra: Any) -> CFP:
        url = f"https://{slug}.example.org/cfp"
        start = today + timedelta(days=start_days)
        return CFP(
            conference_name=name, cfp_url=url, source_url=url, cfp_deadline=today + timedelta(days=deadline_days),
            event_start=start, event_end=start + timedelta(days=1), confidence=0.9,
            evidence=f"CFP closes {today + timedelta(days=deadline_days):%d %B %Y}", **extra,
        )

    specs = [
        ("faraway pycon|2027", cfp("Faraway PyCon 2027", "faraway", 20, 120, city="Lisbon", country="Portugal",
                                   topics=["Python"]),
         TravelSupport(status="none", note="Speakers pay their own way."),
         TripCost(airport_codes=["LIS"], flight_inr=140_000, nights=3, hotel_nightly_inr=20_000,
                  hotel_inr=60_000, total_inr=200_000, status="ok"),
         0.9),
        ("ai agents summit|2027", cfp("AI Agents Summit 2027", "aiagents", 45, 150, city="Singapore",
                                      country="Singapore", topics=["AI agents"]),
         TravelSupport(status="covered", evidence_url="https://aiagents.example.org/travel",
                       note="Flights and hotel are paid for speakers."),
         TripCost(airport_codes=["SIN"], flight_inr=32_000, nights=3, hotel_nightly_inr=10_000,
                  hotel_inr=30_000, total_inr=62_000, status="ok"),
         0.8),
        ("async python online|2026", cfp("Async Python Online 2026", "asyncpy", 10, 60, is_online=True,
                                         topics=["asyncio"]),
         TravelSupport(),
         TripCost(flight_inr=0, hotel_nightly_inr=0, hotel_inr=0, total_inr=0, status="ok", reason="Online event"),
         0.7),
    ]
    rows = [
        score_cfp(key=key, cfp=c, travel=travel, trip=trip, fit=fit, fit_reason="matches your topics",
                  budget_inr=profile.budget_inr, today=today)
        for key, c, travel, trip, fit in specs
    ]
    return PipelineResult(
        profile=profile,
        ranked=sort_and_number(rows),
        stats=RunStats(serpapi_searches=17, serpapi_cache_hits=4, llm_calls=9, valid=5, shortlisted=3, priced=3),
        generated_at=datetime.now(),
        prices_fetched_at=datetime.now(),
    )


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[SpeakerProfile, RunContext]] = []

    def __call__(
        self,
        profile: SpeakerProfile,
        ctx: RunContext,
        *,
        weights: ScoreWeights | None = None,
        progress: Any = None,
    ) -> PipelineResult:
        self.calls.append((profile, ctx))
        for step in range(1, 11):
            progress(Progress(step, f"Step {step}", f"detail {step}"))
            progress(Progress(step, f"Step {step}", f"detail {step} done", done=True))
        return canned_result(profile, ctx.today)


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    fake = FakePipeline()
    # app.py runs "from cfp_compass.pipeline import ... run_pipeline" on every script run.
    monkeypatch.setattr(pipeline, "run_pipeline", fake)
    return fake


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT)


def by_label(elements: Any, label: str) -> Any:
    matches = [e for e in elements if e.label == label]
    assert matches, f"no element labelled {label!r}: {[e.label for e in elements]}"
    return matches[0]


def errors_of(at: AppTest) -> list[str]:
    return [e.value for e in at.exception]


def find_cfps(at: AppTest, topics: str = "Python concurrency, AI agents", airport: str | None = None) -> AppTest:
    by_label(at.sidebar.text_input, "Talk topics").set_value(topics)
    if airport is not None:
        by_label(at.sidebar.text_input, "Home airport (IATA)").set_value(airport)
    by_label(at.sidebar.button, "Find CFPs").click()
    return at.run()


def detail_expanders(at: AppTest) -> list[str]:
    return [e.label for e in at.expander if e.label[:1].isdigit() and ". " in e.label]


# --- Tests ----------------------------------------------------------------------------------------

def test_initial_render() -> None:
    at = new_app().run()
    assert not at.exception, errors_of(at)
    assert by_label(at.sidebar.text_input, "Talk topics").value == ""
    assert by_label(at.sidebar.text_input, "Home airport (IATA)").value == "COK"
    assert by_label(at.sidebar.button, "Find CFPs")
    assert by_label(at.sidebar.toggle, "Cached results only").value is False
    assert at.info and "Find CFPs" in at.info[0].value
    assert not at.error and not at.dataframe


def test_missing_llm_key_stops_with_setup_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    at = new_app().run()
    assert not at.exception, errors_of(at)
    assert at.error and ".env.example" in at.error[0].value
    assert not at.sidebar.button  # st.stop() before the sidebar is drawn


def test_missing_serpapi_key_stops_with_setup_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    at = new_app().run()
    assert not at.exception, errors_of(at)
    assert at.error and ".env.example" in at.error[0].value and "SERPAPI_API_KEY" in at.error[0].value


def test_find_cfps_shows_ranked_table(fake_pipeline: FakePipeline, app_env: Path) -> None:
    at = find_cfps(new_app().run())
    assert not at.exception, errors_of(at)
    assert not at.error, [e.value for e in at.error]

    # The app called the (patched) pipeline once, on the script thread, with the typed profile.
    assert len(fake_pipeline.calls) == 1
    profile, ctx = fake_pipeline.calls[0]
    assert profile.topics == ["Python concurrency", "AI agents"] and profile.home_airport == "COK"
    assert profile.budget_inr == 150_000 and profile.max_conferences == 12
    assert ctx.cache.path == app_env and not ctx.cached_only

    # Progress: all ten steps logged, and the status panel finished.
    assert len(at.session_state["run_log"]) == 10
    assert at.status and at.status[0].label.startswith("Done: 3 conferences ranked")

    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {
        "CFPs found": "5",
        "Conferences priced": "3",
        "Within budget": "2",
        "SerpApi searches used this run": "17",
        "Cache hits": "4",
    }

    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert list(df.columns) == COLUMNS
    assert len(df) == 3
    assert list(df["Rank"]) == [1, 2, 3]
    names = [str(link).rsplit("#", 1)[1].replace("+", " ") for link in df["Conference"]]
    assert set(names) == {"Faraway PyCon 2027", "AI Agents Summit 2027", "Async Python Online 2026"}
    net = dict(zip(names, df["Net cost (₹)"]))
    assert "Over budget" in net["Faraway PyCon 2027"]
    assert net["AI Agents Summit 2027"] == "₹0" and net["Async Python Online 2026"] == "₹0"
    flights = dict(zip(names, df["Flight (₹)"]))
    assert flights["Faraway PyCon 2027"] == "₹1,40,000" and flights["Async Python Online 2026"] == "₹0 (online)"
    assert all(2 <= len(why) <= 4 for why in df["Why"])

    downloads = at.get("download_button")
    labels = {d.proto.label for d in downloads}
    assert labels == {"Download CSV", "Download .ics calendar"}

    assert sorted(detail_expanders(at)) == sorted(f"{r}. {n}" for r, n in zip(df["Rank"], names))

    # Turning on "Within budget only" drops the over-budget row.
    by_label(at.toggle, "Within budget only").set_value(True)
    at.run()
    assert not at.exception, errors_of(at)
    filtered = at.dataframe[0].value
    assert len(filtered) == 2
    assert all("Over budget" not in text for text in filtered["Net cost (₹)"])
    assert len(detail_expanders(at)) == 2
    assert len(fake_pipeline.calls) == 1  # filtering re-renders from session state, no new run

    # Both filters: within budget AND deadline in the next 30 days -> only the online event.
    by_label(at.toggle, "Deadline in the next 30 days").set_value(True)
    at.run()
    assert not at.exception, errors_of(at)
    assert len(at.dataframe[0].value) == 1


def test_invalid_airport_shows_error(fake_pipeline: FakePipeline) -> None:
    at = find_cfps(new_app().run(), airport="12")
    assert not at.exception, errors_of(at)
    assert any("3-letter IATA" in e.value for e in at.error), [e.value for e in at.error]
    assert fake_pipeline.calls == []


def test_empty_topics_shows_error(fake_pipeline: FakePipeline) -> None:
    at = find_cfps(new_app().run(), topics="  ")
    assert not at.exception, errors_of(at)
    assert any("topic" in e.value.lower() for e in at.error), [e.value for e in at.error]
    assert fake_pipeline.calls == []
