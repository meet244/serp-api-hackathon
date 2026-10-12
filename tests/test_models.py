"""Input validation and model defaults."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from cfp_compass.models import CFP, SpeakerProfile, describe_validation_error


def _profile(**kw) -> SpeakerProfile:
    defaults = dict(
        topics="Python concurrency, AI agents",
        home_airport="COK",
        budget_inr=150_000,
        window_start=date(2026, 11, 12),
        window_end=date(2027, 10, 12),
    )
    defaults.update(kw)
    return SpeakerProfile(**defaults)


def _errors(**kw) -> list[str]:
    with pytest.raises(ValidationError) as info:
        _profile(**kw)
    return describe_validation_error(info.value)


# --- SpeakerProfile: accepted input ------------------------------------------------------------

def test_topics_split_and_trimmed():
    profile = _profile()
    assert profile.topics == ["Python concurrency", "AI agents"]
    assert profile.max_conferences == 12
    assert profile.talk_abstract is None


def test_topics_whitespace_and_case_insensitive_dedupe():
    assert _profile(topics="  Python   concurrency ,  python CONCURRENCY, ").topics == ["Python concurrency"]
    assert _profile(topics="Python, python, AI agents").topics == ["Python", "AI agents"]
    assert _profile(topics=["Rust", " rust ", "Go"]).topics == ["Rust", "Go"]


def test_airport_upper_cased():
    assert _profile(home_airport="cok").home_airport == "COK"
    assert _profile(home_airport=" blr ").home_airport == "BLR"


def test_blank_abstract_becomes_none():
    assert _profile(talk_abstract="   ").talk_abstract is None
    assert _profile(talk_abstract="  Async IO in practice. ").talk_abstract == "Async IO in practice."


# --- SpeakerProfile: rejected input --------------------------------------------------------------

@pytest.mark.parametrize("topics", ["", " , ,", []])
def test_rejects_zero_topics(topics):
    messages = _errors(topics=topics)
    assert any("at least one talk topic" in m for m in messages)


def test_rejects_three_topics():
    messages = _errors(topics="Python, AI agents, Rust")
    assert any("at most 2 talk topics" in m for m in messages)


@pytest.mark.parametrize("airport", ["KOCHI", "12", "CO", "C0K", ""])
def test_rejects_bad_airport(airport):
    messages = _errors(home_airport=airport)
    assert "Home airport must be a 3-letter IATA code, for example COK." in messages


@pytest.mark.parametrize("budget", [0, -5])
def test_rejects_non_positive_budget(budget):
    assert "Budget must be greater than 0." in _errors(budget_inr=budget)


@pytest.mark.parametrize(
    ("start", "end"),
    [(date(2027, 1, 1), date(2026, 12, 1)), (date(2027, 1, 1), date(2027, 1, 1))],
)
def test_rejects_reversed_or_equal_window(start, end):
    assert _errors(window_start=start, window_end=end) == ["The travel window must start before it ends."]


@pytest.mark.parametrize("value", [0, 26])
def test_rejects_max_conferences_out_of_range(value):
    messages = _errors(max_conferences=value)
    assert any("between 1 and 25" in m for m in messages)


def test_describe_validation_error_is_friendly():
    messages = _errors(topics="", home_airport="KOCHI", budget_inr=0)
    assert len(messages) == 3
    assert all("Value error" not in m for m in messages)
    assert "Talk topics: Enter at least one talk topic." in messages
    assert "Home airport must be a 3-letter IATA code, for example COK." in messages
    assert "Budget must be greater than 0." in messages


def test_describe_validation_error_labels_type_errors():
    messages = _errors(budget_inr="lots")
    assert len(messages) == 1
    assert messages[0].startswith("Budget")
    assert "Value error" not in messages[0]


# --- CFP ----------------------------------------------------------------------------------------

def test_event_end_defaults_to_start():
    cfp = CFP(conference_name="PyCon US 2027", event_start=date(2027, 5, 14))
    assert cfp.event_end == date(2027, 5, 14)


def test_event_end_before_start_is_fixed():
    cfp = CFP(conference_name="X", event_start=date(2027, 5, 14), event_end=date(2027, 5, 1))
    assert cfp.event_end == date(2027, 5, 14)


def test_event_end_after_start_is_kept():
    cfp = CFP(conference_name="X", event_start=date(2027, 5, 14), event_end=date(2027, 5, 22))
    assert cfp.event_end == date(2027, 5, 22)


def test_no_event_dates():
    cfp = CFP(conference_name="X")
    assert cfp.event_start is None and cfp.event_end is None
    assert cfp.is_cfp_open is True
    assert cfp.confidence == 0.0
    assert cfp.topics == []


def test_cfp_parses_iso_date_strings():
    cfp = CFP.model_validate(
        {"conference_name": "X", "cfp_deadline": "2026-12-15", "event_start": "2027-05-14", "confidence": 0.8}
    )
    assert cfp.cfp_deadline == date(2026, 12, 15)
    assert cfp.event_end == date(2027, 5, 14)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_bounds(confidence):
    with pytest.raises(ValidationError):
        CFP(conference_name="X", confidence=confidence)


def test_source_url_hidden_from_llm_schema():
    schema = CFP.model_json_schema()
    assert "source_url" not in schema["properties"]
    assert "source_url" not in json.dumps(schema)
    assert "conference_name" in schema["properties"]
    assert schema["required"] == ["conference_name"]
    # but the app can still set it
    assert CFP(conference_name="X", source_url="https://hit.example").source_url == "https://hit.example"
