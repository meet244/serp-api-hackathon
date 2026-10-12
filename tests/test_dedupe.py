"""Validation and de-duplication of extracted CFPs."""

from __future__ import annotations

from datetime import date

import pytest

from cfp_compass.dedupe import dedupe_key, drop_reason, event_year, keep_valid, merge_duplicates, normalize_name
from cfp_compass.models import CFP, SpeakerProfile

TODAY = date(2026, 10, 12)
PROFILE = SpeakerProfile(
    topics="Python concurrency, AI agents",
    home_airport="COK",
    budget_inr=150_000,
    window_start=date(2026, 11, 12),
    window_end=date(2027, 10, 12),
)


def _good(**kw) -> CFP:
    defaults = dict(
        conference_name="PyCon US 2027",
        cfp_url="https://us.pycon.example/cfp",
        cfp_deadline=date(2026, 12, 15),
        event_start=date(2027, 5, 14),
        event_end=date(2027, 5, 22),
        city="Long Beach",
        country="USA",
        confidence=0.9,
    )
    defaults.update(kw)
    return CFP(**defaults)


# --- normalize_name / dedupe_key ------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("PyCon US 2027", "pycon us"),
        ("pycon us", "pycon us"),
        ("  PyCon   US  ", "pycon us"),
        ("PyCon US 2027 - Call for Proposals", "pycon us"),
        ("Call for Papers: RustConf 2026!", "rustconf"),
        ("DjangoCon Europe '27 CFP", "djangocon europe"),
        ("PyData London 2027 | Sessionize", "pydata london"),
        ("EuroPython 2027 (Call for Speakers)", "europython"),
        ("JSConf.Asia 2026/2027", "jsconf asia"),
        ("Open Source Summit: pretalx", "open source summit"),
    ],
)
def test_normalize_name(raw, normalized):
    assert normalize_name(raw) == normalized


def test_normalize_name_keeps_numbers_that_are_not_years():
    assert normalize_name("Web3 Summit") == "web3 summit"
    assert normalize_name("C++ Now 2027") == "c now"


def test_pycon_variants_share_a_dedupe_key():
    a = CFP(conference_name="PyCon US 2027", event_start=date(2027, 5, 14), confidence=0.9)
    b = CFP(conference_name="pycon us", event_start=date(2027, 5, 16), confidence=0.6)
    assert dedupe_key(a) == dedupe_key(b) == "pycon us|2027"


def test_different_event_years_do_not_merge():
    a = CFP(conference_name="PyCon US", event_start=date(2026, 5, 14), confidence=0.9)
    b = CFP(conference_name="PyCon US", event_start=date(2027, 5, 14), confidence=0.9)
    assert dedupe_key(a) != dedupe_key(b)


def test_event_year_falls_back_to_name():
    assert event_year(CFP(conference_name="PyCon US 2027")) == 2027
    assert event_year(CFP(conference_name="PyCon US", event_start=date(2027, 1, 1))) == 2027
    assert event_year(CFP(conference_name="PyCon US")) is None
    assert dedupe_key(CFP(conference_name="PyCon US")) == "pycon us|"


# --- merge_duplicates ------------------------------------------------------------------------

def test_merge_keeps_higher_confidence_and_fills_missing_fields():
    strong = CFP(
        conference_name="PyCon US 2027",
        cfp_url="https://us.pycon.example/2027/speaking/",
        event_start=date(2027, 5, 14),
        event_end=date(2027, 5, 22),
        topics=["python"],
        confidence=0.9,
    )
    weak = CFP(
        conference_name="pycon us",
        cfp_url="https://sessionize.example/pycon-us",
        cfp_deadline=date(2026, 12, 18),
        deadline_is_aoe=True,
        evidence="Proposals close December 18, 2026 AoE",
        event_start=date(2027, 5, 15),
        city="Long Beach",
        country="USA",
        venue="Long Beach Convention Center",
        topics=["python", "web"],
        confidence=0.6,
    )
    for order in ([strong, weak], [weak, strong]):
        merged = merge_duplicates(order)
        assert list(merged) == ["pycon us|2027"]
        record = merged["pycon us|2027"]
        # identity and own fields from the higher-confidence record
        assert record.conference_name == "PyCon US 2027"
        assert record.confidence == 0.9
        assert record.cfp_url == "https://us.pycon.example/2027/speaking/"
        assert record.event_start == date(2027, 5, 14)
        assert record.event_end == date(2027, 5, 22)
        # missing fields borrowed from the other record, as consistent groups
        assert record.cfp_deadline == date(2026, 12, 18)
        assert record.deadline_is_aoe is True
        assert record.evidence == "Proposals close December 18, 2026 AoE"
        assert record.city == "Long Beach"
        assert record.country == "USA"
        assert record.venue == "Long Beach Convention Center"
        assert record.topics == ["python", "web"]


def test_merge_does_not_overwrite_present_fields():
    strong = _good(confidence=0.95)
    weak = _good(conference_name="pycon us", cfp_deadline=date(2027, 1, 30), city="Pittsburgh", confidence=0.7)
    record = merge_duplicates([weak, strong])["pycon us|2027"]
    assert record.cfp_deadline == date(2026, 12, 15)
    assert record.city == "Long Beach"


def test_merge_fills_cfp_url_when_missing():
    strong = _good(cfp_url="", confidence=0.9)
    weak = _good(conference_name="pycon us", cfp_url="https://other.example/cfp", confidence=0.6)
    assert merge_duplicates([strong, weak])["pycon us|2027"].cfp_url == "https://other.example/cfp"


def test_merge_equal_confidence_keeps_first():
    first = _good(conference_name="PyCon US 2027", cfp_url="https://first.example", confidence=0.8)
    second = _good(conference_name="PYCON US", cfp_url="https://second.example", confidence=0.8)
    record = merge_duplicates([first, second])["pycon us|2027"]
    assert record.cfp_url == "https://first.example"


def test_merge_keeps_distinct_conferences():
    records = [
        _good(),
        _good(conference_name="DjangoCon US 2027", event_start=date(2027, 9, 1), event_end=None),
        _good(conference_name="PyCon US 2026", event_start=date(2026, 11, 20), event_end=None),
    ]
    merged = merge_duplicates(records)
    assert set(merged) == {"pycon us|2027", "djangocon us|2027", "pycon us|2026"}


def test_merge_does_not_mutate_inputs():
    strong = _good(cfp_deadline=None, evidence="")
    weak = _good(conference_name="pycon us", confidence=0.6, evidence="Closes Dec 15")
    merge_duplicates([strong, weak])
    assert strong.cfp_deadline is None
    assert strong.evidence == ""


# --- drop_reason / keep_valid ------------------------------------------------------------------

def test_good_record_is_kept():
    assert drop_reason(_good(), PROFILE, TODAY) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"is_cfp_open": False}, "CFP closed"),
        ({"confidence": 0.49}, "Low extraction confidence"),
        ({"confidence": 0.0}, "Low extraction confidence"),
        ({"cfp_deadline": date(2026, 10, 11)}, "Deadline has passed"),
        ({"city": None}, "No city and not online"),
        ({"city": ""}, "No city and not online"),
        (
            {"cfp_deadline": None, "event_start": date(2026, 10, 1), "event_end": date(2026, 10, 5)},
            "Event is over",
        ),
        ({"event_start": date(2027, 12, 1), "event_end": None}, "Starts outside your travel window"),
        ({"event_start": date(2026, 11, 1), "event_end": None}, "Starts outside your travel window"),
    ],
)
def test_drop_reasons(overrides, reason):
    assert drop_reason(_good(**overrides), PROFILE, TODAY) == reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 0.5},  # threshold is inclusive
        {"cfp_deadline": TODAY},  # closes today: still open
        {"cfp_deadline": None},  # unknown deadline is kept
        {"city": None, "is_online": True},  # online events need no city
        {"event_start": None, "event_end": None},  # unknown dates kept (trip unavailable)
        {"event_start": PROFILE.window_start, "event_end": None},  # window bounds inclusive
        {"event_start": PROFILE.window_end, "event_end": None},
    ],
)
def test_records_kept(overrides):
    assert drop_reason(_good(**overrides), PROFILE, TODAY) is None


def test_keep_valid_counts_reasons():
    records = [
        _good(),
        _good(conference_name="Closed Conf", is_cfp_open=False),
        _good(conference_name="Also Closed", is_cfp_open=False),
        _good(conference_name="Vague", confidence=0.2),
        _good(conference_name="Late", cfp_deadline=date(2026, 9, 1)),
        _good(conference_name="Nowhere", city=None),
        _good(conference_name="Online Conf", city=None, is_online=True),
        _good(conference_name="Far Future", event_start=date(2028, 3, 1), event_end=None),
    ]
    kept, dropped = keep_valid(records, PROFILE, TODAY)
    assert [c.conference_name for c in kept] == ["PyCon US 2027", "Online Conf"]
    assert dropped == {
        "CFP closed": 2,
        "Low extraction confidence": 1,
        "Deadline has passed": 1,
        "No city and not online": 1,
        "Starts outside your travel window": 1,
    }
    assert len(kept) + sum(dropped.values()) == len(records)


def test_keep_valid_empty():
    assert keep_valid([], PROFILE, TODAY) == ([], {})
