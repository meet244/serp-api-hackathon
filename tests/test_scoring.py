"""Scoring and ranking rules (pure functions, no I/O)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from cfp_compass.models import CFP, RankedCFP, ScoreWeights, TravelSupport, TripCost
from cfp_compass.scoring import (
    AFFORDABILITY_UNKNOWN,
    GRANT_BONUS,
    URGENCY_UNKNOWN,
    affordability_score,
    build_reasons,
    days_until,
    final_score,
    format_inr,
    keyword_fit,
    net_cost_inr,
    rescore,
    score_cfp,
    sort_and_number,
    travel_bonus,
    urgency_score,
)

TODAY = date(2026, 10, 12)
BUDGET = 150_000


# --- urgency ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (None, 0.3),
        (-1, 0.2),
        (0, 0.2),
        (1, 0.2),
        (2, 0.6),
        (6, 0.6),
        (7, 1.0),
        (30, 1.0),
        (31, 0.7),
        (60, 0.7),
        (61, 0.4),
        (365, 0.4),
    ],
)
def test_urgency_buckets_and_boundaries(days, expected):
    assert urgency_score(days) == pytest.approx(expected)


def test_urgency_unknown_constant():
    assert URGENCY_UNKNOWN == 0.3


def test_days_until():
    assert days_until(None, TODAY) is None
    assert days_until(TODAY, TODAY) == 0
    assert days_until(TODAY + timedelta(days=9), TODAY) == 9
    assert days_until(TODAY - timedelta(days=2), TODAY) == -2


# --- affordability -----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("net", "expected"),
    [
        (0, 1.0),
        (BUDGET // 2, 0.5),
        (BUDGET, 0.0),
        (BUDGET + 1, 0.0),
        (BUDGET * 3, 0.0),
        (None, 0.4),
    ],
)
def test_affordability_clamping(net, expected):
    assert affordability_score(net, BUDGET) == pytest.approx(expected)


def test_affordability_always_in_unit_interval():
    for net in (0, 1, 10_000, 75_000, 149_999, 150_000, 150_001, 10**7):
        assert 0.0 <= affordability_score(net, BUDGET) <= 1.0
    assert AFFORDABILITY_UNKNOWN == 0.4


# --- net cost per travel status ------------------------------------------------------------

@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("covered", 0),
        ("partial", 60_000),
        ("grant_available", 120_000),
        ("none", 120_000),
        ("unknown", 120_000),
    ],
)
def test_net_cost_for_every_travel_status(status, expected):
    assert net_cost_inr(120_000, status) == expected


@pytest.mark.parametrize("status", ["covered", "partial", "grant_available", "none", "unknown"])
def test_online_event_costs_nothing(status):
    assert net_cost_inr(120_000, status, is_online=True) == 0
    assert net_cost_inr(None, status, is_online=True) == 0


def test_covered_with_unknown_total_is_zero():
    assert net_cost_inr(None, "covered") == 0


@pytest.mark.parametrize("status", ["partial", "grant_available", "none", "unknown"])
def test_unknown_total_is_unknown_net(status):
    assert net_cost_inr(None, status) is None


def test_partial_rounds_to_whole_rupees():
    result = net_cost_inr(12_347, "partial")  # 6173.5
    assert isinstance(result, int)
    assert result == 6_174
    assert net_cost_inr(99_999, "partial") in (49_999, 50_000)
    assert isinstance(net_cost_inr(99_999, "partial"), int)


def test_travel_bonus():
    assert travel_bonus("grant_available") == pytest.approx(GRANT_BONUS) == pytest.approx(0.05)
    for status in ("covered", "partial", "none", "unknown"):
        assert travel_bonus(status) == 0.0


# --- final score and weights -----------------------------------------------------------------

def test_final_score_default_weights():
    assert final_score(1.0, 0.0, 0.0) == pytest.approx(0.45)
    assert final_score(0.0, 1.0, 0.0) == pytest.approx(0.35)
    assert final_score(0.0, 0.0, 1.0) == pytest.approx(0.20)
    assert final_score(0.8, 0.5, 1.0) == pytest.approx(0.45 * 0.8 + 0.35 * 0.5 + 0.20 * 1.0)


def test_final_score_grant_bonus_added():
    base = final_score(0.5, 0.5, 0.5)
    assert final_score(0.5, 0.5, 0.5, bonus=0.05) == pytest.approx(base + 0.05)


def test_final_score_clamped_to_unit_interval():
    assert final_score(1.0, 1.0, 1.0, bonus=0.05) == pytest.approx(1.0)
    assert final_score(0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert final_score(0.0, 0.0, 0.0, bonus=-0.5) == pytest.approx(0.0)
    assert final_score(2.0, 2.0, 2.0) == pytest.approx(1.0)


def test_weights_normalized_defaults_sum_to_one():
    w = ScoreWeights().normalized()
    assert w.fit + w.affordability + w.urgency == pytest.approx(1.0)
    assert (w.fit, w.affordability, w.urgency) == pytest.approx((0.45, 0.35, 0.20))


def test_weights_normalized_custom():
    w = ScoreWeights(fit=1, affordability=1, urgency=2).normalized()
    assert (w.fit, w.affordability, w.urgency) == pytest.approx((0.25, 0.25, 0.5))


def test_all_zero_weights_fall_back_to_defaults():
    w = ScoreWeights(fit=0, affordability=0, urgency=0).normalized()
    assert (w.fit, w.affordability, w.urgency) == pytest.approx((0.45, 0.35, 0.20))
    assert final_score(1.0, 0.0, 0.0, weights=ScoreWeights(fit=0, affordability=0, urgency=0)) == pytest.approx(0.45)


def test_unnormalised_weights_keep_score_in_range():
    heavy = ScoreWeights(fit=10, affordability=10, urgency=10)
    assert final_score(1.0, 1.0, 1.0, weights=heavy) == pytest.approx(1.0)
    assert final_score(0.6, 0.6, 0.6, weights=heavy) == pytest.approx(0.6)


def test_fit_only_weights():
    assert final_score(0.7, 0.1, 0.9, weights=ScoreWeights(fit=1, affordability=0, urgency=0)) == pytest.approx(0.7)


def test_negative_weight_rejected():
    with pytest.raises(ValidationError):
        ScoreWeights(fit=-0.1)


# --- score_cfp ----------------------------------------------------------------------------------

def _cfp(name: str = "PyCon US 2027", deadline: date | None = TODAY + timedelta(days=20), **kw) -> CFP:
    defaults = dict(
        conference_name=name,
        cfp_url="https://us.pycon.example/cfp",
        cfp_deadline=deadline,
        event_start=date(2027, 5, 14),
        city="Long Beach",
        country="USA",
        confidence=0.9,
    )
    defaults.update(kw)
    return CFP(**defaults)


def _trip(total: int | None, reason: str | None = None) -> TripCost:
    if total is None:
        return TripCost(status="unavailable", reason=reason)
    return TripCost(flight_inr=total - 40_000, hotel_inr=40_000, total_inr=total, nights=4, status="ok")


def test_score_cfp_over_budget():
    row = score_cfp(
        key="pycon us|2027",
        cfp=_cfp(),
        travel=TravelSupport(status="none"),
        trip=_trip(BUDGET + 18_000),
        fit=0.9,
        fit_reason="Python concurrency track",
        budget_inr=BUDGET,
        today=TODAY,
    )
    assert row.net_cost_inr == 168_000
    assert row.over_budget is True
    assert row.affordability_score == 0.0
    assert "Over budget by ₹18,000" in row.reasons
    assert row.days_to_deadline == 20
    assert row.urgency_score == pytest.approx(1.0)
    assert row.final_score == pytest.approx(round(0.45 * 0.9 + 0.20 * 1.0, 3))
    assert 2 <= len(row.reasons) <= 4


def test_score_cfp_within_budget_with_grant():
    row = score_cfp(
        key="k",
        cfp=_cfp(),
        travel=TravelSupport(status="grant_available"),
        trip=_trip(75_000),
        fit=0.6,
        fit_reason=None,
        budget_inr=BUDGET,
        today=TODAY,
    )
    assert row.net_cost_inr == 75_000
    assert row.over_budget is False
    assert row.affordability_score == pytest.approx(0.5)
    assert row.bonus == pytest.approx(0.05)
    expected = 0.45 * 0.6 + 0.35 * 0.5 + 0.20 * 1.0 + 0.05
    assert row.final_score == pytest.approx(round(expected, 3))
    assert "Travel grant available" in row.reasons
    assert "Net cost ₹75,000, within budget" in row.reasons


def test_score_cfp_unknown_cost_is_neutral():
    row = score_cfp(
        key="k",
        cfp=_cfp(event_start=None),
        travel=TravelSupport(),
        trip=_trip(None, reason="Event dates not published"),
        fit=0.5,
        fit_reason="ok",
        budget_inr=BUDGET,
        today=TODAY,
    )
    assert row.net_cost_inr is None
    assert row.over_budget is False
    assert row.affordability_score == pytest.approx(0.4)
    assert "Trip cost unavailable: Event dates not published" in row.reasons


def test_score_cfp_online_and_covered_cost_nothing():
    online = score_cfp(
        key="k", cfp=_cfp(is_online=True, city=None), travel=TravelSupport(), trip=TripCost(),
        fit=0.5, fit_reason=None, budget_inr=BUDGET, today=TODAY,
    )
    assert online.net_cost_inr == 0
    assert online.affordability_score == pytest.approx(1.0)
    assert "Online event: no trip cost" in online.reasons

    covered = score_cfp(
        key="k", cfp=_cfp(), travel=TravelSupport(status="covered"), trip=_trip(None, "Fares not yet published"),
        fit=0.5, fit_reason=None, budget_inr=BUDGET, today=TODAY,
    )
    assert covered.net_cost_inr == 0
    assert covered.affordability_score == pytest.approx(1.0)
    assert "Speaker travel covered" in covered.reasons


def test_score_cfp_clamps_fit_and_final():
    row = score_cfp(
        key="k", cfp=_cfp(is_online=True), travel=TravelSupport(status="grant_available"), trip=TripCost(),
        fit=1.7, fit_reason=None, budget_inr=BUDGET, today=TODAY,
    )
    assert row.fit_score == 1.0
    assert row.final_score == 1.0


# --- reasons -----------------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["covered", "partial", "grant_available", "none", "unknown"])
@pytest.mark.parametrize("days", [None, 0, 1, 9, 90])
@pytest.mark.parametrize("net", [None, 0, 50_000, 200_000])
@pytest.mark.parametrize("is_online", [False, True])
def test_build_reasons_returns_two_to_four_strings(status, days, net, is_online):
    reasons = build_reasons(
        fit=0.8,
        fit_reason="Strong concurrency track with many Python talks and an AI agents day as well, very long reason",
        days_to_deadline=days,
        travel=TravelSupport(status=status),
        trip=TripCost(reason="Fares not yet published"),
        net_cost=net,
        budget_inr=BUDGET,
        is_online=is_online,
    )
    assert 2 <= len(reasons) <= 4
    assert all(isinstance(r, str) and r for r in reasons)
    assert len(reasons[0]) <= len("Strong fit: ") + 70  # long LLM reasons are shortened
    assert reasons[0].endswith("...")


def test_build_reasons_content():
    reasons = build_reasons(
        fit=0.8,
        fit_reason="concurrency track.",
        days_to_deadline=9,
        travel=TravelSupport(status="covered"),
        trip=TripCost(),
        net_cost=0,
        budget_inr=BUDGET,
        is_online=False,
    )
    assert reasons[0] == "Strong fit: concurrency track"
    assert "Deadline in 9 days" in reasons
    assert "Speaker travel covered" in reasons


@pytest.mark.parametrize(
    ("fit", "label"), [(0.75, "Strong fit"), (0.74, "Good fit"), (0.5, "Good fit"), (0.49, "Weak fit")]
)
def test_build_reasons_fit_labels(fit, label):
    reasons = build_reasons(
        fit=fit, fit_reason=None, days_to_deadline=None, travel=TravelSupport(), trip=TripCost(),
        net_cost=None, budget_inr=BUDGET, is_online=False,
    )
    assert reasons[0] == label
    assert "Deadline not published" in reasons


def test_build_reasons_deadline_wording():
    def first_deadline(days):
        return build_reasons(
            fit=0.5, fit_reason=None, days_to_deadline=days, travel=TravelSupport(), trip=TripCost(),
            net_cost=None, budget_inr=BUDGET, is_online=False,
        )[1]

    assert first_deadline(0) == "Deadline today"
    assert first_deadline(1) == "Deadline tomorrow"
    assert first_deadline(9) == "Deadline in 9 days"


# --- ordering ------------------------------------------------------------------------------------

def _row(name: str, final: float, fit: float = 0.5, days: int | None = 10, **kw) -> RankedCFP:
    return RankedCFP(
        key=name.lower(),
        cfp=CFP(conference_name=name, confidence=0.9),
        final_score=final,
        fit_score=fit,
        days_to_deadline=days,
        **kw,
    )


def test_sort_and_number_orders_by_final_score():
    rows = [_row("B", 0.4), _row("A", 0.9), _row("C", 0.6)]
    ranked = sort_and_number(rows)
    assert [r.cfp.conference_name for r in ranked] == ["A", "C", "B"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert [r.rank for r in rows] == [0, 0, 0]  # inputs are not mutated


def test_sort_and_number_tie_breaks():
    rows = [
        _row("No deadline", 0.5, fit=0.6, days=None),
        _row("Later", 0.5, fit=0.6, days=30),
        _row("Better fit", 0.5, fit=0.9, days=60),
        _row("Sooner", 0.5, fit=0.6, days=5),
        _row("beta same", 0.5, fit=0.6, days=None),
        _row("Alpha same", 0.5, fit=0.6, days=None),
    ]
    ranked = sort_and_number(rows)
    assert [r.cfp.conference_name for r in ranked] == [
        "Better fit",
        "Sooner",
        "Later",
        "Alpha same",
        "beta same",
        "No deadline",
    ]
    assert [r.rank for r in ranked] == [1, 2, 3, 4, 5, 6]


def test_sort_and_number_empty():
    assert sort_and_number([]) == []


def test_rescore_changes_order_when_weights_change():
    fit_heavy = _row("Great fit, pricey", 0.0, fit=1.0, days=20, affordability_score=0.0, urgency_score=0.4)
    cheap = _row("Cheap, weak fit", 0.0, fit=0.2, days=20, affordability_score=1.0, urgency_score=0.4)

    default = rescore([cheap, fit_heavy], ScoreWeights())
    assert [r.cfp.conference_name for r in default] == ["Great fit, pricey", "Cheap, weak fit"]
    assert default[0].final_score == pytest.approx(0.45 + 0.08)
    assert default[1].final_score == pytest.approx(0.09 + 0.35 + 0.08)

    money = rescore([cheap, fit_heavy], ScoreWeights(fit=0, affordability=1, urgency=0))
    assert [r.cfp.conference_name for r in money] == ["Cheap, weak fit", "Great fit, pricey"]
    assert money[0].final_score == pytest.approx(1.0)
    assert money[1].final_score == pytest.approx(0.0)
    assert [r.rank for r in money] == [1, 2]


def test_rescore_keeps_grant_bonus():
    row = _row("Grant", 0.0, fit=0.5, affordability_score=0.5, urgency_score=0.5, bonus=0.05)
    (out,) = rescore([row], ScoreWeights())
    assert out.final_score == pytest.approx(0.55)


# --- money formatting ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("amount", "text"),
    [
        (None, "unavailable"),
        (0, "₹0"),
        (999, "₹999"),
        (1000, "₹1,000"),
        (18000, "₹18,000"),
        (99999, "₹99,999"),
        (100000, "₹1,00,000"),
        (150000, "₹1,50,000"),
        (1234567, "₹12,34,567"),
        (123456789, "₹12,34,56,789"),
        (-18000, "-₹18,000"),
        (-500, "-₹500"),
        (1500.6, "₹1,501"),
    ],
)
def test_format_inr(amount, text):
    assert format_inr(amount) == text


# --- keyword fit fallback ----------------------------------------------------------------------------

def test_keyword_fit_alias_match():
    assert keyword_fit(["Python concurrency"], "PyCon US 2027", []) > 0
    assert keyword_fit(["Python"], "PyCon US 2027", []) == pytest.approx(1.0)


def test_keyword_fit_topic_words():
    assert keyword_fit(["AI agents"], "AI Engineer Summit", ["llm", "agents"]) == pytest.approx(1.0)
    # the best topic wins
    assert keyword_fit(["Rust embedded", "Python"], "PyCon India", []) == pytest.approx(1.0)


def test_keyword_fit_unrelated_is_zero():
    assert keyword_fit(["Rust embedded"], "Global Coffee Expo", ["food", "beverage"]) == 0.0
    assert keyword_fit(["Python"], "", []) == 0.0
    assert keyword_fit([], "PyCon US", []) == 0.0


@pytest.mark.parametrize(
    ("topics", "name"),
    [
        (["AI agents"], "Maintainers Summit 2027"),  # "ai" inside "maintainers"
        (["AI agents"], "RailsConf 2027"),  # "ai" inside "railsconf"
        (["AI agents"], "PyCon Spain 2027"),  # "ai" inside "spain"
        (["Go"], "Chicago Ruby Meetup"),  # "go" inside "chicago"
        (["Go"], "Google Cloud Next"),  # "go" inside "google"
    ],
)
def test_keyword_fit_short_words_do_not_match_inside_other_words(topics, name):
    assert keyword_fit(topics, name, []) == 0.0


def test_keyword_fit_short_words_still_match_exactly_or_by_alias():
    assert keyword_fit(["AI"], "AI Summit", []) == pytest.approx(1.0)
    assert keyword_fit(["AI"], "ML Summit", []) == pytest.approx(1.0)  # alias
    assert keyword_fit(["Go"], "GopherCon EU", []) == pytest.approx(1.0)
    assert keyword_fit(["Go"], "Go Days", []) == pytest.approx(1.0)
    # longer words still match inside compound names
    assert keyword_fit(["Python"], "EuroPython 2027", []) == pytest.approx(1.0)
    assert keyword_fit(["Django"], "DjangoCon Europe", []) == pytest.approx(1.0)


def test_keyword_fit_in_unit_interval():
    for topics, name, conf_topics in [
        (["Python concurrency", "AI agents"], "PyCon US 2027", ["python", "ai"]),
        (["Go"], "GopherCon EU", []),
        (["DevOps"], "KubeCon India", ["kubernetes", "cloud"]),
    ]:
        assert 0.0 <= keyword_fit(topics, name, conf_topics) <= 1.0
