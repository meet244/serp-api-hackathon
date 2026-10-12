"""Scoring and ranking. Pure functions only: no I/O, so every rule is easy to unit-test.

final = w_fit * fit + w_afford * affordability + w_urgency * urgency + bonus, clamped to [0, 1].
"""

from __future__ import annotations

import re
from datetime import date

from .models import CFP, RankedCFP, ScoreWeights, TravelStatus, TravelSupport, TripCost

DEFAULT_WEIGHTS = ScoreWeights()
URGENCY_UNKNOWN = 0.3
AFFORDABILITY_UNKNOWN = 0.4
GRANT_BONUS = 0.05
PARTIAL_SHARE = 0.5

TRAVEL_LABELS: dict[str, str] = {
    "covered": "Covered",
    "partial": "Partial",
    "grant_available": "Grant available",
    "none": "None",
    "unknown": "Unknown",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def format_inr(amount: int | float | None) -> str:
    """Format rupees with Indian digit grouping: 150000 -> '₹1,50,000'."""
    if amount is None:
        return "unavailable"
    negative = amount < 0
    digits = str(int(round(abs(amount))))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"{'-' if negative else ''}₹{digits}"


def days_until(deadline: date | None, today: date) -> int | None:
    return None if deadline is None else (deadline - today).days


def urgency_score(days_to_deadline: int | None) -> float:
    """Reward deadlines that leave enough time to write a good proposal."""
    if days_to_deadline is None:
        return URGENCY_UNKNOWN
    if days_to_deadline <= 1:
        return 0.2  # too late to write well
    if days_to_deadline <= 6:
        return 0.6
    if days_to_deadline <= 30:
        return 1.0  # the sweet spot
    if days_to_deadline <= 60:
        return 0.7
    return 0.4


def net_cost_inr(total_inr: int | None, travel_status: TravelStatus, is_online: bool = False) -> int | None:
    """What the speaker pays after the conference's travel support."""
    if is_online or travel_status == "covered":
        return 0
    if total_inr is None:
        return None
    if travel_status == "partial":
        return int(round(total_inr * PARTIAL_SHARE))
    return total_inr  # grant_available, none, unknown: the full cost


def travel_bonus(travel_status: TravelStatus) -> float:
    return GRANT_BONUS if travel_status == "grant_available" else 0.0


def affordability_score(net_cost: int | None, budget_inr: int) -> float:
    """1 - net/budget, clamped. Unknown cost is neutral (0.4) so it is neither rewarded nor buried."""
    if net_cost is None:
        return AFFORDABILITY_UNKNOWN
    if budget_inr <= 0 or net_cost > budget_inr:
        return 0.0
    return clamp(1.0 - net_cost / budget_inr)


def final_score(
    fit: float,
    affordability: float,
    urgency: float,
    bonus: float = 0.0,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    w = weights.normalized()
    return clamp(w.fit * fit + w.affordability * affordability + w.urgency * urgency + bonus)


# --- Keyword-overlap fit: the fallback when the LLM fit call fails -----------------

_ALIASES: dict[str, set[str]] = {
    "python": {"pycon", "django", "djangocon", "flask", "flaskcon", "pydata", "europython", "scipy", "fastapi"},
    "javascript": {"js", "jsconf", "node", "nodejs", "react", "typescript", "frontend"},
    "ruby": {"rails", "rubyconf", "railsconf"},
    "ai": {"ml", "llm", "llms", "genai", "machine", "learning", "artificial", "intelligence", "agents"},
    "agents": {"agentic", "llm", "ai"},
    "rust": {"rustconf", "rustlang"},
    "go": {"golang", "gophercon"},
    "devops": {"sre", "kubernetes", "cloud", "platform", "infrastructure"},
    "data": {"pydata", "analytics", "database", "databases"},
}
_STOPWORDS = {"and", "or", "the", "of", "for", "in", "on", "with", "a", "an", "to", "conference", "conf"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if t not in _STOPWORDS and len(t) > 1}


def keyword_fit(topics: list[str], conference_name: str, conference_topics: list[str]) -> float:
    """Share of a speaker topic's words found in the conference name/topics (best topic wins)."""
    haystack = _tokens(conference_name + " " + " ".join(conference_topics))
    for word in list(haystack):  # "pycon2027" style names
        haystack.update(re.findall(r"[a-z]+", word))
    if not haystack:
        return 0.0
    best = 0.0
    for topic in topics:
        words = _tokens(topic)
        if not words:
            continue
        hits = 0
        for word in words:
            if word in haystack or _ALIASES.get(word, set()) & haystack:
                hits += 1
            elif len(word) > 3 and any(word in h or h in word for h in haystack if len(h) > 3):
                hits += 1
        best = max(best, hits / len(words))
    return round(best, 2)


# --- Reasons -------------------------------------------------------------------------

def _deadline_reason(days: int | None) -> str:
    if days is None:
        return "Deadline not published"
    if days == 0:
        return "Deadline today"
    if days == 1:
        return "Deadline tomorrow"
    return f"Deadline in {days} days"


def build_reasons(
    *,
    fit: float,
    fit_reason: str | None,
    days_to_deadline: int | None,
    travel: TravelSupport,
    trip: TripCost,
    net_cost: int | None,
    budget_inr: int,
    is_online: bool,
) -> list[str]:
    """2 to 4 short, plain-language strings that explain a row's rank."""
    reasons: list[str] = []
    label = "Strong fit" if fit >= 0.75 else "Good fit" if fit >= 0.5 else "Weak fit"
    detail = (fit_reason or "").strip().rstrip(".")
    if len(detail) > 70:
        detail = detail[:67].rstrip() + "..."
    reasons.append(f"{label}: {detail}" if detail else label)

    reasons.append(_deadline_reason(days_to_deadline))

    travel_text = {
        "covered": "Speaker travel covered",
        "partial": "Partial travel support (50% assumed)",
        "grant_available": "Travel grant available",
        "none": "No speaker travel support",
    }.get(travel.status)
    if travel_text:
        reasons.append(travel_text)

    if is_online:
        reasons.append("Online event: no trip cost")
    elif net_cost is None:
        reasons.append(f"Trip cost unavailable: {trip.reason}" if trip.reason else "Trip cost unavailable")
    elif net_cost > budget_inr:
        reasons.append(f"Over budget by {format_inr(net_cost - budget_inr)}")
    else:
        reasons.append(f"Net cost {format_inr(net_cost)}, within budget")
    return reasons[:4]


# --- Ranking -------------------------------------------------------------------------

def score_cfp(
    *,
    key: str,
    cfp: CFP,
    travel: TravelSupport,
    trip: TripCost,
    fit: float,
    fit_reason: str | None,
    budget_inr: int,
    today: date,
    weights: ScoreWeights = DEFAULT_WEIGHTS,
) -> RankedCFP:
    fit = clamp(fit)
    days = days_until(cfp.cfp_deadline, today)
    urgency = urgency_score(days)
    net = net_cost_inr(trip.total_inr, travel.status, cfp.is_online)
    afford = affordability_score(net, budget_inr)
    bonus = travel_bonus(travel.status)
    return RankedCFP(
        key=key,
        cfp=cfp,
        travel=travel,
        trip=trip,
        fit_score=round(fit, 3),
        fit_reason=fit_reason,
        urgency_score=urgency,
        affordability_score=round(afford, 3),
        bonus=bonus,
        final_score=round(final_score(fit, afford, urgency, bonus, weights), 3),
        net_cost_inr=net,
        over_budget=net is not None and net > budget_inr,
        days_to_deadline=days,
        reasons=build_reasons(
            fit=fit,
            fit_reason=fit_reason,
            days_to_deadline=days,
            travel=travel,
            trip=trip,
            net_cost=net,
            budget_inr=budget_inr,
            is_online=cfp.is_online,
        ),
    )


def sort_and_number(rows: list[RankedCFP]) -> list[RankedCFP]:
    """Sort by final score (ties: better fit, then sooner deadline) and set rank 1, 2, 3..."""
    ordered = sorted(
        rows,
        key=lambda r: (
            -r.final_score,
            -r.fit_score,
            r.days_to_deadline if r.days_to_deadline is not None else 10**6,
            r.cfp.conference_name.lower(),
        ),
    )
    return [row.model_copy(update={"rank": i}) for i, row in enumerate(ordered, start=1)]


def rescore(rows: list[RankedCFP], weights: ScoreWeights) -> list[RankedCFP]:
    """Re-rank with new weights without re-running searches (used by the sidebar sliders)."""
    updated = [
        row.model_copy(
            update={
                "final_score": round(
                    final_score(row.fit_score, row.affordability_score, row.urgency_score, row.bonus, weights), 3
                )
            }
        )
        for row in rows
    ]
    return sort_and_number(updated)
