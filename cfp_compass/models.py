"""Typed data model for CFP Compass.

Every field that comes from the web or an LLM is optional, because real pages are
messy. Dates are ``datetime.date`` objects and money is whole rupees (``int``).
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

TravelStatus = Literal["covered", "partial", "grant_available", "none", "unknown"]
TripStatus = Literal["ok", "partial", "unavailable"]

_IATA = re.compile(r"^[A-Z]{3}$")


class SpeakerProfile(BaseModel):
    """What the speaker enters in the sidebar (or on the CLI)."""

    topics: list[str]
    talk_abstract: str | None = None
    home_airport: str
    budget_inr: int
    window_start: date
    window_end: date
    max_conferences: int = 12

    @field_validator("topics", mode="before")
    @classmethod
    def _split_topics(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, (list, tuple)):
            seen: set[str] = set()
            cleaned: list[str] = []
            for item in value:
                topic = " ".join(str(item).split())
                if topic and topic.lower() not in seen:
                    seen.add(topic.lower())
                    cleaned.append(topic)
            return cleaned
        return value

    @field_validator("topics")
    @classmethod
    def _check_topics(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Enter at least one talk topic.")
        if len(value) > 2:
            raise ValueError("Enter at most 2 talk topics, separated by a comma.")
        return value

    @field_validator("talk_abstract", mode="before")
    @classmethod
    def _blank_abstract(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("home_airport", mode="before")
    @classmethod
    def _upper_airport(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("home_airport")
    @classmethod
    def _check_airport(cls, value: str) -> str:
        if not _IATA.fullmatch(value):
            raise ValueError("Home airport must be a 3-letter IATA code, for example COK.")
        return value

    @field_validator("budget_inr")
    @classmethod
    def _check_budget(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Budget must be greater than 0.")
        return value

    @field_validator("max_conferences")
    @classmethod
    def _check_max(cls, value: int) -> int:
        if not 1 <= value <= 25:
            raise ValueError("Conferences to price must be between 1 and 25.")
        return value

    @model_validator(mode="after")
    def _check_window(self) -> SpeakerProfile:
        if self.window_start >= self.window_end:
            raise ValueError("The travel window must start before it ends.")
        return self


def describe_validation_error(error: ValidationError) -> list[str]:
    """Turn a pydantic ValidationError into short, human-readable messages."""
    labels = {
        "topics": "Talk topics",
        "home_airport": "Home airport",
        "budget_inr": "Budget",
        "window_start": "Travel window start",
        "window_end": "Travel window end",
        "max_conferences": "Conferences to price",
    }
    messages = []
    for item in error.errors():
        msg = str(item.get("msg", "Invalid value")).removeprefix("Value error, ")
        field = item.get("loc", ())[0] if item.get("loc") else None
        label = labels.get(str(field)) if field is not None else None
        if label and not msg.startswith(label.split()[0]):
            msg = f"{label}: {msg}"
        messages.append(msg)
    return messages


class Candidate(BaseModel):
    """One raw search hit, before any LLM extraction."""

    title: str = ""
    url: str
    snippet: str = ""
    source_query: str = ""
    date: str | None = None  # Google's own date label for the result, when shown
    position: int | None = None


class CFP(BaseModel):
    """A conference call for proposals, extracted from one search hit by the LLM."""

    conference_name: str = Field(
        description="Official conference name with its edition year if shown, e.g. 'PyCon US 2027'. "
        "Do not include words like 'Call for Proposals' or the site name."
    )
    cfp_url: str = Field(
        default="",
        description="URL where proposals are submitted, copied from the source. Empty if unsure.",
    )
    cfp_deadline: date | None = Field(
        default=None,
        description="Last day to submit proposals (YYYY-MM-DD). Null unless the source states it.",
    )
    deadline_is_aoe: bool = Field(
        default=False,
        description="True only if the deadline is stated as AoE / Anywhere on Earth.",
    )
    event_start: date | None = Field(default=None, description="First day of the event (YYYY-MM-DD), or null.")
    event_end: date | None = Field(default=None, description="Last day of the event (YYYY-MM-DD), or null.")
    city: str | None = Field(default=None, description="City where the event takes place, or null.")
    country: str | None = Field(default=None, description="Country where the event takes place, or null.")
    venue: str | None = Field(default=None, description="Venue name if stated, or null.")
    is_online: bool = Field(default=False, description="True if the event is online-only (virtual).")
    topics: list[str] = Field(default_factory=list, description="Up to 6 short topics the conference covers.")
    is_cfp_open: bool = Field(
        default=True,
        description="False if the source says the CFP has closed or the event is over.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0-1: how clearly the source states these facts (name, deadline, dates, place).",
    )
    evidence: str = Field(
        default="",
        description="A short verbatim excerpt from the source that states the CFP deadline. Empty if none.",
    )
    # Filled in by the app, never by the LLM (hidden from the output schema).
    source_url: SkipJsonSchema[str] = ""

    @model_validator(mode="after")
    def _default_end(self) -> CFP:
        if self.event_start is not None and (self.event_end is None or self.event_end < self.event_start):
            self.event_end = self.event_start
        return self


class TravelSupport(BaseModel):
    """A conference's speaker travel policy."""

    status: TravelStatus = Field(
        default="unknown",
        description="covered = conference pays speaker travel and hotel; partial = pays some of it "
        "(e.g. hotel only or a capped stipend); grant_available = a travel grant / financial aid "
        "programme speakers can apply to; none = explicitly no travel support; unknown = not stated.",
    )
    evidence_url: str | None = Field(default=None, description="The result link the status came from.")
    note: str | None = Field(default=None, description="One short sentence explaining the status.")


class TripCost(BaseModel):
    """The priced trip from the speaker's home airport."""

    airport_codes: list[str] = Field(default_factory=list)
    flight_inr: int | None = None
    nights: int = 0
    hotel_nightly_inr: int | None = None
    hotel_inr: int | None = None
    total_inr: int | None = None
    status: TripStatus = "unavailable"
    reason: str | None = None
    outbound_date: date | None = None
    return_date: date | None = None
    fetched_at: datetime | None = None


class ScoreWeights(BaseModel):
    """Weights for the final score. They are normalised to sum to 1 before use."""

    fit: float = Field(default=0.45, ge=0.0)
    affordability: float = Field(default=0.35, ge=0.0)
    urgency: float = Field(default=0.20, ge=0.0)

    def normalized(self) -> ScoreWeights:
        total = self.fit + self.affordability + self.urgency
        if total <= 0:
            return ScoreWeights()
        return ScoreWeights(
            fit=self.fit / total,
            affordability=self.affordability / total,
            urgency=self.urgency / total,
        )


class RankedCFP(BaseModel):
    """One row of the final table."""

    key: str
    cfp: CFP
    travel: TravelSupport = Field(default_factory=TravelSupport)
    trip: TripCost = Field(default_factory=TripCost)
    fit_score: float = 0.0
    fit_reason: str | None = None
    urgency_score: float = 0.0
    affordability_score: float = 0.0
    bonus: float = 0.0
    final_score: float = 0.0
    net_cost_inr: int | None = None
    over_budget: bool = False
    days_to_deadline: int | None = None
    reasons: list[str] = Field(default_factory=list)
    rank: int = 0


class RunStats(BaseModel):
    """What one pipeline run cost and found."""

    serpapi_searches: int = 0
    serpapi_cache_hits: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    queries: int = 0
    candidates: int = 0
    candidates_after_prefilter: int = 0
    extracted: int = 0
    valid: int = 0
    shortlisted: int = 0
    priced: int = 0
    cached_only: bool = False
    api_disabled_reason: str | None = None
    llm_disabled_reason: str | None = None


class PipelineResult(BaseModel):
    """Everything the UI and CLI need after a run."""

    profile: SpeakerProfile
    ranked: list[RankedCFP]
    unpriced: list[CFP] = Field(default_factory=list)  # valid CFPs beyond the pricing shortlist
    stats: RunStats
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    generated_at: datetime
    prices_fetched_at: datetime | None = None
