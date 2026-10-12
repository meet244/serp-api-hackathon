"""Step 9: price the trip. City to airports, round-trip flight, then hotel, all in INR.

Trip dates: fly out the day before the event starts and home the day after it ends.
The hotel covers the same nights, so nights = event days + 1.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from .context import RunContext
from .models import CFP, SpeakerProfile, TripCost
from .serp import TTL_AIRPORTS, TTL_FLIGHTS, TTL_HOTELS, SerpResponse, search

MAX_FARE_HORIZON_DAYS = 330  # Google Flights rarely lists fares further ahead than this
LATE_FARES_DAYS = 270  # "no results" this far ahead most likely means fares are not out yet
HOTEL_SAMPLE = 10
_IATA = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class TripDates:
    outbound: date
    return_date: date
    check_in: date
    check_out: date
    nights: int


def trip_dates(event_start: date, event_end: date | None = None) -> TripDates:
    """Outbound = start - 1 day, return = end + 1 day; the hotel covers every night between."""
    end = event_end if event_end is not None and event_end >= event_start else event_start
    outbound = event_start - timedelta(days=1)
    return_date = end + timedelta(days=1)
    return TripDates(
        outbound=outbound,
        return_date=return_date,
        check_in=outbound,
        check_out=return_date,
        nights=(return_date - outbound).days,
    )


# --- Google Flights Autocomplete: city to airports -------------------------------------

def airport_query(city: str, country: str | None) -> str:
    """'Portland' is ambiguous, 'Portland, United States' much less so."""
    return f"{city}, {country}" if country else city


def autocomplete_params(query: str) -> dict[str, Any]:
    # exclude_regions must be the string "true": requests would send Python True as "True".
    return {"engine": "google_flights_autocomplete", "q": query, "exclude_regions": "true", "hl": "en", "gl": "us"}


def parse_airports(data: dict[str, Any], country: str | None = None) -> tuple[str | None, list[str]]:
    """First city suggestion with airports (preferring one that names the country): (kgmid, IATA codes)."""
    cities = [
        s
        for s in data.get("suggestions") or []
        if isinstance(s, dict) and s.get("type") == "city" and isinstance(s.get("airports"), list)
    ]
    if not cities:
        return None, []

    def mentions_country(s: dict[str, Any]) -> bool:
        text = f"{s.get('name', '')} {s.get('description', '')}".lower()
        return bool(country) and country.lower() in text

    best = next((s for s in cities if mentions_country(s)), cities[0])
    codes = [
        a["id"]
        for a in best["airports"]
        if isinstance(a, dict) and isinstance(a.get("id"), str) and _IATA.fullmatch(a["id"])
    ]
    kgmid = best.get("id") if isinstance(best.get("id"), str) else None
    return kgmid, list(dict.fromkeys(codes))


def resolve_airports(city: str, country: str | None, ctx: RunContext) -> tuple[str | None, list[str], SerpResponse]:
    response = search(autocomplete_params(airport_query(city, country)), ctx, ttl_seconds=TTL_AIRPORTS)
    kgmid, codes = parse_airports(response.data, country)
    if not codes and not kgmid and country and response.ok:
        # "City, Country" found nothing: fall back to the bare city name.
        response = search(autocomplete_params(city), ctx, ttl_seconds=TTL_AIRPORTS)
        kgmid, codes = parse_airports(response.data, country)
    return kgmid, codes, response


# --- Google Flights: round-trip fare ------------------------------------------------------

def flight_params(home: str, arrival_id: str, dates: TripDates) -> dict[str, Any]:
    return {
        "engine": "google_flights",
        "departure_id": home,
        "arrival_id": arrival_id,
        "type": 1,  # round trip
        "outbound_date": dates.outbound.isoformat(),
        "return_date": dates.return_date.isoformat(),
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "adults": 1,
        "travel_class": 1,  # economy
    }


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def parse_flight_price(data: dict[str, Any]) -> int | None:
    """price_insights.lowest_price, else the cheapest of best_flights[] and other_flights[]."""
    insights = data.get("price_insights")
    if isinstance(insights, dict):
        lowest = _positive_number(insights.get("lowest_price"))
        if lowest is not None:
            return int(round(lowest))
    prices = [
        p
        for group in ("best_flights", "other_flights")
        for option in data.get(group) or []
        if isinstance(option, dict) and (p := _positive_number(option.get("price"))) is not None
    ]
    return int(round(min(prices))) if prices else None


# --- Google Hotels: median nightly rate ---------------------------------------------------

def hotel_query(city: str, country: str | None, venue: str | None) -> str:
    place = f"{venue}, {city}" if venue and venue.lower() != city.lower() else city
    return f"hotels near {place}" + (f", {country}" if country else "")


def hotel_params(query: str, dates: TripDates) -> dict[str, Any]:
    return {
        "engine": "google_hotels",
        "q": query,
        "check_in_date": dates.check_in.isoformat(),
        "check_out_date": dates.check_out.isoformat(),
        "adults": 1,
        "currency": "INR",
        "gl": "in",
        "hl": "en",
    }


def parse_hotel_nightly(data: dict[str, Any], sample: int = HOTEL_SAMPLE) -> int | None:
    """Median of the first ``sample`` nightly rates that exist, so one outlier cannot skew it."""
    properties = data.get("properties") or []
    info = data.get("search_information")
    if not properties and isinstance(info, dict) and "property details" in str(info.get("hotels_results_state", "")):
        properties = [data]  # q matched one hotel: its details sit at the top level
    rates: list[float] = []
    for prop in properties:
        if not isinstance(prop, dict):
            continue
        rate_info = prop.get("rate_per_night")
        rate = _positive_number(rate_info.get("extracted_lowest")) if isinstance(rate_info, dict) else None
        if rate is not None:
            rates.append(rate)
            if len(rates) >= sample:
                break
    return int(round(statistics.median(rates))) if rates else None


# --- Putting a trip together ----------------------------------------------------------------

def _lookup_problem(response: SerpResponse, what: str) -> str:
    if response.error and response.error.startswith("unavailable"):
        return f"{what}: not in cache (cached-only mode)"
    if response.error:
        return f"{what}: lookup failed"
    return f"No {what.lower()} results"


def price_trip(cfp: CFP, profile: SpeakerProfile, ctx: RunContext) -> TripCost:
    """Price one conference trip. Never raises; problems become status + reason."""
    if cfp.is_online:
        return TripCost(flight_inr=0, hotel_nightly_inr=0, hotel_inr=0, total_inr=0, status="ok", reason="Online event")
    if cfp.event_start is None:
        return TripCost(status="unavailable", reason="Event dates not published")
    if not cfp.city:
        return TripCost(status="unavailable", reason="Event city not published")

    dates = trip_dates(cfp.event_start, cfp.event_end)
    base: dict[str, Any] = {"nights": dates.nights, "outbound_date": dates.outbound, "return_date": dates.return_date}

    # Home city first: no trip to pay for, whatever the dates (airport lookups are cached forever).
    kgmid, codes, airport_response = resolve_airports(cfp.city, cfp.country, ctx)
    base["airport_codes"] = codes
    if profile.home_airport in codes:
        return TripCost(
            **base, flight_inr=0, hotel_nightly_inr=0, hotel_inr=0, total_inr=0, status="ok", reason="In your home city"
        )

    days_ahead = (dates.outbound - ctx.today).days
    if days_ahead < 0:
        return TripCost(**base, status="unavailable", reason="Travel dates have already passed")
    if days_ahead > MAX_FARE_HORIZON_DAYS:
        return TripCost(**base, status="unavailable", reason="Fares not yet published")

    problems: list[str] = []
    fetched: list[datetime] = []

    flight_inr: int | None = None
    arrival_id = ",".join(codes) if codes else kgmid
    if not arrival_id:
        problems.append(_lookup_problem(airport_response, "Airport") if not airport_response.ok else "Airport not found")
    else:
        flight_response = search(flight_params(profile.home_airport, arrival_id, dates), ctx, ttl_seconds=TTL_FLIGHTS)
        flight_inr = parse_flight_price(flight_response.data) if flight_response.ok else None
        if flight_response.fetched_at:
            fetched.append(flight_response.fetched_at)
        if flight_inr is None:
            if flight_response.ok and days_ahead > LATE_FARES_DAYS:
                problems.append("Fares not yet published")
            else:
                problems.append(_lookup_problem(flight_response, "Flight"))

    hotel_response = search(hotel_params(hotel_query(cfp.city, cfp.country, cfp.venue), dates), ctx, ttl_seconds=TTL_HOTELS)
    nightly = parse_hotel_nightly(hotel_response.data) if hotel_response.ok else None
    if hotel_response.fetched_at:
        fetched.append(hotel_response.fetched_at)
    if nightly is None:
        problems.append(_lookup_problem(hotel_response, "Hotel"))

    hotel_inr = nightly * dates.nights if nightly is not None else None
    if flight_inr is not None and hotel_inr is not None:
        status, total = "ok", flight_inr + hotel_inr
    elif flight_inr is not None or hotel_inr is not None:
        status, total = "partial", None
    else:
        status, total = "unavailable", None
    return TripCost(
        **base,
        flight_inr=flight_inr,
        hotel_nightly_inr=nightly,
        hotel_inr=hotel_inr,
        total_inr=total,
        status=status,
        reason="; ".join(problems) or None,
        fetched_at=min(fetched) if fetched else None,  # the oldest price in this trip
    )
