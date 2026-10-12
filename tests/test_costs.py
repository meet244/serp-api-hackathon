"""costs.py: trip dates, response parsing, request params and price_trip with a fake SerpApi client."""

from __future__ import annotations

import copy
from datetime import date

import pytest

from cfp_compass.costs import (
    autocomplete_params,
    flight_params,
    hotel_params,
    parse_airports,
    parse_flight_price,
    parse_hotel_nightly,
    price_trip,
    resolve_airports,
    trip_dates,
)
from cfp_compass.models import CFP, SpeakerProfile

from conftest import load_fixture

NO_FLIGHTS = {"error": "Google Flights hasn't returned any results for this query."}
NO_HOTELS = {
    "search_information": {"hotels_results_state": "Fully empty"},
    "error": "Google Hotels hasn't returned any results for this query.",
}


# --- trip_dates -------------------------------------------------------------------------

def test_one_day_event_is_two_nights():
    d = trip_dates(date(2026, 12, 10), date(2026, 12, 10))
    assert d.outbound == date(2026, 12, 9)
    assert d.return_date == date(2026, 12, 11)
    assert d.check_in == d.outbound and d.check_out == d.return_date
    assert d.nights == 2


def test_three_day_event_is_four_nights():
    d = trip_dates(date(2026, 12, 10), date(2026, 12, 12))
    assert d.outbound == date(2026, 12, 9)
    assert d.return_date == date(2026, 12, 13)
    assert d.check_in == date(2026, 12, 9)
    assert d.check_out == date(2026, 12, 13)
    assert d.nights == 4


def test_missing_end_counts_as_one_day():
    assert trip_dates(date(2027, 1, 31)) == trip_dates(date(2027, 1, 31), date(2027, 1, 31))
    d = trip_dates(date(2027, 1, 31), None)
    assert (d.outbound, d.return_date, d.nights) == (date(2027, 1, 30), date(2027, 2, 1), 2)


def test_end_before_start_counts_as_one_day():
    d = trip_dates(date(2027, 3, 10), date(2027, 3, 1))
    assert (d.outbound, d.return_date, d.nights) == (date(2027, 3, 9), date(2027, 3, 11), 2)


# --- parse_flight_price -------------------------------------------------------------------

def test_flight_price_uses_price_insights():
    data = load_fixture("google_flights")
    assert parse_flight_price(data) == 2512
    data["price_insights"]["lowest_price"] = 1999  # lower than any listed flight: proves insights win
    assert parse_flight_price(data) == 1999


def test_flight_price_falls_back_to_min_across_best_and_other():
    data = load_fixture("google_flights")
    del data["price_insights"]
    assert parse_flight_price(data) == 2512  # cheapest of best_flights

    data["other_flights"][0]["price"] = 1500
    assert parse_flight_price(data) == 1500  # cheapest is in other_flights


def test_flight_price_ignores_bad_prices():
    data = {
        "price_insights": {"lowest_price": 0},
        "best_flights": [{"price": 0}, {"price": -10}, {"price": True}, {}, {"price": None}, "junk"],
        "other_flights": [{"price": 4100}, {"price": 5000}],
    }
    assert parse_flight_price(data) == 4100


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"best_flights": [], "other_flights": []},
        {"best_flights": [{"price": 0}], "other_flights": [{"price": False}, {}]},
        {"price_insights": {"lowest_price": -1}},
        NO_FLIGHTS,
    ],
)
def test_flight_price_none_when_nothing(data):
    assert parse_flight_price(data) is None


# --- parse_hotel_nightly ------------------------------------------------------------------

def _prop(rate: int | None) -> dict:
    if rate is None:
        return {"name": "No rate hotel"}
    return {"name": f"Hotel {rate}", "rate_per_night": {"lowest": f"₹{rate}", "extracted_lowest": rate}}


def test_hotel_median_from_fixture():
    data = load_fixture("google_hotels")
    rates = [p["rate_per_night"]["extracted_lowest"] for p in data["properties"] if p.get("rate_per_night")]
    assert sorted(rates) == [18, 114, 247, 332, 484]
    assert parse_hotel_nightly(data) == 247


def test_hotel_properties_without_rate_are_skipped():
    data = load_fixture("google_hotels")
    data["properties"].insert(0, {"name": "Sold out"})
    data["properties"].append({"name": "Weird", "rate_per_night": {"lowest": "n/a"}})
    data["properties"].append({"name": "Zero", "rate_per_night": {"extracted_lowest": 0}})
    assert parse_hotel_nightly(data) == 247

    assert parse_hotel_nightly({"properties": [_prop(None), _prop(100), _prop(None), _prop(300)]}) == 200


def test_hotel_uses_only_first_ten_existing_rates():
    rates = [100 * i for i in range(1, 11)]  # 100..1000: median 550
    props = [_prop(None)] + [_prop(r) for r in rates[:5]] + [_prop(None)] + [_prop(r) for r in rates[5:]]
    props += [_prop(1_000_000)] * 5  # all 15 would give a median of 800
    assert parse_hotel_nightly({"properties": props}) == 550


def test_hotel_single_property_details_shape():
    data = {
        "search_information": {"hotels_results_state": "Showing results for property details"},
        "type": "hotel",
        "name": "H10 Port Vell",
        "rate_per_night": {"lowest": "₹9,000", "extracted_lowest": 9000},
    }
    assert parse_hotel_nightly(data) == 9000


@pytest.mark.parametrize("data", [{}, {"properties": []}, {"properties": [_prop(None)]}, NO_HOTELS])
def test_hotel_none_when_nothing(data):
    assert parse_hotel_nightly(data) is None


# --- parse_airports / autocomplete -------------------------------------------------------

def test_parse_airports_fixture_new_york():
    kgmid, codes = parse_airports(load_fixture("google_flights_autocomplete"))
    assert kgmid == "/m/02_286"
    assert codes == ["JFK", "LGA", "EWR"]


def test_parse_airports_korea():
    data = load_fixture("google_flights_autocomplete_korea")
    assert parse_airports(data, "South Korea") == ("/m/0hsqf", ["GMP", "ICN"])
    assert parse_airports(data) == ("/m/0hsqf", ["GMP", "ICN"])


def test_parse_airports_skips_regions():
    data = {
        "suggestions": [
            {"type": "region", "name": "New Zealand", "id": "/m/0ctw_b"},
            {"type": "region", "name": "Auckland Region", "id": "/m/r", "airports": [{"id": "XXX"}]},
            {"type": "city", "name": "Auckland", "id": "/m/0ctw7", "airports": [{"id": "AKL"}]},
        ]
    }
    assert parse_airports(data) == ("/m/0ctw7", ["AKL"])
    assert parse_airports({"suggestions": data["suggestions"][:2]}) == (None, [])
    assert parse_airports({}) == (None, [])


def test_parse_airports_prefers_suggestion_mentioning_country():
    data = load_fixture("google_flights_autocomplete")
    assert parse_airports(data, "India") == ("/m/0dlv0", ["DEL"])
    # No suggestion mentions the country: the first city wins.
    assert parse_airports(data, "Atlantis") == ("/m/02_286", ["JFK", "LGA", "EWR"])


def test_autocomplete_params_exclude_regions_is_a_string():
    p = autocomplete_params("Porto, Portugal")
    assert p["engine"] == "google_flights_autocomplete"
    assert p["q"] == "Porto, Portugal"
    assert p["exclude_regions"] == "true"
    assert isinstance(p["exclude_regions"], str)


def test_flight_params_match_spec():
    dates = trip_dates(date(2026, 12, 10), date(2026, 12, 12))
    assert flight_params("COK", "JFK,LGA,EWR", dates) == {
        "engine": "google_flights",
        "departure_id": "COK",
        "arrival_id": "JFK,LGA,EWR",
        "type": 1,
        "outbound_date": "2026-12-09",
        "return_date": "2026-12-13",
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "adults": 1,
        "travel_class": 1,
    }


def test_hotel_params_match_dates():
    dates = trip_dates(date(2026, 12, 10), date(2026, 12, 12))
    p = hotel_params("hotels near New York, United States", dates)
    assert p == {
        "engine": "google_hotels",
        "q": "hotels near New York, United States",
        "check_in_date": "2026-12-09",
        "check_out_date": "2026-12-13",
        "adults": 1,
        "currency": "INR",
        "gl": "in",
        "hl": "en",
    }


# --- price_trip with a fake client ---------------------------------------------------------

class EngineFake:
    """Answers by params['engine']; a missing engine raises (it must not be called)."""

    def __init__(self, answers: dict[str, dict]):
        self.answers = answers
        self.calls: list[dict] = []

    def factory(self) -> EngineFake:
        return self

    def search(self, params: dict) -> dict:
        self.calls.append(dict(params))
        params["api_key"] = "FAKE"
        return copy.deepcopy(self.answers[params["engine"]])

    def engines(self) -> list[str]:
        return [c["engine"] for c in self.calls]


def default_answers() -> dict[str, dict]:
    return {
        "google_flights_autocomplete": load_fixture("google_flights_autocomplete"),
        "google_flights": load_fixture("google_flights"),
        "google_hotels": load_fixture("google_hotels"),
    }


@pytest.fixture
def profile() -> SpeakerProfile:
    return SpeakerProfile(
        topics=["Python concurrency"],
        home_airport="COK",
        budget_inr=150_000,
        window_start=date(2026, 11, 12),
        window_end=date(2027, 10, 12),
    )


def make_cfp(**overrides) -> CFP:
    fields = dict(
        conference_name="PyGotham 2026",
        city="New York",
        country="United States",
        event_start=date(2026, 12, 10),
        event_end=date(2026, 12, 12),
        confidence=0.9,
    )
    fields.update(overrides)
    return CFP(**fields)


def use_fake(ctx, answers: dict[str, dict] | None = None) -> EngineFake:
    fake = EngineFake(answers if answers is not None else default_answers())
    ctx.serp_client_factory = fake.factory
    return fake


def test_price_trip_ok_total_is_flight_plus_hotel(ctx, profile):
    fake = use_fake(ctx)
    trip = price_trip(make_cfp(), profile, ctx)

    assert trip.status == "ok"
    assert trip.reason is None
    assert trip.airport_codes == ["JFK", "LGA", "EWR"]
    assert trip.nights == 4
    assert trip.outbound_date == date(2026, 12, 9)
    assert trip.return_date == date(2026, 12, 13)
    assert trip.flight_inr == 2512
    assert trip.hotel_nightly_inr == 247
    assert trip.hotel_inr == 247 * 4
    assert trip.total_inr == 2512 + 247 * 4
    assert trip.fetched_at is not None

    assert sorted(fake.engines()) == ["google_flights", "google_flights_autocomplete", "google_hotels"]
    by_engine = {c["engine"]: c for c in fake.calls}
    assert by_engine["google_flights_autocomplete"]["q"] == "New York, United States"
    assert by_engine["google_flights_autocomplete"]["exclude_regions"] == "true"
    flights = by_engine["google_flights"]
    assert flights["departure_id"] == "COK"
    assert flights["arrival_id"] == "JFK,LGA,EWR"
    assert (flights["outbound_date"], flights["return_date"]) == ("2026-12-09", "2026-12-13")
    hotels = by_engine["google_hotels"]
    assert (hotels["check_in_date"], hotels["check_out_date"]) == ("2026-12-09", "2026-12-13")
    assert "New York" in hotels["q"]
    assert ctx.count("serpapi_searches") == 3


def test_price_trip_rerun_is_free(ctx, profile):
    fake = use_fake(ctx)
    first = price_trip(make_cfp(), profile, ctx)
    second = price_trip(make_cfp(), profile, ctx)

    assert second.total_inr == first.total_inr
    assert len(fake.calls) == 3
    assert ctx.count("serpapi_searches") == 3
    assert ctx.count("serpapi_cache_hits") == 3


def test_price_trip_online_is_free_with_no_calls(ctx, profile):
    fake = use_fake(ctx)
    trip = price_trip(make_cfp(is_online=True, city=None), profile, ctx)
    assert trip.status == "ok"
    assert trip.total_inr == 0
    assert fake.calls == []


def test_price_trip_without_event_start(ctx, profile):
    fake = use_fake(ctx)
    trip = price_trip(make_cfp(event_start=None, event_end=None), profile, ctx)
    assert trip.status == "unavailable"
    assert trip.reason == "Event dates not published"
    assert trip.total_inr is None
    assert fake.calls == []


def test_price_trip_too_far_ahead(ctx, profile):
    fake = use_fake(ctx)
    # today is 2026-10-12; outbound 2027-09-30 is 353 days away (> 330)
    trip = price_trip(make_cfp(event_start=date(2027, 10, 1), event_end=date(2027, 10, 3)), profile, ctx)
    assert trip.status == "unavailable"
    assert trip.reason == "Fares not yet published"
    assert trip.flight_inr is None and trip.hotel_inr is None
    # Only the (forever-cached) airport lookup, to spot a home-city event; no flight or hotel search.
    assert [c["engine"] for c in fake.calls] == ["google_flights_autocomplete"]
    assert ctx.count("serpapi_searches") == 1


def test_price_trip_past_travel_dates(ctx, profile):
    fake = use_fake(ctx)
    trip = price_trip(make_cfp(event_start=date(2026, 10, 12), event_end=date(2026, 10, 13)), profile, ctx)
    assert trip.status == "unavailable"
    # Only the (forever-cached) airport lookup, to spot a home-city event; no flight or hotel search.
    assert [c["engine"] for c in fake.calls] == ["google_flights_autocomplete"]


def test_price_trip_home_airport_in_city(ctx, profile):
    fake = use_fake(ctx)
    home = profile.model_copy(update={"home_airport": "LGA"})
    trip = price_trip(make_cfp(), home, ctx)
    assert trip.status == "ok"
    assert trip.flight_inr == 0 and trip.hotel_inr == 0 and trip.total_inr == 0
    assert fake.engines() == ["google_flights_autocomplete"]


def test_price_trip_no_flight_results_is_partial(ctx, profile):
    answers = default_answers()
    answers["google_flights"] = NO_FLIGHTS
    use_fake(ctx, answers)

    trip = price_trip(make_cfp(), profile, ctx)

    assert trip.status == "partial"
    assert trip.flight_inr is None
    assert trip.hotel_inr == 247 * 4
    assert trip.total_inr is None
    assert "flight" in trip.reason.lower()


def test_price_trip_no_hotel_results_is_partial(ctx, profile):
    answers = default_answers()
    answers["google_hotels"] = NO_HOTELS
    use_fake(ctx, answers)

    trip = price_trip(make_cfp(), profile, ctx)

    assert trip.status == "partial"
    assert trip.flight_inr == 2512
    assert trip.hotel_inr is None and trip.hotel_nightly_inr is None
    assert trip.total_inr is None
    assert "hotel" in trip.reason.lower()


def test_price_trip_no_fares_far_ahead_says_not_published(ctx, profile):
    answers = default_answers()
    answers["google_flights"] = NO_FLIGHTS
    use_fake(ctx, answers)
    # outbound 2027-08-08 is 300 days ahead: inside the 330-day horizon, beyond 270
    trip = price_trip(make_cfp(event_start=date(2027, 8, 9), event_end=date(2027, 8, 11)), profile, ctx)
    assert trip.flight_inr is None
    assert "Fares not yet published" in trip.reason
    assert trip.status != "ok"


def test_price_trip_neither_is_unavailable(ctx, profile):
    answers = default_answers()
    answers["google_flights"] = NO_FLIGHTS
    answers["google_hotels"] = NO_HOTELS
    use_fake(ctx, answers)

    trip = price_trip(make_cfp(), profile, ctx)

    assert trip.status == "unavailable"
    assert trip.flight_inr is None and trip.hotel_inr is None and trip.total_inr is None
    assert trip.reason


def test_price_trip_cached_only_miss(ctx, profile):
    fake = use_fake(ctx)
    ctx.cached_only = True

    trip = price_trip(make_cfp(), profile, ctx)

    assert trip.status == "unavailable"
    assert "cache" in trip.reason.lower()
    assert fake.calls == []


def test_resolve_airports_falls_back_to_bare_city(ctx):
    class Fallback(EngineFake):
        def search(self, params: dict) -> dict:
            self.calls.append(dict(params))
            if params["q"] == "Seoul, South Korea":
                return {"suggestions": []}
            return load_fixture("google_flights_autocomplete_korea")

    fake = Fallback({})
    ctx.serp_client_factory = fake.factory
    kgmid, codes, response = resolve_airports("Seoul", "South Korea", ctx)
    assert (kgmid, codes) == ("/m/0hsqf", ["GMP", "ICN"])
    assert [c["q"] for c in fake.calls] == ["Seoul, South Korea", "Seoul"]
    assert response.ok
