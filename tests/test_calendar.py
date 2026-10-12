"""The .ics export: one all-day event per CFP deadline, with two reminders."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from icalendar import Calendar

from cfp_compass.calendar_export import build_ics, event_uid
from cfp_compass.models import CFP, RankedCFP, TravelSupport, TripCost


def _row(
    key: str,
    name: str,
    deadline: date | None,
    *,
    url: str = "https://us.pycon.example/2027/cfp",
    net: int | None = 168_000,
    status: str = "grant_available",
    **cfp_kw,
) -> RankedCFP:
    return RankedCFP(
        key=key,
        cfp=CFP(conference_name=name, cfp_url=url, cfp_deadline=deadline, confidence=0.9, **cfp_kw),
        travel=TravelSupport(status=status),
        trip=TripCost(total_inr=net, status="ok" if net is not None else "unavailable"),
        net_cost_inr=net,
    )


ROWS = [
    _row(
        "pycon us|2027",
        "PyCon US 2027",
        date(2026, 12, 15),
        event_start=date(2027, 5, 14),
        event_end=date(2027, 5, 22),
        city="Long Beach",
        country="USA",
    ),
    _row("no deadline|2027", "Mystery Conf 2027", None),
    _row(
        "pyconf hyderabad|2027",
        "PyConf Hyderabad 2027",
        date(2026, 11, 2),
        url="",
        net=None,
        status="unknown",
        deadline_is_aoe=True,
    ),
]


def _events(ics: bytes):
    return Calendar.from_ical(ics).walk("VEVENT")


def test_one_vevent_per_row_with_deadline():
    ics = build_ics(ROWS)
    assert isinstance(ics, bytes)
    events = _events(ics)
    assert len(events) == 2
    assert [str(e["SUMMARY"]) for e in events] == ["CFP closes: PyCon US 2027", "CFP closes: PyConf Hyderabad 2027"]


def test_empty_rows_still_valid_calendar():
    cal = Calendar.from_ical(build_ics([]))
    assert cal.walk("VEVENT") == []
    assert str(cal["VERSION"]) == "2.0"


def test_all_day_dates():
    ics = build_ics(ROWS[:1])
    (event,) = _events(ics)
    start = event.decoded("DTSTART")
    end = event.decoded("DTEND")
    assert type(start) is date and start == date(2026, 12, 15)
    assert type(end) is date and end == date(2026, 12, 16)
    assert b"DTSTART;VALUE=DATE:20261215" in ics
    assert b"DTEND;VALUE=DATE:20261216" in ics


def test_description_has_link_cost_and_travel_status():
    first, second = _events(build_ics(ROWS))
    desc = str(first["DESCRIPTION"])
    assert "https://us.pycon.example/2027/cfp" in desc
    assert "₹1,68,000" in desc
    assert "Grant available" in desc
    assert str(first["URL"]) == "https://us.pycon.example/2027/cfp"

    desc2 = str(second["DESCRIPTION"])
    assert "Net cost: unavailable" in desc2
    assert "Unknown" in desc2
    assert "AoE" in desc2


def test_description_falls_back_to_source_url():
    row = _row("k|2027", "Conf 2027", date(2026, 12, 1), url="")
    row.cfp.source_url = "https://sessionize.example/conf-2027"
    (event,) = _events(build_ics([row]))
    assert "https://sessionize.example/conf-2027" in str(event["DESCRIPTION"])
    assert str(event["URL"]) == "https://sessionize.example/conf-2027"


def test_two_alarms_seven_days_and_one_day_before():
    for event in _events(build_ics(ROWS)):
        alarms = event.walk("VALARM")
        assert len(alarms) == 2
        assert [a.decoded("TRIGGER") for a in alarms] == [timedelta(days=-7), timedelta(days=-1)]
        assert all(str(a["ACTION"]) == "DISPLAY" for a in alarms)
        assert all(str(a["DESCRIPTION"]) for a in alarms)


def test_uid_stable_across_builds_and_unique_per_key():
    first = build_ics(ROWS, generated_at=datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc))
    second = build_ics(ROWS, generated_at=datetime(2026, 10, 13, 9, 30, tzinfo=timezone.utc))
    uids_1 = [str(e["UID"]) for e in _events(first)]
    uids_2 = [str(e["UID"]) for e in _events(second)]
    assert uids_1 == uids_2
    assert len(set(uids_1)) == len(uids_1) == 2
    assert uids_1[0] == event_uid("pycon us|2027")
    assert event_uid("pycon us|2027") == event_uid("pycon us|2027")
    assert event_uid("pycon us|2027") != event_uid("pycon us|2026")
    assert event_uid("pycon us|2027").endswith("@cfp-compass")


def test_output_parses_and_round_trips():
    ics = build_ics(ROWS, generated_at=datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc))
    cal = Calendar.from_ical(ics)
    assert str(cal["PRODID"]).startswith("-//CFP Compass")
    assert str(cal["VERSION"]) == "2.0"
    assert cal.to_ical() == ics
    for event in cal.walk("VEVENT"):
        assert event.decoded("DTSTAMP") == datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc)
