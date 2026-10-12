"""Step 5: validate extracted CFPs and merge duplicates.

De-duplication key: the conference name lowercased with punctuation, the year and
CFP boilerplate removed, plus the event year. "PyCon US 2027" and "pycon us" (with a
2027 event date) share the key ``pycon us|2027``.
"""

from __future__ import annotations

import re
from datetime import date

from .models import CFP, SpeakerProfile

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_BOILERPLATE = re.compile(
    r"\b(call for (?:proposals|papers|speakers|talks|presentations|participation)|cfp|cfs|sessionize|pretalx)\b"
)


def normalize_name(name: str) -> str:
    text = name.lower()
    text = _YEAR.sub(" ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = re.sub(r"['’]\d{2}\b", " ", text)  # PyCon '27
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.split())


def event_year(cfp: CFP) -> int | None:
    if cfp.event_start is not None:
        return cfp.event_start.year
    match = _YEAR.search(cfp.conference_name)
    return int(match.group(0)) if match else None


def dedupe_key(cfp: CFP) -> str:
    year = event_year(cfp)
    return f"{normalize_name(cfp.conference_name)}|{year if year is not None else ''}"


def drop_reason(cfp: CFP, profile: SpeakerProfile, today: date) -> str | None:
    """Why a record should be dropped, or None to keep it."""
    if not cfp.is_cfp_open:
        return "CFP closed"
    if cfp.confidence < 0.5:
        return "Low extraction confidence"
    if cfp.cfp_deadline is not None and cfp.cfp_deadline < today:
        return "Deadline has passed"
    if not cfp.city and not cfp.is_online:
        return "No city and not online"
    if cfp.event_end is not None and cfp.event_end < today:
        return "Event is over"
    if cfp.event_start is not None and not (profile.window_start <= cfp.event_start <= profile.window_end):
        return "Starts outside your travel window"
    return None


def keep_valid(cfps: list[CFP], profile: SpeakerProfile, today: date) -> tuple[list[CFP], dict[str, int]]:
    """Return the records worth keeping and a count of drop reasons (for the progress panel)."""
    kept: list[CFP] = []
    dropped: dict[str, int] = {}
    for cfp in cfps:
        reason = drop_reason(cfp, profile, today)
        if reason is None:
            kept.append(cfp)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    return kept, dropped


def _fill_missing(primary: CFP, other: CFP) -> CFP:
    """Keep the primary record; borrow fields it lacks from the duplicate, as consistent groups."""
    update: dict = {}
    if primary.cfp_deadline is None and other.cfp_deadline is not None:
        update.update(
            cfp_deadline=other.cfp_deadline, deadline_is_aoe=other.deadline_is_aoe, evidence=other.evidence
        )
    if primary.event_start is None and other.event_start is not None:
        update.update(event_start=other.event_start, event_end=other.event_end)
    if not primary.city and other.city:
        update.update(city=other.city, country=other.country or primary.country, venue=other.venue or primary.venue)
    if not primary.cfp_url and other.cfp_url:
        update["cfp_url"] = other.cfp_url
    merged_topics = list(dict.fromkeys([*primary.topics, *other.topics]))
    if merged_topics != primary.topics:
        update["topics"] = merged_topics[:8]
    return primary.model_copy(update=update) if update else primary


def merge_duplicates(cfps: list[CFP]) -> dict[str, CFP]:
    """Merge records that share a dedupe key, keeping the higher-confidence one."""
    merged: dict[str, CFP] = {}
    for cfp in cfps:
        key = dedupe_key(cfp)
        current = merged.get(key)
        if current is None:
            merged[key] = cfp
        elif cfp.confidence > current.confidence:
            merged[key] = _fill_missing(cfp, current)
        else:
            merged[key] = _fill_missing(current, cfp)

    # "PyCon US" (no dates, no year) folds into "PyCon US 2027" when exactly one dated edition is known.
    for key in [k for k in merged if k.endswith("|")]:
        name = key[:-1]
        dated = [k for k in merged if k != key and k.partition("|")[0] == name and not k.endswith("|")]
        if name and len(dated) == 1:
            undated, twin = merged.pop(key), merged[dated[0]]
            merged[dated[0]] = (
                _fill_missing(undated, twin) if undated.confidence > twin.confidence else _fill_missing(twin, undated)
            )
    return merged
