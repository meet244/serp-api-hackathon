"""CFP Compass: the Streamlit UI. It only draws; all logic lives in the cfp_compass package.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date, datetime
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from cfp_compass.calendar_export import build_ics
from cfp_compass.config import ConfigError, load_settings
from cfp_compass.models import PipelineResult, RankedCFP, ScoreWeights, SpeakerProfile, describe_validation_error
from cfp_compass.pipeline import Progress, build_context, default_window, run_pipeline
from cfp_compass.scoring import TRAVEL_LABELS, format_inr, rescore

st.set_page_config(page_title="CFP Compass", page_icon="🧭", layout="wide")

def md(text: object) -> str:
    """Escape web or LLM text so it shows literally instead of rendering as Markdown or HTML."""
    out = str(text or "")
    for ch in "\\`*_{}[]()<>#+-.!|$~":
        out = out.replace(ch, "\\" + ch)
    return out


def link(url: str | None) -> str:
    """A Markdown link that only accepts http(s) URLs."""
    if not url or not url.startswith(("http://", "https://")):
        return "not available"
    return f"[{md(url)}](<{url.replace('>', '%3E')}>)"


TRAVEL_ICONS = {"covered": "✅", "partial": "🟡", "grant_available": "🎟️", "none": "❌", "unknown": "❔"}


# --- Startup: keys must exist before anything else -------------------------------------
try:
    settings = load_settings()
except ConfigError as exc:
    st.title("🧭 CFP Compass")
    st.error(f"**Setup needed.** {exc}")
    st.code("cp .env.example .env   # then add SERPAPI_API_KEY and one LLM key", language="bash")
    st.stop()

today = date.today()
default_start, default_end = default_window(today)

# A previous run hit a SerpApi error (e.g. credits ran out): flip the toggle before it renders.
if st.session_state.pop("force_cached_only", False):
    st.session_state["cached_only"] = True


# --- Sidebar --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Speaker profile")
    # A form: editing an input does not rerun (and so cannot cancel) a search in progress.
    with st.form("profile", border=False):
        topics_text = st.text_input(
            "Talk topics", placeholder="Python concurrency, AI agents", help="Comma-separated, at most 2."
        )
        abstract = st.text_area("Talk abstract (optional)", placeholder="Improves topic-fit scoring", height=100)
        home_airport = st.text_input("Home airport (IATA)", value="COK", max_chars=3)
        budget = st.number_input("Total budget (₹)", min_value=1_000, value=150_000, step=5_000)
        window = st.date_input(
            "Travel window (event start dates)", value=(default_start, default_end), min_value=today,
            format="DD/MM/YYYY",
        )
        max_conferences = st.slider(
            "Conferences to price", min_value=5, max_value=15, value=12,
            help="Only the best-fitting conferences are priced, to protect the SerpApi credit budget.",
        )
        cached_only = st.toggle(
            "Cached results only", key="cached_only",
            help="Never spend SerpApi credits: anything not already cached shows as unavailable.",
        )
        find = st.form_submit_button("Find CFPs", type="primary", width="stretch")
    with st.expander("Score weights"):
        w_fit = st.slider("Topic fit", 0.0, 1.0, 0.45, 0.05)
        w_afford = st.slider("Affordability", 0.0, 1.0, 0.35, 0.05)
        w_urgency = st.slider("Deadline urgency", 0.0, 1.0, 0.20, 0.05)
        st.caption("Weights are normalised to sum to 1. Changing them re-ranks instantly, with no new searches.")
    st.caption(f"LLM: `{settings.llm_model}`")

weights = ScoreWeights(fit=w_fit, affordability=w_afford, urgency=w_urgency)

st.title("🧭 CFP Compass")
st.caption(
    "Which conference proposals are worth writing this month? Open CFPs ranked by topic fit, deadline "
    "and the real cost of the trip from your home city, with live flight and hotel prices via SerpApi."
)


# --- Run --------------------------------------------------------------------------------------
def run_from_form() -> None:
    if not (isinstance(window, tuple) and len(window) == 2):
        st.error("Pick both a start and an end date for the travel window.")
        return
    try:
        profile = SpeakerProfile(
            topics=topics_text, talk_abstract=abstract, home_airport=home_airport, budget_inr=int(budget),
            window_start=window[0], window_end=window[1], max_conferences=max_conferences,
        )
    except ValidationError as exc:
        for message in describe_validation_error(exc):
            st.error(message)
        return

    ctx = build_context(settings, cached_only=cached_only, today=today)
    log: list[str] = []
    with st.status("Finding CFPs…", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")

        def on_progress(p: Progress) -> None:  # called on this (script) thread only
            bar.progress(min(p.step / p.total, 1.0), text=f"Step {p.step} of {p.total}: {p.detail}")
            status.update(label=f"Step {p.step} of {p.total}: {p.title}")
            if p.done:
                line = f"**{p.title}**: {md(p.detail)}"
                log.append(line)
                st.write(f"✓ {line}")

        try:
            result = run_pipeline(profile, ctx, weights=weights, progress=on_progress)
        except Exception as exc:  # last-resort guard: keep the app alive
            status.update(label="The run failed", state="error")
            st.error(f"Something went wrong: {type(exc).__name__}. Previous results are kept.")
            return
        status.update(
            label=f"Done: {len(result.ranked)} conferences ranked in {result.stats.duration_seconds}s",
            state="complete", expanded=False,
        )
    st.session_state["result"] = result
    st.session_state["run_log"] = log
    if result.stats.api_disabled_reason and not cached_only:
        st.session_state["force_cached_only"] = True
        st.rerun()


if find:
    run_from_form()


# --- Results ----------------------------------------------------------------------------------
def deadline_text(row: RankedCFP) -> str:
    d = row.cfp.cfp_deadline
    if d is None:
        return "Not published"
    days = row.days_to_deadline
    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
    return f"{d:%d %b %Y} · {when}" + (" · AoE" if row.cfp.deadline_is_aoe else "")


def event_text(row: RankedCFP) -> str:
    cfp = row.cfp
    if cfp.event_start:
        end = cfp.event_end or cfp.event_start
        dates = f"{cfp.event_start:%d %b}" if end == cfp.event_start else f"{cfp.event_start:%d %b}–{end:%d %b %Y}"
        if end == cfp.event_start:
            dates += f" {cfp.event_start:%Y}"
    else:
        dates = "Dates TBA"
    return f"{dates} · Online" if cfp.is_online else f"{dates} · {cfp.city or '?'}"


def money(value: int | None, row: RankedCFP) -> str:
    if row.cfp.is_online:
        return "₹0 (online)"
    return format_inr(value) if value is not None else "unavailable"


def net_text(row: RankedCFP) -> str:
    if row.net_cost_inr is None:
        return "unavailable"
    return format_inr(row.net_cost_inr) + ("  🔴 Over budget" if row.over_budget else "")


def table(rows: list[RankedCFP]) -> pd.DataFrame:
    records = []
    for row in rows:
        url = row.cfp.cfp_url or row.cfp.source_url
        records.append({
            "Rank": row.rank,
            # The link column shows the name after '#': see display_text below.
            "Conference": f"{url}#{quote_plus(row.cfp.conference_name)}",
            "Deadline": deadline_text(row),
            "Event": event_text(row),
            "Fit": row.fit_score,
            "Travel support": f"{TRAVEL_ICONS.get(row.travel.status, '')} {TRAVEL_LABELS.get(row.travel.status)}",
            "Flight (₹)": money(row.trip.flight_inr, row),
            "Hotel (₹)": money(row.trip.hotel_inr, row),
            "Net cost (₹)": net_text(row),
            "Score": row.final_score,
            "Why": row.reasons,
        })
    return pd.DataFrame.from_records(records)


result: PipelineResult | None = st.session_state.get("result")

if result is None:
    st.info("Enter your talk topics in the sidebar and click **Find CFPs**.")
    st.markdown(
        "**How it works.** CFP Compass searches Google for open calls for proposals, extracts each "
        "conference's deadline, dates and city with an LLM, checks whether the conference covers speaker "
        "travel, then prices the round-trip flight and hotel from your home airport, all through SerpApi."
    )
    st.stop()

stats = result.stats
if stats.api_disabled_reason:
    st.warning(
        f"SerpApi stopped answering during the run ({md(stats.api_disabled_reason)}). Switched to cached results "
        "only; showing whatever was cached."
    )
elif stats.cached_only:
    st.info("Cached results only: this run spent no SerpApi credits. Anything not cached shows as unavailable.")
if stats.llm_disabled_reason:
    st.warning(f"{md(stats.llm_disabled_reason)} Results use whatever the LLM had already extracted; try again later.")

rows = rescore(result.ranked, weights)
within = [r for r in rows if r.net_cost_inr is not None and not r.over_budget]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CFPs found", stats.valid)
m2.metric("Conferences priced", stats.priced)
m3.metric("Within budget", len(within))
m4.metric("SerpApi searches used this run", stats.serpapi_searches)
m5.metric("Cache hits", stats.serpapi_cache_hits)

f1, f2, _ = st.columns([1, 1, 2])
only_budget = f1.toggle("Within budget only")
only_soon = f2.toggle("Deadline in the next 30 days")
shown = [
    r for r in rows
    if (not only_budget or (r.net_cost_inr is not None and not r.over_budget))
    and (not only_soon or (r.days_to_deadline is not None and 0 <= r.days_to_deadline <= 30))
]

if not shown:
    st.warning("No conferences match these filters." if rows else "No open CFPs matched your topics and travel window.")
else:
    st.dataframe(
        table(shown),
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Conference": st.column_config.LinkColumn(display_text=r"#([^#]*)$", width="medium",
                                                      help="Opens the CFP page"),
            "Fit": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f", width="small"),
            "Travel support": st.column_config.TextColumn(
                help="Speaker travel policy found by search. The note is in the conference details below."),
            "Net cost (₹)": st.column_config.TextColumn(help="Flight + hotel after the conference's travel support"),
            "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f", width="small"),
            "Why": st.column_config.ListColumn(width="large"),
        },
    )

    st.subheader("Conference details")
    for row in shown:
        cfp, trip, travel = row.cfp, row.trip, row.travel
        with st.expander(f"{row.rank}. {md(cfp.conference_name)}"):
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Deadline evidence:** “{md(cfp.evidence)}”" if cfp.evidence else "**Deadline evidence:** none found")
                st.markdown(f"**CFP source:** {link(cfp.source_url or cfp.cfp_url)}")
                if cfp.cfp_url and cfp.cfp_url != cfp.source_url:
                    st.markdown(f"**Submit at:** {link(cfp.cfp_url)}")
                place = "Online" if cfp.is_online else ", ".join(p for p in (cfp.venue, cfp.city, cfp.country) if p)
                topics = ", ".join(cfp.topics) or "not listed"
                st.markdown(f"**Where:** {md(place or 'not published')} · **Topics:** {md(topics)}")
                st.markdown(f"**Fit:** {row.fit_score:.2f} · {md(row.fit_reason)}")
            with right:
                label = f"{TRAVEL_ICONS.get(travel.status, '')} {TRAVEL_LABELS.get(travel.status)}"
                st.markdown(f"**Travel support:** {label}" + (f": {md(travel.note)}" if travel.note else ""))
                if travel.evidence_url:
                    st.markdown(f"**Travel policy source:** {link(travel.evidence_url)}")
                airports = ", ".join(trip.airport_codes) or "not looked up"
                st.markdown(f"**Airports used:** {md(airports)}")
                if trip.outbound_date and trip.return_date:
                    st.markdown(
                        f"**Trip:** fly {trip.outbound_date:%d %b} → return {trip.return_date:%d %b %Y} "
                        f"· {trip.nights} hotel nights"
                        + (f" at {format_inr(trip.hotel_nightly_inr)}/night (median)" if trip.hotel_nightly_inr else "")
                    )
                st.markdown(f"**Trip status:** {trip.status}" + (f" ({md(trip.reason)})" if trip.reason else ""))

    d1, d2, _ = st.columns([1, 1, 2])
    csv = table(shown).assign(
        Conference=[r.cfp.conference_name for r in shown],
        CFP_URL=[r.cfp.cfp_url or r.cfp.source_url for r in shown],
        Why=["; ".join(r.reasons) for r in shown],
    ).to_csv(index=False).encode("utf-8")
    d1.download_button("Download CSV", csv, file_name="cfp_compass.csv", mime="text/csv", on_click="ignore")
    d2.download_button(
        "Download .ics calendar", build_ics(shown, generated_at=result.generated_at.astimezone()),
        file_name="cfp_deadlines.ics", mime="text/calendar", on_click="ignore",
    )

if result.unpriced:
    with st.expander(f"{len(result.unpriced)} more open CFPs (lower topic fit, not priced)"):
        for cfp in result.unpriced:
            deadline = f"{cfp.cfp_deadline:%d %b %Y}" if cfp.cfp_deadline else "deadline not published"
            st.markdown(f"- {md(cfp.conference_name)} · {deadline} · {link(cfp.cfp_url or cfp.source_url)}")

fetched = result.prices_fetched_at
from_cache = stats.cached_only or bool(stats.api_disabled_reason) or stats.serpapi_searches == 0
st.caption(
    "Flight and hotel prices are live estimates from Google Flights and Google Hotels via SerpApi"
    + (f"; the oldest was fetched {fetched:%d %b %Y at %H:%M}" if fetched else "")
    + (" (served from the cache)." if from_cache and fetched else ".")
    + " CFPs, travel policies, airports, flights and hotels all come from live search."
)

with st.expander("Run log"):
    for line in st.session_state.get("run_log", []):
        st.markdown(f"- {line}")
    s = stats
    st.markdown(
        f"- SerpApi searches: {s.serpapi_searches} · cache hits: {s.serpapi_cache_hits} · LLM calls: {s.llm_calls} "
        f"(cached: {s.llm_cache_hits}) · duration: {s.duration_seconds}s"
    )
    for error in s.errors:
        st.markdown(f"- ⚠️ {md(error)}")
