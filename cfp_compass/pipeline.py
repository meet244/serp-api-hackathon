"""Orchestrates the ten pipeline steps and reports progress through a callback.

Also runnable as a CLI:

    python -m cfp_compass.pipeline run --topics "Python concurrency, AI agents" --home COK --budget 150000
    python -m cfp_compass.pipeline search "call for papers python conference"
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, TypeVar

from pydantic import ValidationError

from .cache import ResponseCache
from .calendar_export import build_ics
from .config import ConfigError, Settings, load_settings
from .context import RunContext
from .costs import price_trip
from .dedupe import keep_valid, merge_duplicates
from .discover import build_queries, discover, discovery_params, plan_search_terms, prefilter
from .enrich import find_travel_support
from .extract import extract_all
from .fit import score_fit
from .models import (
    CFP,
    PipelineResult,
    RunStats,
    ScoreWeights,
    SpeakerProfile,
    TravelSupport,
    TripCost,
    describe_validation_error,
)
from .scoring import TRAVEL_LABELS, format_inr, score_cfp, sort_and_number
from .serp import TTL_DISCOVERY, search

TOTAL_STEPS = 10
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class Progress:
    step: int
    title: str
    detail: str = ""
    done: bool = False
    total: int = TOTAL_STEPS


ProgressCallback = Callable[[Progress], None]


def add_months(day: date, months: int) -> date:
    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def default_window(today: date) -> tuple[date, date]:
    """From 1 month to 12 months from today."""
    return add_months(today, 1), add_months(today, 12)


def build_context(settings: Settings, *, cached_only: bool = False, today: date | None = None) -> RunContext:
    return RunContext(
        cache=ResponseCache(settings.cache_path),
        serpapi_api_key=settings.serpapi_api_key,
        llm_model=settings.llm_model,
        cached_only=cached_only,
        today=today or date.today(),
    )


def _map_parallel(
    items: Sequence[T],
    fn: Callable[[T], R],
    fallback: Callable[[T, Exception], R],
    workers: int,
    on_progress: Callable[[int, int], None],
) -> list[R]:
    """Run ``fn`` over items (order kept). Progress is reported from the calling thread."""
    results: list[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for done, future in enumerate(as_completed(futures), start=1):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:  # a bug in one lookup must not sink the run
                results[i] = fallback(items[i], exc)
            on_progress(done, len(items))
    return results


def run_pipeline(
    profile: SpeakerProfile,
    ctx: RunContext,
    *,
    weights: ScoreWeights | None = None,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    started = time.monotonic()
    weights = weights or ScoreWeights()
    report: ProgressCallback = progress or (lambda _p: None)

    # 1. Validate input (SpeakerProfile already enforces every rule).
    report(Progress(1, "Validate input", (
        f"{' + '.join(profile.topics)} · from {profile.home_airport} · budget {format_inr(profile.budget_inr)} · "
        f"events {profile.window_start:%d %b %Y} to {profile.window_end:%d %b %Y}"
    ), done=True))

    # 2. Discover.
    report(Progress(2, "Discover CFPs", "Planning search terms"))
    terms = plan_search_terms(profile.topics, ctx)
    n_queries = len(build_queries(terms, ctx.today))
    report(Progress(2, "Discover CFPs", f"Running {n_queries} Google searches for: {', '.join(terms)}"))
    candidates, n_queries = discover(profile, ctx, terms)
    report(Progress(2, "Discover CFPs", f"Discovered {len(candidates)} candidates from {n_queries} searches for {', '.join(terms)}", done=True))

    # 3. Cheap pre-filter.
    kept = prefilter(candidates)
    report(Progress(3, "Pre-filter", f"{len(kept)} of {len(candidates)} candidates look like CFPs", done=True))

    # 4. Extract.
    report(Progress(4, "Extract CFP details", f"Extracting {len(kept)} candidates"))
    cfps = extract_all(
        kept, ctx, on_progress=lambda d, t: report(Progress(4, "Extract CFP details", f"Extracted {d} of {t} candidates"))
    )
    # Extractions come back in thread-completion order. Restore search order so merging equally
    # confident duplicates (and every cache key built from the merged records) is the same each run.
    position = {candidate.url: i for i, candidate in enumerate(kept)}
    cfps.sort(key=lambda cfp: position.get(cfp.source_url, len(position)))
    report(Progress(4, "Extract CFP details", f"Extracted {len(cfps)} CFP records from {len(kept)} candidates", done=True))

    # 5. Validate and de-duplicate.
    valid, dropped = keep_valid(cfps, profile, ctx.today)
    merged = merge_duplicates(valid)
    detail = f"{len(merged)} open CFPs in your travel window"
    if dropped:
        detail += " (dropped: " + ", ".join(f"{n} {reason.lower()}" for reason, n in sorted(dropped.items())) + ")"
    report(Progress(5, "Validate and de-duplicate", detail, done=True))

    # 6. Score fit.
    report(Progress(6, "Score topic fit", f"Scoring {len(merged)} conferences"))
    fits, used_llm = score_fit(profile, merged, ctx)
    report(Progress(6, "Score topic fit", f"Scored {len(merged)} conferences" + ("" if used_llm or not merged else " (keyword fallback)"), done=True))

    # 7. Shortlist: only the best-fitting conferences get priced (protects the credit budget).
    ordered = sorted(merged.items(), key=lambda kv: (-fits[kv[0]][0], kv[1].cfp_deadline or date.max, kv[0]))
    shortlist = ordered[: profile.max_conferences]
    unpriced = [cfp for _, cfp in ordered[profile.max_conferences :]]
    report(Progress(7, "Shortlist", f"Top {len(shortlist)} by fit will be priced" + (f"; {len(unpriced)} more not priced" if unpriced else ""), done=True))

    # 8. Travel support.
    def travel_failed(item: tuple[str, CFP], exc: Exception) -> TravelSupport:
        ctx.add_error(f"Travel-policy step failed for {item[1].conference_name}: {type(exc).__name__}")
        return TravelSupport(note="Travel policy could not be checked.")

    travel = _map_parallel(
        shortlist, lambda kv: find_travel_support(kv[1], kv[0], ctx), travel_failed, ctx.max_workers,
        lambda d, t: report(Progress(8, "Find travel support", f"Checked {d} of {t} travel policies")),
    )
    supported = sum(1 for t in travel if t.status in ("covered", "partial", "grant_available"))
    report(Progress(8, "Find travel support", f"Checked {len(travel)} travel policies; {supported} offer support", done=True))

    # 9. Price the trips.
    def pricing_failed(item: tuple[str, CFP], exc: Exception) -> TripCost:
        ctx.add_error(f"Pricing failed for {item[1].conference_name}: {type(exc).__name__}")
        return TripCost(status="unavailable", reason="Pricing failed")

    trips = _map_parallel(
        shortlist, lambda kv: price_trip(kv[1], profile, ctx), pricing_failed, ctx.max_workers,
        lambda d, t: report(Progress(9, "Price trips", f"Priced {d} of {t} trips")),
    )
    priced = sum(1 for t in trips if t.status in ("ok", "partial"))
    report(Progress(9, "Price trips", f"Priced {priced} of {len(trips)} trips (flights + hotels in INR)", done=True))

    # 10. Rank.
    rows = [
        score_cfp(
            key=key, cfp=cfp, travel=travel[i], trip=trips[i], fit=fits[key][0], fit_reason=fits[key][1],
            budget_inr=profile.budget_inr, today=ctx.today, weights=weights,
        )
        for i, (key, cfp) in enumerate(shortlist)
    ]
    ranked = sort_and_number(rows)
    fetched = [t.fetched_at for t in trips if t.fetched_at]
    stats = RunStats(
        serpapi_searches=ctx.count("serpapi_searches"),
        serpapi_cache_hits=ctx.count("serpapi_cache_hits"),
        llm_calls=ctx.count("llm_calls"),
        llm_cache_hits=ctx.count("llm_cache_hits"),
        errors=list(ctx.errors),
        duration_seconds=round(time.monotonic() - started, 1),
        queries=n_queries,
        candidates=len(candidates),
        candidates_after_prefilter=len(kept),
        extracted=len(cfps),
        valid=len(merged),
        shortlisted=len(shortlist),
        priced=priced,
        cached_only=ctx.cached_only,
        api_disabled_reason=ctx.api_disabled_reason,
        llm_disabled_reason=ctx.llm_disabled_reason,
    )
    report(Progress(10, "Rank", f"Ranked {len(ranked)} conferences in {stats.duration_seconds}s", done=True))
    return PipelineResult(
        profile=profile,
        ranked=ranked,
        unpriced=unpriced,
        stats=stats,
        weights=weights,
        generated_at=datetime.now(),
        prices_fetched_at=min(fetched) if fetched else None,  # the oldest price on the page
    )


# --- CLI -----------------------------------------------------------------------------------

def _cell(text: Any, width: int) -> str:
    value = str(text)
    return value if len(value) <= width else value[: width - 1] + "…"


def print_table(result: PipelineResult, out: Any = None) -> None:
    out = out or sys.stdout  # resolved per call, so redirected stdout is honoured
    columns = [("#", 3), ("Conference", 30), ("Deadline", 16), ("Event", 22), ("Fit", 4), ("Travel", 15),
               ("Flight", 10), ("Hotel", 10), ("Net cost", 12), ("Score", 5), ("Why", 60)]
    print("  ".join(_cell(name, width).ljust(width) for name, width in columns), file=out)
    for row in result.ranked:
        cfp, trip = row.cfp, row.trip
        deadline = "unknown"
        if cfp.cfp_deadline:
            deadline = f"{cfp.cfp_deadline:%d %b} ({row.days_to_deadline}d)" + (" AoE" if cfp.deadline_is_aoe else "")
        event = "Online" if cfp.is_online else (cfp.city or "?")
        if cfp.event_start and not cfp.is_online:
            event = f"{cfp.event_start:%d %b %y} {event}"
        net = format_inr(row.net_cost_inr) if row.net_cost_inr is not None else "unavailable"
        if row.over_budget:
            net += " OVER"
        values = [row.rank, cfp.conference_name, deadline, event, f"{row.fit_score:.2f}",
                  TRAVEL_LABELS.get(row.travel.status, row.travel.status),
                  format_inr(trip.flight_inr), format_inr(trip.hotel_inr), net, f"{row.final_score:.2f}",
                  "; ".join(row.reasons)]
        print("  ".join(_cell(v, w).ljust(w) for v, (_, w) in zip(values, columns)), file=out)
    s = result.stats
    print(
        f"\nSerpApi searches used this run: {s.serpapi_searches} · cache hits: {s.serpapi_cache_hits} · "
        f"LLM calls: {s.llm_calls} (cached: {s.llm_cache_hits}) · errors: {len(s.errors)} · {s.duration_seconds}s",
        file=out,
    )
    if s.api_disabled_reason:
        print(f"Switched to cached results only: {s.api_disabled_reason}", file=out)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # never echo request URLs or headers


def _cmd_search(args: argparse.Namespace) -> int:
    settings = load_settings(require_serpapi=not args.cached_only, require_llm=False)
    ctx = build_context(settings, cached_only=args.cached_only)
    response = search(discovery_params(args.query), ctx, ttl_seconds=TTL_DISCOVERY)
    if not response.ok:
        print(f"Search failed: {response.error}", file=sys.stderr)
    for i, item in enumerate((response.data.get("organic_results") or [])[:5], start=1):
        print(f"{i}. {item.get('title', '(no title)')}\n   {item.get('link', '')}")
    print(f"\nSerpApi searches used: {ctx.count('serpapi_searches')} · cache hits: {ctx.count('serpapi_cache_hits')}"
          f"{' (served from cache)' if response.from_cache else ''}")
    return 0 if response.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings(require_serpapi=not args.cached_only, require_llm=not args.cached_only)
    today = date.today()
    start, end = default_window(today)
    try:
        profile = SpeakerProfile(
            topics=args.topics, talk_abstract=args.abstract, home_airport=args.home, budget_inr=args.budget,
            window_start=args.window_start or start, window_end=args.window_end or end,
            max_conferences=args.max_conferences,
        )
    except ValidationError as exc:
        for message in describe_validation_error(exc):
            print(f"Invalid input: {message}", file=sys.stderr)
        return 2
    ctx = build_context(settings, cached_only=args.cached_only, today=today)

    def show(p: Progress) -> None:
        if p.done:
            print(f"[{p.step}/{p.total}] {p.title}: {p.detail}", file=sys.stderr)

    result = run_pipeline(profile, ctx, progress=show)
    print(file=sys.stderr)
    print_table(result)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
    if args.ics_path:
        with open(args.ics_path, "wb") as fh:
            fh.write(build_ics(result.ranked))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cfp_compass.pipeline",
        description="CFP Compass: open CFPs ranked by topic fit, deadline and the real trip cost.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline and print the ranked table")
    run.add_argument("--topics", required=True, help='comma-separated, at most 2, e.g. "Python concurrency, AI agents"')
    run.add_argument("--abstract", default=None, help="optional talk abstract (improves fit scoring)")
    run.add_argument("--home", default="COK", help="home airport IATA code (default COK)")
    run.add_argument("--budget", type=int, default=150_000, help="total budget in INR (default 150000)")
    run.add_argument("--from", dest="window_start", type=date.fromisoformat, help="travel window start, YYYY-MM-DD")
    run.add_argument("--to", dest="window_end", type=date.fromisoformat, help="travel window end, YYYY-MM-DD")
    run.add_argument("--max", dest="max_conferences", type=int, default=12, help="conferences to price (default 12)")
    run.add_argument("--cached-only", action="store_true", help="never call SerpApi; use cached responses only")
    run.add_argument("--json", dest="json_path", help="also write the full result as JSON")
    run.add_argument("--ics", dest="ics_path", help="also write the CFP deadlines as an .ics calendar")
    run.add_argument("-v", "--verbose", action="store_true")
    run.set_defaults(func=_cmd_run)

    probe = sub.add_parser("search", help="run one Google search through the cache and print 5 titles")
    probe.add_argument("query")
    probe.add_argument("--cached-only", action="store_true")
    probe.add_argument("-v", "--verbose", action="store_true")
    probe.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
