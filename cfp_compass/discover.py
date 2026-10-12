"""Steps 2 and 3: discover CFP candidates with Google Search, then a cheap keyword pre-filter."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from .context import RunContext
from .llm import make_agent, run_structured
from .models import Candidate, SpeakerProfile
from .serp import TTL_DISCOVERY, search

MAX_TOPICS = 2
MAX_CANDIDATES = 40
PREFILTER_WORDS = ("cfp", "call for", "proposal", "papers", "speak", "talk", "submit")
# Login walls and video sites: nothing to extract without an account.
SKIP_DOMAINS = ("linkedin.com", "facebook.com", "instagram.com", "x.com", "twitter.com", "youtube.com", "tiktok.com")


PLAN_VERSION = "plan-v1"
PLAN_TTL = 7 * 24 * 60 * 60  # longer than the discovery cache, so re-runs build identical queries


class SearchPlan(BaseModel):
    terms: list[str] = Field(
        description="One search term per talk topic, in the same order: the broad technology or community "
        "name conferences use in their names and tracks (1 to 3 words)."
    )


_plan_agent = make_agent(
    SearchPlan,
    """You turn a speaker's talk topics into conference search terms.
For each topic return the broad technology or community name that conferences use in their names or tracks,
1 to 3 words, in the same order as the topics. Examples: "Python concurrency" -> "Python"; "AI agents" ->
"AI agents"; "Kubernetes security" -> "Kubernetes"; "Rails performance" -> "Ruby on Rails";
"accessible web forms" -> "web accessibility". Keep a topic unchanged if it is already broad.""",
)


def plan_search_terms(topics: list[str], ctx: RunContext) -> list[str]:
    """One cached LLM call: niche talk topics -> the broad terms CFP pages actually use.

    Discovery searches for "Python", not "Python concurrency" (CFP pages rarely name a talk's niche);
    fit scoring still uses the full topics and abstract. Falls back to the topics themselves.
    """
    topics = topics[:MAX_TOPICS]
    try:
        plan = run_structured(
            _plan_agent,
            "Talk topics:\n" + "\n".join(f"- {t}" for t in topics),
            SearchPlan,
            ctx,
            cache_kind="plan",
            cache_params={"topics": [t.lower() for t in topics], "v": PLAN_VERSION},
            ttl_seconds=None if ctx.serp_offline else PLAN_TTL,
            cache_only=ctx.serp_offline,  # no searches possible: do not spend an LLM call either
        )
    except Exception as exc:
        if not ctx.serp_offline:
            ctx.add_error(f"Search planning failed, searching the topics as typed: {type(exc).__name__}")
        return topics
    terms = [" ".join(t.split())[:40] for t in plan.terms[: len(topics)]]
    terms += topics[len(terms):]  # the model returned too few terms
    terms = [term or topic for term, topic in zip(terms, topics)]
    return list(dict.fromkeys(terms))  # "Python" twice -> one set of queries


def search_years(today: date) -> str:
    """This year and next; from July on, next year only (this year's CFPs have mostly closed by then)."""
    return str(today.year + 1) if today.month >= 7 else f"{today.year} OR {today.year + 1}"


def build_queries(topics: list[str], today: date) -> list[str]:
    """Four discovery queries per topic (max 2 topics), aimed at calls that are still open."""
    years = search_years(today)
    queries: list[str] = []
    for topic in topics[:MAX_TOPICS]:
        queries += [
            f'"call for proposals" {topic} conference {years}',
            f'"call for papers" {topic} conference {years}',
            f"site:sessionize.com {topic}",  # speaker profiles are dropped by the pre-filter
            f"site:pretalx.com {topic} cfp",
        ]
    return queries


def discovery_params(query: str) -> dict[str, Any]:
    # CFPs are global, so search Google as seen from the US in English.
    return {"engine": "google", "q": query, "hl": "en", "gl": "us"}


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{parts.query}" if parts.query else "")


def parse_organic(data: dict[str, Any], query: str) -> list[Candidate]:
    """Read organic_results[]; every field is treated as optional."""
    candidates: list[Candidate] = []
    for item in data.get("organic_results") or []:
        if not isinstance(item, dict):
            continue
        link = str(item.get("link") or "")
        if not link.startswith(("http://", "https://")):
            continue
        position = item.get("position")
        candidates.append(
            Candidate(
                title=str(item.get("title") or ""),
                url=link,
                snippet=str(item.get("snippet") or ""),
                source_query=query,
                date=str(item["date"]) if item.get("date") else None,
                position=position if isinstance(position, int) else None,
            )
        )
    return candidates


def merge_candidates(result_lists: list[list[Candidate]], limit: int = MAX_CANDIDATES) -> list[Candidate]:
    """De-duplicate by URL, taking results round-robin so the cap keeps every query's best hits."""
    seen: set[str] = set()
    merged: list[Candidate] = []
    depth = max((len(results) for results in result_lists), default=0)
    for rank in range(depth):
        for results in result_lists:
            if rank >= len(results):
                continue
            candidate = results[rank]
            key = normalize_url(candidate.url)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
            if len(merged) >= limit:
                return merged
    return merged


def discover(
    profile: SpeakerProfile, ctx: RunContext, terms: list[str] | None = None
) -> tuple[list[Candidate], int]:
    """Run the discovery queries in parallel. Returns (candidates, number of queries)."""
    queries = build_queries(terms or profile.topics, ctx.today)
    with ThreadPoolExecutor(max_workers=ctx.max_workers) as pool:
        responses = list(pool.map(lambda q: search(discovery_params(q), ctx, ttl_seconds=TTL_DISCOVERY), queries))
    result_lists = [parse_organic(response.data, query) for query, response in zip(queries, responses)]
    return merge_candidates(result_lists), len(queries)


def _skipped_domain(url: str) -> bool:
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    if host == "sessionize.com" and parts.path.startswith("/s/"):
        return True  # a speaker's session page, not an event's call for speakers
    return any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS)


def prefilter(candidates: list[Candidate]) -> list[Candidate]:
    """Drop hits that cannot be CFPs, without any LLM call: no CFP words, PDFs, login walls."""
    kept: list[Candidate] = []
    for candidate in candidates:
        if urlsplit(candidate.url).path.lower().endswith(".pdf") or _skipped_domain(candidate.url):
            continue
        text = f"{candidate.title} {candidate.snippet}".lower()
        if "speaker profile" in candidate.title.lower():
            continue  # Sessionize speaker profiles mention talks but are not CFPs
        if any(word in text for word in PREFILTER_WORDS):
            kept.append(candidate)
    return kept
