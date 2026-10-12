"""Step 4: turn a search hit into a typed CFP record with a Pydantic AI agent.

When the snippet is thin, the page text is fetched too (8 s timeout, 2 MB cap, http/https
only) and passed to the model as untrusted data.
"""

from __future__ import annotations

import ipaddress
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

from .cache import make_key
from .context import RunContext, log
from .llm import UNTRUSTED_NOTE, make_agent, run_structured, untrusted
from pydantic import Field

from .models import CFP, Candidate

EXTRACT_VERSION = "extract-v2"
EXTRACT_TTL = 72 * 60 * 60
PAGE_TIMEOUT = 8.0
PAGE_MAX_BYTES = 2_000_000
PAGE_MAX_CHARS = 6_000
PAGE_MAX_REDIRECTS = 5
USER_AGENT = "Mozilla/5.0 (compatible; CFP-Compass/1.0; +https://github.com/meet244/serp-api-hackathon)"

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_MONTH_WORD = re.compile(rf"\b(?:{_MONTHS})\b", re.IGNORECASE)
_DATE_MENTION = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b|(?:{_MONTHS})\.?\s+\d{{1,2}}\b|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4})",
    re.IGNORECASE,
)

INSTRUCTIONS = f"""You extract facts about ONE conference's call for proposals (CFP, also called call for papers,
call for speakers or CfS) from a single web search result.

Rules:
- Never invent a date. If the source does not state a date, return null for it.
- Resolve years relative to today's date (given in the prompt). If a date has no year, choose the year that
  fits the context, normally the next occurrence after today's date.
- Set is_cfp_open to false if the source says the CFP has closed, or the event is over.
- Copy a short verbatim excerpt (under 200 characters) that supports the deadline into evidence.
  If the source states no deadline, leave evidence empty.
- confidence reflects how clearly the source states the facts: about 0.9 when the conference name, deadline,
  event dates and place are all explicit; about 0.6 when some are missing; below 0.5 when this is not one
  conference's CFP (a list of many conferences, a news article, a job ad, an unrelated page).
- city is the city name only (e.g. "Porto"); country is the full country name (e.g. "Portugal").
  For online-only events set is_online to true and leave city null. Hybrid events have a city.
- deadline_is_aoe is true only if the deadline is stated as AoE / Anywhere on Earth.
- topics: up to 6 short topics or tracks the conference covers.
- {UNTRUSTED_NOTE}
"""

class ExtractedCFP(CFP):
    """A conference call for proposals extracted from one search result."""

    # Required for the model: with a default, models tend to leave it out (and 0.0 drops the record).
    confidence: float = Field(
        ge=0.0, le=1.0, description="Required. 0-1: how clearly the source states the facts (name, deadline, dates, place)."
    )


_agent = make_agent(ExtractedCFP, INSTRUCTIONS)


def count_date_mentions(text: str) -> int:
    return len(_DATE_MENTION.findall(text or ""))


def evidence_supports_date(evidence: str) -> bool:
    """A deadline needs supporting text with at least a digit or a month name."""
    return bool(evidence) and (any(c.isdigit() for c in evidence) or bool(_MONTH_WORD.search(evidence)))


def html_to_text(html: str, max_chars: int = PAGE_MAX_CHARS) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "template", "iframe", "head"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    return text[:max_chars]


def blocked_host(host: str | None) -> bool:
    """Loopback, private and link-local hosts (literal IPs and local names; no DNS lookup)."""
    host = (host or "").strip("[]").rstrip(".").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def fetch_page_text(url: str, *, timeout: float = PAGE_TIMEOUT, max_bytes: int = PAGE_MAX_BYTES) -> str | None:
    """Visible text of a page, or None if it cannot be fetched quickly and safely.

    http/https only, public hosts only (checked on every redirect hop), at most 5 redirects,
    2 MB, and ``timeout`` seconds overall.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or blocked_host(parts.hostname):
        return None
    started = time.monotonic()

    def check_request(request: httpx.Request) -> None:
        if request.url.scheme not in ("http", "https") or blocked_host(request.url.host):
            raise httpx.UnsupportedProtocol("blocked redirect target")
        if time.monotonic() - started > timeout:
            raise httpx.TimeoutException("page fetch took too long")

    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout, max_redirects=PAGE_MAX_REDIRECTS,
            headers={"User-Agent": USER_AGENT}, event_hooks={"request": [check_request]},
        ) as client:
            with client.stream("GET", url) as response:
                if response.status_code >= 400 or response.url.scheme not in ("http", "https"):
                    return None
                content_type = response.headers.get("content-type", "").lower()
                if content_type and "html" not in content_type and "text/plain" not in content_type:
                    return None
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) >= max_bytes or time.monotonic() - started > timeout:
                        break
                encoding = response.encoding or "utf-8"
        text = html_to_text(bytes(body[:max_bytes]).decode(encoding, errors="replace"))
        return text or None
    except Exception as exc:  # timeouts, TLS errors, blocked pages: carry on with the snippet
        log.info("Page fetch failed for %s: %s", url, type(exc).__name__)
        return None


def build_prompt(candidate: Candidate, today_iso: str, page_text: str | None) -> str:
    parts = [
        f"Today's date: {today_iso}",
        "",
        "<untrusted_source>",
        f"Search result title: {untrusted(candidate.title)}",
        f"URL: {untrusted(candidate.url)}",
        f"Snippet: {untrusted(candidate.snippet)}",
    ]
    if candidate.date:
        parts.append(f"Google's date label for the page: {untrusted(candidate.date)}")
    if page_text:
        parts += ["", "Page text (first part):", untrusted(page_text)]
    parts.append("</untrusted_source>")
    return "\n".join(parts)


def _same_site(url: str, other: str) -> bool:
    a = urlsplit(url).netloc.lower().removeprefix("www.")
    b = urlsplit(other).netloc.lower().removeprefix("www.")
    return bool(a) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def postprocess(cfp: CFP, candidate: Candidate) -> CFP:
    """Apply the evidence rule and keep links honest (a submit link must be on the same site)."""
    update: dict = {"source_url": candidate.url}
    if cfp.cfp_deadline is not None and not evidence_supports_date(cfp.evidence):
        update.update(cfp_deadline=None, deadline_is_aoe=False)
    url = (cfp.cfp_url or "").strip()
    if not url.startswith(("http://", "https://")) or not _same_site(url, candidate.url):
        update["cfp_url"] = candidate.url
    name = " ".join(cfp.conference_name.split())
    update["conference_name"] = name or candidate.title
    update["evidence"] = " ".join(cfp.evidence.split())[:300]
    for field in ("city", "country", "venue"):
        value = getattr(cfp, field)
        if isinstance(value, str):
            update[field] = " ".join(value.split()) or None
    return cfp.model_copy(update=update)


def extract_one(candidate: Candidate, ctx: RunContext) -> CFP | None:
    """Extract one candidate. Returns None (and counts an error) if the model fails."""
    page_text = None
    cache_params = {"url": candidate.url, "v": EXTRACT_VERSION}
    # Snippets rarely carry deadline, dates and city together: fetch the page unless the snippet
    # already mentions at least two dates. A cached extraction skips the fetch entirely.
    needs_page = count_date_mentions(f"{candidate.title} {candidate.snippet}") < 2
    try:
        cached = ctx.cache.get(make_key({"kind": "extract", **cache_params}), None if ctx.serp_offline else EXTRACT_TTL)
        if cached is None and needs_page:
            page_text = fetch_page_text(candidate.url)
        prompt = build_prompt(candidate, ctx.today.isoformat(), page_text)
        extracted = run_structured(
            _agent, prompt, ExtractedCFP, ctx, cache_kind="extract", cache_params=cache_params, ttl_seconds=EXTRACT_TTL
        )
        cfp = CFP.model_validate(extracted.model_dump())
    except Exception as exc:
        ctx.add_error(f"Extraction failed for {candidate.url}: {type(exc).__name__}: {str(exc)[:160]}")
        return None
    return postprocess(cfp, candidate)


def extract_all(
    candidates: list[Candidate],
    ctx: RunContext,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[CFP]:
    """Extract candidates 4 at a time. ``on_progress`` runs in the calling thread.

    Results keep the candidates' order (not completion order), so de-duplication and every
    downstream cache key are identical on a re-run.
    """
    results: list[CFP | None] = [None] * len(candidates)
    total = len(candidates)
    with ThreadPoolExecutor(max_workers=ctx.max_workers) as pool:
        futures = {pool.submit(extract_one, candidate, ctx): i for i, candidate in enumerate(candidates)}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if on_progress:
                on_progress(done, total)
    return [cfp for cfp in results if cfp is not None]
