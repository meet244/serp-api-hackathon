"""Step 8: find each shortlisted conference's speaker travel policy.

One Google search per conference, then a small classification agent reads the top 5
results and returns a TravelSupport record.
"""

from __future__ import annotations

import re
from typing import Any

from .context import RunContext
from .llm import UNTRUSTED_NOTE, make_agent, run_structured, untrusted
from .models import CFP, TravelSupport
from .serp import TTL_TRAVEL_POLICY, search

TRAVEL_VERSION = "travel-v1"
TOP_RESULTS = 5
_YEAR = re.compile(r"\s*\b(?:19|20)\d{2}\b|\s*['’]\d{2}\b")

INSTRUCTIONS = f"""You decide whether a conference supports its speakers' travel costs, using up to 5 web search results.

Choose one status:
- covered: the conference pays travel AND accommodation for accepted speakers.
- partial: it pays part of it, e.g. hotel only, flights only, or a capped stipend or reimbursement.
- grant_available: there is a travel grant, financial aid or opportunity grant programme that speakers can apply to.
- none: the results explicitly say speaker travel is not covered.
- unknown: the results say nothing clear about it.

Rules:
- Only use results that are about this conference (match its name). Other events do not count.
- A free conference ticket alone is not travel support.
- evidence_url must be copied exactly from the link of the result you relied on; null when status is unknown.
- note: one short sentence (under 25 words) explaining the status.
- {UNTRUSTED_NOTE}
"""

_agent = make_agent(TravelSupport, INSTRUCTIONS)


def policy_name(conference_name: str) -> str:
    """'PyCon US 2027' -> 'PyCon US': policy pages often omit the edition year."""
    name = _YEAR.sub("", conference_name).strip(" -–:|,")
    return name or conference_name


def travel_query(conference_name: str) -> str:
    return f'"{policy_name(conference_name)}" speaker travel OR "travel grant" OR "financial aid"'


def travel_params(conference_name: str) -> dict[str, Any]:
    return {"engine": "google", "q": travel_query(conference_name), "hl": "en", "gl": "us"}


def top_results(data: dict[str, Any], limit: int = TOP_RESULTS) -> list[dict[str, str]]:
    results = []
    for item in data.get("organic_results") or []:
        if isinstance(item, dict) and item.get("link"):
            results.append(
                {"title": str(item.get("title") or ""), "link": str(item["link"]), "snippet": str(item.get("snippet") or "")}
            )
        if len(results) >= limit:
            break
    return results


def _norm(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url.strip().lower()).rstrip("/")


def check_evidence(support: TravelSupport, results: list[dict[str, str]]) -> TravelSupport:
    """A status other than unknown must point at one of the results we actually showed the model."""
    links = {_norm(r["link"]): r["link"] for r in results}
    url = links.get(_norm(support.evidence_url or ""))
    if support.status == "unknown":
        return support.model_copy(update={"evidence_url": url})
    if url is None:
        return TravelSupport(status="unknown", note="No clear speaker travel policy found.")
    return support.model_copy(update={"evidence_url": url})


def build_prompt(cfp: CFP, results: list[dict[str, str]]) -> str:
    lines = [f"Conference: {untrusted(cfp.conference_name)}", "", "<untrusted_source>"]
    for i, r in enumerate(results, start=1):
        lines += [f"Result {i}", f"Title: {untrusted(r['title'])}", f"Link: {untrusted(r['link'])}",
                  f"Snippet: {untrusted(r['snippet'])}", ""]
    lines.append("</untrusted_source>")
    return "\n".join(lines)


def find_travel_support(cfp: CFP, key: str, ctx: RunContext) -> TravelSupport:
    """Never raises: failures become status 'unknown' with a note."""
    response = search(travel_params(cfp.conference_name), ctx, ttl_seconds=TTL_TRAVEL_POLICY)
    if not response.ok:
        return TravelSupport(status="unknown", note="Travel policy not checked (search unavailable).")
    results = top_results(response.data)
    if not results:
        return TravelSupport(status="unknown", note="No travel policy found in search results.")
    try:
        support = run_structured(
            _agent,
            build_prompt(cfp, results),
            TravelSupport,
            ctx,
            cache_kind="travel",
            cache_params={"conference": key, "links": [r["link"] for r in results], "v": TRAVEL_VERSION},
            ttl_seconds=TTL_TRAVEL_POLICY,
        )
    except Exception as exc:
        ctx.add_error(f"Travel-policy check failed for {cfp.conference_name}: {type(exc).__name__}")
        return TravelSupport(status="unknown", note="Travel policy could not be classified.")
    return check_evidence(support, results)
