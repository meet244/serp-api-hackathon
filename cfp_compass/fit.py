"""Step 6: topic fit. One LLM call scores every surviving conference; keyword overlap is the fallback."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .context import RunContext
from .llm import UNTRUSTED_NOTE, make_agent, run_structured, untrusted
from .models import CFP, SpeakerProfile
from .scoring import keyword_fit

FIT_VERSION = "fit-v1"
FIT_TTL = 24 * 60 * 60


class FitItem(BaseModel):
    id: int = Field(description="The conference id from the list.")
    fit: float = Field(ge=0.0, le=1.0, description="0-1: how well the speaker's talk fits this conference.")
    reason: str = Field(description="One short line (under 12 words) naming the matching topic or track.")


class FitScores(BaseModel):
    scores: list[FitItem]


INSTRUCTIONS = f"""You match a conference speaker's talk to conferences.
For every conference in the list, return its id, a fit score from 0 to 1, and a one-line reason.
1.0 = the talk is a core topic of the conference; 0.7 = a clear track fits; 0.4 = adjacent audience;
0.1 = unrelated. Judge by the conference's name and topics. Score every id exactly once.
{UNTRUSTED_NOTE}
"""

_agent = make_agent(FitScores, INSTRUCTIONS)


def build_prompt(profile: SpeakerProfile, items: list[tuple[str, CFP]]) -> str:
    lines = [f"Speaker's talk topics: {', '.join(profile.topics)}"]
    if profile.talk_abstract:
        lines.append(f"Talk abstract: {profile.talk_abstract[:1500]}")
    lines += ["", "<untrusted_source>", "Conferences:"]
    for i, (_, cfp) in enumerate(items, start=1):
        where = "online" if cfp.is_online else ", ".join(p for p in (cfp.city, cfp.country) if p)
        topics = ", ".join(cfp.topics[:6]) or "not listed"
        lines.append(untrusted(f"{i}. {cfp.conference_name} | topics: {topics} | {where}"))
    lines.append("</untrusted_source>")
    return "\n".join(lines)


def keyword_scores(profile: SpeakerProfile, items: list[tuple[str, CFP]]) -> dict[str, tuple[float, str]]:
    return {
        key: (keyword_fit(profile.topics, cfp.conference_name, cfp.topics), "keyword match")
        for key, cfp in items
    }


def score_fit(profile: SpeakerProfile, cfps: dict[str, CFP], ctx: RunContext) -> tuple[dict[str, tuple[float, str]], bool]:
    """Returns ({key: (fit, reason)}, used_llm). Any conference the LLM skipped falls back to keywords."""
    items = sorted(cfps.items())
    if not items:
        return {}, False
    fallback = keyword_scores(profile, items)
    try:
        result = run_structured(
            _agent,
            build_prompt(profile, items),
            FitScores,
            ctx,
            cache_kind="fit",
            cache_params={
                "topics": [t.lower() for t in profile.topics],
                "abstract": profile.talk_abstract or "",
                "conferences": [[key, cfp.conference_name, cfp.topics] for key, cfp in items],
                "v": FIT_VERSION,
            },
            ttl_seconds=FIT_TTL,
        )
    except Exception as exc:
        ctx.add_error(f"LLM fit scoring failed, using keyword overlap: {type(exc).__name__}")
        return fallback, False
    scores = dict(fallback)
    for item in result.scores:
        if 1 <= item.id <= len(items):
            key = items[item.id - 1][0]
            scores[key] = (round(item.fit, 3), " ".join(item.reason.split())[:90])
    return scores, True
