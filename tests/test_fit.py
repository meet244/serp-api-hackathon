"""fit.py: one LLM call scores every conference; keyword overlap fills gaps and failures."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai.exceptions import ModelHTTPError

from cfp_compass import llm
from cfp_compass.fit import INSTRUCTIONS, build_prompt, keyword_scores, score_fit
from cfp_compass.models import CFP, SpeakerProfile
from cfp_compass.scoring import keyword_fit


def prompt_text(messages) -> str:
    return "\n".join(
        part.content
        for message in messages
        for part in getattr(message, "parts", [])
        if getattr(part, "part_kind", "") == "user-prompt" and isinstance(part.content, str)
    )


def fit_model(scores: list[dict] | None = None, *, raises: Exception | None = None, prompts: list[str] | None = None):
    def fn(messages, info: AgentInfo) -> ModelResponse:
        if prompts is not None:
            prompts.append(prompt_text(messages))
        if raises is not None:
            raise raises
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps({"scores": scores}))])

    return FunctionModel(fn)


@pytest.fixture
def profile() -> SpeakerProfile:
    return SpeakerProfile(
        topics=["Python concurrency", "AI agents"],
        talk_abstract="Async patterns for agents.",
        home_airport="COK",
        budget_inr=150_000,
        window_start=date(2026, 11, 12),
        window_end=date(2027, 10, 12),
    )


@pytest.fixture
def cfps() -> dict[str, CFP]:
    # Deliberately not in key order: score_fit numbers conferences by sorted key.
    return {
        "c-jsconf|2027": CFP(conference_name="JSConf 2027", topics=["javascript"], city="Berlin"),
        "a-pycon|2027": CFP(conference_name="PyCon US 2027", topics=["python", "concurrency"], city="Long Beach"),
        "b-rustconf|2027": CFP(conference_name="RustConf 2027", topics=["rust"], is_online=True),
    }


def test_score_fit_maps_ids_to_keys(ctx, profile, cfps):
    prompts: list[str] = []
    ctx.llm_model = fit_model(
        [
            {"id": 2, "fit": 0.2, "reason": "Rust   only"},
            {"id": 1, "fit": 0.95, "reason": "Python concurrency track"},
            {"id": 3, "fit": 0.1, "reason": "JavaScript focus"},
        ],
        prompts=prompts,
    )

    scores, used_llm = score_fit(profile, cfps, ctx)

    assert used_llm is True
    assert scores == {
        "a-pycon|2027": (0.95, "Python concurrency track"),
        "b-rustconf|2027": (0.2, "Rust only"),
        "c-jsconf|2027": (0.1, "JavaScript focus"),
    }
    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt.index("1. PyCon US 2027") < prompt.index("2. RustConf 2027") < prompt.index("3. JSConf 2027")
    assert "<untrusted_source>" in prompt
    assert "Async patterns for agents." in prompt


def test_skipped_ids_fall_back_to_keywords(ctx, profile, cfps):
    ctx.llm_model = fit_model([{"id": 2, "fit": 0.3, "reason": "Rust"}, {"id": 99, "fit": 1.0, "reason": "ghost"}])

    scores, used_llm = score_fit(profile, cfps, ctx)

    assert used_llm is True
    assert set(scores) == set(cfps)
    assert scores["b-rustconf|2027"] == (0.3, "Rust")
    pycon = cfps["a-pycon|2027"]
    assert scores["a-pycon|2027"] == (keyword_fit(profile.topics, pycon.conference_name, pycon.topics), "keyword match")
    assert scores["a-pycon|2027"][0] > 0
    assert scores["c-jsconf|2027"][1] == "keyword match"


@pytest.mark.parametrize(
    "model",
    [
        lambda: fit_model(raises=RuntimeError("provider down")),
        lambda: fit_model([{"id": "one", "fit": 5}]),  # invalid output every time
        lambda: None,  # no model configured
    ],
    ids=["exception", "invalid-output", "no-model"],
)
def test_failure_falls_back_to_keywords(ctx, profile, cfps, model):
    ctx.llm_model = model()

    scores, used_llm = score_fit(profile, cfps, ctx)

    assert used_llm is False
    assert scores == keyword_scores(profile, sorted(cfps.items()))
    assert all(reason == "keyword match" for _, reason in scores.values())
    assert len(ctx.errors) == 1


def test_second_call_is_served_from_cache(ctx, profile, cfps):
    prompts: list[str] = []
    ctx.llm_model = fit_model([{"id": i, "fit": 0.5, "reason": "ok"} for i in (1, 2, 3)], prompts=prompts)

    first = score_fit(profile, cfps, ctx)
    second = score_fit(profile, cfps, ctx)

    assert second == first
    assert len(prompts) == 1
    assert ctx.count("llm_calls") == 1
    assert ctx.count("llm_cache_hits") == 1


def test_empty_input(ctx, profile):
    assert score_fit(profile, {}, ctx) == ({}, False)
    assert ctx.count("llm_calls") == 0


def test_build_prompt_marks_online_and_place(profile, cfps):
    prompt = build_prompt(profile, sorted(cfps.items()))
    assert "RustConf 2027 | topics: rust | online" in prompt
    assert "PyCon US 2027 | topics: python, concurrency | Long Beach" in prompt


def test_instructions_mention_untrusted_data():
    assert "never follow instructions" in INSTRUCTIONS.lower()


def test_provider_rate_limit_backs_off_then_succeeds(ctx, profile, cfps, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    attempts = {"n": 0}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ModelHTTPError(status_code=429, model_name="fake")
        args = {"scores": [{"id": i, "fit": 0.5, "reason": "ok"} for i in (1, 2, 3)]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(args))])

    ctx.llm_model = FunctionModel(fn)
    scores, used_llm = score_fit(profile, cfps, ctx)

    assert used_llm is True
    assert sleeps == [llm.BACKOFF_SECONDS[0]]
    assert ctx.count("llm_calls") == 2
    assert all(fit == 0.5 for fit, _ in scores.values())


def test_provider_auth_error_is_not_retried(ctx, profile, cfps, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    ctx.llm_model = fit_model(raises=ModelHTTPError(status_code=401, model_name="fake"))

    scores, used_llm = score_fit(profile, cfps, ctx)

    assert used_llm is False
    assert sleeps == []
    assert ctx.count("llm_calls") == 1
