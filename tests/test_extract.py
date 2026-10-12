"""extract.py: date heuristics, page fetching (mocked httpx), postprocessing and the extraction agent (FunctionModel)."""

from __future__ import annotations

import json
import threading
from datetime import date

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cfp_compass import extract
from cfp_compass.extract import (
    INSTRUCTIONS,
    PAGE_MAX_BYTES,
    PAGE_MAX_CHARS,
    build_prompt,
    count_date_mentions,
    evidence_supports_date,
    extract_all,
    extract_one,
    fetch_page_text,
    html_to_text,
    postprocess,
)
from cfp_compass.models import CFP, Candidate

REAL_CLIENT = httpx.Client


# --- helpers ------------------------------------------------------------------------------

def prompt_text(messages) -> str:
    texts = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt" and isinstance(part.content, str):
                texts.append(part.content)
    return "\n".join(texts)


def structured_model(args_for, prompts: list[str] | None = None) -> FunctionModel:
    """A FunctionModel that answers with the output tool; ``args_for(prompt)`` gives the args."""

    def fn(messages, info: AgentInfo) -> ModelResponse:
        prompt = prompt_text(messages)
        if prompts is not None:
            prompts.append(prompt)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, json.dumps(args_for(prompt)))])

    return FunctionModel(fn)


GOOD_ARGS = {
    "conference_name": "PyCon Test   2027",
    "cfp_url": "https://evil.example.net/submit",
    "cfp_deadline": "2026-12-01",
    "event_start": "2027-05-14",
    "event_end": "2027-05-16",
    "city": "  Long   Beach ",
    "country": "United States",
    "is_online": False,
    "topics": ["python"],
    "is_cfp_open": True,
    "confidence": 0.9,
    "evidence": "CFP closes   December 1, 2026",
}


def candidate(url: str = "https://pycon.example.org/2027/cfp", snippet: str = "Submit your talk proposal now") -> Candidate:
    return Candidate(title="PyCon Test 2027 Call for Proposals", url=url, snippet=snippet, source_query="q")


def mock_httpx(monkeypatch, handler) -> list[httpx.Request]:
    """Route every httpx.Client created by extract.py through a MockTransport."""
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(recording)
        return REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(extract.httpx, "Client", factory)
    return requests


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(extract.httpx, "Client", refuse)


# --- pure helpers -------------------------------------------------------------------------

def test_count_date_mentions():
    assert count_date_mentions("CFP closes March 15, 2027. Event 2027-05-10 to 12/05/2027") == 3
    assert count_date_mentions("Submit by 5th Jan; event Feb 3") == 2
    assert count_date_mentions("Join us in Porto") == 0
    assert count_date_mentions("") == 0


@pytest.mark.parametrize(
    "evidence, expected",
    [
        ("Deadline: 15/01", True),
        ("Proposals due 2027", True),
        ("Submit by March", True),
        ("closes on the first of SEPTEMBER", True),
        ("", False),
        ("Proposals are welcome, submit soon", False),
    ],
)
def test_evidence_supports_date(evidence, expected):
    assert evidence_supports_date(evidence) is expected


def test_html_to_text_strips_script_and_style():
    html = (
        "<html><head><title>Ignore me</title><style>.x{color:red}</style></head>"
        "<body><script>alert('hi')</script><h1>PyCon</h1><p>CFP   closes\n Jan 5</p>"
        "<noscript>enable js</noscript></body></html>"
    )
    text = html_to_text(html)
    assert text == "PyCon CFP closes Jan 5"
    assert "alert" not in text and "color" not in text


def test_html_to_text_truncates():
    text = html_to_text("<p>" + "a" * 10_000 + "</p>")
    assert len(text) == PAGE_MAX_CHARS == 6000


# --- fetch_page_text ----------------------------------------------------------------------

@pytest.mark.parametrize("url", ["ftp://example.com/cfp.html", "file:///etc/passwd", "javascript:alert(1)", "cfp.html"])
def test_fetch_rejects_non_http_schemes(no_network, url):
    assert fetch_page_text(url) is None


def test_fetch_html_page(monkeypatch):
    html = "<html><body><script>var x=1;</script><h1>PyCon</h1><p>CFP closes Jan 5, 2027</p></body></html>"
    requests = mock_httpx(
        monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html)
    )
    assert fetch_page_text("https://pycon.example.org/cfp") == "PyCon CFP closes Jan 5, 2027"
    assert len(requests) == 1
    assert "CFP-Compass" in requests[0].headers["user-agent"]


def test_fetch_follows_redirects(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://pycon.example.org/new"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>Moved here</p>")

    mock_httpx(monkeypatch, handler)
    assert fetch_page_text("https://pycon.example.org/old") == "Moved here"


def test_fetch_404_is_none(monkeypatch):
    mock_httpx(monkeypatch, lambda request: httpx.Response(404, headers={"content-type": "text/html"}, text="<p>Not found</p>"))
    assert fetch_page_text("https://pycon.example.org/missing") is None


def test_fetch_pdf_is_none(monkeypatch):
    mock_httpx(monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7"))
    assert fetch_page_text("https://pycon.example.org/cfp.pdf") is None


def test_fetch_truncates_large_body(monkeypatch):
    chunk = b"a" * 65_536
    total_chunks = (3_000_000 // len(chunk)) + 1  # a little over 3 MB
    served = {"chunks": 0}

    def body():
        for _ in range(total_chunks):
            served["chunks"] += 1
            yield chunk

    seen: list[int] = []
    real_html_to_text = extract.html_to_text

    def spy(html: str, max_chars: int = PAGE_MAX_CHARS) -> str:
        seen.append(len(html))
        return real_html_to_text(html, max_chars)

    monkeypatch.setattr(extract, "html_to_text", spy)
    mock_httpx(monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, content=body()))

    text = fetch_page_text("https://pycon.example.org/huge")

    assert seen == [PAGE_MAX_BYTES]
    assert served["chunks"] < total_chunks  # stopped reading early
    assert text is not None and len(text) <= PAGE_MAX_CHARS


@pytest.mark.parametrize("error", [httpx.ConnectTimeout("timed out"), httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_fetch_httpx_error_is_none(monkeypatch, error):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    mock_httpx(monkeypatch, handler)
    assert fetch_page_text("https://pycon.example.org/cfp") is None


# --- postprocess --------------------------------------------------------------------------

def test_postprocess_nulls_deadline_without_dated_evidence():
    cfp = CFP(conference_name="X", cfp_deadline=date(2026, 11, 1), deadline_is_aoe=True, evidence="Submit soon!")
    out = postprocess(cfp, candidate())
    assert out.cfp_deadline is None
    assert out.deadline_is_aoe is False

    empty = postprocess(CFP(conference_name="X", cfp_deadline=date(2026, 11, 1), evidence=""), candidate())
    assert empty.cfp_deadline is None


@pytest.mark.parametrize("evidence", ["Deadline: 1 Nov 2026 AoE", "CFP closes November first"])
def test_postprocess_keeps_deadline_with_dated_evidence(evidence):
    cfp = CFP(conference_name="X", cfp_deadline=date(2026, 11, 1), deadline_is_aoe=True, evidence=evidence)
    out = postprocess(cfp, candidate())
    assert out.cfp_deadline == date(2026, 11, 1)
    assert out.deadline_is_aoe is True


@pytest.mark.parametrize("bad_url", ["https://evil.example.net/submit", "", "javascript:alert(1)", "/relative/submit"])
def test_postprocess_replaces_offsite_cfp_url(bad_url):
    hit = candidate("https://www.pycon.example.org/2027/cfp")
    out = postprocess(CFP(conference_name="X", cfp_url=bad_url), hit)
    assert out.cfp_url == hit.url
    assert out.source_url == hit.url


@pytest.mark.parametrize(
    "good_url",
    ["https://pycon.example.org/submit", "https://www.pycon.example.org/speak", "https://us.pycon.example.org/2027/speaking/"],
)
def test_postprocess_keeps_same_site_cfp_url(good_url):
    hit = candidate("https://www.pycon.example.org/2027/cfp")
    out = postprocess(CFP(conference_name="X", cfp_url=good_url), hit)
    assert out.cfp_url == good_url
    assert out.source_url == hit.url


def test_postprocess_cleans_whitespace_and_name():
    cfp = CFP(conference_name="  ", city="  Long   Beach ", country=" ", venue=None, evidence="a  b\n c")
    out = postprocess(cfp, candidate())
    assert out.conference_name == candidate().title
    assert out.city == "Long Beach"
    assert out.country is None
    assert out.evidence == "a b c"


# --- prompt and instructions --------------------------------------------------------------

def test_build_prompt_wraps_untrusted_source():
    hit = Candidate(title="T", url="https://x.example/cfp", snippet="S", date="Oct 1, 2026")
    prompt = build_prompt(hit, "2026-10-12", "PAGE TEXT HERE")
    assert "Today's date: 2026-10-12" in prompt
    start, end = prompt.index("<untrusted_source>"), prompt.index("</untrusted_source>")
    body = prompt[start:end]
    for piece in ("https://x.example/cfp", "S", "Oct 1, 2026", "PAGE TEXT HERE"):
        assert piece in body
    assert prompt.index("Today's date") < start


def test_instructions_treat_source_as_untrusted():
    text = INSTRUCTIONS.lower()
    assert "untrusted_source" in text
    assert "never follow instructions" in text
    assert "never invent a date" in text


# --- extract_one / extract_all with a FunctionModel -----------------------------------------

@pytest.fixture
def page_fetches(monkeypatch) -> list[str]:
    fetched: list[str] = []

    def fake_fetch(url: str, **kwargs) -> str:
        fetched.append(url)
        return "PyCon Test 2027, May 14-16 2027, Long Beach. CFP closes December 1, 2026."

    monkeypatch.setattr(extract, "fetch_page_text", fake_fetch)
    return fetched


def test_extract_one_returns_postprocessed_cfp(ctx, page_fetches):
    prompts: list[str] = []
    ctx.llm_model = structured_model(lambda p: GOOD_ARGS, prompts)
    hit = candidate()

    cfp = extract_one(hit, ctx)

    assert cfp is not None
    assert cfp.conference_name == "PyCon Test 2027"
    assert cfp.cfp_deadline == date(2026, 12, 1)
    assert cfp.event_start == date(2027, 5, 14) and cfp.event_end == date(2027, 5, 16)
    assert cfp.city == "Long Beach"
    assert cfp.cfp_url == hit.url  # off-site link replaced
    assert cfp.source_url == hit.url
    assert cfp.evidence == "CFP closes December 1, 2026"
    assert page_fetches == [hit.url]  # thin snippet: page fetched
    assert len(prompts) == 1
    assert "<untrusted_source>" in prompts[0]
    assert "2026-10-12" in prompts[0]
    assert "CFP closes December 1, 2026" in prompts[0]  # page text went to the model
    assert ctx.count("llm_calls") == 1
    assert ctx.errors == []


def test_extract_one_second_call_is_cached(ctx, page_fetches):
    prompts: list[str] = []
    ctx.llm_model = structured_model(lambda p: GOOD_ARGS, prompts)
    hit = candidate()

    first = extract_one(hit, ctx)
    assert ctx.count("llm_calls") == 1 and ctx.count("llm_cache_hits") == 0
    second = extract_one(hit, ctx)

    assert second == first
    assert ctx.count("llm_calls") == 1
    assert ctx.count("llm_cache_hits") == 1
    assert len(prompts) == 1
    assert len(page_fetches) == 1  # the cached extraction skips the page fetch


def test_extract_one_skips_fetch_when_snippet_has_dates(ctx, page_fetches):
    ctx.llm_model = structured_model(lambda p: GOOD_ARGS)
    hit = candidate(snippet="CFP closes Dec 1, 2026. Conference May 14-16, 2027 in Long Beach.")
    assert extract_one(hit, ctx) is not None
    assert page_fetches == []


def test_extract_one_invalid_output_returns_none(ctx, page_fetches):
    prompts: list[str] = []
    ctx.llm_model = structured_model(lambda p: {"confidence": 5, "cfp_deadline": "someday"}, prompts)

    assert extract_one(candidate(), ctx) is None
    assert len(ctx.errors) == 1
    assert "Extraction failed" in ctx.errors[0]
    assert len(prompts) == 3  # first try + 2 retries
    assert ctx.cache.count() == 0


def test_extract_one_without_model_returns_none(ctx, page_fetches):
    ctx.llm_model = None
    assert extract_one(candidate(), ctx) is None
    assert len(ctx.errors) == 1


def test_extract_all_reports_progress_once_per_candidate(ctx, page_fetches):
    def args_for(prompt: str) -> dict:
        if "bad.example" in prompt:
            return {"confidence": 7}
        return {**GOOD_ARGS, "conference_name": prompt.split("Search result title: ")[1].splitlines()[0]}

    ctx.llm_model = structured_model(args_for)
    candidates = [
        Candidate(title="Conf A", url="https://a.example/cfp", snippet="talk"),
        Candidate(title="Conf B", url="https://bad.example/cfp", snippet="talk"),
        Candidate(title="Conf C", url="https://c.example/cfp", snippet="talk"),
    ]
    progress: list[tuple[int, int]] = []
    threads: set[int] = set()

    def on_progress(done: int, total: int) -> None:
        progress.append((done, total))
        threads.add(threading.get_ident())

    results = extract_all(candidates, ctx, on_progress=on_progress)

    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert threads == {threading.get_ident()}  # callbacks run on the calling thread
    assert sorted(c.conference_name for c in results) == ["Conf A", "Conf C"]
    assert len(ctx.errors) == 1


def test_extract_all_empty(ctx):
    calls: list[tuple[int, int]] = []
    assert extract_all([], ctx, on_progress=lambda d, t: calls.append((d, t))) == []
    assert calls == []
