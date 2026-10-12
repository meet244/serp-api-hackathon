"""config.py: required keys, default model per provider key, and keys never shown in repr.

No real .env file is read: ENV_FILE points at a path that does not exist, and the key
variables are cleared for every test.
"""

from __future__ import annotations

import pytest

from cfp_compass import config
from cfp_compass.config import ConfigError, Settings, load_settings, resolve_model

KEY_VARS = (
    "SERPAPI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "CFP_COMPASS_MODEL",
    "CFP_COMPASS_CACHE",
)
FAKE_SERP = "serp-test-value-0001"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist" / "env-file"
    monkeypatch.setattr(config, "ENV_FILE", missing)
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
    assert not missing.exists()


def test_missing_serpapi_key_points_to_env_example(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "llm-test-value")
    with pytest.raises(ConfigError) as info:
        load_settings()
    assert "SERPAPI_API_KEY" in str(info.value)
    assert ".env.example" in str(info.value)


def test_blank_serpapi_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_API_KEY", "llm-test-value")
    with pytest.raises(ConfigError):
        load_settings()


def test_missing_llm_key_points_to_env_example(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", FAKE_SERP)
    with pytest.raises(ConfigError) as info:
        load_settings()
    assert ".env.example" in str(info.value)


def test_requirements_can_be_relaxed():
    settings = load_settings(require_serpapi=False, require_llm=False)
    assert settings.serpapi_api_key is None
    assert settings.llm_model is None


@pytest.mark.parametrize(
    "key_name, model",
    [
        ("ANTHROPIC_API_KEY", "anthropic:claude-haiku-5-5"),
        ("GEMINI_API_KEY", "google:gemini-flash-latest"),
        ("GOOGLE_API_KEY", "google:gemini-flash-latest"),
        ("OPENAI_API_KEY", "openai:gpt-4.1-mini"),
    ],
)
def test_default_model_per_key(monkeypatch, key_name, model):
    monkeypatch.setenv("SERPAPI_API_KEY", FAKE_SERP)
    monkeypatch.setenv(key_name, "llm-test-value")
    assert resolve_model() == model
    settings = load_settings()
    assert settings.llm_model == model
    assert settings.serpapi_api_key == FAKE_SERP


def test_explicit_model_overrides_default(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", FAKE_SERP)
    monkeypatch.setenv("GEMINI_API_KEY", "llm-test-value")
    monkeypatch.setenv("CFP_COMPASS_MODEL", "google:gemini-2.5-pro")
    assert load_settings().llm_model == "google:gemini-2.5-pro"


@pytest.mark.parametrize(
    "model, present, needed",
    [
        ("anthropic:claude-sonnet-4-5", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        ("openai:gpt-4.1", "GEMINI_API_KEY", "OPENAI_API_KEY"),
        ("google:gemini-2.5-flash", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"),
    ],
)
def test_explicit_model_without_its_provider_key_raises(monkeypatch, model, present, needed):
    monkeypatch.setenv("SERPAPI_API_KEY", FAKE_SERP)
    monkeypatch.setenv(present, "llm-test-value")
    monkeypatch.setenv("CFP_COMPASS_MODEL", model)
    with pytest.raises(ConfigError) as info:
        load_settings()
    assert needed in str(info.value)
    assert model in str(info.value)


def test_repr_never_contains_the_key(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", FAKE_SERP)
    monkeypatch.setenv("OPENAI_API_KEY", "llm-test-value")
    settings = load_settings()
    assert settings.serpapi_api_key == FAKE_SERP
    assert FAKE_SERP not in repr(settings)
    assert FAKE_SERP not in str(settings)
    assert FAKE_SERP not in repr(Settings(serpapi_api_key=FAKE_SERP, llm_model=None))


def test_relative_cache_path_is_under_project(monkeypatch):
    monkeypatch.setenv("CFP_COMPASS_CACHE", "tmp-cache/x.sqlite3")
    settings = load_settings(require_serpapi=False, require_llm=False)
    assert settings.cache_path == config.PROJECT_ROOT / "tmp-cache" / "x.sqlite3"
