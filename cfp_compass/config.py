"""Settings and keys, read from the environment and a git-ignored ``.env`` file.

Keys are never printed or logged; ``Settings`` hides them from its repr.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_CACHE_PATH = PROJECT_ROOT / ".cache" / "cfp_compass.sqlite3"

# Pydantic AI provider prefix -> environment variables that can hold its key.
PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai-chat": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
}

# When CFP_COMPASS_MODEL is not set, use a fast, cheap model for the first key found.
DEFAULT_MODELS: tuple[tuple[str, str], ...] = (
    ("ANTHROPIC_API_KEY", "anthropic:claude-haiku-5-5"),
    ("GEMINI_API_KEY", "google:gemini-flash-latest"),
    ("GOOGLE_API_KEY", "google:gemini-flash-latest"),
    ("OPENAI_API_KEY", "openai:gpt-4.1-mini"),
)

SETUP_HINT = "Copy .env.example to .env and fill it in (see the Setup section of README.md)."


class ConfigError(RuntimeError):
    """A required setting is missing or invalid. The message says how to fix it."""


@dataclass(frozen=True)
class Settings:
    serpapi_api_key: str | None = field(repr=False)
    llm_model: str | None
    cache_path: Path = DEFAULT_CACHE_PATH


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


_BARE_MODEL_PREFIXES = (("gemini", "google"), ("claude", "anthropic"), ("gpt-", "openai"), ("o1", "openai"),
                        ("o3", "openai"), ("o4", "openai"))


def with_provider(model: str) -> str:
    """Pydantic AI needs 'provider:model'. Accept a bare 'gemini-flash-latest' and add the prefix."""
    if ":" in model:
        return model
    for prefix, provider in _BARE_MODEL_PREFIXES:
        if model.lower().startswith(prefix):
            return f"{provider}:{model}"
    return model


def resolve_model() -> str | None:
    """The Pydantic AI model string to use, from CFP_COMPASS_MODEL or the first LLM key present."""
    explicit = _env("CFP_COMPASS_MODEL")
    if explicit:
        return with_provider(explicit)
    for key_name, model in DEFAULT_MODELS:
        if _env(key_name):
            return model
    return None


def load_settings(*, require_serpapi: bool = True, require_llm: bool = True) -> Settings:
    """Load settings, failing with a clear message if a required key is missing."""
    load_dotenv(ENV_FILE, override=False)

    serpapi_key = _env("SERPAPI_API_KEY")
    if require_serpapi and not serpapi_key:
        raise ConfigError(f"SERPAPI_API_KEY is missing. {SETUP_HINT} Get a key at https://serpapi.com/manage-api-key.")

    model = resolve_model()
    if require_llm:
        if not model:
            raise ConfigError(
                "No LLM key found. Set one of GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY "
                f"(or CFP_COMPASS_MODEL plus that provider's key). {SETUP_HINT}"
            )
        if ":" not in model:
            raise ConfigError(
                f"CFP_COMPASS_MODEL is '{model}'. Use the Pydantic AI form 'provider:model', "
                f"for example google:gemini-flash-latest. {SETUP_HINT}"
            )
        provider = model.split(":", 1)[0]
        needed = PROVIDER_KEYS.get(provider)
        if needed and not any(_env(name) for name in needed):
            raise ConfigError(f"CFP_COMPASS_MODEL is '{model}' but {' or '.join(needed)} is not set. {SETUP_HINT}")

    cache_path = Path(_env("CFP_COMPASS_CACHE") or DEFAULT_CACHE_PATH)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    return Settings(serpapi_api_key=serpapi_key, llm_model=model, cache_path=cache_path)
