"""
Single wrapper for all LLM calls.

Provider auto-selection (checked at import time):
  GEMINI_KEY set in .env  → Google Gemini  (gemini-2.0-flash, free tier)
  OPENAI_API_KEY only     → OpenAI          (gpt-4o-mini)

Interface is identical for both providers — no other module needs to change.

Usage:
    from src.llm_client import complete, complete_structured

    text = complete("Parse these criteria: ...", system="You are a clinical NLP system.")
    obj  = complete_structured(MyPydanticModel, prompt, system)
"""
from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

from pydantic import BaseModel

from src.config import (
    CACHE_DIR,          # noqa: F401 — imported to trigger load_dotenv
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

GEMINI_KEY: str | None = os.getenv("GEMINI_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MAX_TOKENS: int = 4096

_PROVIDER: str = "gemini" if GEMINI_KEY else "openai"
logger.info("LLM provider: %s", _PROVIDER)

# ---------------------------------------------------------------------------
# Client initialisation (lazy — only the active provider is initialised)
# ---------------------------------------------------------------------------

if _PROVIDER == "gemini":
    from google import genai as _genai
    from google.genai import types as _gtypes
    _gemini_client = _genai.Client(api_key=GEMINI_KEY)
else:
    from openai import OpenAI as _OpenAI
    _openai_client = _OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Cumulative token usage tracker
# ---------------------------------------------------------------------------

_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def get_usage() -> dict[str, int]:
    """Return cumulative token counts since process start (or last reset)."""
    return dict(_usage)


def reset_usage() -> None:
    _usage["prompt_tokens"] = 0
    _usage["completion_tokens"] = 0
    _usage["total_tokens"] = 0


def _track_gemini_usage(meta) -> None:
    if meta is None:
        return
    _usage["prompt_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
    _usage["completion_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
    _usage["total_tokens"] += getattr(meta, "total_token_count", 0) or 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete(
    prompt: str,
    system: str = "You are a precise clinical information extraction system.",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Single-turn chat completion. Returns assistant message text."""
    if _PROVIDER == "gemini":
        response = _gemini_client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=prompt,
            config=_gtypes.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature if temperature is not None else 0.0,
                max_output_tokens=max_tokens or GEMINI_MAX_TOKENS,
            ),
        )
        _track_gemini_usage(response.usage_metadata)
        return response.text or ""

    # OpenAI path
    response = _openai_client.chat.completions.create(
        model=model or OPENAI_MODEL,
        temperature=temperature if temperature is not None else OPENAI_TEMPERATURE,
        max_tokens=max_tokens or OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    usage = response.usage
    if usage:
        _usage["prompt_tokens"] += usage.prompt_tokens
        _usage["completion_tokens"] += usage.completion_tokens
        _usage["total_tokens"] += usage.total_tokens
    logger.debug(
        "LLM call: model=%s, in=%d, out=%d",
        model or OPENAI_MODEL,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )
    return response.choices[0].message.content or ""


def complete_structured(
    schema: Type[T],
    prompt: str,
    system: str = "You are a precise clinical information extraction system.",
    model: str | None = None,
) -> T:
    """
    Completion with structured JSON output parsed into a Pydantic model.

    Gemini path: response_schema=schema enforces output shape; falls back to
    model_validate_json if response.parsed is absent.
    OpenAI path: native beta.parse endpoint with response_format=schema.

    Raises ValidationError if the model returns malformed data — never silently swallows.
    """
    if _PROVIDER == "gemini":
        response = _gemini_client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=prompt,
            config=_gtypes.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
                max_output_tokens=GEMINI_MAX_TOKENS,
            ),
        )
        _track_gemini_usage(response.usage_metadata)
        # Use auto-parsed object if available, else parse from text
        if response.parsed is not None:
            return response.parsed  # type: ignore[return-value]
        return schema.model_validate_json(response.text)

    # OpenAI path
    response = _openai_client.beta.chat.completions.parse(
        model=model or OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format=schema,
    )
    usage = response.usage
    if usage:
        _usage["prompt_tokens"] += usage.prompt_tokens
        _usage["completion_tokens"] += usage.completion_tokens
        _usage["total_tokens"] += usage.total_tokens
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"LLM returned no parsed output for schema {schema.__name__}")
    return parsed


def verify_connection() -> bool:
    """Quick smoke-test that the API key and model are reachable."""
    try:
        result = complete("Reply with the single word: OK")
        ok = "ok" in result.strip().lower()
        print(f"LLM [{_PROVIDER}] connection test: {'PASS' if ok else 'UNEXPECTED'} — '{result.strip()}'")
        return ok
    except Exception as exc:
        print(f"LLM [{_PROVIDER}] connection test: FAIL — {exc}")
        return False


if __name__ == "__main__":
    verify_connection()
    print("Token usage:", get_usage())
