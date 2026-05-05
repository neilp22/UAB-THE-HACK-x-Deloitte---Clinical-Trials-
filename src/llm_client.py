"""
Single wrapper for all LLM calls.

Rules (from CLAUDE.md):
- Model, temperature, max_tokens come from src/config.py
- Swapping model = change OPENAI_MODEL in .env
- All LLM output MUST be validated by the caller with Pydantic before use
- Never call this for: age/gender/numeric comparisons, dates, scoring

Usage:
    from src.llm_client import complete, complete_structured

    text = complete("Parse these criteria: ...", system="You are a clinical NLP system.")
    obj  = complete_structured(MyPydanticModel, prompt, system)
"""
from __future__ import annotations

import logging
from typing import Any, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src.config import OPENAI_API_KEY, OPENAI_MAX_TOKENS, OPENAI_MODEL, OPENAI_TEMPERATURE

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

T = TypeVar("T", bound=BaseModel)

# Cumulative token usage tracker (reset between runs if needed)
_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def get_usage() -> dict[str, int]:
    """Return cumulative token counts since process start."""
    return dict(_usage)


def reset_usage() -> None:
    _usage["prompt_tokens"] = 0
    _usage["completion_tokens"] = 0
    _usage["total_tokens"] = 0


def complete(
    prompt: str,
    system: str = "You are a precise clinical information extraction system.",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Single-turn chat completion.

    Returns the assistant message text.
    Raises openai.APIError on failure (let it propagate — caller handles retry/caching).
    """
    response = _client.chat.completions.create(
        model=model or OPENAI_MODEL,
        temperature=temperature if temperature is not None else OPENAI_TEMPERATURE,
        max_tokens=max_tokens or OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    # Track usage
    usage = response.usage
    if usage:
        _usage["prompt_tokens"] += usage.prompt_tokens
        _usage["completion_tokens"] += usage.completion_tokens
        _usage["total_tokens"] += usage.total_tokens

    content = response.choices[0].message.content or ""
    logger.debug(
        "LLM call: model=%s, in=%d, out=%d",
        model or OPENAI_MODEL,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )
    return content


def complete_structured(
    schema: Type[T],
    prompt: str,
    system: str = "You are a precise clinical information extraction system.",
    model: str | None = None,
) -> T:
    """
    Completion with JSON structured output parsed into a Pydantic model.

    Uses OpenAI's native JSON mode + Pydantic parsing.
    Raises ValidationError if the model returns malformed JSON — never silently swallows.
    """
    response = _client.beta.chat.completions.parse(
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
    """Quick smoke-test that the API key is valid. Prints result."""
    try:
        result = complete("Reply with the single word: OK", max_tokens=5)
        ok = "ok" in result.strip().lower()
        print(f"LLM connection test: {'PASS' if ok else 'UNEXPECTED RESPONSE'} — '{result.strip()}'")
        return ok
    except Exception as exc:
        print(f"LLM connection test: FAIL — {exc}")
        return False


if __name__ == "__main__":
    verify_connection()
    print("Token usage:", get_usage())
