"""
Clinical trial eligibility criteria parser.

Input:  free-text eligibility_criteria string from ClinicalTrials.gov API
Output: ParsedCriteria — flat list of Criterion objects (inclusion + exclusion)

Pipeline:
  1. Regex-split raw text into inclusion / exclusion sections (no LLM)
  2. Extract individual items from each section (no LLM)
  3. If estimated tokens > 3000 or > 20 items total → batch mode (log it)
  4. LLM call per batch → structured Criterion objects
  5. Post-process: normalise operator symbols, validate category, fix type mismatches
  6. Cache result as JSON string keyed by MD5 of criteria_text

LLM is NOT used for: empty input, cache hits, item counting, section splitting.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Literal

import diskcache
from pydantic import BaseModel, Field

from src.config import CACHE_DIR
from src.llm_client import complete_structured

logger = logging.getLogger(__name__)

_criteria_cache = diskcache.Cache(str(CACHE_DIR / "criteria_parser"))

_VALID_CATEGORIES = {"age", "gender", "lab_value", "prior_treatment", "diagnosis", "other"}
_OPERATOR_NORM: dict[str, str] = {
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    ">= ": ">=",
    "<= ": "<=",
    "=": "==",
}
_BATCH_SIZE = 10        # criteria per type per LLM batch
_TOKEN_THRESHOLD = 3000  # rough token estimate that triggers batching
_CHARS_PER_TOKEN = 4    # conservative approximation


# ---------------------------------------------------------------------------
# Public Pydantic models (as specified)
# ---------------------------------------------------------------------------

class Criterion(BaseModel):
    text: str
    type: Literal["inclusion", "exclusion"]
    category: str
    deterministic: bool
    numeric_threshold: float | None = None
    numeric_unit: str | None = None
    numeric_operator: str | None = None  # ">=","<=",">","<","=="


class ParsedCriteria(BaseModel):
    criteria: list[Criterion] = Field(default_factory=list)


# Internal LLM extraction schema (identical shape, used as response_format)
class _CriteriaExtraction(BaseModel):
    criteria: list[Criterion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section splitting and item extraction (deterministic)
# ---------------------------------------------------------------------------

_INCLUSION_RE = re.compile(r"inclusion\s+criteria[:\s]*", re.IGNORECASE)
_EXCLUSION_RE = re.compile(r"exclusion\s+criteria[:\s]*", re.IGNORECASE)
_BULLET_PREFIX = re.compile(r"^\s*(?:[\-\*\•\·\–•]|\d+[\.\)])\s*")


def _split_sections(criteria_text: str) -> tuple[str, str]:
    """
    Regex-split raw criteria text into (inclusion_text, exclusion_text).
    Falls back gracefully when headers are missing.
    """
    inc_m = _INCLUSION_RE.search(criteria_text)
    exc_m = _EXCLUSION_RE.search(criteria_text)

    if inc_m and exc_m:
        if inc_m.start() < exc_m.start():
            return (
                criteria_text[inc_m.end() : exc_m.start()].strip(),
                criteria_text[exc_m.end() :].strip(),
            )
        else:
            return (
                criteria_text[inc_m.end() :].strip(),
                criteria_text[exc_m.end() : inc_m.start()].strip(),
            )
    elif inc_m:
        return criteria_text[inc_m.end() :].strip(), ""
    elif exc_m:
        return "", criteria_text[exc_m.end() :].strip()
    else:
        # No headers — treat entire text as inclusion (conservative)
        return criteria_text.strip(), ""


def _extract_items(section_text: str) -> list[str]:
    """
    Extract individual criterion items from a section block.

    Bulleted lines: continuation (indented) lines are joined to the preceding bullet.
    Non-bulleted sections: each non-empty line is its own item.
    """
    if not section_text.strip():
        return []

    items: list[str] = []
    current: list[str] = []
    current_bulleted = False

    for line in section_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                items.append(" ".join(current))
                current = []
                current_bulleted = False
            continue

        is_new = bool(_BULLET_PREFIX.match(stripped))
        if is_new:
            if current:
                items.append(" ".join(current))
            text = _BULLET_PREFIX.sub("", stripped).strip()
            current = [text] if text else []
            current_bulleted = True
        elif current_bulleted:
            # Indented continuation of a bulleted item
            current.append(stripped)
        else:
            # No bullet in play — each line is its own item
            if current:
                items.append(" ".join(current))
            current = [stripped]
            current_bulleted = False

    if current:
        items.append(" ".join(current))

    return [i for i in items if i]


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# LLM batch parsing
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a precise clinical trial eligibility criteria parser. "
    "Return only valid JSON. No preamble, no markdown, no explanation."
)

_PROMPT_TEMPLATE = """\
Parse each eligibility criterion into a structured JSON object.

Rules:
- type: "inclusion" for items under INCLUSION CRITERIA, "exclusion" for EXCLUSION CRITERIA
- category must be exactly one of: "age", "gender", "lab_value", "prior_treatment", "diagnosis", "other"
  - age: age range requirements
  - gender: biological sex requirements
  - lab_value: numeric lab results, ECOG performance score, vital signs with thresholds
  - prior_treatment: prior therapies, surgeries, chemotherapy, procedures, radiation
  - diagnosis: disease, condition, staging, biomarkers, mutations, histology
  - other: anything else
- deterministic: true ONLY if evaluable from structured fields alone (integer age, gender string, ECOG integer, numeric lab values) WITHOUT semantic reasoning; false for all others
- numeric_threshold / numeric_unit / numeric_operator: fill for explicit numeric cutoffs (e.g. "creatinine ≤ 1.5 mg/dL" → 1.5, "mg/dL", "<="); null otherwise
- numeric_operator must be one of: ">=", "<=", ">", "<", "=="

INCLUSION CRITERIA (type="inclusion"):
{inclusion_block}

EXCLUSION CRITERIA (type="exclusion"):
{exclusion_block}"""


def _fmt_items(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


def _normalize(c: Criterion) -> Criterion:
    """Normalise operator symbols and category value; fix deterministic inconsistency."""
    op = _OPERATOR_NORM.get(c.numeric_operator or "", c.numeric_operator)
    cat = (c.category or "other").lower().strip()
    if cat not in _VALID_CATEGORIES:
        cat = "other"

    # Downgrade deterministic if category implies numeric data but threshold is absent
    det = c.deterministic
    if det and cat in ("lab_value", "age") and c.numeric_threshold is None:
        det = False

    return Criterion(
        text=c.text,
        type=c.type,
        category=cat,
        deterministic=det,
        numeric_threshold=c.numeric_threshold,
        numeric_unit=c.numeric_unit,
        numeric_operator=op,
    )


def _fix_types(
    criteria: list[Criterion],
    inc_items: list[str],
    exc_items: list[str],
) -> list[Criterion]:
    """
    Override type field when it mismatches the source section.
    Uses first-50-char text prefix for matching (LLM may paraphrase slightly).
    """
    inc_keys = {item.lower()[:60] for item in inc_items}
    exc_keys = {item.lower()[:60] for item in exc_items}
    fixed = []
    for c in criteria:
        key = c.text.lower()[:60]
        if key in exc_keys and c.type != "exclusion":
            c = c.model_copy(update={"type": "exclusion"})
        elif key in inc_keys and c.type != "inclusion":
            c = c.model_copy(update={"type": "inclusion"})
        fixed.append(c)
    return fixed


def _llm_parse_batch(
    inc_batch: list[str],
    exc_batch: list[str],
) -> list[Criterion]:
    """
    One LLM call to parse up to _BATCH_SIZE inclusion + _BATCH_SIZE exclusion criteria.
    Retries once with identical prompt on failure; returns raw-text fallback on second failure.
    """
    prompt = _PROMPT_TEMPLATE.format(
        inclusion_block=_fmt_items(inc_batch),
        exclusion_block=_fmt_items(exc_batch),
    )

    for attempt in range(1, 3):
        try:
            result = complete_structured(_CriteriaExtraction, prompt, system=_SYSTEM)
            criteria = [_normalize(c) for c in result.criteria]
            criteria = _fix_types(criteria, inc_batch, exc_batch)
            return criteria
        except Exception as exc:
            logger.warning("Criteria LLM batch failed (attempt %d): %s", attempt, exc)

    # Both attempts failed — produce minimal fallback (no numeric data)
    logger.error("Both LLM attempts failed; using raw-text fallback for batch.")
    fallback: list[Criterion] = []
    for item in inc_batch:
        fallback.append(
            Criterion(text=item, type="inclusion", category="other", deterministic=False)
        )
    for item in exc_batch:
        fallback.append(
            Criterion(text=item, type="exclusion", category="other", deterministic=False)
        )
    return fallback


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_criteria(
    criteria_text: str,
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> ParsedCriteria:
    """
    Parse a free-text eligibility criteria block into structured Criterion objects.

    Splits inclusion/exclusion sections with regex, then calls LLM for structured
    extraction. Results are cached as JSON strings keyed by MD5 of criteria_text.

    Never raises; returns empty ParsedCriteria on any failure.
    """
    if not criteria_text or not criteria_text.strip():
        return ParsedCriteria()

    if use_cache and cache is None:
        cache = _criteria_cache

    cache_key = hashlib.md5(criteria_text.encode()).hexdigest()

    # Cache read
    if use_cache and cache is not None:
        raw = cache.get(cache_key)
        if raw is not None:
            try:
                return ParsedCriteria.model_validate_json(raw)
            except Exception as exc:
                logger.warning("Cache deserialise failed (%s…): %s", cache_key[:8], exc)

    try:
        result = _do_parse(criteria_text)
    except Exception as exc:
        logger.error("parse_criteria failed: %s", exc)
        result = ParsedCriteria()

    # Cache write (raw JSON string as specified)
    if use_cache and cache is not None:
        try:
            cache.set(cache_key, result.model_dump_json())
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    return result


def _do_parse(criteria_text: str) -> ParsedCriteria:
    """Core parse logic without caching."""
    inc_text, exc_text = _split_sections(criteria_text)
    inc_items = _extract_items(inc_text)
    exc_items = _extract_items(exc_text)

    total_items = len(inc_items) + len(exc_items)
    token_est = _estimate_tokens(criteria_text)
    use_batching = token_est > _TOKEN_THRESHOLD or total_items > 20

    all_criteria: list[Criterion] = []

    if use_batching:
        num_batches = max(
            (len(inc_items) + _BATCH_SIZE - 1) // _BATCH_SIZE,
            (len(exc_items) + _BATCH_SIZE - 1) // _BATCH_SIZE,
            1,
        )
        logger.info(
            "Batching criteria parse: %d inc + %d exc items (~%d tokens) → %d batches",
            len(inc_items), len(exc_items), token_est, num_batches,
        )
        for b in range(num_batches):
            i_start, i_end = b * _BATCH_SIZE, (b + 1) * _BATCH_SIZE
            e_start, e_end = b * _BATCH_SIZE, (b + 1) * _BATCH_SIZE
            inc_batch = inc_items[i_start:i_end]
            exc_batch = exc_items[e_start:e_end]
            logger.info("Batch %d/%d: %d inc, %d exc", b + 1, num_batches, len(inc_batch), len(exc_batch))
            all_criteria.extend(_llm_parse_batch(inc_batch, exc_batch))
    else:
        all_criteria = _llm_parse_batch(inc_items, exc_items)

    return ParsedCriteria(criteria=all_criteria)
