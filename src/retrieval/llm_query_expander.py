"""
LLM-based condition variant expansion for clinical trial retrieval.

Uses gpt-4o-mini via src/llm_client.py (maintains token tracking + model routing).
Cached by (conditions, patient_text) hash — zero cost on repeated calls.
"""
from __future__ import annotations

import hashlib
import json
import logging

import diskcache

from src.config import CACHE_DIR
from src.parsing.patient_normalizer import PatientProfile

logger = logging.getLogger(__name__)

_cache = diskcache.Cache(str(CACHE_DIR / "llm_query_expander"))


def get_expanded_condition_variants(
    profile: PatientProfile,
    patient_text: str,
    use_cache: bool = True,
) -> list[str]:
    """
    Use gpt-4o-mini to generate 4-6 precise MeSH-compatible search terms
    from the patient profile, suitable for CT API query.cond / query.term.

    Returns a list of short term strings (1-4 words each).
    Falls back to [] on any error — never raises.
    """
    key = "llm_exp:" + hashlib.md5(
        (str(profile.conditions) + patient_text[:300]).encode()
    ).hexdigest()

    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    prompt = (
        "You are a biomedical search expert. Given the patient below, "
        "return a JSON array of 4-6 short, precise medical terms suitable "
        "for searching ClinicalTrials.gov. Prefer full MeSH names over acronyms. "
        "Each term must be 1-4 words. Return ONLY the JSON array, no explanation.\n"
        "Example: [\"non-small cell lung carcinoma\", \"EGFR mutation\", "
        "\"osimertinib resistant\", \"stage IV adenocarcinoma\"]\n\n"
        f"Patient: {patient_text[:400]}"
    )

    try:
        from src.llm_client import complete
        raw = complete(prompt, temperature=0, max_tokens=150)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        terms = json.loads(raw)
        result = [str(t).strip() for t in terms if isinstance(t, str) and t.strip()][:6]
    except Exception as exc:
        logger.warning("LLM query expansion failed (%s) — returning []", exc)
        result = []

    if use_cache:
        _cache[key] = result

    return result
