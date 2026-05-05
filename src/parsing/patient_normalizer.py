"""
Patient profile extractor: free-text clinical note → PatientProfile.

Deterministic (regex): age, gender, ECOG, lab values — no LLM, no failures.
LLM (gpt-4o-mini): conditions, medications, prior treatments, relevant history.
LLM output is cached by note MD5 hash and validated with Pydantic before use.
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

_llm_cache = diskcache.Cache(str(CACHE_DIR / "patient_normalizer"))


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    age: int | None = None
    gender: Literal["male", "female", "unknown"] = "unknown"
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    ecog_score: int | None = None
    lab_values: dict[str, str] = Field(default_factory=dict)
    prior_treatments: list[str] = Field(default_factory=list)
    relevant_history: list[str] = Field(default_factory=list)


# Internal schema for LLM structured output
class _LLMExtraction(BaseModel):
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    prior_treatments: list[str] = Field(default_factory=list)
    relevant_history: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic extractors (no LLM, no network)
# ---------------------------------------------------------------------------

def extract_age(text: str) -> int | None:
    """Extract integer patient age from free text. Returns None if not found."""
    patterns = [
        r"\b(\d{1,3})-year-old\b",
        r"\b(\d{1,3})\s*yo\b",
        r"\b(\d{1,3})\s*yr[s]?[- ]?old\b",
        r"\bage[d]?\s+(\d{1,3})\b",
        r"\b(\d{1,3})\s+year[s]?\s+old\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 119:
                return val
    return None


def extract_gender(text: str) -> Literal["male", "female", "unknown"]:
    """Extract patient gender from first 500 chars of text (most reliable signal)."""
    snippet = text[:500].lower()
    female_hits = len(re.findall(r"\b(female|woman|girl|she\b|her\b|mrs\.?|ms\.?)\b", snippet))
    male_hits = len(re.findall(r"\b(male|man|boy|he\b|him\b|his\b|mr\.?)\b", snippet))
    if female_hits > male_hits:
        return "female"
    if male_hits > female_hits:
        return "male"
    return "unknown"


def extract_ecog(text: str) -> int | None:
    """Extract ECOG performance status integer (0-5)."""
    m = re.search(
        r"\bECOG\s*(?:performance\s*status|PS|score)?\s*(?:of\s*)?([0-5])\b",
        text, re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None


# Lab value patterns: (regex, canonical_name)
_LAB_PATTERNS: list[tuple[str, str]] = [
    (r"creatinine\s+(?:of\s+)?(\d+\.?\d*)\s*(mg/dL|mg/dl|umol/L)?", "creatinine"),
    (r"h(?:aemo|emo)globin\s+(?:of\s+)?(\d+\.?\d*)\s*(g/dL|g/dl)?", "hemoglobin"),
    (r"bilirubin\s+(?:(?:total\s+)?(?:of\s+))?(\d+\.?\d*)\s*(mg/dL|mg/dl)?", "bilirubin"),
    (r"\bALT\s+(?:of\s+)?(\d+)\s*(U/L|IU/L)?", "alt"),
    (r"\bAST\s+(?:of\s+)?(\d+)\s*(U/L|IU/L)?", "ast"),
    (r"\bWBC\s+(?:of\s+)?(\d+\.?\d*)\s*(k/[uµ]L|x10\^?9/L|K/uL)?", "wbc"),
    (r"platelet[s]?\s+(?:(?:count|level)s?\s+)?(?:of\s+)?(\d+)\s*(k/[uµ]L|K/uL)?", "platelets"),
    (r"\bPSA\s+(?:of\s+)?(\d+\.?\d*)\s*(ng/mL|ng/ml)?", "psa"),
    (r"\b[e]?GFR\s+(?:of\s+)?(\d+\.?\d*)\b", "egfr"),
]


def extract_lab_values(text: str) -> dict[str, str]:
    """Extract numeric lab values from text. Returns {name: 'value unit'}."""
    labs: dict[str, str] = {}
    for pat, name in _LAB_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            value = m.group(1)
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else ""
            labs[name] = f"{value} {unit}".strip()
    return labs


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a clinical information extraction system. "
    "Extract only what is explicitly stated — do not infer or hallucinate."
)

_PROMPT = """\
Extract medical information from the clinical note below.

Return four lists:
- conditions: primary diagnosis + significant comorbidities
- medications: current or recent medications with doses if given
- prior_treatments: past treatments, surgeries, chemotherapy, radiation, procedures
- relevant_history: other clinically significant facts (biomarkers, mutations, staging, performance status, smoking history, allergies)

Aim for 2-8 items per non-empty list. Use empty list [] if none found.

Clinical note:
{note}"""


def _llm_extract(clinical_note: str) -> _LLMExtraction:
    """
    Call LLM for structured extraction. Returns empty defaults on any failure.
    Retries once with a shorter note on first failure.
    """
    for max_chars in (3000, 1500):
        try:
            return complete_structured(
                _LLMExtraction,
                _PROMPT.format(note=clinical_note[:max_chars]),
                system=_SYSTEM,
            )
        except Exception as exc:
            logger.warning("LLM extraction attempt failed (chars=%d): %s", max_chars, exc)

    logger.error("LLM extraction failed after retries; returning empty extraction.")
    return _LLMExtraction()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_patient(
    clinical_note: str,
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> PatientProfile:
    """
    Parse a free-text clinical note into a structured PatientProfile.

    Deterministic fields (age, gender, ECOG, labs) are always extracted via regex.
    LLM is called only for conditions/medications/treatments/history.
    LLM output is cached by MD5 of note text — same note never calls LLM twice.

    Never raises; returns a (possibly partial) PatientProfile on any error.
    """
    if not clinical_note or not clinical_note.strip():
        return PatientProfile()

    if use_cache and cache is None:
        cache = _llm_cache

    # Deterministic extraction (fast, no LLM, no failures)
    age = extract_age(clinical_note)
    gender = extract_gender(clinical_note)
    ecog = extract_ecog(clinical_note)
    labs = extract_lab_values(clinical_note)

    # LLM extraction (cached as plain dict to avoid pickle issues)
    note_hash = hashlib.md5(clinical_note.encode()).hexdigest()
    cache_key = f"llm:{note_hash}"
    llm_data = _LLMExtraction()

    if use_cache and cache is not None:
        raw = cache.get(cache_key)
        if raw is not None:
            try:
                llm_data = _LLMExtraction(**raw)
            except Exception:
                llm_data = _llm_extract(clinical_note)
        else:
            llm_data = _llm_extract(clinical_note)
            try:
                cache.set(cache_key, llm_data.model_dump())
            except Exception as exc:
                logger.warning("Cache write failed: %s", exc)
    else:
        try:
            llm_data = _llm_extract(clinical_note)
        except Exception as exc:
            logger.error("LLM extraction failed: %s", exc)

    return PatientProfile(
        age=age,
        gender=gender,
        ecog_score=ecog,
        lab_values=labs,
        conditions=llm_data.conditions,
        medications=llm_data.medications,
        prior_treatments=llm_data.prior_treatments,
        relevant_history=llm_data.relevant_history,
    )
