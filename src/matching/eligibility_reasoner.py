"""
Eligibility reasoner — evaluates one criterion against a patient profile.

Returns CriterionVerdict: MET / NOT_MET / NEI

Logic:
  - deterministic=True  → rule-based (age / gender / ECOG / lab) → MET or NOT_MET
                          Falls back to NEI when patient field is None or data is missing.
  - deterministic=False → LLM call (Pydantic structured output) → retry once → NEI on failure

Caching:
  - Deterministic path: not cached (pure arithmetic, negligible cost)
  - LLM path: keyed by MD5(criterion.text + "|" + criterion.type + "|" + profile_json)

Caller is responsible for deciding which criteria to evaluate and aggregating verdicts.
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
from src.matching.hard_filter import (
    _LAB_MIN_KEY_LEN,
    _compare,
    _is_ecog,
    _parse_first_float,
    _units_compatible,
)
from src.parsing.criteria_parser import Criterion
from src.parsing.patient_normalizer import PatientProfile

logger = logging.getLogger(__name__)

_reasoner_cache = diskcache.Cache(str(CACHE_DIR / "eligibility_reasoner"))

_FEMALE_RE = re.compile(r"\b(?:female|women|woman)\b", re.IGNORECASE)
_MALE_RE = re.compile(r"\bmale\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------

class CriterionVerdict(BaseModel):
    verdict: Literal["MET", "NOT_MET", "NEI"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# LLM prompt (verbatim per spec)
# ---------------------------------------------------------------------------

_SYSTEM = "You are a clinical trial eligibility assessor. Be conservative. Default to NEI."

_PROMPT_TEMPLATE = """\
Patient profile: {profile_json}
Criterion: {criterion_text}
Type: {criterion_type}
Category: {criterion_category}

Does this patient meet this criterion?
- MET: the patient clearly satisfies this criterion
- NOT_MET: the patient clearly does not satisfy this criterion
- NEI: insufficient information to determine

Return verdict, confidence (0.0-1.0), and a brief reasoning."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_json(profile: PatientProfile) -> str:
    return profile.model_dump_json(exclude_none=True)


def _cache_key(criterion: Criterion, profile: PatientProfile) -> str:
    raw = criterion.text + "|" + criterion.type + "|" + _profile_json(profile)
    return hashlib.md5(raw.encode()).hexdigest()


def _verdict_from_condition(
    condition_met: bool, criterion: Criterion, reasoning: str
) -> CriterionVerdict:
    """
    Map a boolean numeric/gender check to MET/NOT_MET.

    For exclusion criteria: condition_met=True means the exclusion condition fires
    (patient violates the exclusion) → NOT_MET.
    For inclusion criteria: condition_met=True means the inclusion requirement is
    satisfied → MET.
    """
    if criterion.type == "exclusion":
        verdict = "NOT_MET" if condition_met else "MET"
    else:
        verdict = "MET" if condition_met else "NOT_MET"
    return CriterionVerdict(verdict=verdict, confidence=1.0, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Deterministic evaluators (return NEI when patient data is insufficient)
# ---------------------------------------------------------------------------

def _eval_age(profile: PatientProfile, criterion: Criterion) -> CriterionVerdict:
    if profile.age is None:
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="Patient age not recorded")
    if criterion.numeric_threshold is None or not criterion.numeric_operator:
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="No numeric threshold/operator in criterion")
    val = float(profile.age)
    met = _compare(val, criterion.numeric_operator, criterion.numeric_threshold)
    return _verdict_from_condition(
        met, criterion,
        f"age {val} {criterion.numeric_operator} {criterion.numeric_threshold} → {met}",
    )


def _eval_gender(profile: PatientProfile, criterion: Criterion) -> CriterionVerdict:
    if profile.gender == "unknown":
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="Patient gender not recorded")
    text = criterion.text
    if _FEMALE_RE.search(text):
        target, met = "female", profile.gender == "female"
    elif _MALE_RE.search(text):
        target, met = "male", profile.gender == "male"
    else:
        return CriterionVerdict(verdict="NEI", confidence=0.3,
                                reasoning="Cannot identify target gender from criterion text")
    return _verdict_from_condition(
        met, criterion,
        f"criterion targets {target}; patient is {profile.gender}",
    )


def _eval_ecog(profile: PatientProfile, criterion: Criterion) -> CriterionVerdict:
    if profile.ecog_score is None:
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="Patient ECOG score not recorded")
    if criterion.numeric_threshold is None or not criterion.numeric_operator:
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="No numeric threshold/operator in criterion")
    val = float(profile.ecog_score)
    met = _compare(val, criterion.numeric_operator, criterion.numeric_threshold)
    return _verdict_from_condition(
        met, criterion,
        f"ECOG {val} {criterion.numeric_operator} {criterion.numeric_threshold} → {met}",
    )


def _eval_lab(profile: PatientProfile, criterion: Criterion) -> CriterionVerdict:
    if not profile.lab_values:
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="No lab values in patient profile")
    if criterion.numeric_threshold is None or not criterion.numeric_operator:
        return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="No numeric threshold/operator in criterion")

    crit_lower = criterion.text.lower()
    matched_key: str | None = None
    for key in profile.lab_values:
        if len(key) < _LAB_MIN_KEY_LEN:
            continue
        if re.search(r"\b" + re.escape(key) + r"\b", crit_lower):
            matched_key = key
            break

    if matched_key is None:
        return CriterionVerdict(verdict="NEI", confidence=0.5,
                                reasoning="Lab value not found in patient profile")

    value_str = profile.lab_values[matched_key]
    patient_val = _parse_first_float(value_str)
    if patient_val is None:
        return CriterionVerdict(verdict="NEI", confidence=0.5,
                                reasoning=f"Could not parse numeric value from '{value_str}'")

    if not _units_compatible(criterion.numeric_unit, value_str):
        return CriterionVerdict(verdict="NEI", confidence=0.3,
                                reasoning="Unit mismatch — cannot safely compare")

    met = _compare(patient_val, criterion.numeric_operator, criterion.numeric_threshold)
    unit_str = criterion.numeric_unit or ""
    return _verdict_from_condition(
        met, criterion,
        f"{matched_key}={patient_val}{' ' + unit_str if unit_str else ''} "
        f"{criterion.numeric_operator} {criterion.numeric_threshold} → {met}",
    )


def _evaluate_deterministic(profile: PatientProfile, criterion: Criterion) -> CriterionVerdict:
    if criterion.category == "age":
        return _eval_age(profile, criterion)
    if criterion.category == "gender":
        return _eval_gender(profile, criterion)
    if criterion.category == "lab_value":
        if _is_ecog(criterion):
            return _eval_ecog(profile, criterion)
        return _eval_lab(profile, criterion)
    return CriterionVerdict(verdict="NEI", confidence=0.3,
                            reasoning="Category not evaluable deterministically")


# ---------------------------------------------------------------------------
# LLM evaluator
# ---------------------------------------------------------------------------

def _llm_evaluate(profile: PatientProfile, criterion: Criterion) -> CriterionVerdict:
    prompt = _PROMPT_TEMPLATE.format(
        profile_json=_profile_json(profile),
        criterion_text=criterion.text,
        criterion_type=criterion.type,
        criterion_category=criterion.category,
    )
    for attempt in range(1, 3):
        try:
            return complete_structured(CriterionVerdict, prompt, system=_SYSTEM)
        except Exception as exc:
            logger.warning("Eligibility LLM failed (attempt %d): %s", attempt, exc)
    logger.error("Both LLM attempts failed: %s", criterion.text[:80])
    return CriterionVerdict(verdict="NEI", confidence=0.0,
                            reasoning="LLM evaluation failed after 2 attempts")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_criterion(
    profile: PatientProfile,
    criterion: Criterion,
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> CriterionVerdict:
    """
    Evaluate a single criterion against a patient profile.

    deterministic=True  → rule-based, no LLM, no cache
    deterministic=False → LLM with diskcache

    Never raises. Returns NEI on any unrecoverable error.
    """
    if criterion.deterministic:
        return _evaluate_deterministic(profile, criterion)

    if use_cache and cache is None:
        cache = _reasoner_cache

    key = _cache_key(criterion, profile)
    if use_cache and cache is not None:
        raw = cache.get(key)
        if raw is not None:
            try:
                return CriterionVerdict.model_validate_json(raw)
            except Exception as exc:
                logger.warning("Cache deserialise failed (%s…): %s", key[:8], exc)

    verdict = _llm_evaluate(profile, criterion)

    if use_cache and cache is not None:
        try:
            cache.set(key, verdict.model_dump_json())
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    return verdict
