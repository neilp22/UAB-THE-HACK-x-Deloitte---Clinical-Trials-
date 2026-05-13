"""
Eligibility reasoner — evaluates criteria against a patient profile.

Two public entry points:

evaluate_criterion(profile, criterion)
    Evaluates one criterion. deterministic=True → rule-based; False → one LLM call.
    Cached per-criterion by MD5(text + type + profile_json).

evaluate_trial(profile, criteria, nct_id)
    Evaluates ALL criteria for a trial. Deterministic criteria → rule-based.
    All non-deterministic criteria bundled into one LLM call (batched in groups of
    _TRIAL_BATCH_SIZE if token estimate exceeds _TRIAL_TOKEN_THRESHOLD).
    Cached per-trial by MD5(nct_id + "|" + profile_json).

Both return CriterionVerdict: MET / NOT_MET / NEI
evaluate_trial returns list[(criterion_type, CriterionVerdict)] in criterion order.
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

_SYSTEM = "You are a clinical trial eligibility assessor. Be conservative. Return NEI only when required information is explicitly missing."

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


# ---------------------------------------------------------------------------
# Trial-level evaluator — one LLM call for all non-deterministic criteria
# ---------------------------------------------------------------------------

_TRIAL_BATCH_SIZE = 10
_TRIAL_TOKEN_THRESHOLD = 3000
_CHARS_PER_TOKEN = 4

_TRIAL_SYSTEM = "You are a clinical trial eligibility assessor. Be conservative. Return NEI only when required information is explicitly missing."

_TRIAL_PROMPT_TEMPLATE = """\
Patient profile: {profile_json}

Assess each eligibility criterion below. Return exactly {n} verdicts in the same order as listed.

{numbered_criteria}

For each criterion return verdict (MET/NOT_MET/NEI), confidence (0.0-1.0), and brief reasoning."""


class _TrialVerdicts(BaseModel):
    verdicts: list[CriterionVerdict]


def _fmt_criteria_list(criteria: list[Criterion]) -> str:
    return "\n".join(
        f"{i}. [{c.type}] {c.text}" for i, c in enumerate(criteria, 1)
    )


def _llm_batch_trial(
    profile: PatientProfile, criteria: list[Criterion]
) -> list[CriterionVerdict]:
    """One LLM call for a batch of ≤_TRIAL_BATCH_SIZE non-deterministic criteria."""
    prompt = _TRIAL_PROMPT_TEMPLATE.format(
        profile_json=_profile_json(profile),
        n=len(criteria),
        numbered_criteria=_fmt_criteria_list(criteria),
    )
    for attempt in range(1, 3):
        try:
            result = complete_structured(_TrialVerdicts, prompt, system=_TRIAL_SYSTEM)
            verdicts = result.verdicts
            # Pad to match expected count if LLM returned fewer
            nei = CriterionVerdict(verdict="NEI", confidence=0.0,
                                   reasoning="LLM returned fewer verdicts than expected")
            while len(verdicts) < len(criteria):
                verdicts.append(nei)
            return verdicts[: len(criteria)]
        except Exception as exc:
            logger.warning("Trial LLM batch failed (attempt %d): %s", attempt, exc)
    logger.error("Both trial LLM batch attempts failed (%d criteria)", len(criteria))
    return [
        CriterionVerdict(verdict="NEI", confidence=0.0, reasoning="LLM batch failed")
        for _ in criteria
    ]


def evaluate_trial(
    profile: PatientProfile,
    criteria: list[Criterion],
    nct_id: str = "",
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> list[tuple[str, CriterionVerdict]]:
    """
    Evaluate all criteria for one trial in as few LLM calls as possible.

    Deterministic criteria → rule-based (no LLM, not cached).
    Non-deterministic criteria → bundled into one LLM call; batched in groups of
    _TRIAL_BATCH_SIZE only when the prompt would exceed _TRIAL_TOKEN_THRESHOLD tokens.

    Cache key: MD5(nct_id + "|" + profile_json)  — per-trial, not per-criterion.
    Falls back to MD5(all criterion texts + profile_json) when nct_id is empty.

    Returns list[(criterion.type, CriterionVerdict)] in the same order as input criteria.
    Never raises; returns NEI for any criterion that cannot be evaluated.
    """
    if not criteria:
        return []

    if use_cache and cache is None:
        cache = _reasoner_cache

    # Build cache key
    profile_j = _profile_json(profile)
    if nct_id:
        raw_key = nct_id + "|" + profile_j
    else:
        texts = "".join(c.text for c in criteria)
        raw_key = hashlib.md5(texts.encode()).hexdigest() + "|" + profile_j
    trial_cache_key = "trial:" + hashlib.md5(raw_key.encode()).hexdigest()

    if use_cache and cache is not None:
        raw = cache.get(trial_cache_key)
        if raw is not None:
            try:
                import json as _json
                stored = _json.loads(raw)
                return [(item[0], CriterionVerdict.model_validate(item[1])) for item in stored]
            except Exception as exc:
                logger.warning("Trial cache deserialise failed (%s…): %s",
                               trial_cache_key[:12], exc)

    # ---- evaluate ----
    results: list[tuple[str, CriterionVerdict] | None] = [None] * len(criteria)
    non_det_indices: list[int] = []

    for i, c in enumerate(criteria):
        if c.deterministic:
            results[i] = (c.type, _evaluate_deterministic(profile, c))
        else:
            non_det_indices.append(i)

    if non_det_indices:
        non_det = [criteria[i] for i in non_det_indices]

        # Decide whether to batch: estimate single-call prompt size
        single_prompt = _TRIAL_PROMPT_TEMPLATE.format(
            profile_json=profile_j,
            n=len(non_det),
            numbered_criteria=_fmt_criteria_list(non_det),
        )
        if (len(single_prompt) // _CHARS_PER_TOKEN <= _TRIAL_TOKEN_THRESHOLD
                and len(non_det) <= _TRIAL_BATCH_SIZE):
            llm_verdicts = _llm_batch_trial(profile, non_det)
        else:
            llm_verdicts = []
            for start in range(0, len(non_det), _TRIAL_BATCH_SIZE):
                batch = non_det[start: start + _TRIAL_BATCH_SIZE]
                llm_verdicts.extend(_llm_batch_trial(profile, batch))

        for orig_i, verdict in zip(non_det_indices, llm_verdicts):
            results[orig_i] = (criteria[orig_i].type, verdict)

    final: list[tuple[str, CriterionVerdict]] = []
    for i, r in enumerate(results):
        if r is None:
            final.append((criteria[i].type,
                          CriterionVerdict(verdict="NEI", confidence=0.0,
                                           reasoning="Evaluation skipped")))
        else:
            final.append(r)

    if use_cache and cache is not None:
        try:
            import json as _json
            cache.set(trial_cache_key,
                      _json.dumps([[t, v.model_dump()] for t, v in final]))
        except Exception as exc:
            logger.warning("Trial cache write failed: %s", exc)

    return final
