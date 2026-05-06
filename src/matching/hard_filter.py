"""
Hard exclusion filter — deterministic, zero LLM.

Evaluates pre-parsed exclusion criteria that are flagged deterministic=True
against structured fields in PatientProfile.

Rules:
- One confirmed violation → return False immediately (log reason)
- Patient field is None / lab key absent → skip criterion (NEI, pass through)
- Missing threshold or operator → skip criterion (pass through)
- Unit mismatch → skip criterion (pass through; unit conversion out of scope)
- Non-age/gender/lab_value categories → skip (left to LLM eligibility reasoner)

Input:  PatientProfile  +  list[Criterion]  (caller must pre-filter to
        type="exclusion" AND deterministic=True)
Output: bool — True = survives, False = eliminated
"""
from __future__ import annotations

import logging
import re
from operator import eq, ge, gt, le, lt

from src.parsing.criteria_parser import Criterion
from src.parsing.patient_normalizer import PatientProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_OPS: dict[str, object] = {
    ">=": ge,
    "<=": le,
    ">": gt,
    "<": lt,
    "==": eq,
}

# Minimum lab key length for safe word-boundary matching (avoids "ast" in "last")
_LAB_MIN_KEY_LEN = 4


def _compare(value: float, operator: str, threshold: float) -> bool:
    """Evaluate `value OP threshold`. Returns False for unknown operators."""
    fn = _OPS.get(operator)
    return bool(fn(value, threshold)) if fn is not None else False


def _parse_first_float(text: str) -> float | None:
    """Extract the first numeric value from a string like '2.5 mg/dL'."""
    m = re.search(r"[\d]+(?:[.,][\d]+)?", text)
    if m:
        try:
            return float(m.group().replace(",", "."))
        except ValueError:
            pass
    return None


def _units_compatible(criterion_unit: str | None, value_str: str) -> bool:
    """
    Returns True if units are compatible for comparison.

    - No criterion unit → compatible (assume same units)
    - No stored unit → compatible (stored values often omit units for simple cases)
    - Both present → compare case-insensitively
    """
    if criterion_unit is None:
        return True
    m = re.search(r"[\d]+(?:[.,][\d]+)?\s*(.*)", value_str.strip())
    if not m:
        return True
    stored_unit = m.group(1).strip()
    if not stored_unit:
        return True
    return stored_unit.lower() == criterion_unit.lower()


# ---------------------------------------------------------------------------
# Per-category checkers (return True = criterion violated = eliminate)
# ---------------------------------------------------------------------------

def _check_age(profile: PatientProfile, criterion: Criterion) -> bool:
    if profile.age is None:
        return False
    if criterion.numeric_threshold is None or not criterion.numeric_operator:
        return False
    return _compare(float(profile.age), criterion.numeric_operator, criterion.numeric_threshold)


def _check_ecog(profile: PatientProfile, criterion: Criterion) -> bool:
    if profile.ecog_score is None:
        return False
    if criterion.numeric_threshold is None or not criterion.numeric_operator:
        return False
    return _compare(float(profile.ecog_score), criterion.numeric_operator, criterion.numeric_threshold)


def _check_lab(profile: PatientProfile, criterion: Criterion) -> bool:
    if not profile.lab_values:
        return False
    if criterion.numeric_threshold is None or not criterion.numeric_operator:
        return False

    crit_lower = criterion.text.lower()

    # Find matching lab key using word-boundary search (safe against "ast"/"last" clash)
    matched_key: str | None = None
    for key in profile.lab_values:
        if len(key) < _LAB_MIN_KEY_LEN:
            continue
        if re.search(r"\b" + re.escape(key) + r"\b", crit_lower):
            matched_key = key
            break

    if matched_key is None:
        return False

    value_str = profile.lab_values[matched_key]
    patient_val = _parse_first_float(value_str)
    if patient_val is None:
        return False

    if not _units_compatible(criterion.numeric_unit, value_str):
        return False

    return _compare(patient_val, criterion.numeric_operator, criterion.numeric_threshold)


# \b ensures "male" does not match inside "female"
_FEMALE_RE = re.compile(r"\b(?:female|women|woman)\b", re.IGNORECASE)
_MALE_RE = re.compile(r"\bmale\b", re.IGNORECASE)


def _check_gender(profile: PatientProfile, criterion: Criterion) -> bool:
    if profile.gender == "unknown":
        return False
    text = criterion.text
    if _FEMALE_RE.search(text):
        return profile.gender == "female"
    if _MALE_RE.search(text):
        return profile.gender == "male"
    return False  # can't identify target gender → skip


def _is_ecog(criterion: Criterion) -> bool:
    low = criterion.text.lower()
    return "ecog" in low or "performance status" in low or " ps " in low


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_hard_filter(
    profile: PatientProfile,
    exclusion_criteria: list[Criterion],
) -> bool:
    """
    Evaluate deterministic exclusion criteria against a patient profile.

    Caller is responsible for pre-filtering to type="exclusion" + deterministic=True.

    Returns True  (survives) unless a criterion is confirmed violated.
    Returns False (eliminated) on the first confirmed violation.
    Any doubt (missing data, unknown operator, unit mismatch) → skip → pass.
    """
    for criterion in exclusion_criteria:
        violated = False

        if criterion.category == "age":
            violated = _check_age(profile, criterion)
        elif criterion.category == "gender":
            violated = _check_gender(profile, criterion)
        elif criterion.category == "lab_value":
            if _is_ecog(criterion):
                violated = _check_ecog(profile, criterion)
            else:
                violated = _check_lab(profile, criterion)
        # diagnosis / prior_treatment / other → not evaluable deterministically → skip

        if violated:
            logger.info(
                "Hard filter ELIMINATED | category=%s | %s",
                criterion.category,
                criterion.text[:120],
            )
            return False

    return True
