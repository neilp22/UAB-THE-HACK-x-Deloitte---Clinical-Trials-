"""
NEI question generator — produces one specific clinical question for each
criterion that could not be determined from the available patient profile.
"""
from __future__ import annotations

import hashlib
import logging

import diskcache

from src.config import CACHE_DIR
from src.llm_client import complete
from src.matching.eligibility_reasoner import CriterionVerdict
from src.parsing.patient_normalizer import PatientProfile

logger = logging.getLogger(__name__)

_cache = diskcache.Cache(str(CACHE_DIR / "nei_questions"))

_SYSTEM = (
    "You are a clinical research coordinator. "
    "Generate one specific, answerable clinical question."
)

_PROMPT = """\
Trial: {trial_title}
Criterion: {criterion_text}
Patient context: {patient_profile_json}

The patient record lacks information to evaluate this criterion.
Write one specific clinical question a doctor must answer \
to determine if this criterion is met.

Rules:
- One question only, no preamble, no explanation
- Must be specific and measurable
- Must reference the exact threshold if one exists
- End with a question mark
- Example good: 'Has the patient received ≥2 prior lines of platinum-based chemotherapy?'
- Example bad: 'Does the patient meet this criterion?'"""


def generate_nei_question(
    verdict: CriterionVerdict,
    criterion_text: str,
    trial_title: str,
    profile: PatientProfile,
    use_cache: bool = True,
    cache: diskcache.Cache | None = None,
) -> str:
    """
    Generate a targeted clinical question for a NEI verdict.

    Args:
        verdict: CriterionVerdict with verdict == "NEI".
        criterion_text: Original eligibility criterion text.
        trial_title: Trial title for context.
        profile: Normalised patient profile.
        use_cache: Whether to use diskcache.
        cache: Optional existing Cache instance.

    Returns:
        A single specific clinical question ending with '?'.
    """
    if use_cache and cache is None:
        cache = _cache

    key = hashlib.md5((criterion_text + trial_title).encode()).hexdigest()
    if use_cache and cache is not None:
        result = cache.get(key)
        if result is not None:
            return result

    prompt = _PROMPT.format(
        trial_title=trial_title,
        criterion_text=criterion_text,
        patient_profile_json=profile.model_dump_json(exclude_none=True),
    )
    try:
        question = complete(prompt, system=_SYSTEM).strip()
        # Ensure it ends with a question mark
        if question and not question.endswith("?"):
            question += "?"
    except Exception as exc:
        logger.warning("NEI question generation failed for '%s': %s", criterion_text[:60], exc)
        question = f"Does the patient meet the following criterion: {criterion_text[:100]}?"

    if use_cache and cache is not None:
        cache.set(key, question)

    return question
