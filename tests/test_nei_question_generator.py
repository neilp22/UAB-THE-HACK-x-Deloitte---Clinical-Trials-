"""Tests for NEI question generator."""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import diskcache
import pytest

from src.matching.eligibility_reasoner import CriterionVerdict
from src.output.nei_question_generator import generate_nei_question
from src.parsing.patient_normalizer import PatientProfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROFILE = PatientProfile(
    age=45,
    gender="male",
    conditions=["anaplastic astrocytoma"],
    lab_values={},
)

# Three real NEI verdicts drawn from the live eligibility reasoner cache
_NEI_CASES = [
    {
        "criterion_text": "No known allergy or hypersensitivity to foreign proteins or study drug components",
        "trial_title": "Phase II Study of Novel Immunotherapy",
        "reasoning": "No information provided about allergies to foreign proteins.",
    },
    {
        "criterion_text": "Measurable lesion ≥ 10 mm on contrast-enhanced MRI",
        "trial_title": "Brain Tumor Trial NCT00369590",
        "reasoning": "There is no information provided about the size of the lesion; "
                     "therefore, it cannot be determined if it is measurable.",
    },
    {
        "criterion_text": "Tumour volume must not exceed 50 ccm as measured by MRI",
        "trial_title": "Astrocytoma Surgery Trial",
        "reasoning": "Without specific volume measurements, it is unclear if the "
                     "lesion exceeds 50 ccm.",
    },
]


def _make_verdict(reasoning: str) -> CriterionVerdict:
    return CriterionVerdict(verdict="NEI", confidence=0.5, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_case(case: dict, cache: diskcache.Cache) -> str:
    verdict = _make_verdict(case["reasoning"])
    return generate_nei_question(
        verdict=verdict,
        criterion_text=case["criterion_text"],
        trial_title=case["trial_title"],
        profile=_PROFILE,
        use_cache=True,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_is_nonempty_string_for_three_real_nei_verdicts():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        for case in _NEI_CASES:
            result = _run_case(case, cache)
            assert isinstance(result, str)
            assert len(result) > 10, f"Question too short: {result!r}"


def test_output_ends_with_question_mark():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        for case in _NEI_CASES:
            result = _run_case(case, cache)
            assert result.endswith("?"), f"Does not end with '?': {result!r}"


def test_output_is_not_generic():
    """Output must not contain the banned generic phrase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        for case in _NEI_CASES:
            result = _run_case(case, cache)
            assert "Does the patient meet" not in result, (
                f"Generic fallback phrase found in: {result!r}"
            )


def test_cache_hit_returns_same_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        case = _NEI_CASES[0]
        verdict = _make_verdict(case["reasoning"])

        first = generate_nei_question(
            verdict=verdict, criterion_text=case["criterion_text"],
            trial_title=case["trial_title"], profile=_PROFILE,
            use_cache=True, cache=cache,
        )
        # Second call must return same string without hitting LLM
        with patch("src.output.nei_question_generator.complete") as mock_llm:
            second = generate_nei_question(
                verdict=verdict, criterion_text=case["criterion_text"],
                trial_title=case["trial_title"], profile=_PROFILE,
                use_cache=True, cache=cache,
            )
            mock_llm.assert_not_called()

        assert first == second


def test_cache_miss_calls_llm():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        case = _NEI_CASES[0]
        verdict = _make_verdict(case["reasoning"])

        with patch("src.output.nei_question_generator.complete",
                   return_value="Has the patient received prior chemotherapy?") as mock_llm:
            result = generate_nei_question(
                verdict=verdict, criterion_text=case["criterion_text"],
                trial_title=case["trial_title"], profile=_PROFILE,
                use_cache=True, cache=cache,
            )
            mock_llm.assert_called_once()
        assert result == "Has the patient received prior chemotherapy?"


def test_llm_failure_returns_fallback_with_question_mark():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        case = _NEI_CASES[0]
        verdict = _make_verdict(case["reasoning"])

        with patch("src.output.nei_question_generator.complete",
                   side_effect=Exception("API error")):
            result = generate_nei_question(
                verdict=verdict, criterion_text=case["criterion_text"],
                trial_title=case["trial_title"], profile=_PROFILE,
                use_cache=True, cache=cache,
            )
        assert isinstance(result, str)
        assert result.endswith("?")


def test_question_without_trailing_question_mark_gets_one_appended():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = diskcache.Cache(tmpdir)
        case = _NEI_CASES[0]
        verdict = _make_verdict(case["reasoning"])

        with patch("src.output.nei_question_generator.complete",
                   return_value="Has the patient had prior platinum chemotherapy"):
            result = generate_nei_question(
                verdict=verdict, criterion_text=case["criterion_text"],
                trial_title=case["trial_title"], profile=_PROFILE,
                use_cache=True, cache=cache,
            )
        assert result.endswith("?")
