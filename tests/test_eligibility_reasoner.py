"""Tests for src/matching/eligibility_reasoner.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _profile(**kwargs):
    from src.parsing.patient_normalizer import PatientProfile
    defaults = dict(age=None, gender="unknown", conditions=[], medications=[],
                    ecog_score=None, lab_values={}, prior_treatments=[], relevant_history=[])
    defaults.update(kwargs)
    return PatientProfile(**defaults)


def _criterion(text, category, ctype="exclusion", operator=None, threshold=None,
               unit=None, deterministic=True):
    from src.parsing.criteria_parser import Criterion
    return Criterion(
        text=text, type=ctype, category=category, deterministic=deterministic,
        numeric_operator=operator, numeric_threshold=threshold, numeric_unit=unit,
    )


def _verdict(v, c=1.0, r=""):
    from src.matching.eligibility_reasoner import CriterionVerdict
    return CriterionVerdict(verdict=v, confidence=c, reasoning=r)


# ---------------------------------------------------------------------------
# Age — exclusion
# ---------------------------------------------------------------------------

class TestDeterministicAgeExclusion:
    def test_not_met_when_age_violates_exclusion(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=75)
        c = _criterion("Age > 70", "age", "exclusion", ">", 70.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"
        assert v.confidence == 1.0

    def test_met_when_age_does_not_violate_exclusion(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=60)
        c = _criterion("Age > 70", "age", "exclusion", ">", 70.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_nei_when_age_none(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=None)
        c = _criterion("Age > 70", "age", "exclusion", ">", 70.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_nei_when_threshold_missing(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=80)
        c = _criterion("Elderly patients", "age", "exclusion", None, None)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"


# ---------------------------------------------------------------------------
# Age — inclusion
# ---------------------------------------------------------------------------

class TestDeterministicAgeInclusion:
    def test_met_when_inclusion_age_satisfied(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=25)
        c = _criterion("Age >= 18", "age", "inclusion", ">=", 18.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_not_met_when_inclusion_age_not_satisfied(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=15)
        c = _criterion("Age >= 18", "age", "inclusion", ">=", 18.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"

    def test_exact_boundary_inclusive(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=18)
        c = _criterion("Age >= 18", "age", "inclusion", ">=", 18.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"


# ---------------------------------------------------------------------------
# Gender
# ---------------------------------------------------------------------------

class TestDeterministicGender:
    def test_not_met_female_exclusion_for_female_patient(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="female")
        c = _criterion("Female patients excluded", "gender", "exclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"

    def test_met_female_exclusion_for_male_patient(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="male")
        c = _criterion("Female patients excluded", "gender", "exclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_met_female_inclusion_for_female_patient(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="female")
        c = _criterion("Must be female", "gender", "inclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_not_met_female_inclusion_for_male_patient(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="male")
        c = _criterion("Must be female", "gender", "inclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"

    def test_nei_unknown_gender(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="unknown")
        c = _criterion("Male patients", "gender", "exclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_nei_no_gender_keyword(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="female")
        c = _criterion("Patients of childbearing potential", "gender", "exclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_male_word_boundary_not_matched_in_female(self):
        """'male' within 'female' must NOT trigger the male check."""
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(gender="female")
        c = _criterion("Female patients only", "gender", "exclusion")
        v = evaluate_criterion(profile, c, use_cache=False)
        # Criterion targets female → female patient gets NOT_MET (exclusion violated)
        assert v.verdict == "NOT_MET"


# ---------------------------------------------------------------------------
# ECOG
# ---------------------------------------------------------------------------

class TestDeterministicEcog:
    def test_not_met_when_ecog_violates_exclusion(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(ecog_score=3)
        c = _criterion("ECOG PS > 2", "lab_value", "exclusion", ">", 2.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"

    def test_met_when_ecog_within_threshold(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(ecog_score=1)
        c = _criterion("ECOG PS > 2", "lab_value", "exclusion", ">", 2.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_nei_when_ecog_none(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(ecog_score=None)
        c = _criterion("ECOG > 2", "lab_value", "exclusion", ">", 2.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_performance_status_text_detected(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(ecog_score=4)
        c = _criterion("Performance status > 2", "lab_value", "exclusion", ">", 2.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"


# ---------------------------------------------------------------------------
# Lab values
# ---------------------------------------------------------------------------

class TestDeterministicLab:
    def test_not_met_high_creatinine_exclusion(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(lab_values={"creatinine": "2.5 mg/dL"})
        c = _criterion("Creatinine > 2.0 mg/dL", "lab_value", "exclusion", ">", 2.0, "mg/dL")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"

    def test_met_normal_creatinine_exclusion(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(lab_values={"creatinine": "1.0 mg/dL"})
        c = _criterion("Creatinine > 2.0 mg/dL", "lab_value", "exclusion", ">", 2.0, "mg/dL")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_met_hemoglobin_inclusion(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(lab_values={"hemoglobin": "12.0 g/dL"})
        c = _criterion("Hemoglobin >= 10.0 g/dL", "lab_value", "inclusion", ">=", 10.0, "g/dL")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "MET"

    def test_nei_lab_not_in_profile(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(lab_values={})
        c = _criterion("Creatinine > 2.0", "lab_value", "exclusion", ">", 2.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_nei_unit_mismatch(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(lab_values={"creatinine": "1.2 mg/dL"})
        c = _criterion("Creatinine > 100 umol/L", "lab_value", "exclusion", ">", 100.0, "umol/L")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_nei_short_key_skipped(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(lab_values={"ast": "50 U/L"})
        c = _criterion("Last AST value > 40 U/L", "lab_value", "exclusion", ">", 40.0, "U/L")
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"


# ---------------------------------------------------------------------------
# LLM path (mocked)
# ---------------------------------------------------------------------------

class TestLLMPath:
    def test_returns_llm_verdict(self):
        from src.matching.eligibility_reasoner import evaluate_criterion, CriterionVerdict
        profile = _profile(conditions=["non-small cell lung cancer"])
        c = _criterion("Prior platinum-based chemotherapy", "prior_treatment",
                       "exclusion", deterministic=False)
        fake = CriterionVerdict(verdict="NOT_MET", confidence=0.9,
                                reasoning="Patient has received cisplatin")
        with patch("src.matching.eligibility_reasoner.complete_structured", return_value=fake):
            v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NOT_MET"
        assert v.confidence == 0.9

    def test_nei_on_double_llm_failure(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(conditions=["cancer"])
        c = _criterion("Active CNS metastases", "diagnosis", "exclusion", deterministic=False)
        with patch("src.matching.eligibility_reasoner.complete_structured",
                   side_effect=RuntimeError("API down")):
            v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"
        assert v.confidence == 0.0

    def test_llm_retried_once_on_first_failure(self):
        from src.matching.eligibility_reasoner import evaluate_criterion, CriterionVerdict
        profile = _profile()
        c = _criterion("Active infection", "diagnosis", "exclusion", deterministic=False)
        call_count = [0]
        fake = CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="No info")

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient error")
            return fake

        with patch("src.matching.eligibility_reasoner.complete_structured",
                   side_effect=side_effect):
            v = evaluate_criterion(profile, c, use_cache=False)

        assert call_count[0] == 2
        assert v.verdict == "NEI"


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_hit_skips_llm(self):
        import diskcache
        from src.matching.eligibility_reasoner import evaluate_criterion, CriterionVerdict

        profile = _profile(conditions=["cancer"])
        c = _criterion("Prior immunotherapy", "prior_treatment", "exclusion", deterministic=False)
        fake = CriterionVerdict(verdict="MET", confidence=0.8, reasoning="No prior immunotherapy noted")
        call_count = [0]

        def counted(*a, **kw):
            call_count[0] += 1
            return fake

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = diskcache.Cache(tmpdir)
            with patch("src.matching.eligibility_reasoner.complete_structured",
                       side_effect=counted):
                evaluate_criterion(profile, c, use_cache=True, cache=cache)
                evaluate_criterion(profile, c, use_cache=True, cache=cache)

        assert call_count[0] == 1

    def test_cache_stores_json_string(self):
        import diskcache, hashlib
        from src.matching.eligibility_reasoner import (
            evaluate_criterion, CriterionVerdict, _cache_key, _profile_json,
        )

        profile = _profile(conditions=["diabetes"])
        c = _criterion("Active autoimmune disease", "diagnosis", "exclusion", deterministic=False)
        fake = CriterionVerdict(verdict="NEI", confidence=0.5, reasoning="Unclear")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = diskcache.Cache(tmpdir)
            with patch("src.matching.eligibility_reasoner.complete_structured", return_value=fake):
                evaluate_criterion(profile, c, use_cache=True, cache=cache)

            key = _cache_key(c, profile)
            stored = cache.get(key)
            assert isinstance(stored, str)
            assert stored.startswith("{")

    def test_deterministic_path_not_cached(self):
        import diskcache
        from src.matching.eligibility_reasoner import evaluate_criterion, _cache_key

        profile = _profile(age=65)
        c = _criterion("Age > 70", "age", "exclusion", ">", 70.0, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = diskcache.Cache(tmpdir)
            evaluate_criterion(profile, c, use_cache=True, cache=cache)
            key = _cache_key(c, profile)
            assert cache.get(key) is None


# ---------------------------------------------------------------------------
# Non-deterministic category marked deterministic
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_category_marked_deterministic_returns_nei(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        from src.parsing.criteria_parser import Criterion
        profile = _profile(age=45)
        c = Criterion(text="Prior platinum therapy", type="exclusion",
                      category="prior_treatment", deterministic=True)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert v.verdict == "NEI"

    def test_confidence_in_valid_range(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=75)
        c = _criterion("Age > 70", "age", "exclusion", ">", 70.0)
        v = evaluate_criterion(profile, c, use_cache=False)
        assert 0.0 <= v.confidence <= 1.0

    def test_reasoning_is_string(self):
        from src.matching.eligibility_reasoner import evaluate_criterion
        profile = _profile(age=30, gender="female", ecog_score=1,
                           lab_values={"creatinine": "0.9 mg/dL"})
        criteria = [
            _criterion("Age > 70", "age", "exclusion", ">", 70.0),
            _criterion("Female patients", "gender", "exclusion"),
            _criterion("ECOG > 2", "lab_value", "exclusion", ">", 2.0),
            _criterion("Creatinine > 2.0 mg/dL", "lab_value", "exclusion", ">", 2.0, "mg/dL"),
        ]
        for crit in criteria:
            v = evaluate_criterion(profile, crit, use_cache=False)
            assert isinstance(v.reasoning, str) and v.reasoning
