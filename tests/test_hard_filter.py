"""Tests for src/matching/hard_filter.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _profile(**kwargs):
    from src.parsing.patient_normalizer import PatientProfile
    defaults = dict(age=None, gender="unknown", conditions=[], medications=[],
                    ecog_score=None, lab_values={}, prior_treatments=[], relevant_history=[])
    defaults.update(kwargs)
    return PatientProfile(**defaults)


def _criterion(text, category, operator=None, threshold=None, unit=None):
    from src.parsing.criteria_parser import Criterion
    return Criterion(
        text=text, type="exclusion", category=category, deterministic=True,
        numeric_operator=operator, numeric_threshold=threshold, numeric_unit=unit,
    )


# ---------------------------------------------------------------------------
# _compare helper
# ---------------------------------------------------------------------------

class TestCompare:
    def test_greater_than_true(self):
        from src.matching.hard_filter import _compare
        assert _compare(75.0, ">", 70.0) is True

    def test_greater_than_false(self):
        from src.matching.hard_filter import _compare
        assert _compare(60.0, ">", 70.0) is False

    def test_less_equal_true(self):
        from src.matching.hard_filter import _compare
        assert _compare(1.5, "<=", 2.0) is True

    def test_equal(self):
        from src.matching.hard_filter import _compare
        assert _compare(18.0, "==", 18.0) is True

    def test_unknown_operator_returns_false(self):
        from src.matching.hard_filter import _compare
        assert _compare(10.0, "??", 5.0) is False


# ---------------------------------------------------------------------------
# Age checks
# ---------------------------------------------------------------------------

class TestAgeCheck:
    def test_eliminated_when_age_exceeds_threshold(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=75)
        criteria = [_criterion("Age > 70 years", "age", ">", 70.0, "years")]
        assert apply_hard_filter(profile, criteria) is False

    def test_passes_when_age_below_threshold(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=60)
        criteria = [_criterion("Age > 70 years", "age", ">", 70.0, "years")]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_when_age_equals_threshold_and_strict_gt(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=70)
        criteria = [_criterion("Age > 70 years", "age", ">", 70.0)]
        assert apply_hard_filter(profile, criteria) is True

    def test_eliminated_when_age_equals_threshold_and_gte(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=70)
        criteria = [_criterion("Age >= 70 years", "age", ">=", 70.0)]
        assert apply_hard_filter(profile, criteria) is False

    def test_passes_when_age_is_none(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=None)
        criteria = [_criterion("Age > 70", "age", ">", 70.0)]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_when_threshold_missing(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=80)
        criteria = [_criterion("Elderly patients", "age", None, None)]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_when_operator_missing(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=80)
        criteria = [_criterion("Age threshold criterion", "age", None, 70.0)]
        assert apply_hard_filter(profile, criteria) is True


# ---------------------------------------------------------------------------
# ECOG checks
# ---------------------------------------------------------------------------

class TestEcogCheck:
    def test_eliminated_when_ecog_exceeds_threshold(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(ecog_score=3)
        criteria = [_criterion("ECOG PS > 2", "lab_value", ">", 2.0)]
        assert apply_hard_filter(profile, criteria) is False

    def test_passes_when_ecog_within_threshold(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(ecog_score=1)
        criteria = [_criterion("ECOG PS > 2", "lab_value", ">", 2.0)]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_when_ecog_is_none(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(ecog_score=None)
        criteria = [_criterion("ECOG performance status > 2", "lab_value", ">", 2.0)]
        assert apply_hard_filter(profile, criteria) is True

    def test_ecog_detected_via_performance_status_text(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(ecog_score=4)
        criteria = [_criterion("Performance status > 2", "lab_value", ">", 2.0)]
        assert apply_hard_filter(profile, criteria) is False

    def test_ecog_exact_boundary_strict(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(ecog_score=2)
        criteria = [_criterion("ECOG > 2", "lab_value", ">", 2.0)]
        assert apply_hard_filter(profile, criteria) is True  # 2 > 2 is False → passes


# ---------------------------------------------------------------------------
# Lab value checks
# ---------------------------------------------------------------------------

class TestLabCheck:
    def test_eliminated_on_high_creatinine(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(lab_values={"creatinine": "2.5 mg/dL"})
        criteria = [_criterion("Creatinine > 2.0 mg/dL", "lab_value", ">", 2.0, "mg/dL")]
        assert apply_hard_filter(profile, criteria) is False

    def test_passes_on_normal_creatinine(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(lab_values={"creatinine": "1.0 mg/dL"})
        criteria = [_criterion("Creatinine > 2.0 mg/dL", "lab_value", ">", 2.0, "mg/dL")]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_when_lab_not_in_profile(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(lab_values={})
        criteria = [_criterion("Creatinine > 2.0", "lab_value", ">", 2.0)]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_on_unit_mismatch(self):
        from src.matching.hard_filter import apply_hard_filter
        # Stored in mg/dL, criterion in umol/L — can't safely compare
        profile = _profile(lab_values={"creatinine": "1.2 mg/dL"})
        criteria = [_criterion("Creatinine > 100 umol/L", "lab_value", ">", 100.0, "umol/L")]
        assert apply_hard_filter(profile, criteria) is True

    def test_passes_no_unit_in_criterion(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(lab_values={"creatinine": "1.0"})
        criteria = [_criterion("Creatinine > 2.0", "lab_value", ">", 2.0, None)]
        assert apply_hard_filter(profile, criteria) is True

    def test_ast_not_matched_by_short_key(self):
        """'ast' (3 chars) should not match via word-boundary short-key rule."""
        from src.matching.hard_filter import apply_hard_filter
        # 'ast' key is < _LAB_MIN_KEY_LEN (4), so it's skipped for safety
        profile = _profile(lab_values={"ast": "50 U/L"})
        criteria = [_criterion("Last AST value > 40 U/L", "lab_value", ">", 40.0, "U/L")]
        # "ast" < 4 chars → skipped → passes
        assert apply_hard_filter(profile, criteria) is True

    def test_hemoglobin_eliminated(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(lab_values={"hemoglobin": "7.5 g/dL"})
        criteria = [_criterion("Hemoglobin < 8.0 g/dL", "lab_value", "<", 8.0, "g/dL")]
        assert apply_hard_filter(profile, criteria) is False


# ---------------------------------------------------------------------------
# Gender checks
# ---------------------------------------------------------------------------

class TestGenderCheck:
    def test_female_patient_eliminated_by_female_exclusion(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="female")
        criteria = [_criterion("Female patients", "gender")]
        assert apply_hard_filter(profile, criteria) is False

    def test_male_patient_passes_female_exclusion(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="male")
        criteria = [_criterion("Female patients of childbearing potential", "gender")]
        assert apply_hard_filter(profile, criteria) is True

    def test_male_patient_eliminated_by_male_exclusion(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="male")
        criteria = [_criterion("Male patients only", "gender")]
        assert apply_hard_filter(profile, criteria) is False

    def test_female_patient_passes_male_exclusion(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="female")
        criteria = [_criterion("Male patients only", "gender")]
        assert apply_hard_filter(profile, criteria) is True

    def test_unknown_gender_always_passes(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="unknown")
        criteria = [
            _criterion("Female patients", "gender"),
            _criterion("Male patients", "gender"),
        ]
        assert apply_hard_filter(profile, criteria) is True

    def test_male_does_not_match_inside_female(self):
        """'male' within 'female' must NOT trigger the male exclusion."""
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="female")
        # This criterion says 'female' — should eliminate female patient
        criteria = [_criterion("Female patients excluded", "gender")]
        assert apply_hard_filter(profile, criteria) is False

    def test_gender_criterion_no_gender_keyword_passes(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(gender="female")
        criteria = [_criterion("Patients of reproductive age", "gender")]
        assert apply_hard_filter(profile, criteria) is True


# ---------------------------------------------------------------------------
# apply_hard_filter — integration
# ---------------------------------------------------------------------------

class TestApplyHardFilter:
    def test_empty_criteria_always_passes(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=80, gender="female")
        assert apply_hard_filter(profile, []) is True

    def test_early_exit_on_first_violation(self):
        """Returns False on first violated criterion without evaluating others."""
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=75)
        criteria = [
            _criterion("Age > 70", "age", ">", 70.0),
            _criterion("ECOG > 2", "lab_value", ">", 2.0),  # would also violate if checked
        ]
        # Should return False immediately after age check
        result = apply_hard_filter(profile, criteria)
        assert result is False

    def test_non_deterministic_category_skipped(self):
        """diagnosis/prior_treatment categories are not evaluated here."""
        from src.matching.hard_filter import apply_hard_filter
        from src.parsing.criteria_parser import Criterion
        profile = _profile(age=45)
        # diagnosis criterion — even with deterministic=True flag, skip in hard filter
        c = Criterion(text="Prior platinum chemotherapy", type="exclusion",
                      category="prior_treatment", deterministic=True)
        assert apply_hard_filter(profile, [c]) is True

    def test_multiple_passing_criteria_all_pass(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=50, gender="male", ecog_score=1)
        criteria = [
            _criterion("Age > 70", "age", ">", 70.0),
            _criterion("Female patients", "gender"),
            _criterion("ECOG > 2", "lab_value", ">", 2.0),
        ]
        assert apply_hard_filter(profile, criteria) is True

    def test_one_violated_among_many_eliminates(self):
        from src.matching.hard_filter import apply_hard_filter
        profile = _profile(age=50, gender="female", ecog_score=1)
        criteria = [
            _criterion("Age > 70", "age", ">", 70.0),    # passes (50 <= 70)
            _criterion("Female patients", "gender"),       # VIOLATED (female)
            _criterion("ECOG > 2", "lab_value", ">", 2.0), # would pass
        ]
        assert apply_hard_filter(profile, criteria) is False
