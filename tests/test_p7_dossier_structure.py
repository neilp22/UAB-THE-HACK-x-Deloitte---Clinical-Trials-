"""P7 tests for structured dossier completeness."""

from __future__ import annotations

from src.matching.eligibility_reasoner import CriterionVerdict
from src.output.dossier_generator import Dossier, generate_dossier
from src.parsing.patient_normalizer import PatientProfile


_PROFILE = PatientProfile(age=45, gender="male", conditions=["astrocytoma"])

_META = {
    "nct_id": "NCT12345678",
    "title": "Phase III Astrocytoma Trial",
    "phases": ["PHASE3"],
    "status": "RECRUITING",
    "brief_summary": "A randomised controlled trial for astrocytoma patients.",
}


def _v(
    verdict: str,
    confidence: float = 1.0,
    reasoning: str = "test reasoning",
) -> CriterionVerdict:
    return CriterionVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
    )


def _dossier(verdicts, nei_questions=None, score_explanation=None) -> Dossier:
    return generate_dossier(
        profile=_PROFILE,
        verdicts=verdicts,
        trial_metadata=_META,
        nei_questions=nei_questions or {},
        score_explanation=score_explanation,
    )


def test_dossier_contains_p7_minimum_structure():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "ECOG <= 2", _v("NEI", 0.5, "Unknown ECOG")),
        ("exclusion", "No active infection", _v("MET")),
    ]
    nei_questions = {"ECOG <= 2": "What is the patient's current ECOG status?"}

    d = _dossier(verdicts, nei_questions=nei_questions)

    assert d.trial_summary
    assert d.eligibility_overview.total_criteria == 3
    assert len(d.eligibility_table) == 3
    assert len(d.missing_information_questions) == 1
    assert d.structured_flags is not None
    assert d.score_explanation
    assert d.final_recommendation == d.overall_recommendation


def test_red_flag_exclusion_risk_is_structured():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("exclusion", "No prior brain surgery", _v("NOT_MET")),
    ]

    d = _dossier(verdicts)

    codes = {flag.code for flag in d.structured_flags}
    assert "RED_FLAG_EXCLUSION_RISK" in codes
    assert any("EXCLUSION VIOLATED" in flag for flag in d.attention_flags)
    assert d.final_recommendation == "NOT_ELIGIBLE"


def test_amber_flag_critical_nei_for_unclear_exclusion():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("exclusion", "No active infection", _v("NEI", 0.5, "Unknown")),
    ]
    nei_questions = {
        "No active infection": "Does the patient currently have an active infection?"
    }

    d = _dossier(verdicts, nei_questions=nei_questions)

    codes = {flag.code for flag in d.structured_flags}
    assert "AMBER_FLAG_CRITICAL_NEI" in codes
    assert d.missing_information_questions[0].question.endswith("?")


def test_blue_flag_good_match_is_structured_but_legacy_attention_flags_stay_empty():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "ECOG <= 2", _v("MET")),
        ("inclusion", "Signed consent", _v("MET")),
        ("exclusion", "No allergies", _v("MET")),
    ]

    d = _dossier(verdicts)

    codes = {flag.code for flag in d.structured_flags}
    assert "BLUE_FLAG_GOOD_MATCH" in codes
    assert d.attention_flags == []


def test_grey_flag_low_evidence_is_structured():
    verdicts = [
        ("inclusion", "Criterion 1", _v("NOT_MET")),
        ("inclusion", "Criterion 2", _v("NOT_MET")),
        ("inclusion", "Criterion 3", _v("NEI", 0.4, "Unknown")),
        ("exclusion", "No allergies", _v("MET")),
    ]

    d = _dossier(verdicts)

    codes = {flag.code for flag in d.structured_flags}
    assert "GREY_FLAG_LOW_EVIDENCE" in codes
    assert any("LOW INCLUSION MATCH" in flag for flag in d.attention_flags)


def test_score_explanation_can_be_injected_from_ranking():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("exclusion", "No allergies", _v("MET")),
    ]
    explanation = "Ranked based on strong inclusion match and no exclusion violation."

    d = _dossier(verdicts, score_explanation=explanation)

    assert d.score_explanation == explanation


def test_missing_information_questions_are_extracted_from_nei_only():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "Prior platinum therapy", _v("NEI", 0.5, "Unknown")),
        ("exclusion", "No allergies", _v("MET")),
    ]
    nei_questions = {
        "Prior platinum therapy": "Has the patient received prior platinum therapy?"
    }

    d = _dossier(verdicts, nei_questions=nei_questions)

    assert len(d.missing_information_questions) == 1
    assert d.missing_information_questions[0].criterion == "Prior platinum therapy"