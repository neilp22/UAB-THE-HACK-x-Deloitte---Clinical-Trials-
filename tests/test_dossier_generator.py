"""Tests for dossier generator."""
from __future__ import annotations

import pytest

from src.matching.eligibility_reasoner import CriterionVerdict
from src.output.dossier_generator import Dossier, generate_dossier
from src.parsing.patient_normalizer import PatientProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILE = PatientProfile(age=45, gender="male", conditions=["astrocytoma"])

_META = {
    "nct_id": "NCT12345678",
    "title": "Phase III Astrocytoma Trial",
    "phases": ["PHASE3"],
    "status": "RECRUITING",
    "brief_summary": "A randomised controlled trial for astrocytoma patients.",
}


def _v(verdict: str, confidence: float = 1.0, reasoning: str = "test") -> CriterionVerdict:
    return CriterionVerdict(verdict=verdict, confidence=confidence, reasoning=reasoning)


def _dossier(verdicts, nei_questions=None, meta=None) -> Dossier:
    return generate_dossier(
        profile=_PROFILE,
        verdicts=verdicts,
        trial_metadata=meta or _META,
        nei_questions=nei_questions or {},
    )


# ---------------------------------------------------------------------------
# overall_recommendation
# ---------------------------------------------------------------------------

def test_not_eligible_when_exclusion_violated():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("exclusion", "No prior brain surgery", _v("NOT_MET")),
    ]
    d = _dossier(verdicts)
    assert d.overall_recommendation == "NOT_ELIGIBLE"


def test_eligible_when_inclusion_met_no_violations_no_nei():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "ECOG <= 2", _v("MET")),
        ("exclusion", "No prior allergies", _v("MET")),
    ]
    d = _dossier(verdicts)
    assert d.overall_recommendation == "ELIGIBLE"


def test_eligible_pending_clarification_when_inclusion_met_but_has_nei():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "Prior platinum therapy", _v("NEI", 0.5, "Unknown")),
        ("exclusion", "No CNS metastases", _v("MET")),
    ]
    d = _dossier(verdicts)
    assert d.overall_recommendation == "ELIGIBLE_PENDING_CLARIFICATION"


def test_eligible_pending_clarification_when_no_inclusion_met():
    verdicts = [
        ("inclusion", "Age >= 18", _v("NEI")),
        ("inclusion", "Prior therapy", _v("NEI")),
    ]
    d = _dossier(verdicts)
    assert d.overall_recommendation == "ELIGIBLE_PENDING_CLARIFICATION"


# ---------------------------------------------------------------------------
# attention_flags
# ---------------------------------------------------------------------------

def test_exclusion_violated_flag():
    verdicts = [("exclusion", "No prior brain surgery", _v("NOT_MET"))]
    d = _dossier(verdicts)
    assert any("EXCLUSION VIOLATED" in f for f in d.attention_flags)


def test_exclusion_unclear_flag():
    verdicts = [("exclusion", "No active infection", _v("NEI", 0.5, "Unknown"))]
    d = _dossier(verdicts)
    assert any("EXCLUSION UNCLEAR" in f for f in d.attention_flags)


def test_low_inclusion_match_flag():
    verdicts = [
        ("inclusion", f"Criterion {i}", _v("NOT_MET")) for i in range(10)
    ]
    d = _dossier(verdicts)
    assert any("LOW INCLUSION MATCH" in f for f in d.attention_flags)


def test_no_flags_when_all_met():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "ECOG <= 2", _v("MET")),
        ("inclusion", "Signed consent", _v("MET")),
        ("exclusion", "No allergies", _v("MET")),
    ]
    d = _dossier(verdicts)
    assert d.attention_flags == []


# ---------------------------------------------------------------------------
# confidence_score
# ---------------------------------------------------------------------------

def test_confidence_score_all_resolved():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "ECOG <= 2", _v("NOT_MET")),
    ]
    d = _dossier(verdicts)
    assert d.confidence_score == 1.0


def test_confidence_score_half_resolved():
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", "Prior therapy", _v("NEI")),
    ]
    d = _dossier(verdicts)
    assert d.confidence_score == 0.5


def test_confidence_score_empty_verdicts():
    d = _dossier([])
    assert d.confidence_score == 0.0


# ---------------------------------------------------------------------------
# clinical_question only for NEI
# ---------------------------------------------------------------------------

def test_clinical_question_only_on_nei_verdicts():
    nei_text = "Prior platinum therapy"
    verdicts = [
        ("inclusion", "Age >= 18", _v("MET")),
        ("inclusion", nei_text, _v("NEI", 0.5, "Unknown")),
        ("exclusion", "No allergies", _v("NOT_MET")),
    ]
    nei_questions = {nei_text: "Has the patient received prior platinum therapy?"}
    d = _dossier(verdicts, nei_questions=nei_questions)

    for dc in d.eligibility_table:
        if dc.verdict == "NEI":
            assert dc.clinical_question is not None
        else:
            assert dc.clinical_question is None


def test_nei_criterion_without_question_gets_none():
    verdicts = [("inclusion", "Prior therapy", _v("NEI", 0.5, "Unknown"))]
    d = _dossier(verdicts, nei_questions={})
    assert d.eligibility_table[0].clinical_question is None


# ---------------------------------------------------------------------------
# brief_summary truncation
# ---------------------------------------------------------------------------

def test_brief_summary_truncated_at_500_chars():
    long_summary = "A" * 600
    meta = {**_META, "brief_summary": long_summary}
    d = _dossier([("inclusion", "Age >= 18", _v("MET"))], meta=meta)
    assert len(d.brief_summary) == 500


def test_brief_summary_short_not_truncated():
    meta = {**_META, "brief_summary": "Short summary."}
    d = _dossier([("inclusion", "Age >= 18", _v("MET"))], meta=meta)
    assert d.brief_summary == "Short summary."


# ---------------------------------------------------------------------------
# metadata fields
# ---------------------------------------------------------------------------

def test_metadata_fields_populated():
    verdicts = [("inclusion", "Age >= 18", _v("MET"))]
    d = _dossier(verdicts)
    assert d.nct_id == "NCT12345678"
    assert d.trial_title == "Phase III Astrocytoma Trial"
    assert d.phase == "PHASE3"
    assert d.status == "RECRUITING"
