"""Tests for explainable score breakdown."""

from __future__ import annotations

import pytest

from src.matching.eligibility_reasoner import CriterionVerdict
from src.ranking.scorer import (
    explain_score_breakdown,
    score_batch_breakdown,
    score_trial,
    score_trial_breakdown,
)


def _v(verdict: str, confidence: float = 1.0) -> CriterionVerdict:
    return CriterionVerdict(
        verdict=verdict,
        confidence=confidence,
        reasoning="test reasoning",
    )


def test_score_breakdown_matches_score_trial():
    verdicts = [
        ("inclusion", _v("MET")),
        ("inclusion", _v("NEI", 0.5)),
        ("exclusion", _v("MET")),
    ]
    metadata = {"phase": "PHASE2", "status": "RECRUITING"}

    breakdown = score_trial_breakdown(verdicts, metadata)

    assert breakdown["final_score"] == pytest.approx(score_trial(verdicts, metadata))


def test_score_breakdown_contains_expected_keys():
    verdicts = [
        ("inclusion", _v("MET")),
        ("exclusion", _v("MET")),
    ]
    metadata = {"phase": "PHASE3", "status": "RECRUITING"}

    breakdown = score_trial_breakdown(verdicts, metadata)

    expected = {
        "inclusion_met",
        "total_inclusion",
        "inclusion_met_ratio",
        "exclusion_penalty",
        "nei_count",
        "total_criteria",
        "nei_ratio",
        "phase_bonus",
        "recruiting_bonus",
        "weighted_inclusion",
        "weighted_exclusion",
        "weighted_phase",
        "weighted_recruiting",
        "weighted_nei_penalty",
        "raw_score",
        "final_score",
    }

    assert expected.issubset(set(breakdown.keys()))


def test_exclusion_violation_drives_score_to_zero():
    verdicts = [
        ("inclusion", _v("MET")),
        ("exclusion", _v("NOT_MET")),
    ]
    metadata = {"phase": "PHASE3", "status": "RECRUITING"}

    breakdown = score_trial_breakdown(verdicts, metadata)

    assert breakdown["exclusion_penalty"] == 1.0
    assert breakdown["final_score"] == 0.0


def test_nei_penalty_is_visible_in_breakdown():
    verdicts = [
        ("inclusion", _v("MET")),
        ("inclusion", _v("NEI")),
        ("exclusion", _v("MET")),
    ]
    metadata = {"phase": "PHASE2", "status": "RECRUITING"}

    breakdown = score_trial_breakdown(verdicts, metadata)

    assert breakdown["nei_count"] == 1
    assert breakdown["nei_ratio"] > 0
    assert breakdown["weighted_nei_penalty"] < 0


def test_explanation_is_deterministic_text():
    verdicts = [
        ("inclusion", _v("MET")),
        ("inclusion", _v("MET")),
        ("exclusion", _v("MET")),
    ]
    metadata = {"phase": "PHASE3", "status": "RECRUITING"}

    breakdown = score_trial_breakdown(verdicts, metadata)
    explanation = explain_score_breakdown(breakdown)

    assert isinstance(explanation, str)
    assert "Ranked based on" in explanation
    assert "inclusion match" in explanation


def test_score_batch_breakdown_returns_one_breakdown_per_trial():
    verdicts_by_nct = {
        "NCT001": [("inclusion", _v("MET"))],
        "NCT002": [("exclusion", _v("NOT_MET"))],
    }
    metadata_by_nct = {
        "NCT001": {"phase": "PHASE3", "status": "RECRUITING"},
        "NCT002": {"phase": "PHASE1", "status": "COMPLETED"},
    }

    result = score_batch_breakdown(verdicts_by_nct, metadata_by_nct)

    assert set(result.keys()) == {"NCT001", "NCT002"}
    assert "final_score" in result["NCT001"]
    assert "final_score" in result["NCT002"]