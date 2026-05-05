"""Tests for src/ranking/scorer.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _v(verdict, ctype="inclusion", confidence=1.0):
    from src.matching.eligibility_reasoner import CriterionVerdict
    return (ctype, CriterionVerdict(verdict=verdict, confidence=confidence, reasoning="test"))


def _bare(verdict):
    from src.matching.eligibility_reasoner import CriterionVerdict
    return CriterionVerdict(verdict=verdict, confidence=1.0, reasoning="test")


META_P3_REC = {"phase": "PHASE3", "status": "RECRUITING"}
META_P2_REC = {"phase": "PHASE2", "status": "RECRUITING"}
META_P1 = {"phase": "PHASE1", "status": "COMPLETED"}
META_EMPTY = {}


# ---------------------------------------------------------------------------
# Core formula cases
# ---------------------------------------------------------------------------

class TestScoreTrial:
    def test_all_met_inclusion_no_exclusion_violations(self):
        """Best case: full inclusion match, no exclusion problems, Phase3 recruiting."""
        from src.ranking.scorer import score_trial
        verdicts = [
            _v("MET", "inclusion"),
            _v("MET", "inclusion"),
            _v("MET", "inclusion"),
        ]
        score = score_trial(verdicts, META_P3_REC)
        # inclusion_met_ratio=1.0, penalty=0, nei=0, phase=1.0, recruiting=1.0
        # 0.45*1.0 + 0.10*1.0 + 0.15*1.0 = 0.70
        assert abs(score - 0.70) < 1e-9

    def test_one_exclusion_violated_clamps_to_zero(self):
        """One NOT_MET exclusion forces raw score negative → clamped to 0."""
        from src.ranking.scorer import score_trial
        verdicts = [
            _v("MET", "inclusion"),
            _v("MET", "inclusion"),
            _v("NOT_MET", "exclusion"),
        ]
        score = score_trial(verdicts, META_P3_REC)
        # raw = 0.45 - 1.0 + 0.10 + 0.15 = -0.30 → clamped to 0
        assert score == 0.0

    def test_all_nei_returns_low_but_not_zero(self):
        """All NEI: no met inclusions, no exclusion violations, full NEI penalty."""
        from src.ranking.scorer import score_trial
        verdicts = [
            _v("NEI", "inclusion"),
            _v("NEI", "inclusion"),
            _v("NEI", "exclusion"),
        ]
        score = score_trial(verdicts, META_P2_REC)
        # inclusion_met=0, ratio=0.0, penalty=0, nei_ratio=1.0
        # 0 - 0 + 0.10*0.6 + 0.15*1.0 - 0.20*1.0 = 0.06 + 0.15 - 0.20 = 0.01
        expected = 0.45 * 0.0 - 1.0 * 0.0 + 0.10 * 0.6 + 0.15 * 1.0 - 0.20 * 1.0
        assert abs(score - max(expected, 0.0)) < 1e-9

    def test_mixed_met_nei_inclusion(self):
        """Half met, half NEI, no exclusion violations."""
        from src.ranking.scorer import score_trial
        verdicts = [
            _v("MET", "inclusion"),
            _v("NEI", "inclusion"),
            _v("MET", "exclusion"),   # exclusion MET = not violated
        ]
        score = score_trial(verdicts, META_P3_REC)
        # inclusion: 2 total, 1 met → ratio=0.5
        # exclusion: 0 NOT_MET → penalty=0
        # nei: 1/3
        # raw = 0.45*0.5 - 0 + 0.10*1.0 + 0.15*1.0 - 0.20*(1/3)
        #     = 0.225 + 0.10 + 0.15 - 0.0667 ≈ 0.4083
        expected = 0.45 * 0.5 + 0.10 * 1.0 + 0.15 * 1.0 - 0.20 * (1 / 3)
        assert abs(score - expected) < 1e-9

    def test_empty_criteria(self):
        """No criteria at all: inclusion_met_ratio=0, no penalty, no NEI."""
        from src.ranking.scorer import score_trial
        score = score_trial([], META_P3_REC)
        # 0.45*0 + 0.10*1.0 + 0.15*1.0 - 0.20*0 = 0.25
        assert abs(score - 0.25) < 1e-9

    def test_bare_verdicts_treated_as_inclusion(self):
        """Plain CriterionVerdict (no type tag) counted as inclusion."""
        from src.ranking.scorer import score_trial
        verdicts = [_bare("MET"), _bare("MET")]
        score = score_trial(verdicts, META_P3_REC)
        assert score > 0.0
        # Same as all-inclusion-met: 0.45 + 0.10 + 0.15 = 0.70
        assert abs(score - 0.70) < 1e-9


# ---------------------------------------------------------------------------
# Phase and status normalisation
# ---------------------------------------------------------------------------

class TestMetadataNormalisation:
    def test_phase3_lowercase_normalised(self):
        from src.ranking.scorer import score_trial
        s1 = score_trial([_v("MET", "inclusion")], {"phase": "phase3", "status": "RECRUITING"})
        s2 = score_trial([_v("MET", "inclusion")], {"phase": "PHASE3", "status": "RECRUITING"})
        assert abs(s1 - s2) < 1e-9

    def test_phase3_spaced_normalised(self):
        from src.ranking.scorer import score_trial
        s1 = score_trial([_v("MET", "inclusion")], {"phase": "Phase 3", "status": "RECRUITING"})
        s2 = score_trial([_v("MET", "inclusion")], {"phase": "PHASE3", "status": "RECRUITING"})
        assert abs(s1 - s2) < 1e-9

    def test_unknown_phase_gets_default_bonus(self):
        from src.ranking.scorer import score_trial
        score = score_trial([], {"phase": "PHASE4", "status": "COMPLETED"})
        # 0.45*0 + 0.10*0.1 + 0 - 0 = 0.01
        assert abs(score - 0.01) < 1e-9

    def test_missing_phase_gets_default_bonus(self):
        from src.ranking.scorer import score_trial
        score = score_trial([], {"status": "COMPLETED"})
        assert abs(score - 0.01) < 1e-9

    def test_recruiting_status_case_insensitive(self):
        from src.ranking.scorer import score_trial
        s1 = score_trial([], {"phase": "PHASE3", "status": "recruiting"})
        s2 = score_trial([], {"phase": "PHASE3", "status": "RECRUITING"})
        assert abs(s1 - s2) < 1e-9

    def test_non_recruiting_no_bonus(self):
        from src.ranking.scorer import score_trial
        s = score_trial([], {"phase": "PHASE3", "status": "COMPLETED"})
        # 0.10*1.0 + 0 = 0.10
        assert abs(s - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# Clamp behaviour
# ---------------------------------------------------------------------------

class TestClamp:
    def test_score_never_negative(self):
        from src.ranking.scorer import score_trial
        # Worst case: exclusion violated + all NEI + Phase1 + not recruiting
        verdicts = [_v("NOT_MET", "exclusion"), _v("NEI", "inclusion")]
        score = score_trial(verdicts, META_P1)
        assert score >= 0.0

    def test_exclusion_met_does_not_apply_penalty(self):
        from src.ranking.scorer import score_trial
        verdicts = [_v("MET", "exclusion"), _v("MET", "inclusion")]
        score = score_trial(verdicts, META_P3_REC)
        assert score > 0.0

    def test_exclusion_nei_does_not_apply_penalty(self):
        from src.ranking.scorer import score_trial
        verdicts = [_v("NEI", "exclusion"), _v("MET", "inclusion")]
        score = score_trial(verdicts, META_P3_REC)
        # No penalty; NEI applies to ratio
        assert score > 0.0


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------

class TestScoreBatch:
    def test_returns_score_for_each_nct(self):
        from src.ranking.scorer import score_batch
        vbn = {
            "NCT001": [_v("MET", "inclusion")],
            "NCT002": [_v("NOT_MET", "exclusion")],
        }
        mbn = {
            "NCT001": META_P3_REC,
            "NCT002": META_P3_REC,
        }
        result = score_batch(vbn, mbn)
        assert set(result.keys()) == {"NCT001", "NCT002"}
        assert result["NCT001"] > result["NCT002"]

    def test_missing_metadata_uses_empty_dict(self):
        from src.ranking.scorer import score_batch
        vbn = {"NCT999": [_v("MET", "inclusion")]}
        result = score_batch(vbn, {})
        assert "NCT999" in result
        assert result["NCT999"] >= 0.0

    def test_empty_batch_returns_empty_dict(self):
        from src.ranking.scorer import score_batch
        assert score_batch({}, {}) == {}

    def test_all_scores_non_negative(self):
        from src.ranking.scorer import score_batch
        vbn = {
            "A": [_v("MET", "inclusion"), _v("NOT_MET", "exclusion")],
            "B": [_v("NEI", "inclusion"), _v("NEI", "exclusion")],
            "C": [_v("MET", "inclusion"), _v("MET", "exclusion")],
        }
        mbn = {"A": META_P1, "B": META_EMPTY, "C": META_P2_REC}
        for score in score_batch(vbn, mbn).values():
            assert score >= 0.0
