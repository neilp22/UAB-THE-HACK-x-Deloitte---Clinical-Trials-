"""Tests for src/matching/label_deriver.py — single source of truth for T2 labels."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.matching.label_deriver import derive_label


class TestExclusionViolation:
    def test_any_excl_violation_returns_not_met(self):
        s = {"inclusion_met": 5, "inclusion_not_met": 0, "inclusion_nei": 0, "exclusion_violations": 1}
        assert derive_label(s) == "NOT_MET"

    def test_excl_violation_beats_all_inclusion_met(self):
        s = {"inclusion_met": 10, "inclusion_not_met": 0, "inclusion_nei": 0, "exclusion_violations": 1}
        assert derive_label(s) == "NOT_MET"

    def test_zero_excl_violations_not_eliminated(self):
        s = {"inclusion_met": 3, "inclusion_not_met": 0, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) != "NOT_MET"


class TestNEIGates:
    def test_majority_nei_returns_nei(self):
        # nei_ratio = 6/10 = 0.60 > 0.50 → NEI
        s = {"inclusion_met": 1, "inclusion_not_met": 3, "inclusion_nei": 6, "exclusion_violations": 0}
        assert derive_label(s) == "NEI"

    def test_exactly_50pct_nei_does_not_trigger_gate(self):
        # nei_ratio = 5/10 = 0.50, not > 0.50 → falls through to inc_ratio check
        s = {"inclusion_met": 3, "inclusion_not_met": 2, "inclusion_nei": 5, "exclusion_violations": 0}
        # inc_ratio = 3/10 = 0.30, not < 0.30 → MET
        assert derive_label(s) == "MET"

    def test_weak_inclusion_ratio_returns_nei(self):
        # inc_ratio = 1/5 = 0.20 < 0.30 → NEI
        s = {"inclusion_met": 1, "inclusion_not_met": 4, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "NEI"

    def test_exactly_30pct_inc_ratio_does_not_trigger_gate(self):
        # inc_ratio = 3/10 = 0.30, not < 0.30 → MET
        s = {"inclusion_met": 3, "inclusion_not_met": 7, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "MET"


class TestMETPath:
    def test_clear_met(self):
        s = {"inclusion_met": 5, "inclusion_not_met": 1, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "MET"

    def test_met_with_some_not_met_above_threshold(self):
        # inc_ratio = 4/6 = 0.667 → MET
        s = {"inclusion_met": 4, "inclusion_not_met": 2, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "MET"

    def test_single_met_no_other_criteria(self):
        s = {"inclusion_met": 1, "inclusion_not_met": 0, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "MET"


class TestNEIFallback:
    def test_all_zeros_returns_nei(self):
        s = {"inclusion_met": 0, "inclusion_not_met": 0, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "NEI"

    def test_missing_keys_defaults_to_nei(self):
        assert derive_label({}) == "NEI"

    def test_only_not_met_returns_nei(self):
        # inc_ratio = 0 < 0.30 → NEI
        s = {"inclusion_met": 0, "inclusion_not_met": 3, "inclusion_nei": 0, "exclusion_violations": 0}
        assert derive_label(s) == "NEI"
