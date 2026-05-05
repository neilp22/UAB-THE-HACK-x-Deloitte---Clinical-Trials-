"""Tests for src/retrieval/query_builder.py"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConditionVariants:
    def test_abbreviation_expansion(self):
        from src.retrieval.query_builder import _condition_variants
        with patch("src.retrieval.query_builder.lookup_mesh_term", return_value=None):
            variants = _condition_variants("nsclc")
        # NSCLC expands to "non-small cell lung carcinoma"
        # First word of expansion is "non-small" (hyphenated token)
        assert variants[0] == "non-small"
        assert "non-small cell" in variants

    def test_plain_single_word(self):
        from src.retrieval.query_builder import _condition_variants
        with patch("src.retrieval.query_builder.lookup_mesh_term", return_value=None):
            variants = _condition_variants("hypertension")
        assert "hypertension" in variants

    def test_two_word_condition_gives_1_and_2_word_variants(self):
        from src.retrieval.query_builder import _condition_variants
        with patch("src.retrieval.query_builder.lookup_mesh_term", return_value=None):
            variants = _condition_variants("lung cancer")
        assert "lung" in variants
        assert "lung cancer" in variants

    def test_mesh_inverted_term_strips_after_comma(self):
        """MeSH label 'Carcinoma, Non-Small-Cell Lung' → first word 'carcinoma'."""
        from src.retrieval.query_builder import _condition_variants
        with patch("src.retrieval.query_builder.lookup_mesh_term",
                   return_value="Carcinoma, Non-Small-Cell Lung"):
            variants = _condition_variants("non-small cell lung cancer")
        assert "carcinoma" in variants

    def test_no_short_or_empty_variants(self):
        from src.retrieval.query_builder import _condition_variants
        with patch("src.retrieval.query_builder.lookup_mesh_term", return_value=None):
            variants = _condition_variants("cancer")
        assert all(len(v) >= 3 for v in variants)

    def test_max_3_variants(self):
        from src.retrieval.query_builder import _condition_variants
        with patch("src.retrieval.query_builder.lookup_mesh_term",
                   return_value="Neoplasms, Lung"):
            variants = _condition_variants("lung cancer")
        assert len(variants) <= 3


class TestRetrieveCandidates:
    def test_empty_conditions_returns_empty(self):
        from src.retrieval.query_builder import retrieve_candidates
        result = retrieve_candidates([], use_cache=False)
        assert result == []

    def test_deduplication_by_nct_id(self):
        """Multiple query variants returning the same NCT ID → only one entry."""
        from src.retrieval.query_builder import retrieve_candidates
        trial_a = {"nct_id": "NCT001", "title": "Trial A"}
        trial_b = {"nct_id": "NCT002", "title": "Trial B"}

        with patch("src.retrieval.query_builder._condition_variants", return_value=["lung", "lung cancer"]):
            with patch("src.retrieval.query_builder.search_trials", return_value=[trial_a, trial_b]):
                with patch("src.retrieval.query_builder.time.sleep"):
                    result = retrieve_candidates(["lung cancer"], use_cache=False)

        nct_ids = [t["nct_id"] for t in result]
        assert len(nct_ids) == len(set(nct_ids)), "Duplicate NCT IDs in result"

    def test_max_total_respected(self):
        from src.retrieval.query_builder import retrieve_candidates
        trials = [{"nct_id": f"NCT{i:04d}", "title": f"Trial {i}"} for i in range(100)]

        with patch("src.retrieval.query_builder._condition_variants",
                   return_value=["lung", "cancer", "carcinoma", "neoplasm"]):
            with patch("src.retrieval.query_builder.search_trials", return_value=trials):
                with patch("src.retrieval.query_builder.time.sleep"):
                    result = retrieve_candidates(["lung cancer"], max_total=25, use_cache=False)

        assert len(result) <= 25

    def test_returns_list_of_dicts(self):
        from src.retrieval.query_builder import retrieve_candidates
        fake = {"nct_id": "NCT999", "title": "Test"}

        with patch("src.retrieval.query_builder._condition_variants", return_value=["test"]):
            with patch("src.retrieval.query_builder.search_trials", return_value=[fake]):
                with patch("src.retrieval.query_builder.time.sleep"):
                    result = retrieve_candidates(["test"], use_cache=False)

        assert isinstance(result, list)
        assert all(isinstance(t, dict) for t in result)

    def test_failed_query_does_not_crash(self):
        """A CT API failure for one variant should not crash; others still run."""
        from src.retrieval.query_builder import retrieve_candidates
        good_trial = {"nct_id": "NCT100", "title": "Good Trial"}
        call_count = [0]

        def flaky_search(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CT API timeout")
            return [good_trial]

        with patch("src.retrieval.query_builder._condition_variants", return_value=["lung", "cancer"]):
            with patch("src.retrieval.query_builder.search_trials", side_effect=flaky_search):
                with patch("src.retrieval.query_builder.time.sleep"):
                    result = retrieve_candidates(["lung cancer"], use_cache=False)

        # Second variant succeeded — result should contain the good trial
        assert any(t["nct_id"] == "NCT100" for t in result)

    def test_top_4_conditions_limit(self):
        """Only the first 4 conditions are processed."""
        from src.retrieval.query_builder import retrieve_candidates
        variant_calls: list[str] = []

        def capture_variants(cond):
            variant_calls.append(cond)
            return [cond[:4]]  # return a short but valid variant

        with patch("src.retrieval.query_builder._condition_variants", side_effect=capture_variants):
            with patch("src.retrieval.query_builder.search_trials", return_value=[]):
                with patch("src.retrieval.query_builder.time.sleep"):
                    retrieve_candidates(
                        ["c1", "c2", "c3", "c4", "c5", "c6"],
                        use_cache=False,
                    )

        assert len(variant_calls) <= 4
