"""Tests for high-recall retrieval components (Parts 1–4, 7–8)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.retrieval.query_expander import (
    ABBREV,
    THERAPY_CLASS,
    BIOMARKER_EXPAND,
    expand_term,
    expand_query,
    generate_queries,
)
from src.retrieval.rrf import rrf_fuse, rrf_fuse_ids, rrf_fuse_weighted
from src.retrieval.field_bm25 import _split_eligibility, _tok


# ---------------------------------------------------------------------------
# QueryExpander tests (Part 1 & 7)
# ---------------------------------------------------------------------------

class TestExpandTerm:
    def test_abbreviation_expanded(self):
        result = expand_term("nsclc")
        assert "nsclc" in result
        assert "non-small cell lung carcinoma" in result

    def test_therapy_class_expanded(self):
        result = expand_term("osimertinib")
        assert "osimertinib" in result
        assert "EGFR-TKI" in result

    def test_biomarker_expanded(self):
        result = expand_term("egfr")
        assert "EGFR mutation" in result
        assert "EGFR exon 19" in result

    def test_unknown_term_returns_itself(self):
        result = expand_term("unknown_drug_xyz")
        assert result == ["unknown_drug_xyz"]

    def test_no_duplicates(self):
        result = expand_term("egfr")
        assert len(result) == len(set(result))

    def test_her2_expansion(self):
        result = expand_term("her2")
        assert any("ERBB2" in r or "HER2" in r for r in result)


class TestExpandQuery:
    def test_abbreviation_in_query(self):
        variants = expand_query("nsclc stage IV")
        assert "nsclc stage IV" in variants
        assert any("non-small cell lung carcinoma" in v for v in variants)

    def test_no_expansion_for_plain_query(self):
        variants = expand_query("lung cancer")
        # no abbreviation — returns original only
        assert "lung cancer" in variants

    def test_deduplicated(self):
        variants = expand_query("nsclc nsclc")
        assert len(variants) == len(set(variants))


class TestGenerateQueries:
    def _make_profile(self):
        from src.parsing.patient_normalizer import PatientProfile
        return PatientProfile(
            age=58,
            gender="female",
            conditions=["non-small cell lung carcinoma", "NSCLC"],
            medications=["osimertinib"],
            prior_treatments=["carboplatin"],
        )

    def test_returns_list_of_strings(self):
        profile = self._make_profile()
        queries = generate_queries(profile, "58yo female NSCLC EGFR osimertinib")
        assert isinstance(queries, list)
        assert all(isinstance(q, str) for q in queries)

    def test_at_least_3_queries(self):
        profile = self._make_profile()
        queries = generate_queries(profile, "58yo female NSCLC")
        assert len(queries) >= 3

    def test_capped_at_12(self):
        profile = self._make_profile()
        queries = generate_queries(profile, "patient text")
        assert len(queries) <= 12

    def test_no_empty_queries(self):
        profile = self._make_profile()
        queries = generate_queries(profile, "test")
        assert all(len(q) >= 3 for q in queries)

    def test_deduplicated(self):
        profile = self._make_profile()
        queries = generate_queries(profile, "test")
        assert len(queries) == len(set(queries))

    def test_therapy_resistance_query(self):
        profile = self._make_profile()
        queries = generate_queries(profile, "progressed after osimertinib")
        # should include a resistance-flavoured query
        combined = " ".join(queries).lower()
        assert "osimertinib" in combined or "egfr" in combined

    def test_empty_profile(self):
        from src.parsing.patient_normalizer import PatientProfile
        queries = generate_queries(PatientProfile(), "58yo male with NSCLC")
        assert len(queries) >= 1  # at least raw text fallback


# ---------------------------------------------------------------------------
# RRF tests (Part 2)
# ---------------------------------------------------------------------------

class TestRRF:
    def test_basic_fusion(self):
        l1 = ["A", "B", "C"]
        l2 = ["B", "A", "D"]
        result = rrf_fuse_ids([l1, l2])
        # B and A appear in both lists — should rank above C and D
        assert result.index("B") < result.index("C")
        assert result.index("A") < result.index("D")

    def test_document_in_one_list_only(self):
        l1 = ["A", "B"]
        l2 = ["C", "D"]
        result = rrf_fuse_ids([l1, l2])
        assert "A" in result
        assert "C" in result
        assert len(result) == 4

    def test_rrf_formula_values(self):
        l1 = ["A"]
        result = rrf_fuse([l1], k=60)
        nct, score = result[0]
        expected = 1.0 / (60 + 1)
        assert abs(score - expected) < 1e-9

    def test_two_lists_doubles_score(self):
        l1 = ["A"]
        l2 = ["A"]
        result = rrf_fuse([l1, l2], k=60)
        _, score = result[0]
        expected = 2.0 / (60 + 1)
        assert abs(score - expected) < 1e-9

    def test_top_k_truncates(self):
        l1 = [f"NCT{i:08d}" for i in range(200)]
        result = rrf_fuse_ids([l1], top_k=50)
        assert len(result) == 50

    def test_weighted_rrf(self):
        l1 = ["A", "B"]
        l2 = ["B", "A"]
        # equal weights should give B rank 1 (top in l1 × 2.0 + rank 2 in l2 × 1.0)
        result = rrf_fuse_weighted([l1, l2], weights=[2.0, 1.0], k=60)
        ids = [nid for nid, _ in result]
        # A: 2/(60+1) + 1/(60+2) ≈ 0.0328+0.0161 = 0.0489
        # B: 2/(60+2) + 1/(60+1) ≈ 0.0323+0.0164 = 0.0487
        # A should score higher due to weight×rank interaction
        assert ids[0] == "A"

    def test_empty_lists_handled(self):
        result = rrf_fuse_ids([[], []])
        assert result == []

    def test_single_list_preserves_order(self):
        l1 = ["A", "B", "C", "D"]
        result = rrf_fuse_ids([l1])
        assert result == l1


# ---------------------------------------------------------------------------
# Field BM25 helpers (Part 3 & 4)
# ---------------------------------------------------------------------------

class TestSplitEligibility:
    def test_splits_on_exclusion_header(self):
        text = "Inclusion Criteria:\n- age >= 18\n\nExclusion Criteria:\n- pregnancy"
        incl, excl = _split_eligibility(text)
        assert "age" in incl
        assert "pregnancy" in excl
        assert "pregnancy" not in incl

    def test_no_exclusion_header(self):
        text = "Must be 18 years old and have NSCLC"
        incl, excl = _split_eligibility(text)
        assert incl == text
        assert excl == ""

    def test_empty_text(self):
        incl, excl = _split_eligibility("")
        assert incl == "" and excl == ""

    def test_case_insensitive(self):
        text = "inclusion criteria:\n- thing\nexclusion criteria:\n- other"
        incl, excl = _split_eligibility(text)
        assert "thing" in incl
        assert "other" in excl


class TestFieldTok:
    def test_stopwords_removed(self):
        tokens = _tok("the patient must be willing to provide documented consent")
        assert "the" not in tokens
        assert "patient" not in tokens  # in clinical boilerplate stop words
        assert "willing" not in tokens  # also in clinical boilerplate stop words
        assert "documented" in tokens   # not a stop word — preserved

    def test_empty_string(self):
        assert _tok("") == []

    def test_medical_terms_preserved(self):
        tokens = _tok("EGFR L858R non-small cell lung carcinoma")
        assert "egfr" in tokens
        assert "non-small" in tokens or "non" in tokens
