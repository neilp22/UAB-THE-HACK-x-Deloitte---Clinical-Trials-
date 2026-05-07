"""Tests for src/retrieval/clinical_query_planner.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.patient_normalizer import PatientProfile
from src.retrieval.clinical_query_planner import (
    QuerySpec,
    _clean_query,
    _condition_queries,
    _expand_condition,
    _find_terms,
    _remove_duplicates,
    plan_clinical_queries,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NSCLC_TEXT = (
    "58 year old male with non-small cell lung cancer and EGFR mutation. "
    "Prior platinum chemotherapy. Now on osimertinib."
)

NSCLC_PROFILE = PatientProfile(
    age=58,
    gender="male",
    conditions=["non-small cell lung cancer", "EGFR mutation"],
    medications=["osimertinib"],
    prior_treatments=["platinum chemotherapy"],
    relevant_history=["EGFR exon 19 deletion"],
)

EMPTY_PROFILE = PatientProfile()

SIMPLE_PROFILE = PatientProfile(
    age=45,
    gender="female",
    conditions=["breast cancer"],
    medications=[],
    prior_treatments=[],
    relevant_history=[],
)


# ---------------------------------------------------------------------------
# Fix 1: max_words = 3
# ---------------------------------------------------------------------------

class TestCleanQuery:
    def test_max_three_words(self):
        result = _clean_query("non small cell lung cancer metastatic")
        assert len(result.split()) <= 3

    def test_stopwords_removed(self):
        result = _clean_query("a patient with cancer")
        assert "a" not in result.split()
        assert "patient" not in result.split()
        assert "with" not in result.split()

    def test_empty_input(self):
        assert _clean_query("") == ""

    def test_single_word(self):
        assert _clean_query("cancer") == "cancer"

    def test_all_stopwords(self):
        # Only stopwords → empty string
        assert _clean_query("a an the and or") == ""

    def test_special_chars_stripped(self):
        result = _clean_query("NSCLC, EGFR+")
        # commas removed, alphanumeric kept
        assert "," not in result

    def test_respects_3_word_limit_exactly(self):
        result = _clean_query("alpha beta gamma delta epsilon")
        assert len(result.split()) == 3


# ---------------------------------------------------------------------------
# Fix 2: PatientProfile type accepted
# ---------------------------------------------------------------------------

class TestPatientProfileAccepted:
    def test_accepts_patient_profile(self):
        result = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        assert isinstance(result, list)
        assert all(isinstance(q, QuerySpec) for q in result)

    def test_accepts_empty_profile(self):
        result = plan_clinical_queries(EMPTY_PROFILE, NSCLC_TEXT)
        assert isinstance(result, list)

    def test_type_error_on_wrong_type(self):
        with pytest.raises((AttributeError, TypeError)):
            plan_clinical_queries("not a profile", NSCLC_TEXT)  # type: ignore


# ---------------------------------------------------------------------------
# NSCLC / EGFR biomarker query generation
# ---------------------------------------------------------------------------

class TestNSCLCQueries:
    def test_generates_queries(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        assert len(queries) > 0

    def test_contains_egfr_biomarker_query(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        query_texts = [q.query.lower() for q in queries]
        assert any("egfr" in qt for qt in query_texts), (
            f"Expected EGFR biomarker query, got: {query_texts}"
        )

    def test_contains_condition_query(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        cond_queries = [q for q in queries if q.query_type == "condition"]
        assert len(cond_queries) > 0

    def test_no_query_exceeds_3_words(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        for q in queries:
            assert len(q.query.split()) <= 3, (
                f"Query '{q.query}' exceeds 3 words"
            )

    def test_high_priority_condition_first(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        assert queries[0].priority >= queries[-1].priority

    def test_max_queries_respected(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT, max_queries=5)
        assert len(queries) <= 5

    def test_osimertinib_treatment_query(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        query_texts = [q.query.lower() for q in queries]
        assert any("osimertinib" in qt for qt in query_texts)


# ---------------------------------------------------------------------------
# Fallback when no conditions found
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_fires_when_no_conditions(self):
        profile = PatientProfile(
            conditions=[],
            medications=[],
            prior_treatments=[],
            relevant_history=[],
        )
        # Non-empty patient text, no recognized conditions/biomarkers/stages/treatments
        queries = plan_clinical_queries(profile, "hypertension headache mild symptoms")
        # Fallback should produce at least one query
        assert len(queries) >= 1

    def test_fallback_is_term_type(self):
        profile = PatientProfile()
        queries = plan_clinical_queries(profile, "rare genetic syndrome xyz123")
        assert any(q.reason == "Fallback query from patient text" for q in queries)

    def test_no_fallback_when_conditions_present(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        fallback_queries = [q for q in queries if "Fallback" in q.reason]
        assert len(fallback_queries) == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_no_duplicate_query_strings(self):
        queries = plan_clinical_queries(NSCLC_PROFILE, NSCLC_TEXT)
        seen = set()
        for q in queries:
            key = (q.query_type, q.query)
            assert key not in seen, f"Duplicate query: {key}"
            seen.add(key)

    def test_remove_duplicates_keeps_highest_priority(self):
        dupes = [
            QuerySpec(query_type="term", query="egfr", reason="A", priority=80),
            QuerySpec(query_type="term", query="egfr", reason="B", priority=90),
        ]
        result = _remove_duplicates(dupes)
        assert len(result) == 1
        assert result[0].priority == 90

    def test_different_types_not_deduplicated(self):
        mixed = [
            QuerySpec(query_type="condition", query="cancer", reason="A", priority=80),
            QuerySpec(query_type="term", query="cancer", reason="B", priority=80),
        ]
        result = _remove_duplicates(mixed)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Helper internals
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_expand_condition_nsclc(self):
        aliases = _expand_condition("nsclc")
        assert len(aliases) >= 1
        combined = " ".join(aliases).lower()
        assert "nsclc" in combined or "non-small" in combined

    def test_find_terms_biomarkers(self):
        from src.retrieval.clinical_query_planner import BIOMARKERS
        result = _find_terms("patient has EGFR mutation and ALK rearrangement", BIOMARKERS)
        assert "egfr" in result
        assert "alk" in result

    def test_condition_queries_returns_query_spec(self):
        result = _condition_queries(["lung cancer"])
        assert all(isinstance(q, QuerySpec) for q in result)
        assert all(q.query_type == "condition" for q in result)


# ---------------------------------------------------------------------------
# Integration: build_queries_clinical returns NCT IDs for topic 1
# ---------------------------------------------------------------------------

TREC_QRELS = "data/trec/2021/qrels.txt"
TREC_TOPICS = "data/trec/2021/topics.xml"

pytestmark_integration = pytest.mark.skipif(
    not Path(TREC_QRELS).exists(),
    reason="TREC 2021 qrels not present",
)


@pytest.mark.skipif(
    not Path(TREC_TOPICS).exists(),
    reason="TREC 2021 topics not present",
)
def test_build_queries_clinical_returns_nct_ids():
    import xml.etree.ElementTree as ET
    from src.retrieval.query_builder import build_queries_clinical
    from src.parsing.patient_normalizer import normalize_patient

    tree = ET.parse(TREC_TOPICS)
    topic1 = tree.getroot().find('topic[@number="1"]').text.strip()

    profile = normalize_patient(topic1, use_cache=True)
    nct_ids = build_queries_clinical(topic1, profile, max_candidates=20)

    assert isinstance(nct_ids, list)
    assert len(nct_ids) > 0
    for nid in nct_ids:
        assert isinstance(nid, str)
        assert nid.startswith("NCT"), f"Expected NCT ID, got: {nid}"
    # No duplicates
    assert len(nct_ids) == len(set(nct_ids))
