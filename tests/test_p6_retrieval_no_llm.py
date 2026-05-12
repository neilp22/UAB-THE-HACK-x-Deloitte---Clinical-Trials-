"""P6 no-LLM tests for deterministic retrieval/query behavior."""

from __future__ import annotations

from src.parsing.patient_normalizer import PatientProfile
from src.retrieval.query_expander import expand_query, expand_term, generate_queries
from src.retrieval.rrf import rrf_fuse_ids


def test_query_expansion_expands_abbreviation_without_llm():
    result = expand_term("nsclc")

    assert "nsclc" in result
    assert "non-small cell lung carcinoma" in result


def test_query_expansion_expands_therapy_class_without_llm():
    result = expand_term("osimertinib")

    assert "osimertinib" in result
    assert any("EGFR" in item or "EGFR-TKI" in item for item in result)


def test_expand_query_keeps_original_query():
    variants = expand_query("nsclc stage IV")

    assert "nsclc stage IV" in variants
    assert len(variants) == len(set(variants))


def test_generate_queries_keeps_facets_separated_to_reduce_dilution():
    profile = PatientProfile(
        age=58,
        gender="female",
        conditions=[
            "non-small cell lung carcinoma",
            "brain metastases",
            "chronic kidney disease",
        ],
        medications=["osimertinib"],
        prior_treatments=["carboplatin"],
    )

    queries = generate_queries(
        profile,
        raw_text=(
            "58 year old female with NSCLC, brain metastases, chronic kidney "
            "disease, progressed after osimertinib and carboplatin."
        ),
    )

    joined = " | ".join(queries).lower()

    assert len(queries) >= 3
    assert len(queries) <= 12
    assert any("non-small cell lung carcinoma" in q.lower() for q in queries)
    assert any("brain metastases" in q.lower() for q in queries)
    assert "osimertinib" in joined or "egfr" in joined


def test_generate_queries_has_raw_text_fallback_without_llm():
    profile = PatientProfile()

    queries = generate_queries(profile, raw_text="rare syndrome with unusual biomarker")

    assert len(queries) >= 1
    assert any("rare syndrome" in q.lower() for q in queries)


def test_rrf_fusion_promotes_documents_seen_in_multiple_lists():
    result = rrf_fuse_ids(
        [
            ["NCT001", "NCT002", "NCT003"],
            ["NCT002", "NCT004", "NCT001"],
        ]
    )

    assert "NCT001" in result
    assert "NCT002" in result
    assert result.index("NCT001") < result.index("NCT003")
    assert result.index("NCT002") < result.index("NCT004")