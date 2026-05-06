"""
Validation tests for src/retrieval/bm25_retriever.py.

Run with: python tests/test_bm25.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever, build_index

_SAMPLE_TRIALS = [
    {
        "nct_id": "NCT00000001",
        "title": "Phase 3 trial of erlotinib in EGFR-mutant non-small cell lung cancer",
        "brief_summary": "This study evaluates erlotinib versus chemotherapy in NSCLC with EGFR mutations.",
        "eligibility_criteria": "Inclusion: NSCLC, EGFR mutation, age >= 18. Exclusion: prior EGFR therapy.",
        "conditions": ["Non-small cell lung cancer", "EGFR mutation"],
    },
    {
        "nct_id": "NCT00000002",
        "title": "Cardiac resynchronization therapy in heart failure patients",
        "brief_summary": "CRT device implantation in patients with severe heart failure and EF < 35%.",
        "eligibility_criteria": "Inclusion: EF <= 35%, NYHA Class III-IV. Exclusion: recent MI.",
        "conditions": ["Heart failure", "Cardiomyopathy"],
    },
    {
        "nct_id": "NCT00000003",
        "title": "Pembrolizumab plus chemotherapy in advanced lung cancer",
        "brief_summary": "PD-1 inhibitor combined with carboplatin/pemetrexed in stage IV NSCLC.",
        "eligibility_criteria": "Inclusion: stage IV NSCLC, no prior immunotherapy. Exclusion: autoimmune disease.",
        "conditions": ["Lung cancer", "NSCLC"],
    },
]


def test_index_builds() -> None:
    print("TEST: BM25 index builds from trial list...")
    retriever = BM25Retriever(_SAMPLE_TRIALS)
    assert len(retriever) == 3
    print(f"  PASS — index size: {len(retriever)}")


def test_relevant_trial_ranks_first() -> None:
    print("TEST: lung cancer query ranks lung trials above cardiac trial...")
    retriever = build_index(_SAMPLE_TRIALS)
    results = retriever.query("45-year-old patient with EGFR-positive non-small cell lung cancer", top_k=3)

    assert len(results) == 3
    top_nct = results[0]["nct_id"]
    assert top_nct in ("NCT00000001", "NCT00000003"), (
        f"Expected lung cancer trial at rank 1, got {top_nct}"
    )
    # Heart failure trial should NOT be rank 1
    assert results[0]["nct_id"] != "NCT00000002"
    print(f"  PASS — top result: {results[0]['nct_id']} (score={results[0]['bm25_score']:.3f})")


def test_bm25_score_present() -> None:
    print("TEST: bm25_score key present in results...")
    retriever = build_index(_SAMPLE_TRIALS)
    results = retriever.query("heart failure", top_k=2)
    for r in results:
        assert "bm25_score" in r, "Missing bm25_score field"
        assert isinstance(r["bm25_score"], float)
    print(f"  PASS — bm25_score found in all {len(results)} results")


def test_top_k_respected() -> None:
    print("TEST: top_k parameter respected...")
    retriever = build_index(_SAMPLE_TRIALS)
    results = retriever.query("cancer", top_k=2)
    assert len(results) <= 2
    print(f"  PASS — got {len(results)} results (top_k=2)")


def test_empty_index_raises() -> None:
    print("TEST: empty trial list raises ValueError...")
    try:
        BM25Retriever([])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  PASS — ValueError raised as expected")


def test_scores_descending() -> None:
    print("TEST: results sorted by score descending...")
    retriever = build_index(_SAMPLE_TRIALS)
    results = retriever.query("lung cancer EGFR mutation therapy", top_k=3)
    scores = [r["bm25_score"] for r in results]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"
    print(f"  PASS — scores descending: {[f'{s:.3f}' for s in scores]}")


if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in [
        ("index_builds", test_index_builds),
        ("relevant_trial_ranks_first", test_relevant_trial_ranks_first),
        ("bm25_score_present", test_bm25_score_present),
        ("top_k_respected", test_top_k_respected),
        ("empty_index_raises", test_empty_index_raises),
        ("scores_descending", test_scores_descending),
    ]:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL — {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
