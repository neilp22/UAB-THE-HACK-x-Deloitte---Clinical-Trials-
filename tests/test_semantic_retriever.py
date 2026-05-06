"""Tests for src/retrieval/semantic_retriever.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBEDDINGS = "data/cache/embeddings/trial_embeddings.npy"
NCT_IDS    = "data/cache/embeddings/trial_nct_ids.pkl"
CACHE_DIR  = "data/cache/trial_data"

pytestmark = pytest.mark.skipif(
    not Path(EMBEDDINGS).exists(),
    reason="Embedding files not present — run Colab notebook first",
)


@pytest.fixture(scope="module")
def retriever():
    from src.retrieval.semantic_retriever import SemanticRetriever
    return SemanticRetriever(EMBEDDINGS, NCT_IDS, CACHE_DIR)


class TestLoad:
    def test_embeddings_shape(self, retriever):
        assert retriever.embeddings.shape == (26170, 768)

    def test_embeddings_dtype(self, retriever):
        assert retriever.embeddings.dtype == np.float32

    def test_nct_ids_count(self, retriever):
        assert len(retriever.nct_ids) == 26170

    def test_reverse_index_consistent(self, retriever):
        for nct, idx in list(retriever.nct_to_idx.items())[:10]:
            assert retriever.nct_ids[idx] == nct

    def test_embeddings_normalised(self, retriever):
        # All rows should have L2 norm ≈ 1.0
        norms = np.linalg.norm(retriever.embeddings[:100], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestQuery:
    PATIENT = "58 year old male non-small cell lung cancer EGFR mutation"

    def test_returns_top_k(self, retriever):
        results = retriever.query(self.PATIENT, top_k=50)
        assert len(results) == 50

    def test_result_keys(self, retriever):
        results = retriever.query(self.PATIENT, top_k=5)
        for r in results:
            assert "nct_id" in r
            assert "score" in r
            assert "title" in r

    def test_scores_between_zero_and_one(self, retriever):
        results = retriever.query(self.PATIENT, top_k=20)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"score out of range: {r['score']}"

    def test_sorted_descending(self, retriever):
        results = retriever.query(self.PATIENT, top_k=20)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_nct_ids_are_strings(self, retriever):
        results = retriever.query(self.PATIENT, top_k=5)
        for r in results:
            assert isinstance(r["nct_id"], str)
            assert r["nct_id"].startswith("NCT")

    def test_top3_nsclc_egfr(self, retriever):
        """Integration smoke test — print top 3 for NSCLC EGFR patient."""
        results = retriever.query(self.PATIENT, top_k=3)
        print("\n=== Top 3 for NSCLC EGFR patient ===")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] {r['nct_id']} — {r['title'][:80]}")
        assert len(results) == 3


class TestGetTrial:
    def test_get_known_trial(self, retriever):
        nct_id = retriever.nct_ids[0]
        trial = retriever.get_trial(nct_id)
        assert trial is not None
        assert trial.get("nct_id") == nct_id or "title" in trial

    def test_get_unknown_returns_none(self, retriever):
        assert retriever.get_trial("NCT99999999") is None
