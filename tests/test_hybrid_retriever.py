"""Tests for src/retrieval/hybrid_retriever.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBEDDINGS  = "data/cache/embeddings/trial_embeddings.npy"
NCT_IDS     = "data/cache/embeddings/trial_nct_ids.pkl"
CACHE_DIR   = "data/cache/trial_data"
BM25_INDEX  = "data/cache/bm25_trec2021_index.pkl"

pytestmark = pytest.mark.skipif(
    not Path(EMBEDDINGS).exists() or not Path(BM25_INDEX).exists(),
    reason="Embedding or BM25 index files not present",
)

NSCLC = "58 year old male non-small cell lung cancer EGFR mutation"


@pytest.fixture(scope="module")
def sem():
    from src.retrieval.semantic_retriever import SemanticRetriever
    return SemanticRetriever(EMBEDDINGS, NCT_IDS, CACHE_DIR)


@pytest.fixture(scope="module")
def hybrid(sem):
    from src.retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(sem, BM25_INDEX)


class TestInit:
    def test_alpha_beta_defaults(self, hybrid):
        assert hybrid.alpha == 0.5
        assert hybrid.beta == 0.5

    def test_bm25_index_loaded(self, hybrid):
        assert hybrid._bm25 is not None
        assert len(hybrid._bm25_nct_ids) > 0

    def test_bm25_nct_ids_aligned(self, hybrid):
        for nct, idx in list(hybrid._bm25_idx.items())[:10]:
            assert hybrid._bm25_nct_ids[idx] == nct


class TestQuery:
    def test_returns_top_k(self, hybrid):
        results = hybrid.query(NSCLC, top_k=50)
        assert len(results) == 50

    def test_returns_strings(self, hybrid):
        results = hybrid.query(NSCLC, top_k=10)
        for r in results:
            assert isinstance(r, str)
            assert r.startswith("NCT")

    def test_no_duplicates(self, hybrid):
        results = hybrid.query(NSCLC, top_k=20)
        assert len(results) == len(set(results))

    def test_egfr_trials_in_top10(self, hybrid, sem):
        """EGFR/lung-cancer-specific trials should appear in top 10."""
        results = hybrid.query(NSCLC, top_k=10)
        egfr_keywords = {"egfr", "epidermal growth factor", "erlotinib",
                         "gefitinib", "osimertinib"}
        hits = 0
        for nid in results:
            trial = sem.get_trial(nid)
            if trial is None:
                continue
            text = (trial.get("title", "") + trial.get("eligibility_criteria", "")).lower()
            if any(kw in text for kw in egfr_keywords):
                hits += 1
        assert hits >= 1, f"Expected ≥1 EGFR trial in top 10, got {hits}"

    def test_hybrid_beats_or_matches_bm25_recall_topic1(self, hybrid):
        """Hybrid recall@20 on topic 1 should be ≥ BM25-only recall."""
        import xml.etree.ElementTree as ET
        tree = ET.parse("data/trec/2021/topics.xml")
        topic1 = tree.getroot().find('topic[@number="1"]').text.strip()

        qrels_t1 = {}
        for line in open("data/trec/2021/qrels.txt"):
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "1":
                qrels_t1[parts[2]] = int(parts[3])
        relevant = {nid for nid, g in qrels_t1.items() if g >= 1}

        from src.retrieval.trec_index_retriever import TrecIndexRetriever
        bm25_top20 = set(TrecIndexRetriever().load_index().query(topic1, top_k=20))
        hyb_top20  = set(hybrid.query(topic1, top_k=20))

        bm25_recall = len(bm25_top20 & relevant) / len(relevant)
        hyb_recall  = len(hyb_top20  & relevant) / len(relevant)
        assert hyb_recall >= bm25_recall - 0.01, (
            f"Hybrid recall {hyb_recall:.4f} < BM25 recall {bm25_recall:.4f} by more than 0.01"
        )


class TestGetTrial:
    def test_delegates_to_semantic(self, hybrid, sem):
        nct_id = sem.nct_ids[100]
        assert hybrid.get_trial(nct_id) == sem.get_trial(nct_id)

    def test_unknown_returns_none(self, hybrid):
        assert hybrid.get_trial("NCT99999999") is None
