"""
Hybrid retriever: alpha * semantic (BioBERT) + beta * BM25.

Both score arrays are normalised to [0, 1] before combining.
Runs over all 26k TREC-judged trials in <2s on CPU/MPS.
"""
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)

_DEFAULT_INDEX = "data/cache/bm25_trec2021_index.pkl"

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "this", "that", "these", "those", "it", "its", "from", "by",
    "as", "he", "she", "they", "we", "his", "her", "their", "our",
    "who", "which", "what", "when", "where", "how",
})


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class HybridRetriever:
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_index_path: str | Path = _DEFAULT_INDEX,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        self.semantic = semantic_retriever
        self.alpha = alpha
        self.beta = beta

        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25: BM25Okapi = data["bm25"]
        self._bm25_nct_ids: list[str] = data["nct_ids"]

        # Map NCT ID → BM25 array index for alignment
        self._bm25_idx: dict[str, int] = {
            nct: i for i, nct in enumerate(self._bm25_nct_ids)
        }

        logger.info(
            "HybridRetriever ready: %d semantic trials, %d BM25 trials, alpha=%.2f beta=%.2f",
            len(self.semantic.nct_ids), len(self._bm25_nct_ids), alpha, beta,
        )

    def query(self, patient_text: str, top_k: int = 100) -> list[str]:
        """
        Combine semantic + BM25 scores and return top_k NCT IDs.

        Both score vectors are normalised to [0,1] before combining.
        The result array is indexed by semantic's nct_ids order.
        """
        sem_nct_ids = self.semantic.nct_ids  # ground-truth order (26170)

        # --- Semantic scores (already normalised: L2-norm embeddings → dot in [0,1]) ---
        query_vec = self.semantic.encode_query(patient_text)          # (768,)
        sem_scores: np.ndarray = self.semantic.embeddings @ query_vec # (26170,)

        # --- BM25 scores, aligned to semantic's index order ---
        tokens = _tokenize(patient_text)
        if tokens:
            bm25_raw: np.ndarray = self._bm25.get_scores(tokens)      # (N_bm25,)
            bm25_max = float(bm25_raw.max()) + 1e-9
            bm25_norm = bm25_raw / bm25_max                           # (N_bm25,) in [0,1]
        else:
            bm25_norm = np.zeros(len(self._bm25_nct_ids), dtype=np.float32)

        # Re-index BM25 scores into semantic's ordering
        bm25_aligned = np.array(
            [bm25_norm[self._bm25_idx[nid]] if nid in self._bm25_idx else 0.0
             for nid in sem_nct_ids],
            dtype=np.float32,
        )

        # --- Combine ---
        hybrid: np.ndarray = self.alpha * sem_scores + self.beta * bm25_aligned

        top_indices = np.argsort(hybrid)[-top_k:][::-1]
        return [sem_nct_ids[i] for i in top_indices]

    def get_trial(self, nct_id: str) -> dict | None:
        return self.semantic.get_trial(nct_id)
