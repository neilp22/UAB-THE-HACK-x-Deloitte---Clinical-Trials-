"""
BM25-based retrieval over a set of clinical trial documents.

Supports:
    build_index(trials)  → BM25Retriever
    retriever.query(patient_text, top_k=100) → list[dict]
"""
from __future__ import annotations

import logging
import re
import string

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Minimal stop-word list — keep medical terms, remove structural noise
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "this", "that", "these", "those", "it", "its", "from", "by",
    "as", "he", "she", "they", "we", "his", "her", "their", "our",
    "who", "which", "what", "when", "where", "how",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation (keep hyphens), filter stop words."""
    text = text.lower()
    # Replace punctuation except hyphens with space
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _trial_text(trial: dict) -> str:
    """Concatenate all searchable fields of a trial into one text block."""
    parts = [
        trial.get("title", ""),
        trial.get("brief_summary", ""),
        trial.get("eligibility_criteria", ""),
        " ".join(trial.get("conditions", [])),
    ]
    return " ".join(p for p in parts if p)


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi with trial-aware indexing.

    Attributes:
        trials: The list of trial dicts passed to build_index.
    """

    def __init__(self, trials: list[dict]) -> None:
        if not trials:
            raise ValueError("Cannot build index from empty trial list.")

        self.trials = trials
        corpus = [_tokenize(_trial_text(t)) for t in trials]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built over %d trials.", len(trials))

    def query(self, patient_text: str, top_k: int = 100) -> list[dict]:
        """
        Score all indexed trials against patient_text.

        Returns:
            List of top_k trial dicts, each augmented with a 'bm25_score' key,
            sorted descending by score.
        """
        query_tokens = _tokenize(patient_text)
        if not query_tokens:
            logger.warning("Empty query tokens — returning empty result.")
            return []

        scores: np.ndarray = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            trial = dict(self.trials[idx])  # shallow copy
            trial["bm25_score"] = float(scores[idx])
            results.append(trial)

        return results

    def __len__(self) -> int:
        return len(self.trials)


def build_index(trials: list[dict]) -> BM25Retriever:
    """Convenience factory: build a BM25Retriever from a list of trial dicts."""
    return BM25Retriever(trials)
