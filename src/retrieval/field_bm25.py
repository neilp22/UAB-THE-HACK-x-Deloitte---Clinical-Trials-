"""
Field-weighted BM25 over the TREC clinical trial corpus.

Parts 3 & 4 of the high-recall retrieval architecture.

Why field separation improves recall:
  - A single "bag of words" index weights all fields equally.
  - Conditions/title are the most discriminative: a trial for NSCLC says
    "non-small cell lung carcinoma" in the title and conditions.
  - Inclusion criteria describe what patients SHOULD HAVE → high signal for recall.
  - Exclusion criteria HURT recall: a trial that excludes "prior EGFR TKI" will
    have "EGFR TKI" in its exclusion text, causing false-positive BM25 matches
    for patients WITH EGFR TKI history. Down-weighting exclusion criteria
    reduces this pollution.

Field weights (empirically motivated, consistent with MS-MARCO literature):
  title:      3.0  — concise, high-precision signal
  summary:    2.0  — describes target population
  inclusion:  1.5  — "what patients should have" → recall booster
  exclusion:  0.3  — "what patients must NOT have" → noise source, down-weight

Index is built once from the diskcache and saved as a pkl.
Load time on subsequent runs: <2s.
"""
from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_PATH = "data/cache/field_bm25_index.pkl"

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "this", "that", "these", "those", "it", "its", "from", "by",
    "as", "he", "she", "they", "we", "his", "her", "their", "our",
    "who", "which", "what", "when", "where", "how",
    # Clinical boilerplate that adds noise
    "criteria", "criterion", "patient", "patients", "study", "trial",
    "must", "least", "able", "willing", "signed", "informed", "consent",
    "include", "included", "inclusion", "exclude", "excluded", "exclusion",
})

# Field weights for linear combination
FIELD_WEIGHTS = {
    "title":     3.0,
    "summary":   2.0,
    "inclusion": 1.5,
    "exclusion": 0.3,
}


def _tok(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    return [t for t in text.split() if t not in _STOP_WORDS and len(t) > 1]


def _split_eligibility(text: str) -> tuple[str, str]:
    """
    Split a raw eligibility_criteria blob into (inclusion_text, exclusion_text).

    Handles the common "Inclusion Criteria: ... Exclusion Criteria: ..." format.
    If no clear boundary is found, treats the full text as inclusion.

    Why this matters: see module docstring — exclusion text causes false-positive
    BM25 matches when a patient has a condition that the trial EXCLUDES.
    """
    if not text:
        return "", ""

    excl_pattern = re.compile(
        r"(?:^|\n)\s*(?:Exclusion Criteria|EXCLUSION CRITERIA|Exclusion:)",
        re.IGNORECASE | re.MULTILINE,
    )
    m = excl_pattern.search(text)
    if m:
        return text[:m.start()].strip(), text[m.start():].strip()
    return text.strip(), ""


class FieldBM25Retriever:
    """
    Multi-field BM25 retriever with configurable field weights.

    On first use, call build_index() or load_or_build().
    Subsequent uses load from pkl (fast).
    """

    def __init__(self) -> None:
        self._nct_ids: list[str] = []
        self._title_bm25: BM25Okapi | None = None
        self._summary_bm25: BM25Okapi | None = None
        self._inclusion_bm25: BM25Okapi | None = None
        self._exclusion_bm25: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build_from_cache(
        self,
        trial_cache_dir: str | Path,
        nct_ids: list[str],
        save_path: str | Path = _DEFAULT_INDEX_PATH,
    ) -> "FieldBM25Retriever":
        """
        Build four BM25 indexes from the diskcache trial store.

        Args:
            trial_cache_dir: path to the diskcache containing trial dicts
            nct_ids: ordered list of NCT IDs to index (typically from bm25_trec2021_index.pkl)
            save_path: where to pickle the built indexes for fast reuse
        """
        import diskcache
        t0 = time.time()
        cache = diskcache.Cache(str(trial_cache_dir))

        title_corpus, summary_corpus, incl_corpus, excl_corpus = [], [], [], []
        valid_ids: list[str] = []

        for nct_id in nct_ids:
            trial = cache.get(nct_id)
            if not trial:
                continue

            title = trial.get("title", "")
            summary = trial.get("brief_summary", "")
            elig = trial.get("eligibility_criteria", "")
            inclusion, exclusion = _split_eligibility(elig)

            title_corpus.append(_tok(title))
            summary_corpus.append(_tok(summary))
            incl_corpus.append(_tok(inclusion))
            excl_corpus.append(_tok(exclusion))
            valid_ids.append(nct_id)

        logger.info("Building field BM25 indexes over %d trials...", len(valid_ids))
        self._nct_ids = valid_ids
        self._title_bm25 = BM25Okapi(title_corpus)
        self._summary_bm25 = BM25Okapi(summary_corpus)
        self._inclusion_bm25 = BM25Okapi(incl_corpus)
        self._exclusion_bm25 = BM25Okapi(excl_corpus)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "nct_ids": self._nct_ids,
                "title": self._title_bm25,
                "summary": self._summary_bm25,
                "inclusion": self._inclusion_bm25,
                "exclusion": self._exclusion_bm25,
            }, f)

        logger.info("FieldBM25 built in %.1fs — saved to %s", time.time() - t0, save_path)
        return self

    def load(self, path: str | Path = _DEFAULT_INDEX_PATH) -> "FieldBM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._nct_ids = data["nct_ids"]
        self._title_bm25 = data["title"]
        self._summary_bm25 = data["summary"]
        self._inclusion_bm25 = data["inclusion"]
        self._exclusion_bm25 = data["exclusion"]
        logger.info("FieldBM25 loaded: %d trials from %s", len(self._nct_ids), path)
        return self

    @classmethod
    def load_or_build(
        cls,
        trial_cache_dir: str | Path,
        nct_ids: list[str],
        index_path: str | Path = _DEFAULT_INDEX_PATH,
    ) -> "FieldBM25Retriever":
        """Load from pkl if it exists; otherwise build and save."""
        inst = cls()
        if Path(index_path).exists():
            return inst.load(index_path)
        return inst.build_from_cache(trial_cache_dir, nct_ids, save_path=index_path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 500,
        weights: dict[str, float] | None = None,
    ) -> list[str]:
        """
        Return top_k NCT IDs scored by field-weighted BM25.

        Args:
            text:    Patient description or retrieval query
            top_k:   Number of candidates to return
            weights: Override FIELD_WEIGHTS if provided
        """
        if not self._nct_ids:
            raise RuntimeError("Index not built. Call load_or_build() first.")

        w = weights or FIELD_WEIGHTS
        tokens = _tok(text)
        if not tokens:
            return []

        # Score each field
        def _scores(bm25: BM25Okapi) -> np.ndarray:
            raw = bm25.get_scores(tokens)
            mx = float(raw.max()) + 1e-9
            return raw / mx  # normalise to [0, 1]

        combined = (
            w.get("title", 3.0)     * _scores(self._title_bm25)
            + w.get("summary", 2.0)   * _scores(self._summary_bm25)
            + w.get("inclusion", 1.5) * _scores(self._inclusion_bm25)
            + w.get("exclusion", 0.3) * _scores(self._exclusion_bm25)
        )

        top_idx = np.argsort(combined)[-top_k:][::-1]
        return [self._nct_ids[i] for i in top_idx]

    def query_inclusion_only(self, text: str, top_k: int = 500) -> list[str]:
        """
        Retrieve using ONLY inclusion criteria text.

        Why: inclusion criteria describe what patients SHOULD HAVE, so they
        directly mirror the patient's conditions and biomarkers.
        Exclusion criteria add noise (they describe what the patient must NOT have).
        Retrieval over inclusion-only systematically avoids false-positive matches
        caused by the exclusion text mentioning a condition the patient has.
        """
        return self.query(text, top_k=top_k, weights={"title": 0, "summary": 0,
                                                        "inclusion": 1.0, "exclusion": 0})

    @property
    def nct_ids(self) -> list[str]:
        return self._nct_ids
