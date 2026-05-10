"""
High-recall retriever: multi-query × multi-field × RRF fusion.

Parts 1–8 of the high-recall retrieval architecture.

Architecture
============

  Patient text
      │
      ▼
  ┌──────────────────────────────┐
  │   Query Decomposition        │  → 5–12 focused queries
  │   (QueryExpander)            │    (condition, biomarker, therapy,
  └──────────────┬───────────────┘     resistance, demographics)
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  ┌───────────┐    ┌──────────────┐
  │ HybridBM25│    │  FieldBM25   │
  │ (existing)│    │  4 indexes   │
  │ +Semantic │    │  title/sum/  │
  │           │    │  incl/excl   │
  └─────┬─────┘    └──────┬───────┘
        │                 │
        │  ┌──────────────┘
        │  │
        ▼  ▼
  ┌──────────────────────┐
  │  Reciprocal Rank      │  RRF(d) = Σ 1/(60 + rank(d))
  │  Fusion (RRF)         │  over all (query × index) lists
  └──────────┬────────────┘
             │
             ▼
  Top-N candidates (default 500)
  → downstream: hard filter → LLM eligibility → scorer

Why this improves Recall@20:
  Current single-query hybrid retrieves trials that share vocabulary with ONE
  representation of the patient. Multi-query union covers different phrasings,
  synonyms, and biomarker variants. At cap=500 and 12 queries × 3 indexes = 36
  ranked lists, the union of unique documents is much larger than any single list.

  Theoretical recall ceiling (union of all retrieved sets) grows with:
    C ≈ 1 - (1 - p)^Q   where p = per-query precision, Q = number of queries
  For p=0.07, Q=12: C ≈ 1 - 0.93^12 ≈ 0.58  (vs 0.07 single query)

Usage
=====
  from src.retrieval.high_recall_retriever import HighRecallRetriever

  r = HighRecallRetriever()   # loads indexes once
  nct_ids = r.retrieve(patient_text, profile, top_k=500)
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np

from src.retrieval.field_bm25 import FieldBM25Retriever
from src.retrieval.query_expander import generate_queries
from src.retrieval.rrf import rrf_fuse_weighted

logger = logging.getLogger(__name__)

_DEFAULT_BM25_INDEX    = "data/cache/bm25_trec2021_index.pkl"
_DEFAULT_FIELD_INDEX   = "data/cache/field_bm25_index.pkl"
_DEFAULT_EMBEDDINGS    = "data/cache/embeddings/trial_embeddings.npy"
_DEFAULT_NCT_IDS_PKL   = "data/cache/embeddings/trial_nct_ids.pkl"
_DEFAULT_TRIAL_CACHE   = "data/cache/trial_data"

# RRF weights: higher = more trusted list
# Semantic lists are a bit more reliable per-query than BM25 field variants
_RRF_WEIGHTS = {
    "hybrid_primary":    2.0,   # full text hybrid (BM25 + semantic), primary query
    "semantic":          1.5,   # semantic-only variants
    "field_weighted":    1.5,   # field-weighted BM25, primary query
    "inclusion_only":    1.2,   # inclusion-criteria BM25
    "field_extra":       0.8,   # field-weighted BM25, secondary queries
    "bm25_extra":        0.6,   # single-field BM25, extra queries
}


class HighRecallRetriever:
    """
    Multi-query multi-field retriever with RRF fusion.

    Loads three indexes at construction time (one-time cost):
      - HybridRetriever (existing BM25 + BioBERT)
      - FieldBM25Retriever (4 field indexes)

    All indexes operate over the same 26,162 TREC-judged NCT IDs.
    """

    def __init__(
        self,
        bm25_index_path: str | Path = _DEFAULT_BM25_INDEX,
        field_index_path: str | Path = _DEFAULT_FIELD_INDEX,
        embeddings_path: str | Path = _DEFAULT_EMBEDDINGS,
        nct_ids_pkl: str | Path = _DEFAULT_NCT_IDS_PKL,
        trial_cache_dir: str | Path = _DEFAULT_TRIAL_CACHE,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        t0 = time.time()

        # ---- existing HybridRetriever ----
        from src.retrieval.semantic_retriever import SemanticRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever

        sem = SemanticRetriever(
            embeddings_path=embeddings_path,
            nct_ids_path=nct_ids_pkl,
            trial_cache_dir=trial_cache_dir,
        )
        self._hybrid = HybridRetriever(sem, bm25_index_path, alpha=alpha, beta=beta)

        # Keep BM25-only access for multi-query BM25 scoring
        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)
        from rank_bm25 import BM25Okapi
        import re
        _STOP = frozenset({
            "a","an","the","and","or","but","in","on","at","to","for","of",
            "with","is","was","are","were","be","been","has","have","had",
            "do","does","did","will","would","could","should","may","might",
            "this","that","these","those","it","its","from","by","as",
        })
        def _tok(text: str) -> list[str]:
            text = text.lower()
            text = re.sub(r"[^\w\s\-]", " ", text)
            return [t for t in text.split() if t not in _STOP and len(t) > 1]

        self._bm25: BM25Okapi = data["bm25"]
        self._bm25_nct_ids: list[str] = data["nct_ids"]
        self._tok = _tok

        # ---- FieldBM25 (build from cache if not yet built) ----
        with open(bm25_index_path, "rb") as f:
            idx_data = pickle.load(f)
        nct_id_list = idx_data["nct_ids"]

        self._field = FieldBM25Retriever.load_or_build(
            trial_cache_dir=trial_cache_dir,
            nct_ids=nct_id_list,
            index_path=field_index_path,
        )

        logger.info(
            "HighRecallRetriever ready in %.1fs — %d trials",
            time.time() - t0,
            len(self._bm25_nct_ids),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bm25_rank(self, query: str, top_k: int) -> list[str]:
        tokens = self._tok(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self._bm25_nct_ids[i] for i in top_idx]

    def _semantic_rank(self, query: str, top_k: int) -> list[str]:
        return self._hybrid.query(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        patient_text: str,
        profile=None,
        top_k: int = 500,
        use_llm_queries: bool = False,
    ) -> list[str]:
        """
        High-recall multi-query retrieval with RRF fusion.

        Args:
            patient_text:  Raw patient description
            profile:       PatientProfile (optional — if None, text-only queries used)
            top_k:         Number of candidates to return (recommend 300–1000)
            use_llm_queries: Add LLM-generated queries (costs ~$0.001/call)

        Returns:
            Ordered list of NCT IDs, highest-recall first (RRF score descending).
        """
        t0 = time.time()

        # ---- Step 1: generate query variants ----
        if profile is not None:
            if use_llm_queries:
                from src.retrieval.query_expander import generate_queries_llm
                queries = generate_queries_llm(profile, patient_text)
            else:
                queries = generate_queries(profile, patient_text)
        else:
            queries = [patient_text]

        # Always include the raw patient text as the primary query
        if patient_text not in queries:
            queries = [patient_text] + queries

        logger.debug("HighRecall: %d queries for retrieval", len(queries))

        # ---- Step 2: retrieve + assign per-list weights for RRF ----
        ranked_lists: list[list[str]] = []
        weights: list[float] = []

        primary_q = queries[0]

        # Primary query: full hybrid (highest trust)
        ranked_lists.append(self._semantic_rank(primary_q, top_k=top_k))
        weights.append(_RRF_WEIGHTS["hybrid_primary"])

        # Primary query: field-weighted BM25
        ranked_lists.append(self._field.query(primary_q, top_k=top_k))
        weights.append(_RRF_WEIGHTS["field_weighted"])

        # Primary query: inclusion-only BM25 (catches biomarker-specific criteria)
        ranked_lists.append(self._field.query_inclusion_only(primary_q, top_k=top_k))
        weights.append(_RRF_WEIGHTS["inclusion_only"])

        # Secondary queries: semantic + BM25
        for q in queries[1:]:
            ranked_lists.append(self._semantic_rank(q, top_k=top_k))
            weights.append(_RRF_WEIGHTS["semantic"])

            ranked_lists.append(self._bm25_rank(q, top_k=top_k))
            weights.append(_RRF_WEIGHTS["bm25_extra"])

            ranked_lists.append(self._field.query(q, top_k=top_k))
            weights.append(_RRF_WEIGHTS["field_extra"])

        # ---- Step 3: RRF fusion ----
        fused = rrf_fuse_weighted(ranked_lists, weights=weights, k=60, top_k=top_k)
        result = [nct_id for nct_id, _ in fused]

        elapsed = time.time() - t0
        unique_before_cap = len({nid for lst in ranked_lists for nid in lst})
        logger.info(
            "HighRecall: %d queries × %d lists → %d unique before cap → top %d (%.2fs)",
            len(queries), len(ranked_lists), unique_before_cap, len(result), elapsed,
        )

        return result

    def get_trial(self, nct_id: str) -> dict | None:
        return self._hybrid.get_trial(nct_id)


# ---------------------------------------------------------------------------
# Part 5: Dense retrieval model recommendations
# ---------------------------------------------------------------------------

DENSE_MODEL_RECOMMENDATIONS = """
## Dense Retrieval Model Recommendations (Part 5)

Current model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Trained: MS-MARCO passage retrieval (general web text)
- Clinical relevance: moderate (biomedical BERT backbone, but IR trained on web queries)

### Recommended Alternatives (best → acceptable)

| Model | Why better | Inference speed | Re-embedding cost | Recommendation |
|-------|-----------|-----------------|-------------------|----------------|
| `ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder` | Trained on PubMed click-through data (clinical queries → abstracts). Dual encoder fine-tuned specifically for biomedical retrieval. Best domain match. | Fast (147M params each) | ~30 min on MPS for 26k trials | **BEST for this task** |
| `BAAI/bge-large-en-v1.5` | State-of-art general-purpose bi-encoder. Outperforms MedCPT on non-clinical benchmarks. May generalise better to new topics. | Medium (335M params) | ~45 min | Good backup |
| `BioLinkBERT-large` | Pre-trained on PubMed + citation links. Better entity-level biomedical understanding. | Medium (340M params) | ~45 min | Good for rare conditions |
| `SapBERT` | Fine-tuned for biomedical entity normalisation. Excellent for biomarker/drug matching. | Fast (110M params) | ~20 min | Best for drug/biomarker facets |
| Current: `S-PubMedBERT-MS-MARCO` | Baseline | Fast | Already done | Baseline |

### Hackathon Recommendation: MedCPT

Reasoning:
1. Trained directly on clinical query → article retrieval (closest to our task)
2. Dual-encoder: separate query and document encoders (better alignment)
3. Available on HuggingFace: `ncbi/MedCPT-Query-Encoder`
4. Re-embedding 26k trials takes ~30 min on M3 MPS — feasible in 1 day
5. Expected Recall@20 improvement: +0.03–0.08 vs current model

To use MedCPT, change `_MODEL_NAME` in semantic_retriever.py and
use asymmetric encoding (query encoder for patient, article encoder for trials).
"""

# ---------------------------------------------------------------------------
# Part 6: Two-stage retrieval design
# ---------------------------------------------------------------------------

TWO_STAGE_DESIGN = """
## Two-Stage Retrieval Design (Part 6)

Stage 1: High-Recall Retrieval (HighRecallRetriever)
  - Multi-query × multi-field × RRF
  - Target: top-500 candidates
  - Latency: ~2–5s per topic (all CPU/MPS, no LLM)
  - Expected Recall@500: 0.35–0.55

Stage 2: Cross-Encoder Reranking
  - Model: `BAAI/bge-reranker-base` (278M params)
    - Why: trained on MS-MARCO + BEIR, fast on CPU, strong biomedical transfer
    - Alternative: `cross-encoder/ms-marco-MiniLM-L-12-v2` (faster, lower quality)
    - NOT recommended: T5-based rerankers (too slow for 500 pairs on CPU)
  - Input: top-500 (patient text, trial title + summary) pairs
  - Output: top-100 reranked candidates
  - Latency: ~8–15s for 500 pairs on M3 MPS
  - Expected Recall@100: 0.40–0.60

Stage 3: LLM Eligibility Reasoning (existing)
  - Input: top-20–30 trials from Stage 2
  - Output: MET/NOT_MET/NEI per criterion
  - Cost: ~$0.03/topic at cap=30

Implementation note:
  The cross-encoder reranker is optional — adding Stage 2 requires installing
  `sentence-transformers` cross-encoder classes (already installed as a dep).

  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder("BAAI/bge-reranker-base")
  scores = reranker.predict([(patient_text, trial["title"] + " " + trial["brief_summary"])
                              for trial in candidates])
"""
