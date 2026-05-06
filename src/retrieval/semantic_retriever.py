"""
Semantic retriever using BioBERT embeddings (pritamdeka/S-PubMedBert-MS-MARCO).

Embeddings are L2-normalised → cosine similarity == dot product.
Matrix multiply over 26k trials completes in <1s on CPU/MPS.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import diskcache
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDINGS = "data/cache/embeddings/trial_embeddings.npy"
_DEFAULT_NCT_IDS    = "data/cache/embeddings/trial_nct_ids.pkl"
_DEFAULT_CACHE_DIR  = "data/cache/trial_data"
_MODEL_NAME         = "pritamdeka/S-PubMedBert-MS-MARCO"


class SemanticRetriever:
    def __init__(
        self,
        embeddings_path: str | Path = _DEFAULT_EMBEDDINGS,
        nct_ids_path: str | Path = _DEFAULT_NCT_IDS,
        trial_cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    ) -> None:
        self.embeddings: np.ndarray = np.load(str(embeddings_path))  # (N, 768)
        with open(nct_ids_path, "rb") as f:
            self.nct_ids: list[str] = pickle.load(f)
        self.nct_to_idx: dict[str, int] = {nct: i for i, nct in enumerate(self.nct_ids)}
        self._cache = diskcache.Cache(str(trial_cache_dir))
        self._model: _ST | None = None
        print(f"SemanticRetriever loaded: {len(self.nct_ids)} trials")

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info("Loading %s on %s …", _MODEL_NAME, device)
            self._model = SentenceTransformer(_MODEL_NAME, device=device)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Encode text → L2-normalised (768,) float32 vector."""
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def query(self, patient_text: str, top_k: int = 100) -> list[dict]:
        """
        Return top_k trials ranked by cosine similarity to patient_text.

        Each result: {"nct_id": str, "score": float, "title": str}
        """
        query_vec = self.encode_query(patient_text)               # (768,)
        scores: np.ndarray = self.embeddings @ query_vec          # (N,)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            nct_id = self.nct_ids[idx]
            trial = self._cache.get(nct_id)
            title = trial.get("title", "") if trial is not None else ""
            results.append({
                "nct_id": nct_id,
                "score": float(scores[idx]),
                "title": title,
            })
        return results

    def get_trial(self, nct_id: str) -> dict | None:
        """Return full trial dict from cache, or None if not found."""
        result = self._cache.get(nct_id)
        return result if result is not None else None
