"""
BM25 retriever over all NCT IDs present in a TREC qrels file.

Build once, query forever:
    r = TrecIndexRetriever()
    r.build_index("data/trec/2021/qrels.txt", "data/cache/trial_data")
    ids = r.query(patient_text, top_k=50)
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import diskcache
import requests
from rank_bm25 import BM25Okapi

from src.config import CT_API_BASE

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}
_BATCH_SIZE = 100
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "this", "that", "these", "those", "it", "its", "from", "by",
    "as", "he", "she", "they", "we", "his", "her", "their", "our",
    "who", "which", "what", "when", "where", "how",
})


def _tokenize(text: str) -> list[str]:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _parse_study(raw: dict) -> dict:
    ps = raw.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    status_mod = ps.get("statusModule", {})
    desc_mod = ps.get("descriptionModule", {})
    elig_mod = ps.get("eligibilityModule", {})
    design_mod = ps.get("designModule", {})
    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "status": status_mod.get("overallStatus", ""),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
        "phases": design_mod.get("phases", []),
    }


def _fetch_batch(nct_ids: list[str], session: requests.Session) -> list[dict]:
    url = f"{CT_API_BASE}/studies"
    params = {
        "filter.ids": ",".join(nct_ids),
        "pageSize": len(nct_ids),
        "format": "json",
    }
    for attempt in range(2):
        try:
            resp = session.get(url, params=params, timeout=60.0)
            if resp.status_code == 429:
                logger.warning("429 rate limit — sleeping 5s")
                time.sleep(5)
                continue
            resp.raise_for_status()
            return resp.json().get("studies", [])
        except requests.RequestException as exc:
            logger.error("Batch fetch failed (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)
    return []


class TrecIndexRetriever:
    def __init__(self) -> None:
        self._nct_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._cache: diskcache.Cache | None = None

    def build_index(self, qrels_path: str | Path, cache_dir: str | Path) -> "TrecIndexRetriever":
        t0 = time.time()
        qrels_path = Path(qrels_path)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Parse unique NCT IDs from qrels
        unique_ids: list[str] = sorted({
            line.split()[2]
            for line in qrels_path.read_text().splitlines()
            if line.strip() and len(line.split()) >= 4
        })
        print(f"[TrecIndex] {len(unique_ids)} unique NCT IDs in qrels")

        self._cache = diskcache.Cache(str(cache_dir))

        # Split into batches, check cache first
        session = requests.Session()
        session.headers.update(_HEADERS)

        fetched = 0
        cache_hits = 0
        failed = 0

        batches = [unique_ids[i:i + _BATCH_SIZE] for i in range(0, len(unique_ids), _BATCH_SIZE)]
        total_batches = len(batches)

        for batch_num, batch in enumerate(batches, 1):
            uncached = [nid for nid in batch if self._cache.get(nid) is None]
            cache_hits += len(batch) - len(uncached)

            if uncached:
                raw_studies = _fetch_batch(uncached, session)
                for raw in raw_studies:
                    trial = _parse_study(raw)
                    nid = trial["nct_id"]
                    if nid:
                        self._cache.set(nid, trial)
                        fetched += 1

                # Any IDs the API didn't return → store empty sentinel so we skip them
                returned_ids = {_parse_study(r)["nct_id"] for r in raw_studies}
                for nid in uncached:
                    if nid not in returned_ids:
                        self._cache.set(nid, {"nct_id": nid, "title": "", "status": "",
                                               "brief_summary": "", "eligibility_criteria": "",
                                               "phases": []})
                        failed += 1

            if batch_num % 50 == 0 or batch_num == total_batches:
                elapsed = time.time() - t0
                print(f"[TrecIndex] batch {batch_num}/{total_batches} — "
                      f"fetched={fetched} cache_hits={cache_hits} failed={failed} "
                      f"elapsed={elapsed:.0f}s")

        # Build BM25 index
        print("[TrecIndex] Building BM25 index...")
        trials = []
        for nid in unique_ids:
            trial = self._cache.get(nid)
            if trial is not None:
                trials.append(trial)

        self._nct_ids = [t["nct_id"] for t in trials]
        corpus = [_tokenize(t["title"] + " " + t["brief_summary"])
                  for t in trials]
        self._bm25 = BM25Okapi(corpus)

        # Save index
        index_path = Path("data/cache/bm25_trec2021_index.pkl")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({"nct_ids": self._nct_ids, "bm25": self._bm25}, f)

        elapsed = time.time() - t0
        print(f"[TrecIndex] Done. trials_indexed={len(self._nct_ids)} "
              f"fetched={fetched} cache_hits={cache_hits} failed={failed} "
              f"elapsed={elapsed:.0f}s")
        print(f"[TrecIndex] Index saved to {index_path}")
        return self

    def load_index(self, index_path: str | Path = "data/cache/bm25_trec2021_index.pkl",
                   cache_dir: str | Path = "data/cache/trial_data") -> "TrecIndexRetriever":
        index_path = Path(index_path)
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self._nct_ids = data["nct_ids"]
        self._bm25 = data["bm25"]
        self._cache = diskcache.Cache(str(cache_dir))
        logger.info("Loaded BM25 index: %d trials", len(self._nct_ids))
        return self

    def query(self, patient_text: str, top_k: int = 50) -> list[str]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")
        import numpy as np
        tokens = _tokenize(patient_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._nct_ids[i] for i in top_indices]

    def get_trial(self, nct_id: str) -> dict | None:
        if self._cache is None:
            return None
        return self._cache.get(nct_id)
