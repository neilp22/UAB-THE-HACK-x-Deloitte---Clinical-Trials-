#!/usr/bin/env python3
"""
Build the field-weighted BM25 index required for --retrieval-mode high-recall.

Run once before using HighRecallRetriever:
    python scripts/build_field_index.py

The script reads the existing BM25 index (data/cache/bm25_trec2021_index.pkl)
to get the canonical NCT ID list, then loads each trial from the diskcache,
splits inclusion/exclusion criteria, and builds four BM25Okapi indexes:
  title / summary / inclusion_criteria / exclusion_criteria

Output: data/cache/field_bm25_index.pkl (~120 MB, loads in <2s)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR
from src.retrieval.field_bm25 import FieldBM25Retriever


def main() -> None:
    bm25_path = CACHE_DIR / "bm25_trec2021_index.pkl"
    field_path = CACHE_DIR / "field_bm25_index.pkl"
    trial_cache = CACHE_DIR / "trial_data"

    if not bm25_path.exists():
        print(f"ERROR: BM25 index not found at {bm25_path}", file=sys.stderr)
        print("Run TrecIndexRetriever().build_index() first.", file=sys.stderr)
        sys.exit(1)

    if field_path.exists():
        print(f"Field BM25 index already exists at {field_path}")
        print("Delete it and re-run to rebuild.")
        return

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    nct_ids = data["nct_ids"]
    print(f"Building field BM25 index over {len(nct_ids)} NCT IDs...")

    FieldBM25Retriever().build_from_cache(
        trial_cache_dir=trial_cache,
        nct_ids=nct_ids,
        save_path=field_path,
    )
    print(f"Done. Index saved to {field_path}")
    size_mb = field_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
