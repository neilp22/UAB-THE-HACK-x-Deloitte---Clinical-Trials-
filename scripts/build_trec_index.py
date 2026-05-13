#!/usr/bin/env python3
"""
Build (or rebuild) the TREC BM25 index from cached trial data.

Corpus: title + brief_summary only (eligibility_criteria excluded).
Output: data/cache/bm25_trec2021_index.pkl

Must be run from the project root directory.

Usage
-----
  python scripts/build_trec_index.py
  python scripts/build_trec_index.py --year 2022
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CACHE_DIR, TREC_2021_DIR, TREC_2022_DIR
from src.retrieval.trec_index_retriever import TrecIndexRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild TREC BM25 index")
    parser.add_argument("--year", type=int, choices=[2021, 2022], default=2021,
                        help="TREC qrels year to index (default: 2021)")
    args = parser.parse_args()

    trec_dir = TREC_2021_DIR if args.year == 2021 else TREC_2022_DIR
    qrels_path = trec_dir / "qrels.txt"

    if not qrels_path.exists():
        print(f"ERROR: qrels not found at {qrels_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Rebuilding TREC {args.year} BM25 index")
    print(f"  qrels      : {qrels_path}")
    print(f"  trial cache: {CACHE_DIR / 'trial_data'}")
    print(f"  output     : data/cache/bm25_trec{args.year}_index.pkl")
    print()

    TrecIndexRetriever().build_index(
        qrels_path=str(qrels_path),
        cache_dir=str(CACHE_DIR / "trial_data"),
    )


if __name__ == "__main__":
    main()
