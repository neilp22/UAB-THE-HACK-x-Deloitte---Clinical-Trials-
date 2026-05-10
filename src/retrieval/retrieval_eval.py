"""
Retrieval evaluation instrumentation.

Part 8 of the high-recall retrieval architecture.

Measures:
  - Recall@K (K = 20, 50, 100, 200, 500)
  - Candidate pool coverage (how many judged-relevant trials are reachable)
  - Retrieval list overlap (how similar are two retrieval strategies)
  - Retrieval diversity (unique NCT IDs across multiple query lists)

Usage:
  from src.retrieval.retrieval_eval import RetrievalEvaluator, print_recall_table
  ev = RetrievalEvaluator(qrels_path="data/trec/2021/qrels.txt")
  ev.add_run("hybrid",      {tid: nct_ids_for_topic})
  ev.add_run("high_recall", {tid: nct_ids_for_topic})
  print_recall_table(ev)
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load a TREC qrels file → {topic_id: {nct_id: grade}}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            tid, _, nct_id, grade = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(tid, {})[nct_id] = grade
    return qrels


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant documents found in top-K retrieved."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / k


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant)


class RetrievalEvaluator:
    """
    Accumulates per-topic retrieval results across multiple runs and
    computes aggregate recall metrics.
    """

    def __init__(self, qrels_path: str | Path, min_grade: int = 1) -> None:
        """
        Args:
            qrels_path: TREC qrels file path
            min_grade:  Minimum grade to count as relevant (1 = excluded, 2 = eligible).
                        Default 1 means we care about finding ALL judged trials (incl. excluded).
                        Use 2 for eligible-only recall.
        """
        self.qrels = load_qrels(qrels_path)
        self.min_grade = min_grade
        self._runs: dict[str, dict[str, list[str]]] = {}  # run_name → {tid → [nct_ids]}

    def add_run(self, name: str, results: dict[str, list[str]]) -> None:
        """Register a retrieval run. results = {topic_id: [ranked nct_ids]}"""
        self._runs[name] = results

    def evaluate(
        self,
        ks: list[int] | None = None,
        grade: Literal["eligible", "all_judged"] = "eligible",
    ) -> dict[str, dict[str, float]]:
        """
        Compute per-run mean Recall@K and MAP.

        Returns:
            {run_name: {"recall@20": float, "recall@100": float, ..., "map": float}}
        """
        if ks is None:
            ks = [20, 50, 100, 200, 500]

        min_grade = 2 if grade == "eligible" else 1
        results: dict[str, dict[str, float]] = {}

        for run_name, run in self._runs.items():
            per_topic_r: dict[int, list[float]] = defaultdict(list)
            per_topic_ap: list[float] = []

            for tid, retrieved in run.items():
                topic_qrels = self.qrels.get(tid, {})
                relevant = {nct for nct, g in topic_qrels.items() if g >= min_grade}
                if not relevant:
                    continue
                for k in ks:
                    per_topic_r[k].append(recall_at_k(retrieved, relevant, k))
                per_topic_ap.append(average_precision(retrieved, relevant))

            run_result: dict[str, float] = {
                f"recall@{k}": statistics.mean(per_topic_r[k]) if per_topic_r[k] else 0.0
                for k in ks
            }
            run_result["map"] = statistics.mean(per_topic_ap) if per_topic_ap else 0.0
            run_result["n_topics"] = len(per_topic_ap)
            results[run_name] = run_result

        return results

    def coverage(self, run_name: str, k: int = 500) -> dict[str, float]:
        """
        Candidate pool coverage: what fraction of the total judged-relevant
        population is reachable in the top-K candidate pool?

        This is the theoretical upper bound on downstream T1/T3 metrics.
        """
        run = self._runs.get(run_name, {})
        total_relevant = 0
        total_covered = 0
        for tid, retrieved in run.items():
            topic_qrels = self.qrels.get(tid, {})
            relevant = {nct for nct, g in topic_qrels.items() if g >= 2}
            pool = set(retrieved[:k])
            total_relevant += len(relevant)
            total_covered += len(pool & relevant)
        return {
            "coverage": total_covered / max(total_relevant, 1),
            "covered": total_covered,
            "total_relevant": total_relevant,
        }

    def overlap(self, run_a: str, run_b: str, k: int = 100) -> float:
        """
        Jaccard overlap between two retrieval run candidate pools at top-K.
        High overlap = redundant; low overlap = complementary (good for fusion).
        """
        runs = self._runs
        if run_a not in runs or run_b not in runs:
            return 0.0
        a_pools = {tid: set(lst[:k]) for tid, lst in runs[run_a].items()}
        b_pools = {tid: set(lst[:k]) for tid, lst in runs[run_b].items()}
        jaccards = []
        for tid in a_pools:
            if tid in b_pools:
                a, b = a_pools[tid], b_pools[tid]
                jaccards.append(len(a & b) / len(a | b) if (a | b) else 0.0)
        return statistics.mean(jaccards) if jaccards else 0.0

    def diversity(self, run_name: str, k: int = 100) -> float:
        """
        Intra-topic diversity: average fraction of unique NCT IDs across
        the multiple query lists (before RRF fusion). Only applicable if
        the run was produced by a multi-query retriever.

        Currently measured as: unique candidates per topic / k
        (ratio of 1.0 = all different, <1.0 = duplicates across queries)
        """
        run = self._runs.get(run_name, {})
        ratios = []
        for retrieved in run.values():
            unique = len(set(retrieved[:k]))
            ratios.append(unique / k if k > 0 else 0.0)
        return statistics.mean(ratios) if ratios else 0.0


def print_recall_table(evaluator: RetrievalEvaluator, ks: list[int] | None = None) -> None:
    """Print a formatted comparison table of all registered runs."""
    if ks is None:
        ks = [20, 50, 100, 200, 500]

    results = evaluator.evaluate(ks=ks, grade="eligible")

    # Header
    k_headers = "  ".join(f"R@{k:>4}" for k in ks)
    print(f"\n{'Run':<22}  {'Topics':>6}  {k_headers}  {'MAP':>6}")
    print("-" * (22 + 6 + len(ks) * 9 + 8))

    for run_name, metrics in sorted(results.items()):
        k_vals = "  ".join(f"{metrics.get(f'recall@{k}', 0.0):>6.4f}" for k in ks)
        print(
            f"{run_name:<22}  {int(metrics.get('n_topics', 0)):>6}  "
            f"{k_vals}  {metrics.get('map', 0.0):>6.4f}"
        )

    print()

    # Coverage per run at K=500
    print("Candidate pool coverage @ top-500 (eligible trials only):")
    for run_name in evaluator._runs:
        cov = evaluator.coverage(run_name, k=500)
        print(f"  {run_name:<22}  coverage={cov['coverage']:.3f}  "
              f"({cov['covered']}/{cov['total_relevant']} eligible trials in pool)")

    # Pairwise overlap
    run_names = list(evaluator._runs.keys())
    if len(run_names) >= 2:
        print("\nPairwise Jaccard overlap @ top-100:")
        for i in range(len(run_names)):
            for j in range(i + 1, len(run_names)):
                ov = evaluator.overlap(run_names[i], run_names[j], k=100)
                print(f"  {run_names[i]} ↔ {run_names[j]}: {ov:.3f}")


# ---------------------------------------------------------------------------
# Prioritization table (Part 9)
# ---------------------------------------------------------------------------

PRIORITIZATION_TABLE = """
## Retrieval Improvements: Prioritization by Expected Recall Gain (Part 9)

| # | Improvement | Expected Recall@20 Gain | Difficulty | Inference Cost | Priority |
|---|-------------|------------------------|------------|----------------|----------|
| 1 | Multi-query generation (5–12 queries/topic) | +0.04–0.10 | Low | Negligible (deterministic) | **P0** |
| 2 | Increase candidate cap 50→500 (HighRecallRetriever) | +0.03–0.08 | Low | ~$0.10/topic extra LLM | **P0** |
| 3 | Field-weighted BM25 (title×3, inclusion×1.5, exclusion×0.3) | +0.02–0.05 | Low | None | **P0** |
| 4 | Inclusion-only retrieval path | +0.01–0.03 | Low | None | **P0** |
| 5 | RRF fusion over all query×field lists | +0.02–0.04 (on top of above) | Low | None | **P0** |
| 6 | Replace S-PubMedBERT with MedCPT (re-embed 26k trials) | +0.03–0.08 | Medium (30 min re-embed) | None extra | **P1** |
| 7 | LLM query expansion (gpt-4o-mini, 5 extra queries) | +0.01–0.03 | Low | ~$0.001/topic | **P1** |
| 8 | Cross-encoder reranker (BAAI/bge-reranker-base, Stage 2) | +0.01–0.03 on NDCG (reranking, not recall) | Medium | CPU: 8–15s/topic | **P2** |
| 9 | Expand to full 500k CT universe | +0.05–0.15 (more coverage) | High (new index) | CT API time | **P3** |
| 10 | BM25+ (BM25L/BM25+) instead of BM25Okapi | +0.01–0.02 | Low | None | **P2** |

Notes:
- Improvements 1–5 are already implemented in HighRecallRetriever (this PR)
- P0 = implement today; P1 = implement if time allows; P2 = optional; P3 = post-hackathon
- "Recall gain" is additive estimate; actual gain depends on topic distribution
- Cap increase (#2) has LLM cost implications: 500 cap × $0.001/trial × 75 topics = $37.50
  → run eligibility on top-20–30 after Stage-2 reranking, not all 500
"""
