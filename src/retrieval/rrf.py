"""
Reciprocal Rank Fusion (RRF).

Part 2 of the high-recall retrieval architecture.

RRF(d) = Σ_i  1 / (k + rank_i(d))

Properties:
- Robust to score magnitude differences across ranked lists
- Does not require score normalisation
- Naturally down-weights documents ranked low in ALL lists
- Documents not present in a list contribute 0 for that list (not penalised)

Reference: Cormack, Clarke, Buettcher (2009) "Reciprocal Rank Fusion outperforms
Condorcet and individual rank learning methods." SIGIR.
"""
from __future__ import annotations

from collections import defaultdict


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists of NCT IDs using RRF.

    Args:
        ranked_lists: Each inner list is a ranking of NCT IDs (best first).
        k:            RRF smoothing constant (default 60, standard TREC value).
        top_k:        Truncate output to top_k. None = return all.

    Returns:
        List of (nct_id, rrf_score) sorted descending by rrf_score.

    Why k=60:
        Empirically optimal across TREC tasks. Lower k amplifies the importance
        of top-1 positions; higher k gives more uniform weight. k=60 balances
        precision-at-1 gains vs recall breadth.
    """
    scores: dict[str, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank_0based, nct_id in enumerate(ranked):
            rank_1based = rank_0based + 1
            scores[nct_id] += 1.0 / (k + rank_1based)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        fused = fused[:top_k]

    return fused


def rrf_fuse_ids(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[str]:
    """Convenience wrapper — returns only the ordered NCT IDs."""
    return [nct_id for nct_id, _ in rrf_fuse(ranked_lists, k=k, top_k=top_k)]


def rrf_fuse_weighted(
    ranked_lists: list[list[str]],
    weights: list[float],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Weighted RRF: multiply each list's contribution by a weight.

    Useful for down-weighting noisy retrieval lists (e.g. exclusion-criteria BM25).

    RRF_w(d) = Σ_i  w_i / (k + rank_i(d))
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must have the same length")

    scores: dict[str, float] = defaultdict(float)

    for ranked, w in zip(ranked_lists, weights):
        for rank_0based, nct_id in enumerate(ranked):
            rank_1based = rank_0based + 1
            scores[nct_id] += w / (k + rank_1based)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        fused = fused[:top_k]

    return fused
