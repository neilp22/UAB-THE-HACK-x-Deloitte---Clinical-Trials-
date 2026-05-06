"""
Trial eligibility scorer.

Converts per-criterion verdicts + trial metadata into a single float score ≥ 0.0.

Formula
-------
  inclusion_met_ratio = inclusion_met / max(total_inclusion, 1)
  exclusion_penalty   = 1.0 if any exclusion verdict is NOT_MET else 0.0
  nei_ratio           = nei_count / max(total_criteria, 1)
  phase_bonus         = {"PHASE3": 1.0, "PHASE2": 0.6, "PHASE1": 0.3}.get(phase, 0.1)
  recruiting_bonus    = 1.0 if status == "RECRUITING" else 0.0

  score = (
      0.55 * inclusion_met_ratio
    - 1.00 * exclusion_penalty
    + 0.15 * phase_bonus
    + 0.10 * recruiting_bonus
    - 0.10 * nei_ratio
  )
  return max(score, 0.0)

CriterionVerdict carries no criterion type, so callers pass
list[tuple[str, CriterionVerdict]] where the str is "inclusion" | "exclusion".
Plain list[CriterionVerdict] is also accepted (all treated as inclusion).
"""
from __future__ import annotations

from typing import Union

from src.matching.eligibility_reasoner import CriterionVerdict

_PHASE_MAP: dict[str, float] = {
    "PHASE3": 1.0,
    "PHASE2": 0.6,
    "PHASE1": 0.3,
}

# A verdict entry is either a bare CriterionVerdict (→ treated as inclusion)
# or a (type_str, CriterionVerdict) tuple.
VerdictEntry = Union[CriterionVerdict, tuple[str, CriterionVerdict]]


def _norm_phase(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.upper().replace(" ", "").replace("_", "")


def _norm_status(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.upper().strip()


def _unpack(entry: VerdictEntry) -> tuple[str, CriterionVerdict]:
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return "inclusion", entry


def score_trial(
    verdicts: list[VerdictEntry],
    metadata: dict,
) -> float:
    """
    Score a single trial.

    verdicts: list of CriterionVerdict or (type, CriterionVerdict) tuples.
    metadata: dict with keys "phase" and "status" (both optional).
    Returns float ≥ 0.0.
    """
    typed = [_unpack(e) for e in verdicts]
    total_criteria = len(typed)

    inclusion_verdicts = [v for (t, v) in typed if t == "inclusion"]
    exclusion_verdicts = [v for (t, v) in typed if t == "exclusion"]

    inclusion_met = sum(1 for v in inclusion_verdicts if v.verdict == "MET")
    total_inclusion = len(inclusion_verdicts)
    inclusion_met_ratio = inclusion_met / max(total_inclusion, 1)

    exclusion_penalty = 1.0 if any(v.verdict == "NOT_MET" for v in exclusion_verdicts) else 0.0

    nei_count = sum(1 for (_, v) in typed if v.verdict == "NEI")
    nei_ratio = nei_count / max(total_criteria, 1)

    phase_bonus = _PHASE_MAP.get(_norm_phase(metadata.get("phase")), 0.1)
    recruiting_bonus = 1.0 if _norm_status(metadata.get("status")) == "RECRUITING" else 0.0

    raw = (
        0.55 * inclusion_met_ratio
        - 1.00 * exclusion_penalty
        + 0.15 * phase_bonus
        + 0.10 * recruiting_bonus
        - 0.10 * nei_ratio
    )
    return max(raw, 0.0)


def score_batch(
    verdicts_by_nct: dict[str, list[VerdictEntry]],
    metadata_by_nct: dict[str, dict],
) -> dict[str, float]:
    """
    Score all trials. Returns nct_id → score (unsorted).
    NCT IDs absent from metadata_by_nct get an empty metadata dict.
    """
    return {
        nct_id: score_trial(verdicts, metadata_by_nct.get(nct_id, {}))
        for nct_id, verdicts in verdicts_by_nct.items()
    }
