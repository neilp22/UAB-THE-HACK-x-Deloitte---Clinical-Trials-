"""
Trial eligibility scorer.

Converts per-criterion verdicts + trial metadata into:
1. a single float score >= 0.0
2. an explainable score_breakdown dictionary

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

_WEIGHTS: dict[str, float] = {
    "inclusion": 0.55,
    "exclusion": -1.00,
    "phase": 0.15,
    "recruiting": 0.10,
    "nei": -0.10,
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


def score_trial_breakdown(
    verdicts: list[VerdictEntry],
    metadata: dict,
) -> dict:
    """
    Return an explainable score breakdown for a single trial.

    This function keeps the same scoring philosophy as score_trial(),
    but exposes each component so the ranking is auditable.
    """
    typed = [_unpack(e) for e in verdicts]
    total_criteria = len(typed)

    inclusion_verdicts = [v for (t, v) in typed if t == "inclusion"]
    exclusion_verdicts = [v for (t, v) in typed if t == "exclusion"]

    inclusion_met = sum(1 for v in inclusion_verdicts if v.verdict == "MET")
    total_inclusion = len(inclusion_verdicts)
    inclusion_met_ratio = inclusion_met / max(total_inclusion, 1)

    exclusion_penalty = (
        1.0 if any(v.verdict == "NOT_MET" for v in exclusion_verdicts) else 0.0
    )

    nei_count = sum(1 for (_, v) in typed if v.verdict == "NEI")
    nei_ratio = nei_count / max(total_criteria, 1)

    phase_bonus = _PHASE_MAP.get(_norm_phase(metadata.get("phase")), 0.1)
    recruiting_bonus = (
        1.0 if _norm_status(metadata.get("status")) == "RECRUITING" else 0.0
    )

    weighted_inclusion = _WEIGHTS["inclusion"] * inclusion_met_ratio
    weighted_exclusion = _WEIGHTS["exclusion"] * exclusion_penalty
    weighted_phase = _WEIGHTS["phase"] * phase_bonus
    weighted_recruiting = _WEIGHTS["recruiting"] * recruiting_bonus
    weighted_nei_penalty = _WEIGHTS["nei"] * nei_ratio

    raw_score = (
        weighted_inclusion
        + weighted_exclusion
        + weighted_phase
        + weighted_recruiting
        + weighted_nei_penalty
    )

    final_score = max(raw_score, 0.0)

    return {
        "inclusion_met": inclusion_met,
        "total_inclusion": total_inclusion,
        "inclusion_met_ratio": inclusion_met_ratio,
        "exclusion_penalty": exclusion_penalty,
        "nei_count": nei_count,
        "total_criteria": total_criteria,
        "nei_ratio": nei_ratio,
        "phase_bonus": phase_bonus,
        "recruiting_bonus": recruiting_bonus,
        "weighted_inclusion": weighted_inclusion,
        "weighted_exclusion": weighted_exclusion,
        "weighted_phase": weighted_phase,
        "weighted_recruiting": weighted_recruiting,
        "weighted_nei_penalty": weighted_nei_penalty,
        "raw_score": raw_score,
        "final_score": final_score,
    }


def explain_score_breakdown(breakdown: dict) -> str:
    """
    Generate a short deterministic explanation of the score.

    No LLM call is used here.
    """
    reasons: list[str] = []

    inclusion_ratio = breakdown.get("inclusion_met_ratio", 0.0)
    exclusion_penalty = breakdown.get("exclusion_penalty", 0.0)
    nei_ratio = breakdown.get("nei_ratio", 0.0)
    recruiting_bonus = breakdown.get("recruiting_bonus", 0.0)
    phase_bonus = breakdown.get("phase_bonus", 0.0)

    if inclusion_ratio >= 0.7:
        reasons.append("strong inclusion match")
    elif inclusion_ratio >= 0.3:
        reasons.append("partial inclusion match")
    else:
        reasons.append("weak inclusion match")

    if exclusion_penalty > 0:
        reasons.append("possible exclusion violation detected")
    else:
        reasons.append("no exclusion violation detected")

    if recruiting_bonus > 0:
        reasons.append("trial is recruiting")
    else:
        reasons.append("trial is not marked as recruiting")

    if phase_bonus >= 1.0:
        reasons.append("high phase suitability")
    elif phase_bonus >= 0.6:
        reasons.append("moderate phase suitability")
    else:
        reasons.append("low or unclear phase suitability")

    if nei_ratio > 0.4:
        reasons.append("many criteria remain uncertain")
    elif nei_ratio > 0:
        reasons.append("some criteria remain uncertain")

    return "Ranked based on " + ", ".join(reasons) + "."


def score_trial(
    verdicts: list[VerdictEntry],
    metadata: dict,
) -> float:
    """
    Score a single trial.

    Kept for backwards compatibility.
    """
    return score_trial_breakdown(verdicts, metadata)["final_score"]


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


def score_batch_breakdown(
    verdicts_by_nct: dict[str, list[VerdictEntry]],
    metadata_by_nct: dict[str, dict],
) -> dict[str, dict]:
    """
    Score all trials and return nct_id → score_breakdown.
    """
    return {
        nct_id: score_trial_breakdown(verdicts, metadata_by_nct.get(nct_id, {}))
        for nct_id, verdicts in verdicts_by_nct.items()
    }