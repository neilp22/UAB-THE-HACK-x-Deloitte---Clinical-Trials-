"""
Single source of truth for T2 label derivation.

Maps an eligibility_summary dict (raw counts from the pipeline) to
one of: "MET" | "NOT_MET" | "NEI".

All pipeline code that produces T2 predictions must use this function.
"""
from __future__ import annotations


def derive_label(eligibility_summary: dict) -> str:
    """
    Derive T2 eligibility label from raw verdict counts.

    Args:
        eligibility_summary: dict with keys:
            inclusion_met        (int)
            inclusion_not_met    (int)
            inclusion_nei        (int)
            exclusion_violations (int)

    Returns:
        "NOT_MET"  — any confirmed exclusion violation
        "NEI"      — majority uncertain, or weak inclusion match
        "MET"      — clear inclusion evidence, no exclusion violation
    """
    inc_met = eligibility_summary.get("inclusion_met", 0)
    inc_not_met = eligibility_summary.get("inclusion_not_met", 0)
    inc_nei = eligibility_summary.get("inclusion_nei", 0)
    excl_violations = eligibility_summary.get("exclusion_violations", 0)

    inc_total = max(inc_met + inc_not_met + inc_nei, 1)
    inc_ratio = inc_met / inc_total
    nei_ratio = inc_nei / inc_total

    if excl_violations > 0:
        return "NOT_MET"
    if inc_not_met > inc_met:
        return "NOT_MET"
    if nei_ratio > 0.50:
        return "NEI"
    if inc_ratio < 0.30:
        return "NEI"
    if inc_met > 0:
        return "MET"
    return "NEI"
