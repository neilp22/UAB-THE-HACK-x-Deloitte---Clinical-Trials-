"""
Dossier generator — assembles a structured per-trial patient dossier from
eligibility verdicts, NEI questions, and trial metadata.

Zero LLM calls — fully deterministic.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.matching.eligibility_reasoner import CriterionVerdict
from src.parsing.patient_normalizer import PatientProfile

_BRIEF_SUMMARY_MAX = 500


class DossierCriterion(BaseModel):
    criterion: str
    type: Literal["inclusion", "exclusion"]
    verdict: Literal["MET", "NOT_MET", "NEI"]
    justification: str
    clinical_question: str | None = None  # only if NEI


class Dossier(BaseModel):
    nct_id: str
    trial_title: str
    phase: str
    status: str
    brief_summary: str
    eligibility_table: list[DossierCriterion]
    attention_flags: list[str]
    overall_recommendation: Literal[
        "ELIGIBLE",
        "NOT_ELIGIBLE",
        "ELIGIBLE_PENDING_CLARIFICATION",
    ]
    confidence_score: float = Field(ge=0.0, le=1.0)


def generate_dossier(
    profile: PatientProfile,
    verdicts: list[tuple[str, str, CriterionVerdict]],
    trial_metadata: dict,
    nei_questions: dict[str, str],
) -> Dossier:
    """
    Assemble a Dossier for one trial.

    Args:
        profile: Normalised patient profile (unused in logic but kept for
                 future extensions, e.g. patient-specific flags).
        verdicts: List of (criterion_type, criterion_text, CriterionVerdict)
                  in the same order as input criteria.
        trial_metadata: Flat trial dict from CT API — keys used:
                        nct_id, title, phase, status, brief_summary.
        nei_questions: criterion_text → clinical question (from
                       generate_nei_question; only NEI criteria present).

    Returns:
        Populated Dossier instance.
    """
    # Step 1 — eligibility table
    eligibility_table: list[DossierCriterion] = []
    for ctype, ctext, v in verdicts:
        clinical_question = nei_questions.get(ctext) if v.verdict == "NEI" else None
        eligibility_table.append(DossierCriterion(
            criterion=ctext,
            type=ctype,
            verdict=v.verdict,
            justification=v.reasoning,
            clinical_question=clinical_question,
        ))

    # Step 2 — overall recommendation
    has_exclusion_violated = any(
        ctype == "exclusion" and v.verdict == "NOT_MET"
        for ctype, _, v in verdicts
    )
    has_inclusion_met = any(
        ctype == "inclusion" and v.verdict == "MET"
        for ctype, _, v in verdicts
    )
    has_any_nei = any(v.verdict == "NEI" for _, _, v in verdicts)

    if has_exclusion_violated:
        overall_recommendation = "NOT_ELIGIBLE"
    elif has_inclusion_met:
        overall_recommendation = (
            "ELIGIBLE_PENDING_CLARIFICATION" if has_any_nei else "ELIGIBLE"
        )
    else:
        overall_recommendation = "ELIGIBLE_PENDING_CLARIFICATION"

    # Step 3 — attention flags
    attention_flags: list[str] = []
    for ctype, ctext, v in verdicts:
        if ctype == "exclusion" and v.verdict == "NOT_MET":
            attention_flags.append(f"EXCLUSION VIOLATED: {ctext[:80]}")
        elif ctype == "exclusion" and v.verdict == "NEI":
            attention_flags.append(f"EXCLUSION UNCLEAR: {ctext[:80]}")

    inclusion_total = sum(1 for ctype, _, _ in verdicts if ctype == "inclusion")
    inclusion_met = sum(
        1 for ctype, _, v in verdicts if ctype == "inclusion" and v.verdict == "MET"
    )
    if inclusion_total > 0 and inclusion_met / inclusion_total < 0.3:
        attention_flags.append(
            f"LOW INCLUSION MATCH: only {inclusion_met} of {inclusion_total} "
            "inclusion criteria met"
        )

    # Step 4 — confidence score
    resolved = sum(1 for _, _, v in verdicts if v.verdict in ("MET", "NOT_MET"))
    total = max(len(verdicts), 1)
    confidence_score = resolved / total

    # Step 5 — brief summary (truncated)
    raw_summary = trial_metadata.get("brief_summary", "No summary available.")
    brief_summary = (
        raw_summary[:_BRIEF_SUMMARY_MAX] if len(raw_summary) > _BRIEF_SUMMARY_MAX
        else raw_summary
    )

    phase_list = trial_metadata.get("phases") or []
    return Dossier(
        nct_id=trial_metadata.get("nct_id", ""),
        trial_title=trial_metadata.get("title", ""),
        phase=phase_list[0] if phase_list else trial_metadata.get("phase", ""),
        status=trial_metadata.get("status", ""),
        brief_summary=brief_summary,
        eligibility_table=eligibility_table,
        attention_flags=attention_flags,
        overall_recommendation=overall_recommendation,
        confidence_score=round(confidence_score, 4),
    )
