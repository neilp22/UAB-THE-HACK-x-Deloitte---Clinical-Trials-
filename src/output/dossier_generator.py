"""
Dossier generator — assembles a structured per-trial patient dossier from
eligibility verdicts, NEI questions, and trial metadata.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.matching.eligibility_reasoner import CriterionVerdict
from src.parsing.patient_normalizer import PatientProfile


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
    verdicts: list[tuple[str, CriterionVerdict]],
    trial_metadata: dict,
    nei_questions: dict[int, str],
) -> Dossier:
    """
    Assemble a Dossier for one trial.

    Args:
        profile: Normalised patient profile.
        verdicts: Ordered list of (criterion_type, CriterionVerdict) from evaluate_trial.
        trial_metadata: Flat trial dict from CT API (nct_id, title, phase, status,
                        brief_summary, eligibility_criteria, ...).
        nei_questions: Mapping of verdict index → generated clinical question
                       (only for NEI verdicts; produced by generate_nei_question).

    Returns:
        Populated Dossier instance.

    TODO: implement — build eligibility_table from verdicts + nei_questions,
    derive overall_recommendation (NOT_ELIGIBLE if any exclusion NOT_MET,
    ELIGIBLE if no exclusion NOT_MET and ≥1 inclusion MET,
    ELIGIBLE_PENDING_CLARIFICATION otherwise),
    populate attention_flags for high-confidence NOT_MET and NEI exclusions,
    compute confidence_score as mean(verdict.confidence).
    """
    raise NotImplementedError
