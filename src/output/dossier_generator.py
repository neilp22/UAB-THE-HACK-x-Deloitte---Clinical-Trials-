"""Dossier generator.

Assembles a structured per-trial patient dossier from eligibility verdicts,
NEI questions, trial metadata and optional score explanations.

Zero LLM calls — fully deterministic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.matching.eligibility_reasoner import CriterionVerdict
from src.parsing.patient_normalizer import PatientProfile


_BRIEF_SUMMARY_MAX = 500

Recommendation = Literal[
    "ELIGIBLE",
    "NOT_ELIGIBLE",
    "ELIGIBLE_PENDING_CLARIFICATION",
]

FlagCode = Literal[
    "RED_FLAG_EXCLUSION_RISK",
    "AMBER_FLAG_CRITICAL_NEI",
    "BLUE_FLAG_GOOD_MATCH",
    "GREY_FLAG_LOW_EVIDENCE",
]


class DossierCriterion(BaseModel):
    criterion: str
    type: Literal["inclusion", "exclusion"]
    verdict: Literal["MET", "NOT_MET", "NEI"]
    justification: str
    clinical_question: str | None = None


class MissingInformationQuestion(BaseModel):
    criterion: str
    criterion_type: Literal["inclusion", "exclusion"]
    question: str


class EligibilityOverview(BaseModel):
    inclusion_met: int
    inclusion_not_met: int
    inclusion_nei: int
    exclusion_violations: int
    exclusion_nei: int
    total_criteria: int
    confidence_score: float = Field(ge=0.0, le=1.0)


class DossierFlag(BaseModel):
    code: FlagCode
    severity: Literal["red", "amber", "blue", "grey"]
    message: str


class Dossier(BaseModel):
    nct_id: str
    trial_title: str
    phase: str
    status: str

    # Legacy field kept for backwards compatibility.
    brief_summary: str

    # Explicit P7 fields.
    trial_summary: str
    eligibility_overview: EligibilityOverview
    eligibility_table: list[DossierCriterion]
    missing_information_questions: list[MissingInformationQuestion] = Field(
        default_factory=list
    )

    # Legacy attention flags kept for backwards compatibility with old tests.
    attention_flags: list[str]

    # New structured flags.
    structured_flags: list[DossierFlag] = Field(default_factory=list)

    # Ranking explanation can be injected from P3 when available.
    score_explanation: str | None = None

    # Legacy recommendation kept for backwards compatibility.
    overall_recommendation: Recommendation

    # Explicit P7 final recommendation.
    final_recommendation: Recommendation

    confidence_score: float = Field(ge=0.0, le=1.0)


def _truncate_summary(raw_summary: str) -> str:
    return (
        raw_summary[:_BRIEF_SUMMARY_MAX]
        if len(raw_summary) > _BRIEF_SUMMARY_MAX
        else raw_summary
    )


def _build_default_score_explanation(
    recommendation: Recommendation,
    overview: EligibilityOverview,
) -> str:
    if recommendation == "NOT_ELIGIBLE":
        return (
            "The trial is not recommended because at least one exclusion "
            "criterion appears to be violated."
        )

    if recommendation == "ELIGIBLE":
        return (
            "The trial is recommended because inclusion criteria are met, "
            "no exclusion violation is detected, and no clarification is needed."
        )

    if overview.exclusion_nei > 0:
        return (
            "The trial may be eligible, but clarification is required because "
            "at least one exclusion criterion remains uncertain."
        )

    return (
        "The trial may be eligible, but clarification is required because "
        "some eligibility criteria remain unresolved."
    )


def _make_flag(
    code: FlagCode,
    severity: Literal["red", "amber", "blue", "grey"],
    message: str,
) -> DossierFlag:
    return DossierFlag(code=code, severity=severity, message=message)


def generate_dossier(
    profile: PatientProfile,
    verdicts: list[tuple[str, str, CriterionVerdict]],
    trial_metadata: dict,
    nei_questions: dict[str, str],
    score_explanation: str | None = None,
    score_breakdown: dict | None = None,
) -> Dossier:
    """Assemble a structured dossier for one trial.

    Args:
        profile: Normalised patient profile. Kept for future patient-specific flags.
        verdicts: List of (criterion_type, criterion_text, CriterionVerdict).
        trial_metadata: Flat trial dict from CT API.
        nei_questions: criterion_text -> clinical question.
        score_explanation: Optional deterministic ranking explanation from P3.
        score_breakdown: Optional ranking breakdown from P3. Currently not required.

    Returns:
        Populated Dossier instance.
    """

    del profile
    del score_breakdown

    # 1. Eligibility table and missing-information questions.
    eligibility_table: list[DossierCriterion] = []
    missing_information_questions: list[MissingInformationQuestion] = []

    for ctype, ctext, verdict in verdicts:
        clinical_question = nei_questions.get(ctext) if verdict.verdict == "NEI" else None

        eligibility_table.append(
            DossierCriterion(
                criterion=ctext,
                type=ctype,
                verdict=verdict.verdict,
                justification=verdict.reasoning,
                clinical_question=clinical_question,
            )
        )

        if verdict.verdict == "NEI" and clinical_question:
            missing_information_questions.append(
                MissingInformationQuestion(
                    criterion=ctext,
                    criterion_type=ctype,
                    question=clinical_question,
                )
            )

    # 2. Eligibility overview.
    inclusion_met = sum(
        1
        for ctype, _, verdict in verdicts
        if ctype == "inclusion" and verdict.verdict == "MET"
    )
    inclusion_not_met = sum(
        1
        for ctype, _, verdict in verdicts
        if ctype == "inclusion" and verdict.verdict == "NOT_MET"
    )
    inclusion_nei = sum(
        1
        for ctype, _, verdict in verdicts
        if ctype == "inclusion" and verdict.verdict == "NEI"
    )
    exclusion_violations = sum(
        1
        for ctype, _, verdict in verdicts
        if ctype == "exclusion" and verdict.verdict == "NOT_MET"
    )
    exclusion_nei = sum(
        1
        for ctype, _, verdict in verdicts
        if ctype == "exclusion" and verdict.verdict == "NEI"
    )

    total_criteria = len(verdicts)
    resolved = sum(
        1 for _, _, verdict in verdicts if verdict.verdict in ("MET", "NOT_MET")
    )
    confidence_score = resolved / max(total_criteria, 1)

    overview = EligibilityOverview(
        inclusion_met=inclusion_met,
        inclusion_not_met=inclusion_not_met,
        inclusion_nei=inclusion_nei,
        exclusion_violations=exclusion_violations,
        exclusion_nei=exclusion_nei,
        total_criteria=total_criteria,
        confidence_score=round(confidence_score, 4),
    )

    # 3. Recommendation.
    has_exclusion_violated = exclusion_violations > 0
    has_inclusion_met = inclusion_met > 0
    has_any_nei = inclusion_nei > 0 or exclusion_nei > 0

    if has_exclusion_violated:
        recommendation: Recommendation = "NOT_ELIGIBLE"
    elif has_inclusion_met:
        recommendation = (
            "ELIGIBLE_PENDING_CLARIFICATION" if has_any_nei else "ELIGIBLE"
        )
    else:
        recommendation = "ELIGIBLE_PENDING_CLARIFICATION"

    # 4. Legacy attention flags + new structured flags.
    attention_flags: list[str] = []
    structured_flags: list[DossierFlag] = []

    for ctype, ctext, verdict in verdicts:
        if ctype == "exclusion" and verdict.verdict == "NOT_MET":
            attention_flags.append(
                f"RED_FLAG_EXCLUSION_RISK: EXCLUSION VIOLATED: {ctext[:80]}"
            )
            structured_flags.append(
                _make_flag(
                    "RED_FLAG_EXCLUSION_RISK",
                    "red",
                    f"Possible exclusion violation: {ctext[:120]}",
                )
            )

        elif ctype == "exclusion" and verdict.verdict == "NEI":
            attention_flags.append(
                f"AMBER_FLAG_CRITICAL_NEI: EXCLUSION UNCLEAR: {ctext[:80]}"
            )
            structured_flags.append(
                _make_flag(
                    "AMBER_FLAG_CRITICAL_NEI",
                    "amber",
                    f"Critical exclusion information is missing: {ctext[:120]}",
                )
            )

    inclusion_total = sum(1 for ctype, _, _ in verdicts if ctype == "inclusion")
    inclusion_ratio = inclusion_met / max(inclusion_total, 1)

    if inclusion_total > 0 and inclusion_ratio < 0.3:
        attention_flags.append(
            "GREY_FLAG_LOW_EVIDENCE: LOW INCLUSION MATCH: "
            f"only {inclusion_met} of {inclusion_total} inclusion criteria met"
        )
        structured_flags.append(
            _make_flag(
                "GREY_FLAG_LOW_EVIDENCE",
                "grey",
                f"Low inclusion match: {inclusion_met}/{inclusion_total} met.",
            )
        )

    if (
        recommendation == "ELIGIBLE"
        and inclusion_ratio >= 0.7
        and exclusion_violations == 0
        and has_any_nei is False
    ):
        structured_flags.append(
            _make_flag(
                "BLUE_FLAG_GOOD_MATCH",
                "blue",
                "Good match: strong inclusion support and no unresolved criteria.",
            )
        )

    # 5. Summary and score explanation.
    raw_summary = trial_metadata.get("brief_summary", "No summary available.")
    trial_summary = _truncate_summary(raw_summary)

    final_score_explanation = score_explanation or _build_default_score_explanation(
        recommendation,
        overview,
    )

    phase_list = trial_metadata.get("phases") or []

    return Dossier(
        nct_id=trial_metadata.get("nct_id", ""),
        trial_title=trial_metadata.get("title", ""),
        phase=phase_list[0] if phase_list else trial_metadata.get("phase", ""),
        status=trial_metadata.get("status", ""),
        brief_summary=trial_summary,
        trial_summary=trial_summary,
        eligibility_overview=overview,
        eligibility_table=eligibility_table,
        missing_information_questions=missing_information_questions,
        attention_flags=attention_flags,
        structured_flags=structured_flags,
        score_explanation=final_score_explanation,
        overall_recommendation=recommendation,
        final_recommendation=recommendation,
        confidence_score=round(confidence_score, 4),
    )