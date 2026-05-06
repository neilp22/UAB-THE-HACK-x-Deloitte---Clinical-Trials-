"""
NEI question generator — produces one specific clinical question for each
criterion that could not be determined from the available patient profile.
"""
from __future__ import annotations

from src.matching.eligibility_reasoner import CriterionVerdict


def generate_nei_question(
    verdict: CriterionVerdict,
    criterion_text: str,
    trial_title: str = "",
) -> str:
    """
    Generate a targeted clinical question for a NEI verdict.

    Args:
        verdict: CriterionVerdict with verdict == "NEI".
        criterion_text: Original eligibility criterion text.
        trial_title: Trial title for context (optional).

    Returns:
        A single specific clinical question to resolve the NEI.

    TODO: implement with LLM call — prompt should ask for one concise
    yes/no or value-eliciting question that would directly resolve the
    criterion, grounded in criterion_text and the reasoning in verdict.
    """
    raise NotImplementedError
