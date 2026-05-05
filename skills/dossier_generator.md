# Skill: dossier_generator

## Purpose
Produce the per-trial structured dossier JSON ready for medical review.

## Output Schema
```json
{
  "nct_id": "NCT...",
  "trial_title": "...",
  "phase": "PHASE3",
  "status": "RECRUITING",
  "brief_summary": "...",
  "eligibility_table": [...],
  "attention_flags": ["..."],
  "overall_recommendation": "ELIGIBLE|NOT_ELIGIBLE|ELIGIBLE_PENDING_CLARIFICATION",
  "confidence_score": 0.73
}
```

## Attention Flags (auto-generated)
- EXCLUSION_VIOLATION: hard disqualifier found
- MULTIPLE_NEI: >3 unresolved criteria
- PHASE_1_ONLY: low priority phase
- NOT_RECRUITING: trial not currently enrolling

## LLM Usage
Allowed for brief_summary narrative. All structured fields deterministic.

## Status
[ ] Not implemented
