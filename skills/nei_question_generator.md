# Skill: nei_question_generator

## Purpose
For each NEI criterion verdict, generate the precise clinical question that, if answered,
would resolve the uncertainty. Not a generic "please provide more info" — a specific,
clinically actionable question.

## Input
- criterion text
- patient profile context
- evidence_span from reasoner

## Output
```json
{
  "criterion_id": "inc_1",
  "clinical_question": "What is the patient's most recent eGFR value (mL/min/1.73m²) and date of measurement?"
}
```

## Quality Criteria (for T4 eval)
- Specifies exact data type / unit requested
- References the specific criterion constraint
- Is answerable by a clinician from chart review

## LLM Usage
Allowed. Claude Sonnet. Temperature = 0 for reproducibility.

## Status
[ ] Not implemented
