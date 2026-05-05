# Skill: eligibility_reasoner

## Purpose
Evaluate each structured criterion against patient profile data.
Returns MET / NOT_MET / NEI with justification and evidence span.

## Hard Rules (deterministic — no LLM)
- Age: compare patient age (integer) to threshold
- Gender: exact string match
- Numeric labs/scores: float comparison with unit normalization
- Date-based windows: datetime arithmetic

## LLM Rules (semantic evaluation)
- Medical concept matching (e.g., "severe renal impairment" vs. CKD stage)
- Negation handling in criteria text
- Temporal reasoning beyond simple date math
- Ambiguous synonyms requiring clinical judgment

## Output per criterion
```json
{
  "criterion_id": "inc_1",
  "verdict": "MET|NOT_MET|NEI",
  "justification": "...",
  "evidence_span": "..."
}
```

## NEI Invariant
Every NEI verdict MUST have a corresponding clinical question (see nei_question_generator).

## Status
[ ] Not implemented
