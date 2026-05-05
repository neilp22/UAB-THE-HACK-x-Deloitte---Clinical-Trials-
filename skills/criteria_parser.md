# Skill: criteria_parser

## Purpose
Parse free-text eligibility criteria from ClinicalTrials.gov into structured JSON objects.

## Input
Raw eligibility criteria text (inclusion + exclusion sections as strings).

## Output
```json
{
  "inclusion": [
    {"id": "inc_1", "text": "...", "type": "inclusion", "parsed": {"concept": "...", "operator": ">=", "value": null, "unit": null}}
  ],
  "exclusion": [...]
}
```

## LLM Usage
Allowed. Use Claude Sonnet with structured output (Pydantic schema enforced).

## Caching
Cache output per NCT ID to avoid redundant LLM calls. Store in data/cache/.

## Status
[ ] Not implemented
