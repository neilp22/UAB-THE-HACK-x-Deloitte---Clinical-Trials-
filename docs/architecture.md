# Architecture

> Status: DRAFT — to be finalized after Phase 1 plan approval

## System Overview

```
Patient Profile (free text)
        │
        ▼
[1] Patient Normalizer          ← deterministic NLP / regex + LLM extraction
        │  structured patient dict (age, gender, conditions, labs, meds...)
        ▼
[2] Query Builder               ← deterministic (MeSH mapping + rule-based)
        │  ClinicalTrials.gov API queries (chained, refined)
        ▼
[3] Retrieval Engine            ← BM25 baseline + semantic re-rank
        │  candidate trial list (NCT IDs + metadata)
        ▼
[4] Criteria Parser             ← LLM (Claude Sonnet, cached per NCT ID)
        │  structured criteria JSON
        ▼
[5] Hard Filter                 ← deterministic only
        │  eliminates trials with exclusion violations
        ▼
[6] Eligibility Reasoner        ← hybrid: deterministic for numeric/date, LLM for semantic
        │  per-criterion: MET / NOT_MET / NEI + justification
        ▼
[7] NEI Question Generator      ← LLM (Claude Sonnet)
        │  clinical question per NEI criterion
        ▼
[8] Scoring & Ranking           ← deterministic scoring function
        │  ranked trial list with scores
        ▼
[9] Dossier Generator           ← deterministic structure + LLM narrative
        │
        ▼
    JSON Output (predictions file + per-trial dossiers)
```

## Module Responsibilities
TBD — filled in after Phase 1 approval
