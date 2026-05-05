# Data Strategy

> Status: DRAFT — to be finalized after Phase 1 plan approval

## Data Sources

### ClinicalTrials.gov API
- Endpoint: GET /studies
- No auth required
- Rate limits: TBD — implement exponential backoff
- Cache all responses in data/cache/ (gitignored)
- Key fields: NCT ID, title, phase, status, eligibility criteria, conditions, interventions

### TREC Clinical Trials 2021 & 2022
- data/trec/2021/: topics (patient profiles), qrels (relevance judgments)
- data/trec/2022/: same structure — DO NOT USE FOR TUNING
- Download: https://trec.nist.gov/data/trials.html

### MeSH Ontology
- Used for synonym normalization and concept mapping
- API: https://id.nlm.nih.gov/mesh/
- Cache lookups in data/cache/mesh/

## Caching Strategy
- Parsed criteria: keyed by NCT ID + schema version hash
- API responses: keyed by query hash, TTL = 24h during dev
- LLM outputs: keyed by (NCT ID, patient_hash) — use for eval reproducibility

## TREC Data Usage Rules
- 2021: free to inspect, tune, iterate
- 2022: treat as a black box — run once on Day 7 only
