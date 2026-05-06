# CLAUDE.md — Clinical Trial Matching Agent

## Project Goal
Build an agentic AI system that, given a free-text patient profile, retrieves relevant clinical
trials from ClinicalTrials.gov, evaluates eligibility criterion-by-criterion (MET / NOT_MET / NEI),
ranks trials, generates clinical questions for uncertain criteria, and produces a structured
dossier per trial — benchmarked on TREC Clinical Trials 2021 & 2022.

Hackathon: UAB x Deloitte | Deadline: ~2026-05-12

## Tech Stack
- Language: Python 3.11+
- LLM: gpt-4o-mini (all modules, temperature=0 except dossier=0.3)
- Retrieval: HybridRetriever (BioBERT + BM25, default) | TrecIndexRetriever | CT API
- Semantic model: pritamdeka/S-PubMedBert-MS-MARCO (loaded lazily, MPS if available)
- Validation: Pydantic v2
- Eval: pytrec_eval (TREC official)
- Parallelism: ThreadPoolExecutor(max_workers=5) per-trial in pipeline

## Key Files & Folders
```
src/retrieval/    — CT API client, BM25, QueryBuilder, TrecIndex, Semantic, Hybrid
src/parsing/      — Eligibility criteria parser (free-text → structured JSON)
src/matching/     — Hard-filter engine (deterministic) + LLM eligibility reasoner
src/ranking/      — Scoring function, weight tuning
src/output/       — Dossier generator, NEI question generator
data/trec/        — TREC 2021 (dev) and 2022 (validation) datasets
data/cache/       — All caches (gitignored): criteria, reasoner, trial_data, embeddings
docs/             — architecture.md, data_strategy.md
tests/            — Unit/integration tests (one per module)
scripts/          — CLI runners, eval scripts
```

## Evaluation Weights
| Task | Metric       | Weight |
|------|-------------|--------|
| T1   | Recall@20   | 0.20   |
| T2   | Micro-F1    | 0.30   |
| T3   | NDCG@10     | 0.25   |
| T4   | NEI Q quality | 0.15 |
| T5   | Dossier completeness | 0.10 |

TREC 2021 = dev/tuning set. TREC 2022 = validation (touch only on Day 7).

## Rules & Conventions
- Never write app code until the plan is approved by the user
- Always work on feature branches; merge to dev; merge to main only when stable
- Never commit broken code to main
- Never delete or overwrite files without asking
- Keep this file under 200 lines

### Retrieval & Embeddings
- SemanticRetriever loads the sentence-transformer model **lazily** (not at __init__)
- Embeddings live at `data/cache/embeddings/` — **never commit** (gitignored via /data/cache/)
- `trial_embeddings.npy` (77 MB) and `trial_nct_ids.pkl` must stay out of git
- HybridRetriever: alpha=0.5 (semantic) / beta=0.5 (BM25) — tunable, not buried in logic
- diskcache truthiness: always use `cache.get(key) is not None`, never `if cache.get(key)`

### Parallelism
- ThreadPoolExecutor(max_workers=5) for per-trial LLM calls — reduce to 3 on 429 errors
- Pipeline flag: `--retrieval-mode {api,trec-index,semantic,hybrid}` (default: hybrid)
- `--max-workers N` to tune concurrency at runtime

### LLM Usage — Strict Boundaries
**ALLOWED:**
- Parsing free-text eligibility criteria → structured JSON
- Semantic eligibility evaluation (concept matching, temporal logic)
- Generating NEI clinical questions
- Writing dossier narrative summaries

**NEVER ALLOWED:**
- Age comparisons, gender matching
- Numeric threshold checks (lab values, ECOG scores, dates)
- Assigning the final ranking score
- Any operation solvable with a deterministic rule

### Verdict Rules
- `NEI` = not enough info — default when uncertain; never hallucinate eligibility
- Exclusion criteria are hard eliminators — one NOT_MET exclusion disqualifies
- Every NEI verdict must produce a generated clinical question

## Commit Convention
```
feat:      new feature
fix:       bug fix
eval:      benchmark / metric update
docs:      documentation only
refactor:  code restructure, no behavior change
```

## Sub-docs
- [Architecture](docs/architecture.md)
- [Data Strategy](docs/data_strategy.md)
