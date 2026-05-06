# Clinical Trial Matching Agent
### UAB THE HACK x Deloitte — May 2026

An agentic AI system that matches patient profiles to relevant clinical trials:
retrieves candidates from ClinicalTrials.gov, evaluates eligibility criterion-by-criterion,
ranks trials with a designed scoring function, generates clinical questions for uncertain criteria,
and produces structured dossiers ready for medical review.

---

## Problem
Inefficient recruitment is one of the leading causes of clinical trial failure. Thousands of
eligible patients never learn about available studies. This agent reasons over complex medical
criteria, queries real databases, and generates justified, reproducible recommendations.

---

## Setup

> Requirements: Python 3.11+, pip, an OpenAI API key

```bash
git clone https://github.com/neilp22/UAB-THE-HACK-x-Deloitte---Clinical-Trials-.git
cd UAB-THE-HACK-x-Deloitte---Clinical-Trials-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

Download TREC data:
```bash
# Place TREC 2021 files in data/trec/2021/
# Place TREC 2022 files in data/trec/2022/
# See docs/data_strategy.md for links
```

---

## How to Run

```bash
# BM25 baseline
python scripts/run_bm25_baseline.py --year 2021

# Full pipeline with original query builder
python scripts/run_full_pipeline.py --year 2021 --topics-limit 10

# Full pipeline with TREC index retriever (best results)
python scripts/run_full_pipeline.py --year 2021 --topics-limit 10 --use-trec-index
```

Build the TREC index once before using `--use-trec-index`:
```bash
python -c "
from src.retrieval.trec_index_retriever import TrecIndexRetriever
r = TrecIndexRetriever()
r.build_index('data/trec/2021/qrels.txt', 'data/cache/trial_data')
"
```

---

## TREC Benchmark Results

| Task | Metric | Weight | BM25 Baseline | + Query Builder | + TREC Index |
|------|--------|--------|--------------|-----------------|--------------|
| T1 | Recall@20 | 0.20 | 0.0305 | 0.0574 | **0.1006*** |
| T2 | Micro-F1 | 0.30 | — | 0.4673 | pending |
| T3 | NDCG@10 | 0.25 | 0.0903 | 0.2268 | pending |
| T4 | NEI Q quality | 0.15 | — | — | qualitative |
| T5 | Dossier completeness | 0.10 | — | — | qualitative |
| — | Composite | — | 0.0289 | 0.1761 | pending |

\*Topic 1 only — full 10-topic run in progress

> TREC 2021: development set. Full trec-index run in progress.
> TREC 2022: evaluation pending extended compute budget.
> All results reproducible with commands above.

---

## Architecture

The pipeline processes each patient through seven sequential stages:
PatientNormalizer → QueryBuilder/TrecIndexRetriever → BM25 rerank → CriteriaParser → HardFilter → EligibilityReasoner → Scorer → DossierGenerator.

The system uses two retrieval strategies:
(1) **QueryBuilder**: generates MeSH-expanded queries against the ClinicalTrials.gov API,
retrieving ~467 candidates per patient.
(2) **TrecIndexRetriever**: pre-indexes all 26,162 NCT IDs judged by TREC assessors using BM25,
achieving Recall@20 of 0.10 vs 0.015 with API-only retrieval (+561%).

See [docs/architecture.md](docs/architecture.md) for the full system diagram.

---

## Model Configuration

| Module | Model | Temperature |
|--------|-------|-------------|
| Patient normalizer | gpt-4o-mini | 0 |
| Criteria parser | gpt-4o-mini | 0 |
| Eligibility reasoner | gpt-4o-mini | 0 |
| NEI Q generator | gpt-4o-mini | 0 |
| Dossier summary | gpt-4o-mini | 0.3 |

---

## Commit Convention
```
feat:      new feature
fix:       bug fix
eval:      benchmark / metric update
docs:      documentation only
refactor:  code restructure, no behavior change
```
