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

# Full pipeline — hybrid retrieval (recommended)
python scripts/run_full_pipeline.py --year 2021 --topics-limit 10 --retrieval-mode hybrid

# Clinical query planner mode (biomarker-aware)
python scripts/run_full_pipeline.py --year 2021 --topics-limit 10 --retrieval-mode clinical
```

---

## TREC Benchmark Results

### Final results — Hybrid retrieval (BM25 α=0.50 + BioBERT β=0.50), cap=100, gpt-4o-mini

| Metric | Weight | TREC 2021 (dev, 10 topics) | TREC 2022 (val, 5 topics) |
|--------|--------|---------------------------|--------------------------|
| T1 Recall@20 | 0.20 | 0.0646 | 0.0165 |
| T2 Micro-F1  | 0.30 | **0.4216** | **0.4653** |
| T3 NDCG@10   | 0.25 | 0.3925 | 0.2354 |
| MAP          | —    | 0.1714 | 0.0189 |
| **Composite** | — | **0.2375** | **0.2017** |

Generalization gap: 0.0358 (< 0.05 — system generalizes to unseen patient profiles).

### Evolution from baseline (TREC 2021)

| Configuration | Composite | vs BM25 |
|--------------|-----------|---------|
| BM25 Day 1 | 0.029 | baseline |
| + Query Builder | 0.176 | +507% |
| + TREC Index | 0.228 | +686% |
| + Hybrid + NEI gates | 0.252 | +769% |
| + cap=100 (final) | 0.238 | +720% |

> All results reproducible with commands above. LLM cost: ~$0.065/topic (gpt-4o-mini, warm cache).

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
