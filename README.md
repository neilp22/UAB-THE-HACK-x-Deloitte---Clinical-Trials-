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
# BM25 baseline (Day 1)
python scripts/run_bm25_baseline.py --year 2021

# BM25 + Query Builder (Day 2 — recommended)
python scripts/run_bm25_baseline.py --year 2021 --use-query-builder

# Quick smoke test on 5 topics
python scripts/run_bm25_baseline.py --year 2021 --use-query-builder --topics-limit 5
```

---

## TREC Benchmark Results (TREC 2021 dev set, 75 topics)

| Task | Metric       | Weight | BM25 Baseline | + Query Builder | Our System |
|------|-------------|--------|--------------|-----------------|------------|
| T1   | Recall@20   | 0.20   | 0.0305       | **0.0574**      | —          |
| T2   | Micro-F1    | 0.30   | —            | —               | —          |
| T3   | NDCG@10     | 0.25   | 0.0903       | **0.2268**      | —          |
| T4   | NEI Q quality | 0.15 | —            | —               | —          |
| T5   | Dossier completeness | 0.10 | —       | —               | —          |
| **T1+T3** | **Composite** | — | 0.0289 | **0.0682** | —   |

> Day 1 BM25 baseline: 1-2 term query, 200 results/topic.
> Day 2 Query Builder: LLM condition extraction + MeSH synonyms → 467 avg candidates/topic (+133%).
> Recall@20 +88%, NDCG@10 +151% vs Day 1 baseline.

---

## Model Configuration

| Module              | Model         | Temperature |
|--------------------|---------------|-------------|
| Patient normalizer  | gpt-4o-mini   | 0           |
| Criteria parser     | gpt-4o-mini   | 0           |
| Eligibility reasoner | gpt-4o-mini  | 0           |
| NEI Q generator     | gpt-4o-mini   | 0           |
| Dossier summary     | gpt-4o-mini   | 0.3         |

---

## Commit Convention
```
feat:      new feature
fix:       bug fix
eval:      benchmark / metric update
docs:      documentation only
refactor:  code restructure, no behavior change
```

---

## Architecture
See [docs/architecture.md](docs/architecture.md) for the full system diagram.
