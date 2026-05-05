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

> Requirements: Python 3.11+, pip, an Anthropic API key

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
# Single patient profile
python scripts/run_agent.py --patient data/examples/patient_001.json

# Batch (non-interactive, TREC-style)
python scripts/run_batch.py --input data/trec/2021/topics.xml --output output/predictions.run

# Evaluate on TREC 2021 (dev set)
python scripts/evaluate.py --year 2021 --run output/predictions.run
```

---

## TREC Benchmark Results

| Task | Metric       | Weight | BM25 Baseline | Our System |
|------|-------------|--------|--------------|------------|
| T1   | Recall@20   | 0.20   | —            | —          |
| T2   | Micro-F1    | 0.30   | —            | —          |
| T3   | NDCG@10     | 0.25   | —            | —          |
| T4   | NEI Q quality | 0.15 | —            | —          |
| T5   | Dossier completeness | 0.10 | —       | —          |
| **Total** | **Score** | **1.00** | —       | —          |

---

## Model Configuration

| Module            | Model              | Version | Temperature |
|------------------|--------------------|---------|-------------|
| Criteria parser  | TBD                | TBD     | 0           |
| Eligibility reasoner | TBD           | TBD     | 0           |
| NEI Q generator  | TBD                | TBD     | 0           |
| Dossier summary  | TBD                | TBD     | 0.3         |

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
