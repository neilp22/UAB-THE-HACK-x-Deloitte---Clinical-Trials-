# Clinical Trial Matching Agent
### UAB THE HACK × Deloitte — Clinical AI Challenge, May 2026

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Model: gpt-4o-mini](https://img.shields.io/badge/LLM-gpt--4o--mini-green.svg)](https://platform.openai.com)
[![Benchmark: TREC Clinical Trials 2021/2022](https://img.shields.io/badge/benchmark-TREC%202021%2F2022-orange.svg)](https://www.trec-cds.org/)

An agentic AI pipeline that, given a free-text patient profile, retrieves relevant clinical trials
from ClinicalTrials.gov, evaluates eligibility criterion-by-criterion (MET / NOT_MET / NEI),
produces a ranked list with justified scores, generates targeted clinical questions for uncertain
criteria, and writes structured per-trial dossiers ready for physician review.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Environment Variables](#environment-variables)
7. [Non-interactive Batch Prediction (Evaluators start here)](#non-interactive-batch-prediction)
8. [Quick Demo](#quick-demo)
9. [Benchmark Runs](#benchmark-runs)
10. [Retrieval Modes](#retrieval-modes)
11. [Pipeline Explained](#pipeline-explained)
12. [Input / Output Format](#input--output-format)
13. [Model Configuration](#model-configuration)
14. [Evaluation Metrics](#evaluation-metrics)
15. [Results](#results)
16. [Test Run Outputs — What's in `data/`](#test-run-outputs)
15. [Error Analysis](#error-analysis)
16. [Limitations](#limitations)
17. [Troubleshooting](#troubleshooting)

---

## Problem Statement

Clinical trial recruitment failure affects 80% of studies and delays drug development by years.
A key bottleneck: oncologists cannot manually cross-reference a patient's full clinical history
against thousands of structured eligibility criteria across hundreds of ongoing trials.

This system automates that process. Given a free-text clinical note (the kind a physician would
write), it:

- Extracts structured clinical facts (age, diagnoses, labs, biomarkers, prior treatments)
- Retrieves up to 250 candidate trials from a 26,000-trial TREC corpus
- Parses each trial's raw eligibility text into structured inclusion/exclusion criteria
- Evaluates every criterion individually — deterministically for age/gender/labs, via LLM otherwise
- Ranks trials by a weighted eligibility score
- Flags information gaps with targeted clinical questions
- Outputs per-trial dossiers with full audit trail

---

## Architecture Overview

```
Free-text Patient Note
        │
        ▼
┌─────────────────────┐
│  PatientNormalizer  │  Extracts: age, gender, conditions, labs, ECOG,
│  (LLM + regex)      │  medications, biomarkers, prior treatments
└────────┬────────────┘
         │  PatientProfile (Pydantic)
         ▼
┌─────────────────────┐
│  HighRecallRetriever│  Multi-query × FieldBM25 × HybridBERT × RRF(k=60)
│  (Index: 26k trials)│  → top-250 candidate NCT IDs
└────────┬────────────┘
         │  list[trial_dict]
         ▼
┌─────────────────────┐    per trial (parallel, max_workers=5)
│  CriteriaParser     │  ───────────────────────────────────────
│  (LLM, batched)     │  Raw eligibility text → Criterion[]
└────────┬────────────┘  (type, category, deterministic, threshold)
         ▼
┌─────────────────────┐
│  HardFilter         │  Age / Gender / Lab / ECOG — rule-based only
│  (deterministic)    │  One exclusion violation → trial eliminated
└────────┬────────────┘
         │  survivors
         ▼
┌─────────────────────┐
│ EligibilityReasoner │  MET / NOT_MET / NEI per criterion
│ (LLM, concurrent)   │  Deterministic for structured fields; LLM for rest
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Scorer             │  Weighted formula (no LLM)
│  (deterministic)    │  score = 0.55·inc_ratio − 1.00·excl_pen + …
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  DossierGenerator   │  Per-trial summary + NEI questions (LLM)
│  + NEI Q Generator  │  Fully auditable output
└────────┬────────────┘
         ▼
   Ranked Trial List + JSON Predictions + TREC Run File + Dossiers
```

**Key design principles:**
- Deterministic-first: age/gender/lab checks use pure rules; LLM only where necessary
- Conservative NEI default: never hallucinate eligibility; flag uncertainty explicitly
- Full auditability: every verdict includes reasoning + confidence score
- Caching by content hash: identical input always produces identical output across runs

---

## Technology Stack

| Layer | Tool / Library | Version |
|---|---|---|
| Language | Python | 3.11+ |
| LLM | OpenAI gpt-4o-mini | API v1 |
| Semantic retrieval | pritamdeka/S-PubMedBert-MS-MARCO | sentence-transformers 2.7+ |
| BM25 retrieval | rank-bm25 (BM25Okapi) | 0.2.2+ |
| Data validation | Pydantic | v2.5+ |
| Disk caching | diskcache | 5.6.3+ |
| TREC evaluation | pytrec_eval-terrier | 0.5.9+ |
| Classification metrics | scikit-learn | 1.4+ |
| HTTP client | requests (NOT httpx — Cloudflare blocks httpx) | 2.31+ |
| Accelerator | torch (MPS on Apple Silicon) | 2.3+ |

---

## Repository Structure

```
.
├── src/
│   ├── config.py                   # All paths, constants, env loading
│   ├── llm_client.py               # OpenAI / Gemini wrapper + usage tracking
│   ├── parsing/
│   │   ├── patient_normalizer.py   # Free-text → PatientProfile
│   │   └── criteria_parser.py      # Raw criteria text → Criterion[]
│   ├── retrieval/
│   │   ├── ct_client.py            # ClinicalTrials.gov API v2 client
│   │   ├── bm25_retriever.py       # BM25Okapi on live trial fetch
│   │   ├── trec_index_retriever.py # Pre-built BM25 on 26k TREC corpus
│   │   ├── semantic_retriever.py   # BioBERT cosine similarity (26k embeddings)
│   │   ├── hybrid_retriever.py     # BM25 (α=0.5) + BioBERT (β=0.5) fusion
│   │   ├── high_recall_retriever.py# Multi-query × FieldBM25 × RRF (primary)
│   │   ├── field_bm25.py           # Field-weighted BM25 (title=3, inc=1.5, exc=0.3)
│   │   ├── rrf.py                  # Reciprocal Rank Fusion (k=60)
│   │   ├── query_builder.py        # MeSH expansion + multi-variant query builder
│   │   ├── query_expander.py       # Deterministic multi-facet query generation
│   │   ├── llm_query_expander.py   # LLM-based MeSH-compatible term expansion
│   │   └── clinical_query_planner.py # Biomarker-aware structured query planning
│   ├── matching/
│   │   ├── hard_filter.py          # Deterministic exclusion filter (no LLM)
│   │   ├── eligibility_reasoner.py # Per-criterion LLM verdict + deterministic rules
│   │   ├── label_deriver.py        # Verdict counts → MET/NOT_MET/NEI label (T2)
│   │   └── ...
│   ├── ranking/
│   │   └── scorer.py               # Weighted score formula + breakdown explainer
│   └── output/
│       ├── dossier_generator.py    # Structured per-trial dossier (deterministic)
│       └── nei_question_generator.py # Clinical question per NEI verdict (LLM)
├── scripts/
│   ├── run_full_pipeline.py        # ★ Main entry point — TREC benchmark runner
│   ├── demo_pipeline.py            # Interactive demo (5 fictional patients)
│   ├── run_bm25_baseline.py        # BM25-only baseline (no LLM)
│   ├── build_trec_index.py         # One-time: builds BM25 index from TREC corpus
│   ├── build_field_index.py        # One-time: builds field-weighted BM25 index
│   ├── evaluate.py                 # Runs 2021+2022, reports metrics + cost
│   ├── recompute_metrics.py        # Re-evaluates from existing run files (free)
│   └── tune_weights.py             # Grid-searches scorer weights
├── tests/                          # One test file per module (pytest)
├── data/
│   ├── trec/
│   │   ├── 2021/topics.xml         # 75 patient profiles (dev set)
│   │   ├── 2021/qrels.txt          # 35,832 relevance judgments
│   │   ├── 2022/topics.xml         # Validation set
│   │   └── 2022/qrels.txt
│   ├── runs/                       # TREC run files (output)
│   ├── predictions/                # JSON predictions (output)
│   └── cache/                      # All LLM + API caches (gitignored)
│       ├── bm25_trec2021_index.pkl # Pre-built BM25 index (19 MB)
│       ├── field_bm25_index.pkl    # Field-weighted index (55 MB)
│       ├── embeddings/
│       │   ├── trial_embeddings.npy # BioBERT vectors (77 MB, 26k trials × 768)
│       │   └── trial_nct_ids.pkl
│       ├── criteria_parser/        # Cached parsed criteria
│       ├── eligibility_reasoner/   # Cached verdicts
│       └── query_builder/          # Cached API responses
├── docs/
│   ├── architecture.md
│   └── data_strategy.md
├── requirements.txt
└── CLAUDE.md                       # Developer guidance for AI assistant
```

---

## Installation

### Prerequisites

- Python 3.11+
- An OpenAI API key (gpt-4o-mini)
- ~2 GB disk for indexes and embeddings

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/neilp22/UAB-THE-HACK-x-Deloitte---Clinical-Trials-.git
cd UAB-THE-HACK-x-Deloitte---Clinical-Trials-

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-proj-...

# 5. Download TREC data (required for benchmark evaluation)
# Place files at:
#   data/trec/2021/topics.xml
#   data/trec/2021/qrels.txt
#   data/trec/2022/topics.xml
#   data/trec/2022/qrels.txt
# Download from: https://www.trec-cds.org/2021.html

# 6. Build retrieval indexes (one-time, ~10 minutes)
python scripts/build_trec_index.py --year 2021
python scripts/build_field_index.py

# 7. (Optional) Pre-compute BioBERT embeddings for all 26k trials
# Requires sentence-transformers + ~77 MB disk + ~20 min on GPU
# If you already have trial_embeddings.npy, skip this step
```

### Verify installation

```bash
python -c "from src.llm_client import verify_connection; verify_connection()"
# Expected: ✓ LLM connection verified (gpt-4o-mini)
```

---

## Environment Variables

Create a `.env` file in the repository root:

```env
# Required
OPENAI_API_KEY=sk-proj-...         # OpenAI API key

# Optional — override defaults
OPENAI_MODEL=gpt-4o-mini           # default: gpt-4o-mini
OPENAI_TEMPERATURE=0               # default: 0 (deterministic)
OPENAI_MAX_TOKENS=2048             # default: 2048

# Optional — Gemini fallback (free tier)
GEMINI_KEY=...                     # if set, overrides OpenAI for all LLM calls

# Optional — ClinicalTrials.gov API tuning
CT_PAGE_SIZE=200                   # default: 200
CT_MAX_RESULTS=1000                # default: 1000
CT_POLITE_DELAY=0.3                # seconds between API calls (default: 0.3)
```

---

## Non-interactive Batch Prediction

> **This is the entry point for evaluators.** No TREC data required. No interactive prompts.

### Step 1 — Prepare your patient list

Create a JSON file with one object per patient:

```json
[
  {"id": "P001", "text": "Patient is a 45-year-old man with anaplastic astrocytoma..."},
  {"id": "P002", "text": "58-year-old woman with NSCLC stage IV, EGFR exon 19 deletion..."},
  {"id": "P003", "text": "67-year-old male with metastatic colorectal cancer, KRAS WT..."}
]
```

A ready-to-use example is included at `data/example_patients.json`.

Also accepted: plain text file, one patient note per line (IDs auto-assigned P001, P002…).

### Step 2 — Run

```bash
python scripts/predict.py --input patients.json
```

That's it. The script is fully non-interactive.

### Step 3 — Collect results

| Output | Path |
|---|---|
| **Ranked predictions (JSON)** | `data/predictions/batch_predictions.json` |
| **Per-trial dossiers** | `data/dossiers/batch/<patient_id>/<nct_id>.json` |

### Full options

```bash
# Use the included example patients
python scripts/predict.py --input data/example_patients.json

# Custom output path
python scripts/predict.py --input patients.json --output results/my_predictions.json

# Faster run with fewer candidates (cheaper, good for testing)
python scripts/predict.py --input patients.json --candidate-cap 50

# Change retrieval mode (default: high-recall)
python scripts/predict.py --input patients.json --retrieval-mode hybrid

# Cap LLM spend (stops processing if cost exceeds limit)
python scripts/predict.py --input patients.json --budget 2.0

# Disable caching (forces full re-evaluation)
python scripts/predict.py --input patients.json --no-cache
```

### Output format

```json
[
  {
    "patient_id": "P001",
    "ranked_trials": [
      {
        "rank": 1,
        "nct_id": "NCT00362570",
        "score": 0.325,
        "score_breakdown": {
          "inclusion_met": 2,
          "total_inclusion": 4,
          "inclusion_met_ratio": 0.5,
          "exclusion_penalty": 0.0,
          "nei_count": 1,
          "phase_bonus": 0.6,
          "recruiting_bonus": 1.0,
          "final_score": 0.325
        },
        "score_explanation": "Ranked based on partial inclusion match, no exclusion violation detected, trial is recruiting, moderate phase suitability.",
        "title": "Temozolomide in Patients With Newly Diagnosed Anaplastic Oligodendroglioma",
        "phase": "PHASE2",
        "status": "RECRUITING",
        "eligibility_summary": {
          "inclusion_met": 2,
          "inclusion_not_met": 1,
          "inclusion_nei": 1,
          "exclusion_violations": 0
        },
        "dossier_path": "data/dossiers/batch/P001/NCT00362570.json"
      }
    ]
  }
]
```

### Cost and runtime (reference)

| Setup | Candidates | Patients | Time | LLM cost |
|---|---|---|---|---|
| Example (3 patients, cap=50, high-recall) | 50/patient | 3 | ~20 min | ~$0.21 |
| Single topic benchmark (cap=250, high-recall) | 250 | 1 | ~7 min | ~$0.09 |
| Full TREC 2021 (75 topics, cap=250) | 250/topic | 75 | ~9h | ~$6–7 |

> First run without cache is 3–5× slower. Subsequent runs reuse cached LLM outputs.

---

## Quick Demo

Run an end-to-end pipeline on pre-built fictional patients without TREC data:

```bash
# Patient 1 only (Maria García, 58F, NSCLC EGFR+), cap=10 trials, ~1 min
yes '' | python scripts/demo_pipeline.py --patients 1 --cap 10 --mode hybrid

# All 5 demo patients, full cap=50, ~15 min
yes '' | python scripts/demo_pipeline.py --patients 5 --cap 50 --mode hybrid
```

The `yes ''` auto-answers interactive pauses. Remove it to step through each phase manually.

**Demo patients:**
1. Maria García — 58F, NSCLC stage IV, EGFR exon 19 del, erlotinib failure
2. Joan Martínez — 67M, mCRC, oxaliplatin/irinotecan failure, KRAS WT
3. Ana Puig — 45F, HR+/HER2- breast cancer, CDK4/6i failure
4. Carles Soler — 72M, mCRPC, post-docetaxel, PSA rising
5. Laura Vidal — 34F, AML relapsed/refractory, FLT3-ITD+

---

## Benchmark Runs

All commands assume TREC data is in `data/trec/` and indexes are built.

```bash
# ─── BM25 baseline (no LLM, fast, free) ───────────────────────────────────
python scripts/run_bm25_baseline.py --year 2021

# ─── Full pipeline — recommended production command ────────────────────────
python scripts/run_full_pipeline.py --year 2021 --retrieval-mode high-recall

# ─── Single-topic debug (fast, cheap ~$0.09) ──────────────────────────────
python scripts/run_full_pipeline.py --topic 1 --retrieval-mode high-recall

# ─── First 5 topics smoke test ────────────────────────────────────────────
python scripts/run_full_pipeline.py --topics-limit 5 --retrieval-mode high-recall

# ─── Specific topics ──────────────────────────────────────────────────────
python scripts/run_full_pipeline.py --topics 3,7,12 --retrieval-mode high-recall

# ─── Validate on TREC 2022 (unseen patients) ─────────────────────────────
python scripts/run_full_pipeline.py --year 2022 --retrieval-mode high-recall

# ─── No-cache run (forces full re-fetch, expensive) ───────────────────────
python scripts/run_full_pipeline.py --topic 1 --retrieval-mode high-recall --no-cache

# ─── Recompute metrics from existing run files (zero API cost) ────────────
python scripts/recompute_metrics.py

# ─── Tune scorer weights via grid search ─────────────────────────────────
python scripts/tune_weights.py
```

**Key flags for `run_full_pipeline.py`:**

| Flag | Default | Description |
|---|---|---|
| `--year {2021,2022}` | 2021 | TREC dataset year |
| `--retrieval-mode MODE` | combined | See [Retrieval Modes](#retrieval-modes) |
| `--topic N` | all | Run single topic N only |
| `--topics N,M,...` | all | Run specific topic IDs |
| `--topics-limit N` | all | Run first N topics |
| `--no-cache` | off | Disable disk cache (re-fetches everything) |
| `--budget USD` | 1.75 | Stop if estimated LLM cost exceeds this |
| `--max-workers N` | 5 | ThreadPoolExecutor concurrency per topic |
| `--candidate-cap N` | 250 | Max trials retrieved per topic |

---

## Retrieval Modes

| Mode | Strategy | Recall | Speed | Notes |
|---|---|---|---|---|
| `api` | Live CT API + MeSH queries | Low | Medium | Production use, no index needed |
| `trec-index` | Pre-built BM25 on 26k TREC corpus | Medium | Fast | Requires `build_trec_index.py` |
| `semantic` | BioBERT cosine similarity (26k embeddings) | Medium | Slow | Requires embeddings |
| `hybrid` | BM25 α=0.5 + BioBERT β=0.5 | High | Medium | Recommended for most runs |
| `clinical` | Biomarker-aware structured query planning | High | Medium | Best for biomarker-driven patients |
| `combined` | CT API + clinical planner union | High | Slow | Default |
| `high-recall` | Multi-query × FieldBM25 × BioBERT × RRF | Highest | Medium | **Best for T3/T1; recommended** |

**`high-recall` detail:** Generates 5–13 queries per patient (condition variants, biomarker queries,
therapy combinations, demographics), retrieves from HybridBERT + 4 FieldBM25 indexes, fuses
results via Reciprocal Rank Fusion (k=60) with per-list weights. 2427 unique candidates before
capping to 250.

---

## Pipeline Explained

### Stage 1 — Patient Normalization (`src/parsing/patient_normalizer.py`)

Input: free-text clinical note.

1. **Deterministic extractors (no LLM, regex):**
   - Age: patterns like "45-year-old", "45 yo", "age 45"
   - Gender: word-boundary regex for "male"/"female"
   - ECOG: patterns for "ECOG 1", "PS 2", "performance status 0"
   - Lab values: creatinine, hemoglobin, bilirubin, ALT, AST, WBC, platelets, PSA, eGFR

2. **LLM extraction (gpt-4o-mini, temp=0, cached by MD5):**
   - Conditions/diagnoses, medications, prior treatments, relevant history (mutations, staging, biomarkers)
   - Fallback: empty lists on double failure (never crashes)

Output: `PatientProfile` (Pydantic v2 model).

### Stage 2 — Candidate Retrieval (`src/retrieval/`)

In `high-recall` mode:
1. Query decomposition: condition synonyms, biomarker-specific terms, therapy combinations (13 queries for typical oncology patient)
2. FieldBM25 on 4 fields: title (weight=3.0), summary (2.0), inclusion criteria (1.5), exclusion criteria (0.3)
3. HybridRetriever: BioBERT semantic (α=0.5) + BM25 (β=0.5)
4. RRF fusion across all ranked lists (k=60), weighted by source reliability
5. Top-250 unique NCT IDs returned

> **Why requests, not httpx?** ClinicalTrials.gov API uses Cloudflare WAF which blocks httpx via TLS fingerprint. Only `requests` works reliably.

### Stage 3 — Criteria Parsing (`src/parsing/criteria_parser.py`)

Input: raw `eligibility_criteria` text from CT API.

1. Regex split into inclusion/exclusion sections
2. Extract individual items (deterministic)
3. LLM batch classification (batch size=10, threshold=3000 tokens):
   - Each criterion → `{type, category, deterministic, numeric_threshold, numeric_unit, numeric_operator}`
   - 6 categories: `age, gender, lab_value, prior_treatment, diagnosis, other`
4. Cached by MD5(criteria_text)

### Stage 4 — Hard Filter (`src/matching/hard_filter.py`)

Evaluates only `deterministic=True` exclusion criteria. No LLM.

- **Age:** float comparison against threshold
- **Gender:** word-boundary regex match
- **Lab value:** unit-aware numeric comparison
- **ECOG:** numeric comparison

One violated exclusion → trial eliminated (score = -0.5). Doubt → pass (conservative).

### Stage 5 — Eligibility Reasoning (`src/matching/eligibility_reasoner.py`)

Runs concurrently across trials (ThreadPoolExecutor, max_workers=5).

For each surviving trial:
1. Deterministic criteria → rule-based verdict (confidence=1.0)
2. Non-deterministic criteria → LLM batch evaluation
   - Prompt: patient profile JSON + numbered criteria list
   - Returns: `{verdict: MET|NOT_MET|NEI, confidence: 0.0-1.0, reasoning: str}` per criterion
   - Cached by MD5(nct_id | profile_json)

### Stage 6 — Scoring (`src/ranking/scorer.py`)

No LLM. Fully deterministic:

```
inclusion_met_ratio = inclusion_met / max(total_inclusion, 1)
exclusion_penalty   = 1.0 if any exclusion NOT_MET else 0.0
nei_ratio           = nei_count / max(total_criteria, 1)
phase_bonus         = {PHASE3: 1.0, PHASE2: 0.6, PHASE1: 0.3, other: 0.1}
recruiting_bonus    = 1.0 if status == "RECRUITING" else 0.0

score = 0.55 * inclusion_met_ratio
      - 1.00 * exclusion_penalty
      + 0.15 * phase_bonus
      + 0.10 * recruiting_bonus
      - 0.10 * nei_ratio
final_score = max(score, 0.0)
```

### Stage 7 — Output Generation

**DossierGenerator** (`src/output/dossier_generator.py`) — fully deterministic, no LLM:
- Per-trial eligibility table (one row per criterion, with verdict + reasoning)
- Attention flags (exclusion violations, low inclusion match, unclear exclusions)
- Overall recommendation: `ELIGIBLE` / `NOT_ELIGIBLE` / `ELIGIBLE_PENDING_CLARIFICATION`
- Confidence score: resolved_verdicts / total_verdicts

**NEI Question Generator** (`src/output/nei_question_generator.py`) — LLM (gpt-4o-mini, temp=0):
- One specific, measurable clinical question per NEI verdict
- Cached by MD5(criterion_text + trial_title)
- Fallback: "Does the patient meet the following criterion: {criterion_text}?"

---

## Input / Output Format

### Input: Patient Note (free text)

```
Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine
complicated by severe lower extremity weakness and urinary retention s/p Foley catheter,
high-dose steroids, hypertension, and hyperlipidemia. He is currently on levetiracetam,
dexamethasone, oxycodone, terazosin. Performance status ECOG 2.
Creatinine 1.1 mg/dL, ALT 38 U/L.
```

### Output: JSON Predictions

```json
[
  {
    "patient_id": "1",
    "ranked_trials": [
      {
        "rank": 1,
        "nct_id": "NCT00613093",
        "score": 0.536,
        "score_breakdown": {
          "inclusion_met": 5,
          "total_inclusion": 5,
          "inclusion_met_ratio": 1.0,
          "exclusion_penalty": 0.0,
          "nei_count": 0,
          "phase_bonus": 1.0,
          "recruiting_bonus": 1.0,
          "final_score": 0.536
        },
        "score_explanation": "Ranked based on strong inclusion match, no exclusion violation detected, trial is recruiting, high phase suitability.",
        "title": "Phase 3 Trial of Radiation Therapy...",
        "phase": "PHASE3",
        "status": "RECRUITING",
        "eligibility_summary": {
          "inclusion_met": 5,
          "inclusion_not_met": 0,
          "inclusion_nei": 0,
          "exclusion_violations": 0
        },
        "dossier_path": "data/dossiers/2021/1/NCT00613093.json"
      }
    ]
  }
]
```

### Output: TREC Run File (standard format)

```
1 Q0 NCT00613093 1 0.536000 clinical_agent
1 Q0 NCT00876499 2 0.498000 clinical_agent
...
```

### Output: Per-Trial Dossier (JSON)

```json
{
  "nct_id": "NCT00613093",
  "trial_title": "Phase 3 Trial of Radiation Therapy...",
  "phase": "PHASE3",
  "status": "RECRUITING",
  "overall_recommendation": "ELIGIBLE_PENDING_CLARIFICATION",
  "confidence_score": 0.85,
  "attention_flags": ["1 NEI verdict requires clinical clarification"],
  "eligibility_table": [
    {
      "criterion_type": "inclusion",
      "criterion_text": "Age 18 years or older",
      "verdict": "MET",
      "confidence": 1.0,
      "justification": "Patient is 45 years old.",
      "clinical_question": null
    },
    {
      "criterion_type": "inclusion",
      "criterion_text": "Prior temozolomide-based chemotherapy",
      "verdict": "NEI",
      "confidence": 0.0,
      "justification": "No mention of temozolomide in patient record.",
      "clinical_question": "Has the patient received prior temozolomide-based chemotherapy for their astrocytoma?"
    }
  ]
}
```

---

## Model Configuration

| Module | Model | Temperature | Notes |
|---|---|---|---|
| Patient normalizer | gpt-4o-mini | 0 | Deterministic extraction |
| Criteria parser | gpt-4o-mini | 0 | Deterministic classification |
| Eligibility reasoner | gpt-4o-mini | 0 | Conservative, consistent |
| NEI Q generator | gpt-4o-mini | 0 | Precise clinical questions |
| Dossier summary | gpt-4o-mini | 0.3 | Slight variation for fluency |
| Semantic retrieval | S-PubMedBert-MS-MARCO | — | Pre-computed embeddings |

**Gemini fallback:** Set `GEMINI_KEY` in `.env` to route all LLM calls through `gemini-2.0-flash`
(free tier). Useful for cost reduction but not validated for accuracy parity.

---

## Evaluation Metrics

| Metric | Task | Weight | Description |
|---|---|---|---|
| Recall@20 | T1 | 0.20 | Fraction of judged-relevant trials in top-20 |
| Micro-F1 | T2 | 0.30 | Classification accuracy: MET/NOT_MET/NEI |
| NDCG@10 | T3 | 0.25 | Normalized discounted cumulative gain at rank 10 |
| MAP | — | — | Mean average precision (retrieval quality) |
| **Composite** | — | — | `0.20·T1 + 0.30·T2 + 0.25·T3` |

**T2 label derivation** (`src/matching/label_deriver.py`):
```
exclusion_violations > 0                → NOT_MET
inclusion_not_met > inclusion_met       → NOT_MET
nei_ratio > 0.50 OR inc_ratio < 0.30   → NEI
inclusion_met > 0                       → MET
else                                    → NEI
```

---

## Results

### TREC 2021 — topic 1, high-recall mode (single topic benchmark)

| Metric | Score |
|---|---|
| T1 Recall@20 | 0.0533 |
| T2 Micro-F1 | 0.5472 |
| T3 NDCG@10 | **0.2916** |
| MAP | 0.1949 |
| **Composite** | **0.2477** |
| LLM cost | $0.09 |
| Runtime | 434s |

### TREC 2021 — hybrid mode, 10 topics

| Metric | Score |
|---|---|
| T1 Recall@20 | 0.0646 |
| T2 Micro-F1 | 0.4216 |
| T3 NDCG@10 | 0.3925 |
| MAP | 0.1714 |
| **Composite** | **0.2375** |

### TREC 2022 — hybrid mode, 5 topics (validation, unseen)

| Metric | Score |
|---|---|
| T2 Micro-F1 | 0.4653 |
| T3 NDCG@10 | 0.2354 |
| **Composite** | **0.2017** |

Generalization gap: 0.0358 (< 0.05 — system generalizes well to unseen patient profiles).

### Baseline Progression (TREC 2021)

| Configuration | Composite | vs BM25 |
|---|---|---|
| BM25 baseline (Day 1) | 0.029 | — |
| + Query Builder + MeSH | 0.176 | +507% |
| + TREC Index retrieval | 0.228 | +686% |
| + Hybrid + NEI gate | 0.252 | +769% |
| + high-recall mode | 0.248 | +755% |

---

## Error Analysis

### Retrieval failures (T1/T3)
- Root cause: CT API live mode (`combined`) retrieves 250 trials from live search, missing ~85% of relevant trials judged in TREC qrels. TREC qrels have 407 judged trials for topic 1; live API returns 250 with only 7/47 truly eligible trials.
- Fix: Use `--retrieval-mode high-recall` (indexes full 26k TREC corpus).

### NEI overgeneration
- Some trials receive NEI on criteria that could be evaluated. Arises when the patient note omits relevant history that would let the LLM make a decision.
- Not a bug — this is the correct conservative behavior. The generated clinical question surfaces it.

### Exclusion false positives
- Exclusion criteria mentioning condition names (e.g., "No prior NSCLC") cause BM25 false positives for NSCLC patients. Mitigated by down-weighting exclusion field (weight=0.3) in FieldBM25.

### Top-1 ranking errors
- Trial NCT00876499 ranked #1 for topic 1 in high-recall mode (grade=0) despite score=0.536.
- Cause: Trial had 5/5 inclusion criteria MET, zero exclusion violations, but TREC assessors judged it not relevant — mismatch between structured eligibility and clinical appropriateness.

---

## Limitations

1. **Recall ceiling at 250:** Only 250 candidates evaluated per topic. With 407 judged trials per topic on average, ~40% of relevant trials are structurally unreachable.
2. **LLM hallucination on complex criteria:** Multi-conditional criteria ("At least 2 of the following 5...") may be mis-evaluated.
3. **Temporal logic not handled:** Criteria like "progression within 6 months of last treatment" are classified as NEI (correctly conservative but reduces MET recall).
4. **Numeric lab extraction:** Regex-based; fails on non-standard notation (">ULN", "within normal limits").
5. **Live CT API mode:** Blocked by Cloudflare when using httpx. Must use `requests`.
6. **BioBERT model size:** 77MB embeddings file required for semantic/hybrid/high-recall modes.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| HTTP 400 on CT API | Query too long (>4 words in query.term) | Use `--retrieval-mode trec-index` or `high-recall` |
| HTTP 429 on OpenAI | Rate limit hit | Reduce `--max-workers` to 3 |
| `if cache:` returns False | diskcache truthiness bug | Always use `cache.get(key) is not None` |
| Low T1 Recall@20 | Wrong retrieval mode | Switch to `--retrieval-mode high-recall` |
| RuntimeError on MPS | torch/MPS incompatibility | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Empty predictions for topic | No cached trials in trial_data | Run with `--no-cache` once to populate |
| Budget exceeded | LLM cost > $1.75 | Increase `--budget 5.0` or pre-warm cache |

---

## Test Run Outputs

This repository includes real outputs from runs executed during development.
All results are in `data/` and ready to inspect.

### `data/predictions/`

| File | Description |
|---|---|
| `full_pipeline_2021.json` | TREC 2021 — Topic 1, high-recall mode, 250 candidates. T1=0.053, T2=0.547, T3=0.292, Composite=0.248. |
| `full_pipeline_2022.json` | TREC 2022 — validation set run. |
| `batch_predictions.json` | **Batch prediction demo** — 3 fictional patients (`data/example_patients.json`), 50 candidates each, high-recall mode. $0.21, ~20 min. |

### `data/runs/`

TREC-format run files (used by `pytrec_eval` for official metric computation):

| File | Description |
|---|---|
| `full_pipeline_2021.txt` | TREC 2021 run file — 250 lines (1 topic × 250 ranked trials) |
| `full_pipeline_2022.txt` | TREC 2022 run file |

Format: `<topic_id> Q0 <nct_id> <rank> <score> clinical_agent`

### `data/dossiers/`

Structured per-trial clinical dossiers — one JSON file per (topic, NCT ID) pair.

```
data/dossiers/
├── 2021/               ← TREC 2021 benchmark dossiers
│   ├── 1/              ← Topic 1 (anaplastic astrocytoma patient)
│   │   ├── NCT00031538.json
│   │   ├── NCT00613093.json
│   │   └── ... (top-10 trials per topic)
│   ├── 2/              ← Topic 2
│   └── ...
├── 2022/               ← TREC 2022 benchmark dossiers
└── batch/              ← Batch prediction dossiers (from predict.py)
    ├── P001/           ← Patient P001 (astrocytoma)
    │   ├── NCT00362570.json
    │   └── ...
    ├── P002/           ← Patient P002 (NSCLC EGFR+)
    └── P003/           ← Patient P003 (mCRC KRAS WT)
```

Each dossier contains:

```json
{
  "nct_id": "NCT00362570",
  "trial_title": "Temozolomide in Patients With Newly Diagnosed Anaplastic Oligodendroglioma",
  "phase": "PHASE2",
  "status": "RECRUITING",
  "overall_recommendation": "ELIGIBLE_PENDING_CLARIFICATION",
  "confidence_score": 0.75,
  "attention_flags": ["1 NEI verdict requires clinical clarification"],
  "eligibility_table": [
    {
      "criterion_type": "inclusion",
      "criterion_text": "Histologically confirmed anaplastic oligodendroglioma",
      "verdict": "NEI",
      "confidence": 0.4,
      "justification": "Patient has anaplastic astrocytoma; oligodendroglioma subtype not confirmed.",
      "clinical_question": "Has the pathology report confirmed oligodendroglioma histology specifically?"
    },
    {
      "criterion_type": "inclusion",
      "criterion_text": "Age 18 years or older",
      "verdict": "MET",
      "confidence": 1.0,
      "justification": "Patient is 45 years old.",
      "clinical_question": null
    }
  ]
}
```

### `data/example_patients.json`

Three ready-to-use fictional patient profiles for testing `predict.py`:

| ID | Profile |
|---|---|
| P001 | 45M, anaplastic astrocytoma, ECOG 2, on dexamethasone + levetiracetam |
| P002 | 58F, NSCLC stage IV, EGFR exon 19 del, post-erlotinib progression |
| P003 | 67M, metastatic CRC, KRAS WT, post-FOLFOX + FOLFIRI, MSS |
